#!/usr/bin/env python3
# file: yt_comments_insights_refactor.py
"""
YouTube Comments Insights (refactor)
- Separate cleaning for summarization vs sentiment.
- Token-based chunking for BART to avoid over-cleaning/garbling.
- Light dedupe & noise reduction.
- Slow crawl: fetch ONE page per poll, sleeping 10s between pages.

Usage:
    python yt_comments_insights_refactor.py

Requirements:
    pip install transformers torch requests tqdm
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Dict, List, Sequence, Tuple, Optional, Union
from urllib.parse import parse_qs, urlparse


import requests
import random

# Optional progress bar for long NLP steps (not used for download now)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from transformers import AutoTokenizer, pipeline


# =========================
# 🔑 API key
# =========================
# .env support (optional)
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# =========================
# 🌐 HTTP config (timeouts/retries)
# =========================
API_TIMEOUT = int(os.getenv("YOUTUBE_HTTP_TIMEOUT", "60"))      # seconds
API_MAX_RETRIES = int(os.getenv("YOUTUBE_HTTP_RETRIES", "6"))   # attempts
API_BACKOFF_BASE = float(os.getenv("YOUTUBE_HTTP_BACKOFF", "1.8"))  # exponential base
API_BACKOFF_CAP = int(os.getenv("YOUTUBE_HTTP_BACKOFF_CAP", "60"))  # max sleep seconds per retry

class YouTubeAPIError(RuntimeError):
    pass

def http_get_json(url: str, params: dict, timeout: Union[int, float] = API_TIMEOUT) -> dict:
    """GET with retries & exponential backoff + jitter. Raises YouTubeAPIError on failure."""
    last_err = None
    for attempt in range(API_MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            # Handle quota / rate limiting explicitly
            if r.status_code in (429, 503):
                # fall through to backoff
                last_err = RuntimeError(f"HTTP {r.status_code}: possible rate limit. Body: {r.text[:200]}")
            else:
                r.raise_for_status()
                data = r.json()
                # If API returns structured error
                if isinstance(data, dict) and "error" in data:
                    err = data["error"]
                    # Extract reason if present
                    reason = None
                    try:
                        reason = err.get("errors", [{}])[0].get("reason")
                    except Exception:
                        reason = None
                    # Retry on rate/availability errors; otherwise raise
                    if reason in {"rateLimitExceeded", "quotaExceeded", "backendError"}:
                        last_err = RuntimeError(f"YouTube API error: {reason}")
                    else:
                        raise YouTubeAPIError(f"YouTube API error: {err}")
                else:
                    return data
        except requests.exceptions.ReadTimeout as e:
            last_err = e
        except requests.exceptions.RequestException as e:
            last_err = e

        # Backoff with jitter
        sleep_s = min(API_BACKOFF_CAP, (API_BACKOFF_BASE ** attempt)) + random.uniform(0, 0.5)
        print(f"Retrying in ~{sleep_s:.1f}s (attempt {attempt + 1}/{API_MAX_RETRIES})...")
        time.sleep(sleep_s)

    raise YouTubeAPIError(f"HTTP GET failed after {API_MAX_RETRIES} retries. Last error: {last_err}")

# WARNING: Do not hardcode API keys in real projects.
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") or "AIzaSyCqxoq-Vvf6kWjwSp1nG8Td2cRS-s7X0WY"

# =========================
# 🔧 Transformers pipelines
# =========================
_SUMMARIZER_MODEL = "google/pegasus-cnn_dailymail"
_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def get_summarizer():
    return pipeline("summarization", model=_SUMMARIZER_MODEL)


def get_sentiment_analyzer():
    return pipeline("sentiment-analysis", model=_SENTIMENT_MODEL)


def get_summarizer_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(_SUMMARIZER_MODEL, use_fast=True)


# =========================
# 🧩 URL utils
# =========================
def extract_video_id(youtube_url: str) -> str:
    """Extract a video ID from common YouTube URL formats."""
    try:
        u = urlparse(youtube_url)
        if u.netloc in ("youtu.be", "www.youtu.be"):
            vid = u.path.strip("/")
            if vid:
                return vid
        if u.path.startswith("/shorts/"):
            return u.path.split("/shorts/")[1].split("/")[0]
        qs = parse_qs(u.query)
        if "v" in qs:
            return qs["v"][0]
    except Exception:
        pass
    raise ValueError("Could not extract a video ID from that URL. Please paste a full YouTube link.")


# =========================
# 📥 YouTube Data API calls
# =========================
def fetch_video_title(video_id: str) -> str:
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {"part": "snippet", "id": video_id, "key": YOUTUBE_API_KEY}
    data = http_get_json(url, params, timeout=API_TIMEOUT)
    items = data.get("items", [])
    if not items:
        return "(title unavailable)"
    return items[0]["snippet"].get("title", "(title unavailable)")


def fetch_comments_page(
    video_id: str,
    page_token: Optional[str] = None,
    order: str = "relevance",
    max_results: int = 100,
) -> Tuple[List[str], Optional[str]]:
    """Fetch a *single* page of up to `max_results` top-level comments.
    Returns (comments, next_page_token). Use in a slow polling loop.
    """
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max(1, min(max_results, 100)),
        "textFormat": "plainText",
        "key": YOUTUBE_API_KEY,
        "order": order,
    }
    if page_token:
        params["pageToken"] = page_token

    data = http_get_json(base_url, params, timeout=API_TIMEOUT)

    items = data.get("items", [])
    comments: List[str] = []
    for it in items:
        try:
            txt = it["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
        except KeyError:
            continue
        comments.append(txt)

    return comments, data.get("nextPageToken")


# =========================
# 🧼 Cleaning (dual pipelines)
# =========================
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_HANDLE_RE = re.compile(r"[@#]\w+")
_MULTI_WS_RE = re.compile(r"\s+")
_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_NON_ALNUM_SOFT_RE = re.compile(r"[^a-zA-Z0-9\s\.,!?:;\-()'\"]+")


def clean_for_summary(s: str) -> str:
    s = s.strip()
    s = _URL_RE.sub(" ", s)
    s = _TIMESTAMP_RE.sub(" ", s)
    s = _MULTI_WS_RE.sub(" ", s)
    return s.strip()


def clean_for_sentiment(s: str) -> str:
    s = s.strip().lower()
    s = _URL_RE.sub(" ", s)
    s = _HANDLE_RE.sub(" ", s)
    s = _TIMESTAMP_RE.sub(" ", s)
    s = _NON_ALNUM_SOFT_RE.sub(" ", s)
    s = _MULTI_WS_RE.sub(" ", s)
    return s.strip()


# =========================
# 🔎 Summary quality helpers
# =========================
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
_NONWORD_RATIO_CUTOFF = 0.6  # if >60% of chars are non-letters, drop for summary

def is_informative_for_summary(s: str) -> bool:
    """Heuristic filter to keep comments that add information for summarization.
    (Sentiment pipeline should still use lightly-cleaned full set.)"""
    if not s:
        return False
    t = _EMOJI_RE.sub("", s)
    t = _MULTI_WS_RE.sub(" ", t).strip()
    if len(t) < 20:  # too short → likely noise like 'still amazing', '🔥'
        return False
    letters = sum(ch.isalpha() for ch in t)
    nonletters = max(1, len(t) - letters)
    if nonletters / (letters + nonletters) > _NONWORD_RATIO_CUTOFF:
        return False
    low = t.lower()
    # common low-signal fragments in YT threads
    bad_fragments = ("subscribe", "check my channel")
    if any(k in low for k in bad_fragments) and len(t) < 60:
        return False
    return True

def dedupe_loose(lines: Sequence[str]) -> List[str]:
    """Collapse near-duplicates (e.g., 'sooooo goooood' vs 'soo good')."""
    seen = set()
    out: List[str] = []
    for c in lines:
        # normalize by removing non-word chars and lowering
        norm = re.sub(r"\W+", " ", c.lower()).strip()
        # trim long char repeats: "sooooo" -> "soo"
        norm = re.sub(r"(.)\1{2,}", r"\1\1", norm)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(c)
    return out


# =========================
# ✂️ Token-based chunking for BART
# =========================
def split_by_tokens(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> List[str]:
    """Split text into chunks under max_tokens at sentence-ish boundaries."""
    if not text:
        return []
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []

    def token_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    cur_tokens = 0
    for sent in sentences:
        tlen = token_len(sent)
        if tlen >= max_tokens:
            words = sent.split()
            tmp: List[str] = []
            for w in words:
                if token_len(" ".join(tmp + [w])) >= max_tokens:
                    if tmp:
                        chunks.append(" ".join(tmp))
                        tmp = []
                tmp.append(w)
            if tmp:
                chunks.append(" ".join(tmp))
            continue

        if cur_tokens + tlen > max_tokens:
            if current:
                chunks.append(" ".join(current))
            current = [sent]
            cur_tokens = tlen
        else:
            current.append(sent)
            cur_tokens += tlen

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


# =========================
# 📝 Summarization (map-reduce)
# =========================
# Tuning for map-reduce summarization
SUM_FINAL_MAX_WORDS = 150
MAP_MAX_LEN = 160
MAP_MIN_LEN = 60
REDUCE_MAX_LEN = 120
REDUCE_MIN_LEN = 60

def _summarize_chunk(summarizer, text: str, max_len: int, min_len: int) -> str:
    return summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,            # deterministic
        num_beams=4,                # more coherent than greedy
        no_repeat_ngram_size=3,     # reduce repetitive phrasing
        length_penalty=1.0,
    )[0]["summary_text"]


def summarize_comments(comments: Sequence[str], max_words: int = SUM_FINAL_MAX_WORDS) -> str:
    if not comments:
        return "No comments available to summarize."

    # Heavier dedupe to avoid echo-chamber lines
    deduped = dedupe_loose(comments)
    if not deduped:
        return "Not enough textual content to summarize."

    text = " ".join(deduped)
    summarizer = get_summarizer()
    tokenizer = get_summarizer_tokenizer()

    # Keep chunks under the model's limit with a safety margin
    input_limit = min(getattr(tokenizer, "model_max_length", 1024), 1024) - 64
    chunks = split_by_tokens(text, tokenizer, input_limit)
    if not chunks:
        return "Not enough textual content to summarize."

    # Map step
    partials: List[str] = []
    for ch in tqdm(chunks, desc="Summarizing (map)"):
        try:
            partials.append(_summarize_chunk(summarizer, ch, MAP_MAX_LEN, MAP_MIN_LEN))
        except Exception:
            continue

    if not partials:
        return "Summarization failed for all chunks."

    # Reduce step
    reduce_text = " ".join(partials)
    reduce_chunks = split_by_tokens(reduce_text, tokenizer, input_limit)
    finals: List[str] = []
    for ch in reduce_chunks:
        try:
            finals.append(_summarize_chunk(summarizer, ch, REDUCE_MAX_LEN, REDUCE_MIN_LEN))
        except Exception:
            continue
    if not finals:
        finals = partials[:3]

    final = " ".join(finals)
    words = final.split()
    if len(words) > max_words:
        final = " ".join(words[:max_words]) + "..."
    return final


# =========================
# 🙂 Sentiment
# =========================
def analyze_sentiment(comments: Sequence[str], neutral_threshold: float = 0.60) -> Dict:
    if not comments:
        return {"total": 0, "pos": 0, "neg": 0, "neu": 0, "overall": 0.0, "top_pos": [], "top_neg": []}

    analyzer = get_sentiment_analyzer()

    B = 64
    labels_scores: List[Dict] = []
    for i in tqdm(range(0, len(comments), B), desc="Running sentiment"):
        batch = list(comments[i : i + B])
        res = analyzer(batch, truncation=True)
        labels_scores.extend(res)

    pos = neg = neu = 0
    overall_scores: List[float] = []
    scored_examples: List[Tuple[float, str]] = []

    for txt, r in zip(comments, labels_scores):
        label = r["label"].upper()
        conf = float(r["score"])
        if conf < neutral_threshold:
            neu += 1
            signed = 0.0
        elif label == "POSITIVE":
            pos += 1
            signed = +conf
        else:
            neg += 1
            signed = -conf
        overall_scores.append(signed)
        scored_examples.append((signed, txt))

    scored_examples.sort(key=lambda x: x[0], reverse=True)
    top_pos = [t for s, t in scored_examples if s > 0][:3]
    top_neg = [t for s, t in reversed(scored_examples) if s < 0][:3]

    overall = sum(overall_scores) / max(len(overall_scores), 1)

    return {
        "total": len(comments),
        "pos": pos,
        "neg": neg,
        "neu": neu,
        "overall": round(overall, 4),
        "top_pos": top_pos,
        "top_neg": top_neg,
    }


# =========================
# 📊 Reporting (console only)
# =========================
def pretty_print_report(title: str, n_requested: int, n_fetched: int, summary: str, sent: Dict) -> None:
    print("\n" + "=" * 72)
    print(f"🎬  Video: {title}")
    print(f"💬  Comments requested: {n_requested}  |  fetched: {n_fetched}")
    print("-" * 72)
    print("🧾 Summary of viewer comments:")
    print(summary)
    print("-" * 72)
    print("🧠 Sentiment overview:")
    total = sent["total"] or 1

    def pct(x: int) -> str:
        return f"{(100.0 * x / total):.1f}%"

    print(f"  • Positive: {sent['pos']} ({pct(sent['pos'])})")
    print(f"  • Neutral:  {sent['neu']} ({pct(sent['neu'])})")
    print(f"  • Negative: {sent['neg']} ({pct(sent['neg'])})")
    print(f"  • Overall score [-1..1]: {sent['overall']}")
    if sent["top_pos"]:
        print("\n  🌟 Top positive examples:")
        for i, t in enumerate(sent["top_pos"], 1):
            print(f"    {i}. {t[:200]}{'...' if len(t) > 200 else ''}")
    if sent["top_neg"]:
        print("\n  ⚠️  Top negative examples:")
        for i, t in enumerate(sent["top_neg"], 1):
            print(f"    {i}. {t[:200]}{'...' if len(t) > 200 else ''}")
    print("=" * 72)


# =========================
# 🏃 Main
# =========================
def main() -> None:
    if not YOUTUBE_API_KEY:
        print("Error: YOUTUBE_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        yt_url = input("Paste YouTube URL: ").strip()
        video_id = extract_video_id(yt_url)

        # Ask the user how many comments to analyze
        try:
            n = int(input("How many comments should I analyze? ").strip())
        except Exception:
            n = 200
            print("Invalid number; defaulting to 200.")
        n = max(1, n)

        order = "relevance"  # change to "time" for newest-first
        title = fetch_video_title(video_id)

        print(f"\nStarting collection for: {title}")
        print(f"Target: {n} comments. Polling YouTube every 10 seconds...\n")

        collected: List[str] = []
        seen_norm = set()
        next_token: Optional[str] = None

        while len(collected) < n:
            batch, next_token = fetch_comments_page(
                video_id, page_token=next_token, order=order, max_results=30
            )

            if not batch and next_token is None:
                print("No more pages from API.")
                break

            # Deduplicate across polls by a normalized key
            for c in batch:
                norm = re.sub(r"\W+", " ", c.lower()).strip()
                if norm and norm not in seen_norm:
                    seen_norm.add(norm)
                    collected.append(c)
                    if len(collected) >= n:
                        break

            print(f"Fetched so far: {len(collected)}/{n}")

            if len(collected) >= n:
                break
            if next_token is None:
                print("Reached the end of available comments before target.")
                break

            time.sleep(10)  # slow crawl: one page per 10s

        # Split cleaning per task
        summary_pool = [clean_for_summary(c) for c in collected if c and c.strip()]
        summary_pool = [c for c in summary_pool if is_informative_for_summary(c)]
        summary_pool = dedupe_loose(summary_pool)

        sentiment_ready = [clean_for_sentiment(c) for c in collected if c and c.strip()]
        sentiment_ready = [c for c in sentiment_ready if c]

        summary = summarize_comments(summary_pool, max_words=SUM_FINAL_MAX_WORDS)
        sent = analyze_sentiment(sentiment_ready, neutral_threshold=0.60)

        pretty_print_report(title, n, len(collected), summary, sent)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
