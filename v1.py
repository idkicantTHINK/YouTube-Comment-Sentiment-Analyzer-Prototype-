#!/usr/bin/env python3
# file: yt_comments_insights_refactor.py
"""
YouTube Comments Insights (refactor)
- Separate cleaning for summarization vs sentiment.
- Token-based chunking for BART to avoid over-cleaning/garbling.
- Light dedupe & noise reduction.
- Polls the YouTube API every 10s until the requested number of comments is reached.

Usage:
    python yt_comments_insights_refactor.py

Requirements:
    pip install transformers torch requests tqdm

Notes:
- Keep your API key secure; never commit it to a public repo.
- Summarizer: facebook/bart-large-cnn
- Sentiment: distilbert-base-uncased-finetuned-sst-2-english
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Dict, List, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import requests

# Optional progress bar
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

# WARNING: Do not hardcode API keys in real projects.
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") or "AIzaSyCqxoq-Vvf6kWjwSp1nG8Td2cRS-s7X0WY"

# =========================
# 🔧 Transformers pipelines
# =========================
_SUMMARIZER_MODEL = "facebook/bart-large-cnn"
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
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return "(title unavailable)"
    return items[0]["snippet"].get("title", "(title unavailable)")


def fetch_comments(video_id: str, target_n: int, order: str = "relevance") -> List[str]:
    """Fetch up to target_n top-level comments (plain text) in one pass (single crawl)."""
    comments: List[str] = []
    page_token = None
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": YOUTUBE_API_KEY,
        "order": order,
    }

    with tqdm(total=target_n, desc="Downloading comments") as pbar:
        while len(comments) < target_n:
            if page_token:
                params["pageToken"] = page_token
            else:
                params.pop("pageToken", None)

            r = requests.get(base_url, params=params, timeout=30)
            if r.status_code != 200:
                time.sleep(1.5)
                r = requests.get(base_url, params=params, timeout=30)
                r.raise_for_status()
            data = r.json()
            items = data.get("items", [])
            if not items:
                break

            for it in items:
                try:
                    txt = it["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
                except KeyError:
                    continue
                comments.append(txt)
                pbar.update(1)
                if len(comments) >= target_n:
                    break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    return comments


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

def summarize_comments(comments: Sequence[str], max_words: int = 150) -> str:
    if not comments:
        return "No comments available to summarize."

    seen = set()
    deduped: List[str] = []
    for c in comments:
        norm = re.sub(r"\W+", " ", c.lower()).strip()
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(c)

    text = " ".join(deduped)
    summarizer = get_summarizer()
    tokenizer = get_summarizer_tokenizer()

    input_limit = min(getattr(tokenizer, "model_max_length", 1024), 1024) - 64

    chunks = split_by_tokens(text, tokenizer, input_limit)
    if not chunks:
        return "Not enough textual content to summarize."

    partials: List[str] = []
    for ch in tqdm(chunks, desc="Summarizing (map)"):
        try:
            out = summarizer(ch, max_length=160, min_length=40, do_sample=False)
            partials.append(out[0]["summary_text"])
        except Exception:
            continue

    if not partials:
        return "Summarization failed for all chunks."

    reduce_text = " ".join(partials)
    reduce_chunks = split_by_tokens(reduce_text, tokenizer, input_limit)

    finals: List[str] = []
    for ch in reduce_chunks:
        try:
            out = summarizer(ch, max_length=120, min_length=40, do_sample=False)
            finals.append(out[0]["summary_text"])
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

        order = "relevance"  # keep simple; change to "time" if needed
        title = fetch_video_title(video_id)

        # Poll every 10 seconds until we reach the target count
        print(f"\nStarting collection for: {title}")
        print(f"Target: {n} comments. Polling YouTube every 10 seconds...\n")

        collected: List[str] = []
        seen_norm = set()

        while len(collected) < n:
            batch = fetch_comments(video_id, n, order=order)
            # Deduplicate across polls by a normalized key
            for c in batch:
                norm = re.sub(r"\W+", " ", c.lower()).strip()
                if norm and norm not in seen_norm:
                    seen_norm.add(norm)
                    collected.append(c)
                    if len(collected) >= n:
                        break

            print(f"Fetched so far: {len(collected)}/{n}")
            if len(collected) < n:
                time.sleep(10)

        # Split cleaning per task
        summary_ready = [clean_for_summary(c) for c in collected if c and c.strip()]
        summary_ready = [c for c in summary_ready if c]

        sentiment_ready = [clean_for_sentiment(c) for c in collected if c and c.strip()]
        sentiment_ready = [c for c in sentiment_ready if c]

        summary = summarize_comments(summary_ready, max_words=150)
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