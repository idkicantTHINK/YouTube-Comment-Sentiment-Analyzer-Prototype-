#!/usr/bin/env python3
"""
YouTube Comments Insights
- Paste a YouTube URL
- Enter how many comments (N) you want analyzed (fetches 100 per API call until N or no more)
- Cleans text, summarizes with a transformer, and runs sentiment analysis
- Prints a concise viewer-comments summary and sentiment breakdown

Usage:
    python yt_comments_insights.py

Notes:
- Keep your API key secure; never commit it to a public repo.
- Summarizer model: facebook/bart-large-cnn (robust news summarizer)
- Sentiment model: distilbert-base-uncased-finetuned-sst-2-english
"""

import os
import re
import sys
import time
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Tuple
import requests

# Optional progress bar (pip install tqdm); fallback if not installed
try:
    from tqdm import tqdm
except:
    def tqdm(x, **kwargs): return x

# =========================
# 🔑 Your YouTube API key
# =========================
YOUTUBE_API_KEY = os.getenv(
    "YOUTUBE_API_KEY",
    "AIzaSyCqxoq-Vvf6kWjwSp1nG8Td2cRS-s7X0WY"  # provided by user; consider setting env var instead
)

# =========================
# 🔧 Transformers pipelines
# =========================
from transformers import pipeline

def get_summarizer():
    # BART is strong at abstractive summarization of long-ish prose
    return pipeline("summarization", model="facebook/bart-large-cnn")

def get_sentiment_analyzer():
    # Returns POSITIVE or NEGATIVE + confidence; we’ll add a neutral band
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# =========================
# 🧩 Utility functions
# =========================
def extract_video_id(youtube_url: str) -> str:
    """
    Supports:
      - https://www.youtube.com/watch?v=VIDEOID
      - https://youtu.be/VIDEOID
      - https://www.youtube.com/shorts/VIDEOID
    """
    try:
        u = urlparse(youtube_url)
        if u.netloc in ("youtu.be", "www.youtu.be"):
            return u.path.strip("/")

        if u.path.startswith("/shorts/"):
            return u.path.split("/shorts/")[1].split("/")[0]

        qs = parse_qs(u.query)
        if "v" in qs:
            return qs["v"][0]
    except Exception:
        pass
    raise ValueError("Could not extract a video ID from that URL. Please paste a full YouTube link.")

def fetch_video_title(video_id: str) -> str:
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": video_id,
        "key": YOUTUBE_API_KEY
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return "(title unavailable)"
    return items[0]["snippet"]["title"]

def fetch_comments(video_id: str, target_n: int, order: str = "relevance") -> List[str]:
    """
    Fetches up to target_n top-level comments (plain text).
    Uses commentThreads endpoint with pagination, maxResults=100.
    """
    comments = []
    page_token = None
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": YOUTUBE_API_KEY,
        "order": order
    }

    with tqdm(total=target_n, desc="Downloading comments") as pbar:
        while len(comments) < target_n:
            if page_token:
                params["pageToken"] = page_token
            else:
                params.pop("pageToken", None)

            r = requests.get(base_url, params=params, timeout=30)
            if r.status_code != 200:
                # basic retry on quota/network hiccups
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

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
HANDLE_RE = re.compile(r"[@#]\w+")
MULTISPACE_RE = re.compile(r"\s+")
NON_PRINTABLE_RE = re.compile(r"[^a-zA-Z0-9\s\.\,\!\?\:\;\-\(\)\'\"]+")

def clean_text(s: str) -> str:
    s = s.strip()
    s = URL_RE.sub(" ", s)
    s = HANDLE_RE.sub(" ", s)
    s = NON_PRINTABLE_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s)
    return s.strip()

def chunk_joined_text(text: str, max_chars: int = 2800) -> List[str]:
    """
    Splits a long string at sentence-ish boundaries to respect model input limits.
    """
    if len(text) <= max_chars:
        return [text]
    chunks, cur = [], []
    cur_len = 0
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    for sent in sentences:
        if cur_len + len(sent) + 1 > max_chars:
            chunks.append(" ".join(cur).strip())
            cur, cur_len = [sent], len(sent) + 1
        else:
            cur.append(sent)
            cur_len += len(sent) + 1
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks

def summarize_comments(comments: List[str], max_words: int = 150) -> str:
    """
    Map-Reduce summarization:
      - join all cleaned comments
      - summarize in chunks
      - summarize the summaries
    """
    if not comments:
        return "No comments available to summarize."

    joined = " ".join(comments)
    chunks = chunk_joined_text(joined, max_chars=2800)
    summarizer = get_summarizer()

    partial_summaries = []
    for ch in tqdm(chunks, desc="Summarizing (map)"):
        summary = summarizer(
            ch,
            max_length=160,  # tokens, ~120-160 words
            min_length=40,
            do_sample=False
        )[0]["summary_text"]
        partial_summaries.append(summary)

    # Reduce step
    reduce_in = " ".join(partial_summaries)
    reduce_chunks = chunk_joined_text(reduce_in, max_chars=2800)

    reduce_summaries = []
    for ch in reduce_chunks:
        summary = summarizer(
            ch,
            max_length=120,
            min_length=40,
            do_sample=False
        )[0]["summary_text"]
        reduce_summaries.append(summary)

    final = " ".join(reduce_summaries)
    # Optional: gently trim to ~N words
    words = final.split()
    if len(words) > max_words:
        final = " ".join(words[:max_words]) + "..."
    return final

def analyze_sentiment(comments: List[str]) -> Dict:
    """
    Batches comments through a binary sentiment model, and:
      - counts positive/negative with a neutral band (confidence < 0.60)
      - computes an overall score in [-1, 1]
      - returns top 3 examples for each extreme
    """
    if not comments:
        return {"total": 0, "pos": 0, "neg": 0, "neu": 0, "overall": 0.0, "top_pos": [], "top_neg": []}

    analyzer = get_sentiment_analyzer()

    # Run in batches for speed and robustness
    B = 64
    labels_scores = []
    for i in tqdm(range(0, len(comments), B), desc="Running sentiment"):
        batch = comments[i:i+B]
        res = analyzer(batch, truncation=True)
        labels_scores.extend(res)

    pos = neg = neu = 0
    overall_scores = []
    scored_examples = []  # [(score_signed, text)]

    for txt, r in zip(comments, labels_scores):
        label = r["label"]
        conf = float(r["score"])

        # Define a neutral band
        if conf < 0.60:
            neu += 1
            signed = 0.0
        elif label.upper() == "POSITIVE":
            pos += 1
            signed = +conf
        else:
            neg += 1
            signed = -conf

        overall_scores.append(signed)
        scored_examples.append((signed, txt))

    # Top examples
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

def pretty_print_report(title: str, n_requested: int, n_fetched: int, summary: str, sent: Dict):
    print("\n" + "="*72)
    print(f"🎬  Video: {title}")
    print(f"💬  Comments requested: {n_requested}  |  fetched: {n_fetched}")
    print("-"*72)
    print("🧾 Summary of viewer comments:")
    print(summary)
    print("-"*72)
    print("🧠 Sentiment overview:")
    total = sent["total"] or 1
    def pct(x): return f"{(100.0 * x / total):.1f}%"
    print(f"  • Positive: {sent['pos']} ({pct(sent['pos'])})")
    print(f"  • Neutral:  {sent['neu']} ({pct(sent['neu'])})")
    print(f"  • Negative: {sent['neg']} ({pct(sent['neg'])})")
    print(f"  • Overall score [-1..1]: {sent['overall']}")
    if sent["top_pos"]:
        print("\n  🌟 Top positive examples:")
        for i, t in enumerate(sent["top_pos"], 1):
            print(f"    {i}. {t[:200]}{'...' if len(t)>200 else ''}")
    if sent["top_neg"]:
        print("\n  ⚠️  Top negative examples:")
        for i, t in enumerate(sent["top_neg"], 1):
            print(f"    {i}. {t[:200]}{'...' if len(t)>200 else ''}")
    print("="*72)

def main():
    try:
        yt_url = input("Paste YouTube URL: ").strip()
        video_id = extract_video_id(yt_url)

        n_str = input("How many comments should we analyze? (e.g., 100, 500): ").strip()
        try:
            n = max(1, int(n_str))
        except ValueError:
            print("Invalid number; defaulting to 100.")
            n = 100

        title = fetch_video_title(video_id)
        raw_comments = fetch_comments(video_id, n, order="relevance")
        if not raw_comments:
            print("No comments found or API returned none.")
            sys.exit(0)

        # Clean comments for both summarization and sentiment
        cleaned = [clean_text(c) for c in raw_comments if c and c.strip()]
        cleaned = [c for c in cleaned if len(c) > 0]

        summary = summarize_comments(cleaned, max_words=150)
        sent = analyze_sentiment(cleaned)
        pretty_print_report(title, n, len(raw_comments), summary, sent)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()