

YouTube Comment Sentiment Analyzer (Prototype)

A simple CLI tool that fetches YouTube comments, cleans them, summarizes discussions using a Transformer model, and performs sentiment analysis.

Built as a learning prototype—may have limitations with API quota, disabled comments, or restricted videos.

⸻

Features
	•	Fetch YouTube comments via YouTube Data API v3
	•	Clean & dedupe noisy comments
	•	Summarize threads using Google Pegasus
	•	Sentiment analysis with DistilBERT
	•	Console report with sentiment breakdown & example comments

⸻

Installation

git clone https://github.com/YOUR_USERNAME/YouTube-Comment-Sentiment-Analyzer-Prototype-.git
cd YouTube-Comment-Sentiment-Analyzer-Prototype-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Add your API key to a .env file:

YOUTUBE_API_KEY=YOUR_KEY_HERE


⸻

Usage

python v2.py

Follow the prompts:
	•	Paste a YouTube URL
	•	Enter comment count

⸻

Limitations
	•	Doesn’t work on videos with comments disabled
	•	API quotas may cause 403 errors
	•	Summarization may be slow on very large threads

⸻

Project Structure

v2.py
v1.py
hdh.py
requirements.txt
.gitignore


