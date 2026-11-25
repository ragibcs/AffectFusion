#!/usr/bin/env python3
"""
AffectFusion: Multimodal Bangla Text + Emoji Dataset Scraper & Preprocessor
Thesis FYDP: AffectFusion (Daffodil International University)
Authors: MD. RAGIB HASAN (232-15-775), MD. ASADULLA HEL GALIB (232-15-270)
Supervisor: Sharmin Akter, Co-Supervisor: Dewan Mamun Raza

Platforms supported:
1. YouTube Comments (Bengali news, drama, music, cricket reactions)
2. Reddit (r/bangladesh, r/kolkata, r/Dhaka)
3. Twitter / X (Bangla keywords, hashtags, viral topics)
4. Facebook Public Comments (News portals, public posts)

Preprocesses data directly into the exact schema of Bangla_Train.csv.
"""

import os
import re
import csv
import json
import argparse
from typing import List, Dict, Any, Optional
import emoji
from apify_client import ApifyClient

APIFY_TOKEN = os.getenv("APIFY_TOKEN", "YOUR_APIFY_TOKEN_HERE")

# Standard 10 Emotion Columns for AffectFusion (GoEmotions aligned + Sarcasm)
EMOTIONS = [
    "Joy", "Sadness", "Anger", "Fear", "Surprise",
    "Disgust", "Love", "Anticipation", "Sarcasm", "Neutral"
]

CSV_HEADERS = [
    "ID", "Data", "text_only", "emoji_stream", "has_emoji",
    "emoji_count", "word_count"
] + EMOTIONS + [
    "label_source", "annotator_agreement", "sarcasm_or_contradiction_flag"
]

def clean_and_anonymize_text(text: str) -> str:
    """Strips URLs, usernames, mentions, emails, and PII."""
    if not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"/?u/[\w_-]+", "", text)
    text = re.sub(r"@[\w_.-]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_bengali_text(text: str, min_chars: int = 2) -> bool:
    """Checks if text contains Bengali Unicode characters (ঀ-৿)."""
    bengali_chars = re.findall(r"[ঀ-৿]", text)
    return len(bengali_chars) >= min_chars

def extract_features(raw_text: str, row_id: int, label_source: str = "raw_scraped") -> Optional[Dict[str, Any]]:
    """Extracts text_only, emoji_stream, counts, and initializes emotion columns."""
    cleaned = clean_and_anonymize_text(raw_text)
    if not cleaned or len(cleaned) < 4:
        return None

    if not is_bengali_text(cleaned):
        return None

    emoji_entries = emoji.emoji_list(cleaned)
    emojis = [entry["emoji"] for entry in emoji_entries]

    emoji_stream = ",".join(emojis) if emojis else ""
    has_emoji = 1 if len(emojis) > 0 else 0
    emoji_count = len(emojis)

    text_only = emoji.replace_emoji(cleaned, "").strip()
    text_only = re.sub(r"\s+", " ", text_only)

    word_count = len(text_only.split()) if text_only else 0
    if word_count < 1:
        return None

    row = {
        "ID": row_id,
        "Data": cleaned,
        "text_only": text_only,
        "emoji_stream": emoji_stream,
        "has_emoji": has_emoji,
        "emoji_count": emoji_count,
        "word_count": word_count,
        "Joy": 0,
        "Sadness": 0,
        "Anger": 0,
        "Fear": 0,
        "Surprise": 0,
        "Disgust": 0,
        "Love": 0,
        "Anticipation": 0,
        "Sarcasm": 0,
        "Neutral": 0,
        "label_source": label_source,
        "annotator_agreement": "",
        "sarcasm_or_contradiction_flag": 0
    }
    return row

def scrape_youtube(client: ApifyClient, video_urls: List[str], max_comments: int = 200, budget_cap: float = 0.50) -> List[str]:
    """Scrapes YouTube video comments (high density of Bangla text + emojis)."""
    print(f"[*] Scraping YouTube: {len(video_urls)} video(s), max_comments={max_comments}, budget_cap=${budget_cap}")
    run_input = {
        "startUrls": [{"url": u} for u in video_urls],
        "maxComments": max_comments
    }
    run = client.actor("streamers/youtube-comments-scraper").call(
        run_input=run_input,
        max_total_charge_usd=budget_cap
    )
    texts = []
    if run and run.status == "SUCCEEDED":
        dataset = client.dataset(run.default_dataset_id)
        for item in dataset.iterate_items():
            content = item.get("comment") or item.get("text")
            if content:
                texts.append(content)
    print(f"[+] Retrieved {len(texts)} comments from YouTube.")
    return texts

def scrape_reddit(client: ApifyClient, subreddits: List[str], max_items: int = 50, budget_cap: float = 0.50) -> List[str]:
    """Scrapes Reddit posts & comments."""
    print(f"[*] Scraping Reddit: Subreddits={subreddits}, max_items={max_items}, budget_cap=${budget_cap}")
    start_urls = [{"url": f"https://www.reddit.com/r/{sub}/hot/"} for sub in subreddits]
    run_input = {
        "startUrls": start_urls,
        "maxItems": max_items,
        "maxComments": max_items * 2,
        "scrapeComments": True
    }
    run = client.actor("trudax/reddit-scraper-lite").call(
        run_input=run_input,
        max_total_charge_usd=budget_cap
    )
    texts = []
    if run and run.status == "SUCCEEDED":
        dataset = client.dataset(run.default_dataset_id)
        for item in dataset.iterate_items():
            if item.get("title"):
                texts.append(item["title"])
            if item.get("selfText"):
                texts.append(item["selfText"])
            if item.get("body"):
                texts.append(item["body"])
    print(f"[+] Retrieved {len(texts)} posts/comments from Reddit.")
    return texts

def scrape_twitter(client: ApifyClient, search_terms: List[str], max_tweets: int = 100, budget_cap: float = 0.50) -> List[str]:
    """Scrapes Twitter/X tweets for Bangla keywords."""
    print(f"[*] Scraping Twitter/X: Queries={search_terms}, max_tweets={max_tweets}, budget_cap=${budget_cap}")
    run_input = {
        "searchTerms": search_terms,
        "maxTweets": max_tweets,
        "sort": "Latest"
    }
    run = client.actor("apidojo/tweet-scraper").call(
        run_input=run_input,
        max_total_charge_usd=budget_cap
    )
    texts = []
    if run and run.status == "SUCCEEDED":
        dataset = client.dataset(run.default_dataset_id)
        for item in dataset.iterate_items():
            text = item.get("text") or item.get("full_text")
            if text:
                texts.append(text)
    print(f"[+] Retrieved {len(texts)} tweets from Twitter/X.")
    return texts

def scrape_facebook_comments(client: ApifyClient, post_urls: List[str], max_comments: int = 100, budget_cap: float = 0.50) -> List[str]:
    """Scrapes public Facebook post comments."""
    print(f"[*] Scraping Facebook: {len(post_urls)} post(s), max_comments={max_comments}, budget_cap=${budget_cap}")
    run_input = {
        "startUrls": [{"url": u} for u in post_urls],
        "resultsLimit": max_comments
    }
    run = client.actor("apify/facebook-comments-scraper").call(
        run_input=run_input,
        max_total_charge_usd=budget_cap
    )
    texts = []
    if run and run.status == "SUCCEEDED":
        dataset = client.dataset(run.default_dataset_id)
        for item in dataset.iterate_items():
            text = item.get("text") or item.get("message")
            if text:
                texts.append(text)
    print(f"[+] Retrieved {len(texts)} comments from Facebook.")
    return texts

def save_to_csv(rows: List[Dict[str, Any]], output_file: str, append: bool = True):
    """Saves or appends processed rows to dataset CSV."""
    file_exists = os.path.exists(output_file)
    mode = "a" if append and file_exists else "w"

    existing_count = 0
    if file_exists and append:
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_count = max(0, sum(1 for _ in reader) - 1)

    with open(output_file, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if mode == "w" or not file_exists:
            writer.writeheader()

        for i, row in enumerate(rows):
            row["ID"] = existing_count + i + 1
            writer.writerow(row)

    print(f"[+] Saved {len(rows)} samples to {output_file} (Total in dataset: {existing_count + len(rows)})")

def main():
    parser = argparse.ArgumentParser(description="AffectFusion Thesis Data Scraper & Preprocessor")
    parser.add_argument("--source", choices=["youtube", "reddit", "twitter", "facebook"], required=True, help="Platform to scrape")
    parser.add_argument("--urls", nargs="+", default=[], help="Target URLs (YouTube video URLs or Facebook post URLs)")
    parser.add_argument("--subreddits", nargs="+", default=["bangladesh", "kolkata", "Dhaka"], help="Subreddits to scrape")
    parser.add_argument("--queries", nargs="+", default=["বাংলাদেশ", "খেলা", "বিনোদন", "রাজনীতি"], help="Twitter search terms")
    parser.add_argument("--max-items", type=int, default=100, help="Max items to fetch per run")
    parser.add_argument("--budget-cap", type=float, default=0.50, help="Max USD budget cap per actor run (min $0.50 on Apify)")
    parser.add_argument("--output", type=str, default="Bangla_Scraped_Dataset.csv", help="Output CSV path")
    parser.add_argument("--emoji-only", action="store_true", help="Keep only samples containing at least 1 emoji")

    args = parser.parse_args()
    client = ApifyClient(APIFY_TOKEN)

    raw_texts = []
    if args.source == "youtube":
        if not args.urls:
            print("[!] Error: Please provide YouTube video URLs with --urls")
            return
        raw_texts = scrape_youtube(client, args.urls, max_comments=args.max_items, budget_cap=args.budget_cap)
    elif args.source == "reddit":
        raw_texts = scrape_reddit(client, args.subreddits, max_items=args.max_items, budget_cap=args.budget_cap)
    elif args.source == "twitter":
        raw_texts = scrape_twitter(client, args.queries, max_tweets=args.max_items, budget_cap=args.budget_cap)
    elif args.source == "facebook":
        if not args.urls:
            print("[!] Error: Please provide Facebook post URLs with --urls")
            return
        raw_texts = scrape_facebook_comments(client, args.urls, max_comments=args.max_items, budget_cap=args.budget_cap)

    print(f"[*] Processing and filtering {len(raw_texts)} raw texts for Bangla + Emoji semantics...")
    processed_rows = []
    seen_texts = set()

    for text in raw_texts:
        res = extract_features(text, row_id=len(processed_rows) + 1)
        if res:
            if args.emoji_only and not res["has_emoji"]:
                continue
            norm = res["text_only"]
            if norm not in seen_texts:
                seen_texts.add(norm)
                processed_rows.append(res)

    print(f"[+] Valid Bangla samples extracted: {len(processed_rows)}")
    if processed_rows:
        emoji_count = sum(1 for r in processed_rows if r["has_emoji"])
        print(f"    - With emojis: {emoji_count} ({emoji_count/len(processed_rows)*100:.1f}%)")
        print(f"    - Text-only: {len(processed_rows) - emoji_count}")
        save_to_csv(processed_rows, args.output, append=True)
    else:
        print("[-] No matching Bangla samples found in this batch.")

if __name__ == "__main__":
    main()
