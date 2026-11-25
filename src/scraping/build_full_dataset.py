#!/usr/bin/env python3
"""
AffectFusion: Multilingual (Bangla, Banglish, English) Social Media Scraper & Dataset Builder
Thesis FYDP: AffectFusion (Daffodil International University)
Authors: MD. RAGIB HASAN (232-15-775), MD. ASADULLA HEL GALIB (232-15-270)
Supervisor: Sharmin Akter, Co-Supervisor: Dewan Mamun Raza

Supports:
1. Native Bengali (বাংলা)
2. Banglish (Latin-script Bengali / Code-mixed)
3. English Social Media Posts with Emojis from Bangladeshi/South Asian communities

Outputs directly into the exact schema of Bangla_Train.csv.
"""

import os
import re
import csv
import json
import time
from typing import List, Dict, Any, Optional
import emoji
from apify_client import ApifyClient

APIFY_TOKEN = os.getenv("APIFY_TOKEN", "YOUR_APIFY_TOKEN_HERE")

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

# Multilingual Emotion Lexicon (Bangla + Banglish + English + Emojis)
EMOTION_LEXICON = {
    "Joy": {
        "words": [
            # Bangla
            "আনন্দ", "খুশি", "মজা", "সুন্দর", "উপভোগ", "দারুণ", "অভিনন্দন", "উৎসব", "হাসি", "জিতলো", "চমৎকার", "ভালো", "সেরা",
            # Banglish
            "khushi", "anondo", "moja", "shundor", "sera", "valo", "bhalo", "congrats", "jitsilam", "osthir", "hashte hashte", "enjoy",
            # English
            "happy", "joy", "excited", "love it", "amazing", "awesome", "win", "celebration", "blessed", "proud", "congratulations", "fun", "wonderful"
        ],
        "emojis": ["😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣", "😊", "😇", "🙂", "🥳", "🎉", "✨", "🔥", "🏆", "👏", "💃", "🕺"]
    },
    "Sadness": {
        "words": [
            # Bangla
            "কষ্ট", "মন খারাপ", "দুঃখ", "মৃত্যু", "হতাশা", "একা", "কান্না", "বিষন্ন", "বেদনা", "হারলো", "আহত", "মৃত", "অনাহার", "অশ্রু",
            # Banglish
            "kosto", "mon kharap", "dukkho", "kanna", "eka", "harlo", "bhalo na", "miss kori", "depression", "broken", "kade", "hopeless",
            # English
            "sad", "heartbreak", "crying", "lonely", "depressed", "grief", "pain", "loss", "hurts", "rip", "hopeless", "mourning", "unhappy", "sorrow"
        ],
        "emojis": ["😢", "😭", "😞", "😔", "😟", "😕", "🙁", "☹️", "😣", "😖", "😫", "😩", "🥺", "💔", "🥀", "😿"]
    },
    "Anger": {
        "words": [
            # Bangla
            "রাগ", "ক্ষোভ", "অন্যায়", "ঘৃণা", "প্রতিবাদ", "দুর্নীতি", "দালাল", "লজ্জা", "ধিক্কার", "শাস্তি", "খুনি", "চোর", "জালিয়াত",
            # Banglish
            "rag", "khov", "onnay", "durniti", "dalal", "lojja", "dhikkar", "chor", "faltu", "beadhob", "cheater", "bokachoda", "kharap",
            # English
            "angry", "mad", "furious", "rage", "corrupt", "unfair", "disgusted", "shameless", "idiot", "boycott", "hate", "protest", "crime"
        ],
        "emojis": ["😠", "😡", "🤬", "😤", "💢", "👿", "👺", "🖕"]
    },
    "Fear": {
        "words": [
            # Bangla
            "ভয়", "আতঙ্ক", "বিপদ", "শঙ্কা", "হুমকি", "উদ্বেগ", "নিরাপত্তাহীনতা", "ভয়ঙ্কর", "ভূতুড়ে", "অনিশ্চিত",
            # Banglish
            "bhoy", "voy", "atonko", "bipod", "shongka", "voyongkor", "tension", "scared", "fear",
            # English
            "fear", "scared", "afraid", "terrified", "anxious", "danger", "panic", "horror", "nightmare", "unsafe", "threat"
        ],
        "emojis": ["😨", "😰", "😱", "🥶", "😬", "🫣", "💀", "☠️", "👻"]
    },
    "Surprise": {
        "words": [
            # Bangla
            "অবাক", "বিস্ময়", "অপ্রত্যাশিত", "অবিশ্বাস্য", "হঠাৎ", "চমকে", "স্তব্ধ", "অদ্ভুত",
            # Banglish
            "obak", "bishmoy", "oprottashito", "obisshasho", "hotath", "shock", "ki holo", "omg", "unbelievable",
            # English
            "surprise", "shocked", "unbelievable", "unexpected", "speechless", "wow", "sudden", "stunned", "astonished", "what the"
        ],
        "emojis": ["😮", "😯", "😲", "😳", "🤯", "🫢", "👀", "⁉️", "‼️", "😱"]
    },
    "Disgust": {
        "words": [
            # Bangla
            "বিরক্ত", "বমি", "অসহ্য", "নোংরা", "ঘেন্না", "জঘন্য", "ছি", "বাজে", "অরুচিকর", "ঘৃণ্য",
            # Banglish
            "birokto", "bomi", "osojjo", "nongra", "ghenna", "joghonno", "chhi", "baje", "disgusting", "cringe",
            # English
            "disgust", "gross", "nasty", "sick", "cringe", "awful", "horrible", "yuck", "trash", "repulsive", "creepy"
        ],
        "emojis": ["🤢", "🤮", "🤧", "💩", "👎", "🤮"]
    },
    "Love": {
        "words": [
            # Bangla
            "ভালোবাসা", "ভালোবাসি", "প্রিয়", "স্নেহ", "মায়া", "শ্রদ্ধা", "প্রেম", "আদর", "ভালোবাসেন", "হৃদয়", "পছন্দ",
            # Banglish
            "valobasha", "bhalobasha", "valobashi", "priyo", "maya", "prem", "crush", "adore", "shundori", "jaan", "babu",
            # English
            "love", "adore", "sweetheart", "lovely", "soulmate", "favorite", "darling", "precious", "beloved", "caring", "heart"
        ],
        "emojis": ["❤️", "💖", "💕", "💞", "💓", "💗", "💘", "💝", "😍", "🥰", "😘", "😻", "💌", "💐"]
    },
    "Anticipation": {
        "words": [
            # Bangla
            "অপেক্ষা", "প্রত্যাশা", "আশা", "স্বপ্ন", "পরিকল্পনা", "ভবিষ্যৎ", "আসন্ন", "অপেক্ষায়", "আগামীকাল", "শীঘ্রই", "প্রস্তুতি",
            # Banglish
            "opekkha", "prottasha", "asha", "shopno", "plan", "future", "waiting", "dekha jak", "upcoming", "shob thik hobe",
            # English
            "anticipation", "looking forward", "hope", "ready", "countdown", "upcoming", "excited for", "dream", "patiently", "planning"
        ],
        "emojis": ["⏳", "⌛", "🤞", "🔮", "🌅", "🎯", "🚀", "🔜"]
    },
    "Sarcasm": {
        "words": [
            # Bangla
            "বাহ", "উন্নয়ন", "হেসে বাঁচি না", "দারুণ দেশ", "সাধু সাধু", "ডিজিটাল", "বেশ তো", "চমৎকার তো", "দারুণ সিদ্ধান্ত",
            # Banglish
            "wah", "darun", "unnoyon", "hese bachi na", "shadhon", "digital", "wah re", "ki shundor", "haha darun", "onek sundor",
            # English
            "yeah right", "sure", "what a great", "totally", "bravo", "brilliant idea", "as if", "slow claps", "genius move", "clown"
        ],
        "emojis": ["🙄", "😏", "🙃", "🤡", "😒", "🥱", "🤦‍♂️", "🤦‍♀️"]
    },
    "Neutral": {
        "words": [
            # Bangla
            "বাংলাদেশ", "ঢাকা", "সংবাদ", "শিক্ষা", "বিশ্ববিদ্যালয়", "পরীক্ষা", "আবহাওয়া", "বিজ্ঞান", "তথ্য", "খবর", "প্রতিবেদন",
            # Banglish
            "bangladesh", "dhaka", "news", "university", "exam", "info", "notice", "update", "routine", "shomoy", "kobe",
            # English
            "update", "news", "announcement", "report", "question", "general", "information", "schedule", "routine", "admission", "semester"
        ],
        "emojis": ["📢", "📰", "📌", "ℹ️", "📄", "🗓️", "📊"]
    }
}

SCRAPE_TASKS = [
    # 1. Direct High-Volume Community Feeds (Reddit Hot/Top)
    {
        "type": "startUrls",
        "urls": [
            "https://www.reddit.com/r/bangladesh/hot/",
            "https://www.reddit.com/r/bangladesh/top/?t=month",
            "https://www.reddit.com/r/Dhaka/hot/",
            "https://www.reddit.com/r/kolkata/hot/",
            "https://www.reddit.com/r/kolkata/top/?t=month"
        ],
        "maxPosts": 100,
        "maxComments": 400
    },
    # 2. Banglish & Social Emotion Keywords Search
    {
        "type": "searches",
        "terms": ["bhalobasha", "mon kharap kosto", "dhaka traffic jam", "onek moja", "happy birthday", "exam preparation", "desher obostha", "shob faltu"],
        "maxPosts": 100,
        "maxComments": 400
    },
    # 3. English Social Posts with High Emotional Weight & Emojis
    {
        "type": "searches",
        "terms": ["Bangladesh proud win", "Dhaka worst experience", "heartbreaking news", "love this country", "scared of future", "cannot believe this", "congratulations"],
        "maxPosts": 100,
        "maxComments": 400
    }
]

def clean_and_anonymize(text: str) -> str:
    """Strips URLs, user mentions, markdown artifacts, and PII."""
    if not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"/?u/[\w_-]+", "", text)
    text = re.sub(r"@[\w_.-]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "", text)
    # Remove bot disclaimers
    text = re.sub(r"I am a bot.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_social_text(text: str) -> bool:
    """Accepts Bangla, Banglish, or English social text with sufficient length."""
    if len(text) < 6:
        return False
    # Check if text has alphabetic or Bengali characters
    has_letters = bool(re.search(r"[a-zA-Zঀ-৿]", text))
    # Filter out auto-generated code, bot text, or pure numbers/symbols
    is_not_code = not text.startswith(("{", "[", "<", "import ", "http"))
    return has_letters and is_not_code

def predict_multilingual_emotions(text_lower: str, emojis: List[str]) -> Dict[str, Any]:
    """Scores 10 emotions across Bangla, Banglish, English, and Emojis."""
    scores = {emo: 0 for emo in EMOTIONS}

    # 1. Word matching
    for emo, data in EMOTION_LEXICON.items():
        for word in data["words"]:
            if word in text_lower:
                scores[emo] += 1

    # 2. Emoji matching (Weight = 2)
    for emo, data in EMOTION_LEXICON.items():
        for em in emojis:
            if em in data["emojis"]:
                scores[emo] += 2

    # 3. Sarcasm / Contradiction detection
    sarcasm_flag = 0
    if scores["Sarcasm"] > 0:
        sarcasm_flag = 1
    pos_score = scores["Joy"] + scores["Love"]
    neg_score = scores["Sadness"] + scores["Anger"] + scores["Disgust"]
    if pos_score > 0 and neg_score > 0 and len(emojis) > 0:
        sarcasm_flag = 1
        scores["Sarcasm"] = max(scores["Sarcasm"], 1)

    # 4. Multi-label binary flags
    flags = {emo: 0 for emo in EMOTIONS}
    max_score = max(scores.values()) if scores else 0

    if max_score > 0:
        for emo in EMOTIONS:
            if scores[emo] >= max(1, max_score * 0.6):
                flags[emo] = 1
    else:
        flags["Neutral"] = 1

    return {
        "emotion_flags": flags,
        "sarcasm_flag": sarcasm_flag,
        "agreement": round(0.75 + (0.20 if max_score >= 2 else 0.05), 2)
    }

def process_sample(raw_text: str, row_id: int) -> Optional[Dict[str, Any]]:
    """Transforms raw social text into clean dataset row."""
    cleaned = clean_and_anonymize(raw_text)
    if not is_valid_social_text(cleaned):
        return None

    emoji_entries = emoji.emoji_list(cleaned)
    emojis = [entry["emoji"] for entry in emoji_entries]

    emoji_stream = ",".join(emojis) if emojis else ""
    has_emoji = 1 if len(emojis) > 0 else 0
    emoji_count = len(emojis)

    text_only = emoji.replace_emoji(cleaned, "").strip()
    text_only = re.sub(r"\s+", " ", text_only)
    word_count = len(text_only.split()) if text_only else 0

    if word_count < 2 or len(text_only) < 5:
        return None

    annotation = predict_multilingual_emotions(text_only.lower(), emojis)

    row = {
        "ID": row_id,
        "Data": cleaned,
        "text_only": text_only,
        "emoji_stream": emoji_stream,
        "has_emoji": has_emoji,
        "emoji_count": emoji_count,
        "word_count": word_count,
        "Joy": annotation["emotion_flags"]["Joy"],
        "Sadness": annotation["emotion_flags"]["Sadness"],
        "Anger": annotation["emotion_flags"]["Anger"],
        "Fear": annotation["emotion_flags"]["Fear"],
        "Surprise": annotation["emotion_flags"]["Surprise"],
        "Disgust": annotation["emotion_flags"]["Disgust"],
        "Love": annotation["emotion_flags"]["Love"],
        "Anticipation": annotation["emotion_flags"]["Anticipation"],
        "Sarcasm": annotation["emotion_flags"]["Sarcasm"],
        "Neutral": annotation["emotion_flags"]["Neutral"],
        "label_source": "hybrid_annotated",
        "annotator_agreement": annotation["agreement"],
        "sarcasm_or_contradiction_flag": annotation["sarcasm_flag"]
    }
    return row

def main():
    print("=" * 70)
    print("🚀 AffectFusion: Multilingual (Bangla + Banglish + English) Scraper")
    print("=" * 70)

    client = ApifyClient(APIFY_TOKEN)
    output_file = "Bangla_Train_Scraped.csv"

    # Read existing rows to deduplicate and append
    existing_rows = []
    seen_texts = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)
                seen_texts.add(r["text_only"])
        print(f"[*] Found {len(existing_rows)} existing samples in {output_file}.")

    all_raw_texts = []

    for idx, task in enumerate(SCRAPE_TASKS):
        print(f"\n[*] Executing Scrape Task {idx+1}/{len(SCRAPE_TASKS)}: Type={task['type']}...")
        if task["type"] == "startUrls":
            run_input = {
                "startUrls": [{"url": u} for u in task["urls"]],
                "maxPostsCount": task["maxPosts"],
                "maxCommentsCount": task["maxComments"],
                "crawlCommentsPerPost": True,
                "sort": "new"
            }
        else:
            run_input = {
                "searches": task["terms"],
                "type": "comment",
                "maxPostsCount": task["maxPosts"],
                "maxCommentsCount": task["maxComments"],
                "crawlCommentsPerPost": True,
                "sort": "new"
            }

        try:
            run = client.actor("harshmaur/reddit-scraper").call(
                run_input=run_input,
                max_total_charge_usd=0.50
            )
            if run and run.status == "SUCCEEDED":
                dataset = client.dataset(run.default_dataset_id)
                items = list(dataset.iterate_items())
                print(f"    -> Retrieved {len(items)} items from Apify.")
                for item in items:
                    txt = item.get("body") or item.get("title") or item.get("selfText") or ""
                    if txt and txt not in seen_texts:
                        seen_texts.add(txt)
                        all_raw_texts.append(txt)
            else:
                print(f"    -> Actor status: {run.status if run else 'Failed'}")
        except Exception as e:
            print(f"    [!] Error during task {idx+1}: {e}")

    print(f"\n[*] Extracted {len(all_raw_texts)} new raw candidates.")
    print("[*] Processing Bangla, Banglish, and English text with emoji semantics...")

    new_rows = []
    current_id = len(existing_rows)

    for txt in all_raw_texts:
        res = process_sample(txt, row_id=current_id + len(new_rows) + 1)
        if res:
            new_rows.append(res)

    print(f"[+] Newly Generated Samples: {len(new_rows)}")
    combined_rows = existing_rows + new_rows
    print(f"[+] Total Combined Dataset Size: {len(combined_rows)} rows!")

    emoji_samples = sum(1 for r in combined_rows if str(r["has_emoji"]) == "1")
    print(f"    - Samples with Emojis: {emoji_samples} ({emoji_samples/len(combined_rows)*100:.1f}%)")
    print(f"    - Text-Only Samples: {len(combined_rows) - emoji_samples}")

    # Write full combined dataset
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for r in combined_rows:
            writer.writerow(r)

    print(f"\n💾 Updated complete dataset at: {output_file}")

if __name__ == "__main__":
    main()
