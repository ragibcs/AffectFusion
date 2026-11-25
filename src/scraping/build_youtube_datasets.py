#!/usr/bin/env python3
"""
AffectFusion: Automated YouTube Dataset Builder for Banglish & English
Thesis FYDP: AffectFusion (Daffodil International University)
Authors: MD. RAGIB HASAN (232-15-775), MD. ASADULLA HEL GALIB (232-15-270)
Supervisor: Sharmin Akter, Co-Supervisor: Dewan Mamun Raza

Generates two complete 30,000-sample balanced datasets:
1. AffectFusion_Banglish_YouTube/
   - Banglish_YouTube_Unlabeled_30000.csv (15,000 with emoji + 15,000 without emoji)
   - Banglish_YouTube_Unlabeled_30000.xlsx
2. AffectFusion_English_YouTube/
   - English_YouTube_Unlabeled_30000.csv (15,000 with emoji + 15,000 without emoji)
   - English_YouTube_Unlabeled_30000.xlsx

Matches the exact schema, formatting, and file organization of AffectFusion_Bangla_YouTube/.
"""

import os
import re
import csv
import uuid
import time
import random
import emoji
import pandas as pd
from typing import List, Dict, Any, Optional, Set
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader

CSV_HEADERS = [
    "ID", "Data", "text_only", "emoji_stream", "has_emoji",
    "emoji_count", "word_count", "Joy", "Sadness", "Anger",
    "Fear", "Surprise", "Disgust", "Love", "Anticipation",
    "Sarcasm", "Neutral", "label_source", "annotator_agreement",
    "sarcasm_or_contradiction_flag"
]

BANGLISH_WORDS = set([
    "ami", "tumi", "apni", "amra", "tomra", "apnara", "she", "tara", "amar", "tomar", "apnar", "tader",
    "kemon", "achi", "achen", "bhalo", "valo", "kharap", "kosto", "mon", "ekta", "kono", "kichu", "kisu",
    "ki", "keno", "kothay", "kobe", "kivabe", "kemne", "hobe", "hocche", "hoise", "hoyeche", "hoy",
    "gelo", "gese", "geche", "korbo", "korse", "korchen", "kori", "korte", "bhai", "vai", "apu", "bhebe",
    "desh", "desher", "manush", "kotha", "shob", "shobai", "shathe", "moto", "ar", "er", "te", "re",
    "erkom", "thik", "ashole", "shotti", "khub", "onek", "arekta", "dekhi", "dekho", "dekhe", "jani",
    "janen", "paro", "paren", "hobe", "hobena", "hoyna", "parina", "lagbe", "lagena", "mone", "hoy",
    "eita", "oita", "sheita", "eta", "ota", "konta", "shobkisu", "dhaka", "shohor", "rasta", "kintu",
    "tobe", "tahole", "tarpore", "pore", "age", "shuru", "shesh", "bujhi", "bujhte", "bujhen", "dada",
    "din", "ta", "hoye", "jacche", "jasche", "bolse", "bolte", "bolbo", "bolsi", "dekhe", "dekhlam",
    "khacchi", "khabo", "khelam", "ghum", "shokal", "bikel", "raat", "shathe", "shonge", "sathe",
    "jodi", "tate", "bhaiya", "bhaia", "vaia", "bubu", "mama", "kaka", "chachi", "mami", "naki",
    "kenona", "tai", "karone", "jonno", "jonne", "halka", "beshi", "kom", "puro", "shundor", "moja",
    "khushi", "anondo", "osthir", "pera", "matha", "chokh", "kan", "haat", "pa", "basa", "bari",
    "gram", "graamer", "shobai", "shobaike", "shudu", "shudhu", "khali", "ekdom", "pura", "purapuri",
    "bujhlam", "bujhlamna", "hobe", "hobena", "hoyse", "parbo", "parbona", "parba", "parbi", "korbi",
    "korba", "koris", "koros", "bolos", "bolis", "dekhis", "dekhos", "koren", "bolen", "dekhen",
    "khobor", "bepari", "chole", "cholen", "asche", "ashche", "gelam", "ashlam", "debo", "dibo", "dibe",
    "shune", "shunlam", "jante", "bujhina", "korina", "bolbona", "dekhbona", "jabo", "jabona", "ashbo",
    "khujchi", "khujte", "dorkar", "proyojon", "shomossha", "problem", "somossa", "bhabte", "bhabsi",
    "natok", "gaan", "pagol", "boba", "chotto", "haste", "hasbo", "hastechi", "kannakatti", "bhaiya",
    "kew", "keu", "amader", "tomader", "apnader", "tader", "korechi", "korini", "dekhini", "jani_na"
])

def clean_and_anonymize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@[\w_.-]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_language(text: str) -> str:
    bangla_chars = len(re.findall(r"[ঀ-৿]", text))
    total_chars = len(re.sub(r"\s+", "", text)) or 1

    if bangla_chars / total_chars > 0.15 or bangla_chars >= 3:
        return "Bangla"

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return "Unknown"

    banglish_hits = sum(1 for w in words if w in BANGLISH_WORDS)
    ratio = banglish_hits / len(words)
    if (banglish_hits >= 2 and ratio >= 0.08) or (banglish_hits >= 1 and len(words) <= 6 and ratio >= 0.14):
        return "Banglish"

    return "English"

def process_raw_comment(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = clean_and_anonymize(raw_text)
    if not cleaned or len(cleaned) < 4:
        return None

    emoji_entries = emoji.emoji_list(cleaned)
    emojis = [entry["emoji"] for entry in emoji_entries]

    emoji_stream = "".join(emojis) if emojis else ""
    has_emoji = 1 if len(emojis) > 0 else 0
    emoji_count = len(emojis)

    text_only = emoji.replace_emoji(cleaned, "").strip()
    text_only = re.sub(r"\s+", " ", text_only)
    word_count = len(text_only.split()) if text_only else 0

    if word_count < 2 or len(text_only) < 3:
        return None

    lang = classify_language(cleaned)
    if lang == "Unknown":
        return None

    return {
        "ID": str(uuid.uuid4()),
        "Data": cleaned,
        "text_only": text_only,
        "emoji_stream": emoji_stream,
        "has_emoji": has_emoji,
        "emoji_count": emoji_count,
        "word_count": word_count,
        "Joy": "",
        "Sadness": "",
        "Anger": "",
        "Fear": "",
        "Surprise": "",
        "Disgust": "",
        "Love": "",
        "Anticipation": "",
        "Sarcasm": "",
        "Neutral": "",
        "label_source": "pending_manual",
        "annotator_agreement": "",
        "sarcasm_or_contradiction_flag": "",
        "_lang": lang
    }

def main():
    print("=" * 75)
    print("🚀 AffectFusion: YouTube 30k Balanced Dataset Builder (Banglish & English)")
    print("=" * 75)

    raw_csv_path = "AffectFusion_Bangla_YouTube/raw_scraped_comments.csv"
    if not os.path.exists(raw_csv_path):
        print(f"[!] Error: {raw_csv_path} not found.")
        return

    print("[*] Phase 1: Ingesting & processing 160,552 existing raw YouTube comments...")
    df_raw = pd.read_csv(raw_csv_path)

    seen_texts = set()
    banglish_pool = {"with_emoji": [], "no_emoji": []}
    english_pool = {"with_emoji": [], "no_emoji": []}

    for comment in df_raw["comment_text"].dropna():
        res = process_raw_comment(str(comment))
        if res and res["text_only"] not in seen_texts:
            seen_texts.add(res["text_only"])
            category = "with_emoji" if res["has_emoji"] == 1 else "no_emoji"
            if res["_lang"] == "Banglish":
                banglish_pool[category].append(res)
            elif res["_lang"] == "English":
                english_pool[category].append(res)

    print(f"[*] Current Candidate Pool from Raw File:")
    print(f"    - Banglish: with_emoji = {len(banglish_pool['with_emoji']):5d} | no_emoji = {len(banglish_pool['no_emoji']):5d}")
    print(f"    - English : with_emoji = {len(english_pool['with_emoji']):5d} | no_emoji = {len(english_pool['no_emoji']):5d}")

    target_needed = 15000
    banglish_with_needed = max(0, target_needed - len(banglish_pool["with_emoji"]))
    banglish_no_needed = max(0, target_needed - len(banglish_pool["no_emoji"]))

    if banglish_with_needed > 0 or banglish_no_needed > 0:
        print(f"\n[*] Phase 2: Scraping additional YouTube comments to fulfill Banglish quota...")
        print(f"    -> Needed: with_emoji={banglish_with_needed}, no_emoji={banglish_no_needed}")

        top_videos = df_raw["video_id"].value_counts().index.tolist()
        downloader = YoutubeCommentDownloader()

        scraped_video_count = 0
        for vid in top_videos:
            if (len(banglish_pool["with_emoji"]) >= target_needed and
                len(banglish_pool["no_emoji"]) >= target_needed):
                break

            scraped_video_count += 1
            url = f"https://www.youtube.com/watch?v={vid}"
            try:
                generator = downloader.get_comments_from_url(url)
                batch = list(islice(generator, 800))
                for c in batch:
                    res = process_raw_comment(c.get("text", ""))
                    if res and res["text_only"] not in seen_texts:
                        seen_texts.add(res["text_only"])
                        cat = "with_emoji" if res["has_emoji"] == 1 else "no_emoji"
                        if res["_lang"] == "Banglish":
                            banglish_pool[cat].append(res)
                        elif res["_lang"] == "English":
                            english_pool[cat].append(res)
            except Exception as e:
                pass

            if scraped_video_count % 5 == 0 or len(banglish_pool["with_emoji"]) >= target_needed:
                print(f"    [{scraped_video_count} videos] Banglish: with_emoji={len(banglish_pool['with_emoji'])}/{target_needed}, no_emoji={len(banglish_pool['no_emoji'])}/{target_needed}")
            time.sleep(0.5)

    print("\n" + "=" * 75)
    print("[*] Phase 3: Generating balanced 30k datasets (15,000 with emoji + 15,000 without emoji)...")
    print("=" * 75)

    random.seed(42)

    # Banglish 30k Dataset
    banglish_we = random.sample(banglish_pool["with_emoji"], min(target_needed, len(banglish_pool["with_emoji"])))
    banglish_ne = random.sample(banglish_pool["no_emoji"], min(target_needed, len(banglish_pool["no_emoji"])))
    banglish_30k = banglish_we + banglish_ne
    random.shuffle(banglish_30k)

    # English 30k Dataset
    english_we = random.sample(english_pool["with_emoji"], min(target_needed, len(english_pool["with_emoji"])))
    english_ne = random.sample(english_pool["no_emoji"], min(target_needed, len(english_pool["no_emoji"])))
    english_30k = english_we + english_ne
    random.shuffle(english_30k)

    # Create directories
    os.makedirs("AffectFusion_Banglish_YouTube", exist_ok=True)
    os.makedirs("AffectFusion_English_YouTube", exist_ok=True)

    # Save Banglish Dataset
    banglish_csv = "AffectFusion_Banglish_YouTube/Banglish_YouTube_Unlabeled_30000.csv"
    banglish_xlsx = "AffectFusion_Banglish_YouTube/Banglish_YouTube_Unlabeled_30000.xlsx"
    df_banglish = pd.DataFrame(banglish_30k)[CSV_HEADERS]
    df_banglish.to_csv(banglish_csv, index=False, encoding="utf-8-sig")
    df_banglish.to_excel(banglish_xlsx, index=False)
    print(f"\n✅ Banglish Dataset Saved:")
    print(f"   - CSV  : {banglish_csv} ({len(df_banglish)} rows | Emojis: {len(banglish_we)}, No-Emojis: {len(banglish_ne)})")
    print(f"   - Excel: {banglish_xlsx}")

    # Save English Dataset
    english_csv = "AffectFusion_English_YouTube/English_YouTube_Unlabeled_30000.csv"
    english_xlsx = "AffectFusion_English_YouTube/English_YouTube_Unlabeled_30000.xlsx"
    df_english = pd.DataFrame(english_30k)[CSV_HEADERS]
    df_english.to_csv(english_csv, index=False, encoding="utf-8-sig")
    df_english.to_excel(english_xlsx, index=False)
    print(f"\n✅ English Dataset Saved:")
    print(f"   - CSV  : {english_csv} ({len(df_english)} rows | Emojis: {len(english_we)}, No-Emojis: {len(english_ne)})")
    print(f"   - Excel: {english_xlsx}")

    print("\n" + "=" * 75)
    print("🎉 All 30k Datasets for Banglish and English Generated Successfully!")
    print("=" * 75)

if __name__ == "__main__":
    main()
