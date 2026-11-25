#!/usr/bin/env python3
"""
AffectFusion: High-Speed Multi-Threaded 30k YouTube Dataset Generator
Thesis FYDP: AffectFusion (Daffodil International University)
Authors: MD. RAGIB HASAN (232-15-775), MD. ASADULLA HEL GALIB (232-15-270)
Supervisor: Sharmin Akter, Co-Supervisor: Dewan Mamun Raza

Generates:
1. AffectFusion_Banglish_YouTube/
   - Banglish_YouTube_Unlabeled_30000.csv (15,000 with emoji + 15,000 without emoji)
   - Banglish_YouTube_Unlabeled_30000.xlsx
   - cleaned_candidate_pool.csv
2. AffectFusion_English_YouTube/
   - English_YouTube_Unlabeled_30000.csv (15,000 with emoji + 15,000 without emoji)
   - English_YouTube_Unlabeled_30000.xlsx
   - cleaned_candidate_pool.csv
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from youtube_comment_downloader import YoutubeCommentDownloader

CSV_HEADERS = [
    "ID", "Data", "text_only", "emoji_stream", "has_emoji",
    "emoji_count", "word_count", "Joy", "Sadness", "Anger",
    "Fear", "Surprise", "Disgust", "Love", "Anticipation",
    "Sarcasm", "Neutral", "label_source", "annotator_agreement",
    "sarcasm_or_contradiction_flag"
]

BANGLISH_PATTERNS = [
    r"\b(ami|tumi|apni|amra|tomra|apnara|tara|amar|tomar|apnar|tader|amader|tomader)\b",
    r"\b(kemon|achi|achen|bhalo|valo|kharap|kosto|mon|ekta|kono|kichu|kisu|kew|keu)\b",
    r"\b(ki|keno|kothay|kobe|kivabe|kemne|hobe|hocche|hoise|hoyeche|hoy|hoyse|hoyna)\b",
    r"\b(gelo|gese|geche|korbo|korse|korchen|kori|korte|korechi|korini|kora|koris|korba)\b",
    r"\b(bhai|vai|apu|bhaiya|vaia|bhebe|bhabte|bhabsi|jani|janen|jani_na|dekhi|dekho|dekhe)\b",
    r"\b(desh|desher|manush|kotha|shob|shobai|shathe|moto|erkom|thik|ashole|shotti|khub|onek)\b",
    r"\b(dhaka|shohor|rasta|kintu|tobe|tahole|tarpore|pore|age|shuru|shesh|bujhi|bujhte)\b",
    r"\b(din|ta|hoye|jacche|jasche|bolse|bolte|bolbo|bolsi|khacchi|khabo|khelam|ghum)\b",
    r"\b(jodi|tate|naki|kenona|tai|karone|jonno|jonne|halka|beshi|kom|puro|shundor|sundor|moja)\b",
    r"\b(khushi|anondo|osthir|pera|matha|chokh|kan|haat|pa|basa|bari|gram|graamer|shudhu|khali)\b",
    r"\b(natok|gaan|pagol|boba|chotto|haste|hasbo|kannakatti|shune|shunlam|jante|jabo|ashbo)\b",
    r"\b(vaiya|bhaia|bro|mama|dost|dosto|boss|ostad|apnar|valo|lagse|laglo|lagche|lagteche)\b",
    r"\b(akash|batas|bristi|sokal|dupur|bikal|sondha|rat|shokal|bikel|raat|sundor|darun)\b",
    r"\b(onek|valobasa|bhalobasha|valobashi|valobasi|maya|prem|crush|pagli|pagol|pagla)\b",
    r"\b(kichui|kisui|bujhlam|bujhina|parina|parbo|parbona|dorkar|somossa|shomossha)\b",
    r"\b(ei|oi|shei|shobar|sob|sobai|vabi|chobi|dekhte|dekhle|khub|baje|besh|shotti)\b"
]
banglish_regex = re.compile("|".join(BANGLISH_PATTERNS), re.IGNORECASE)

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

    banglish_matches = len(banglish_regex.findall(text))
    if banglish_matches >= 1:
        return "Banglish"

    return "English"

def process_raw_comment(raw_text: str) -> Optional[Dict[str, Any]]:
    cleaned = clean_and_anonymize(raw_text)
    if not cleaned or len(cleaned) < 3:
        return None

    emoji_entries = emoji.emoji_list(cleaned)
    emojis = [entry["emoji"] for entry in emoji_entries]

    emoji_stream = "".join(emojis) if emojis else ""
    has_emoji = 1 if len(emojis) > 0 else 0
    emoji_count = len(emojis)

    text_only = emoji.replace_emoji(cleaned, "").strip()
    text_only = re.sub(r"\s+", " ", text_only)
    word_count = len(text_only.split()) if text_only else 0

    if word_count < 1 or (not text_only and not has_emoji):
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

def fetch_comments_for_video(vid: str, max_comments: int = 1500) -> List[Dict[str, Any]]:
    downloader = YoutubeCommentDownloader()
    url = f"https://www.youtube.com/watch?v={vid}"
    results = []
    try:
        comments = islice(downloader.get_comments_from_url(url), max_comments)
        for c in comments:
            res = process_raw_comment(c.get("text", ""))
            if res:
                results.append(res)
    except Exception:
        pass
    return results

def main():
    print("=" * 75, flush=True)
    print("🚀 AffectFusion: Multi-Threaded 30k YouTube Dataset Generator", flush=True)
    print("=" * 75, flush=True)

    raw_csv_path = "AffectFusion_Bangla_YouTube/raw_scraped_comments.csv"
    if not os.path.exists(raw_csv_path):
        print(f"[!] Error: {raw_csv_path} not found.", flush=True)
        return

    print("[*] Ingesting 160,552 comments from raw candidate pool...", flush=True)
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

    print(f"[*] Raw Candidate Pool Counts:", flush=True)
    print(f"    - Banglish: with_emoji = {len(banglish_pool['with_emoji']):5d} | no_emoji = {len(banglish_pool['no_emoji']):5d}", flush=True)
    print(f"    - English : with_emoji = {len(english_pool['with_emoji']):5d} | no_emoji = {len(english_pool['no_emoji']):5d}", flush=True)

    target_needed = 15000

    # Multi-threaded quick harvest if needed
    if len(banglish_pool["with_emoji"]) < target_needed:
        needed = target_needed - len(banglish_pool["with_emoji"])
        print(f"\n[*] Parallel scraping {needed} additional Banglish comments with emojis across 8 threads...", flush=True)

        candidate_vids = df_raw["video_id"].value_counts().index.tolist()[:80]

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_vid = {executor.submit(fetch_comments_for_video, vid, 2000): vid for vid in candidate_vids}

            for future in as_completed(future_to_vid):
                vid = future_to_vid[future]
                try:
                    items = future.result()
                    new_added = 0
                    for res in items:
                        if res["text_only"] not in seen_texts:
                            seen_texts.add(res["text_only"])
                            cat = "with_emoji" if res["has_emoji"] == 1 else "no_emoji"
                            if res["_lang"] == "Banglish":
                                banglish_pool[cat].append(res)
                                new_added += 1
                            elif res["_lang"] == "English":
                                english_pool[cat].append(res)
                    print(f"    [+] Video {vid}: +{new_added} Banglish -> Total with_emoji = {len(banglish_pool['with_emoji'])}/{target_needed}", flush=True)
                except Exception as e:
                    pass

                if len(banglish_pool["with_emoji"]) >= target_needed:
                    print("[*] Quota reached! Finalizing pool...", flush=True)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    print("\n" + "=" * 75, flush=True)
    print("[*] Sampling exactly 15,000 with emoji + 15,000 without emoji for each language...", flush=True)
    print("=" * 75, flush=True)

    random.seed(42)

    # 1. Banglish 30k Dataset
    banglish_we = random.sample(banglish_pool["with_emoji"], min(target_needed, len(banglish_pool["with_emoji"])))
    banglish_ne = random.sample(banglish_pool["no_emoji"], min(target_needed, len(banglish_pool["no_emoji"])))
    banglish_30k = banglish_we + banglish_ne
    random.shuffle(banglish_30k)

    # 2. English 30k Dataset
    english_we = random.sample(english_pool["with_emoji"], min(target_needed, len(english_pool["with_emoji"])))
    english_ne = random.sample(english_pool["no_emoji"], min(target_needed, len(english_pool["no_emoji"])))
    english_30k = english_we + english_ne
    random.shuffle(english_30k)

    # Create directories matching AffectFusion folder structure
    os.makedirs("AffectFusion_Banglish_YouTube", exist_ok=True)
    os.makedirs("AffectFusion_English_YouTube", exist_ok=True)

    # Export Banglish Files
    banglish_csv = "AffectFusion_Banglish_YouTube/Banglish_YouTube_Unlabeled_30000.csv"
    banglish_xlsx = "AffectFusion_Banglish_YouTube/Banglish_YouTube_Unlabeled_30000.xlsx"
    banglish_pool_csv = "AffectFusion_Banglish_YouTube/cleaned_candidate_pool.csv"

    df_banglish = pd.DataFrame(banglish_30k)[CSV_HEADERS]
    df_banglish.to_csv(banglish_csv, index=False, encoding="utf-8-sig")
    df_banglish.to_excel(banglish_xlsx, index=False)
    pd.DataFrame(banglish_pool["with_emoji"] + banglish_pool["no_emoji"]).to_csv(banglish_pool_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ Banglish Dataset Successfully Saved:", flush=True)
    print(f"   - CSV  : {banglish_csv} ({len(df_banglish)} rows | with_emoji: {len(banglish_we)}, no_emoji: {len(banglish_ne)})", flush=True)
    print(f"   - Excel: {banglish_xlsx}", flush=True)
    print(f"   - Pool : {banglish_pool_csv}", flush=True)

    # Export English Files
    english_csv = "AffectFusion_English_YouTube/English_YouTube_Unlabeled_30000.csv"
    english_xlsx = "AffectFusion_English_YouTube/English_YouTube_Unlabeled_30000.xlsx"
    english_pool_csv = "AffectFusion_English_YouTube/cleaned_candidate_pool.csv"

    df_english = pd.DataFrame(english_30k)[CSV_HEADERS]
    df_english.to_csv(english_csv, index=False, encoding="utf-8-sig")
    df_english.to_excel(english_xlsx, index=False)
    pd.DataFrame(english_pool["with_emoji"] + english_pool["no_emoji"]).to_csv(english_pool_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ English Dataset Successfully Saved:", flush=True)
    print(f"   - CSV  : {english_csv} ({len(df_english)} rows | with_emoji: {len(english_we)}, no_emoji: {len(english_ne)})", flush=True)
    print(f"   - Excel: {english_xlsx}", flush=True)
    print(f"   - Pool : {english_pool_csv}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("🎉 All 30k Balanced YouTube Datasets for Banglish & English Complete!", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
