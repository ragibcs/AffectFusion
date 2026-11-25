#!/usr/bin/env python3
"""
AffectFusion: Target-Driven Multi-Token Reddit Scraper & Dataset Generator
Thesis FYDP: AffectFusion (Daffodil International University)
Authors: MD. RAGIB HASAN (232-15-775), MD. ASADULLA HEL GALIB (232-15-270)
Supervisor: Sharmin Akter, Co-Supervisor: Dewan Mamun Raza

Goal: Automatically scrape until:
- 5,000 Bangla samples
- 5,000 Banglish samples
- 5,000 English samples
Rotates across all 10 Apify tokens with real-time deduplication and streaming to disk.
"""

import os
import re
import csv
import uuid
import time
import argparse
import emoji
from typing import List, Dict, Any, Optional, Set
from apify_client import ApifyClient

# Load API tokens from environment or specify your tokens here
env_tokens = os.getenv("APIFY_TOKENS")
if env_tokens:
    APIFY_TOKENS = [t.strip() for t in env_tokens.split(",") if t.strip()]
else:
    APIFY_TOKENS = [
        "YOUR_APIFY_TOKEN_1",
        "YOUR_APIFY_TOKEN_2"
    ]

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
    "khujchi", "khujte", "dorkar", "proyojon", "shomossha", "problem", "somossa", "bhabte", "bhabsi"
])

# Deep High-Density Search Clusters targeting Bangla and Banglish heavily
QUERY_POOLS = {
    "Bangla": [
        ["আনন্দ", "খুশি", "উৎসব", "হাসি", "মজা", "দারুণ", "অভিনন্দন"],
        ["কষ্ট", "মন খারাপ", "দুঃখ", "হতাশা", "একা", "কান্না", "বেদনা", "অশ্রু"],
        ["রাগ", "ক্ষোভ", "অন্যায়", "ঘৃণা", "প্রতিবাদ", "লজ্জা", "দালাল", "শাস্তি"],
        ["অবাক", "অবিশ্বাস্য", "ভয়", "আতঙ্ক", "বিপদ", "শঙ্কা", "চমকে", "স্তব্ধ"],
        ["ভালোবাসা", "ভালোবাসি", "প্রিয়", "প্রেম", "মায়া", "শ্রদ্ধা", "হৃদয়"],
        ["অপেক্ষা", "প্রত্যাশা", "আশা", "স্বপ্ন", "পরিকল্পনা", "ভবিষ্যৎ", "শীঘ্রই"],
        ["বিরক্ত", "অসহ্য", "নোংরা", "জঘন্য", "ছি", "বাজে", "ঘৃণ্য", "বমি"],
        ["বাহ দারুণ", "উন্নয়ন", "হেসে বাঁচি না", "দারুণ দেশ", "সাধু সাধু", "ডিজিটাল"],
        ["বাংলাদেশ ক্রিকেট", "সাকিব", "তামিম", "শান্ত", "ম্যাচ", "জয়", "পরাজয়"],
        ["ঢাকা বিশ্ববিদ্যালয়", "বুয়েট", "ড্যাফোডিল", "জাহাঙ্গীরনগর", "মেডিকেল", "হল"],
        ["চাকরি", "বেকারত্ব", "বিসিএস", "পরীক্ষা", "রেজাল্ট", "পড়াশোনা", "টিউশন"],
        ["যানজট", "মেট্রোরেল", "রিকশা", "মিরপুর", "ধানমন্ডি", "গুলশান", "উত্তরা"],
        ["বৃষ্টি", "বর্ষা", "শীতকাল", "কুয়াশা", "গরম", "লোডশেডিং", "বিদ্যুৎ"],
        ["বিরিয়ানি", "কাচ্চি", "ফুচকা", "চা", "আড্ডা", "ইলিশ", "খিচুড়ি"],
        ["নাটক", "সিনেমা", "গান", "তাহসান", "হাবিব", "অপূর্ব", "মেহজাবীন"],
        ["জুলাই আন্দোলন", "ছাত্র আন্দোলন", "স্বৈরাচার", "স্বাধীনতা", "শহীদ", "বিপ্লব"],
        ["পরিবার", "মা বাবা", "ভাই বোন", "বন্ধু", "সম্পর্ক", "স্মৃতি", "অতীত"],
        ["গ্রামের বাড়ি", "নদী", "পদ্মা সেতু", "কক্সবাজার", "সেন্টমার্টিন", "সাজেক"],
        ["বইমেলা", "একুশে ফেব্রুয়ারি", "পহেলা বৈশাখ", "ঈদ মোবারক", "পূজা"],
        ["ডিপ্রেশন", "মানসিক স্বাস্থ্য", "কাউন্সেলিং", "একাকীত্ব", "ঘুম আসে না"]
    ],
    "Banglish": [
        ["mon kharap", "kosto lagse", "kanna pacche", "eka eka", "depression"],
        ["onek khushi", "onek moja", "shundor din", "congrats bhai", "osthir"],
        ["rag lagse", "faltu manush", "durniti bondho koro", "churi", "shob cheated"],
        ["bhoy lagse", "tension hocche", "ki hobe", "danger situation", "scared"],
        ["valobasha", "valobashi", "crush kheyese", "prem", "moner manush", "jaan"],
        ["opekkha korchi", "asha rakhi", "future plan", "varsity admission", "exam tension"],
        ["birokto lagse", "osojjo", "cringe laglo", "nongra politics", "chhi"],
        ["wah re unnoyon", "hese bachi na", "darun desh", "digital shob", "haha darun"],
        ["dhaka traffic jam", "metrorail experience", "mirpur er rasta", "rickshaw bhara"],
        ["bcs preparation", "chakri khujchi", "porashona hocche na", "cgpa kom"],
        ["cricket match win", "shakib al hasan", "tamim iqbal", "bcs exam routine"],
        ["kacchi biryani", "tong er cha", "addakhana", "dost shobai", "bikel bela"],
        ["natok dekhi", "bangla gaan", "coke studio bangla", "movie review"],
        ["bhai ekta help", "kono solution ache", "suggest korun", "kivabe korbo"],
        ["desher obostha kharap", "bidesh jabo", "ielts preparation", "visa update"],
        ["hall er life", "du campus", "daffodil campus", "buet admission test"]
    ],
    "English": [
        ["Bangladesh politics current affairs", "interim government reforms Dhaka"],
        ["Dhaka city daily life experience", "traffic jam in Dhaka solutions"],
        ["Bangladeshi food cuisine biryani", "travelling to Coxs Bazar Sylhet"],
        ["studying in Bangladesh universities", "career opportunities in Bangladesh"],
        ["mental health depression support Bangladesh", "moving abroad from Bangladesh IELTS"],
        ["Bangladeshi tech startup scene", "freelancing IT industry in Bangladesh"],
        ["Bangladesh cricket team match highlights", "cultural festivals in Bangladesh"]
    ]
}

def clean_and_anonymize(text: str) -> str:
    """Strips URLs, user mentions, markdown formatting, and PII."""
    if not text:
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"/?u/[\w_-]+", "", text)
    text = re.sub(r"@[\w_.-]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "", text)
    text = re.sub(r"I am a bot.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_language(text: str) -> str:
    """Accurately classifies text into Bangla, Banglish, or English."""
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

def process_raw_text(raw_text: str) -> Optional[Dict[str, Any]]:
    """Transforms raw text into exact unlabeled schema row."""
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

    if word_count < 2 or len(text_only) < 4:
        return None

    lang = classify_language(cleaned)
    if lang == "Unknown":
        return None

    row = {
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
    return row

def load_existing_texts(file_path: str) -> Set[str]:
    seen = set()
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("text_only"):
                        seen.add(r["text_only"])
        except Exception:
            pass
    return seen

def append_to_csv(file_path: str, rows: List[Dict[str, Any]]):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-bangla", type=int, default=5000)
    parser.add_argument("--target-banglish", type=int, default=5000)
    parser.add_argument("--target-english", type=int, default=5000)
    parser.add_argument("--budget-per-call", type=float, default=0.50)
    args = parser.parse_args()

    bangla_file = "Bangla_reddit_unlabeled.csv"
    banglish_file = "Banglish_reddit_unlabeled.csv"
    english_file = "English_reddit_unlabeled.csv"
    all_file = "All_Reddit_Unlabeled.csv"

    seen_texts = set()
    for fp in [bangla_file, banglish_file, english_file, all_file]:
        seen_texts.update(load_existing_texts(fp))

    counts = {
        "Bangla": len(load_existing_texts(bangla_file)),
        "Banglish": len(load_existing_texts(banglish_file)),
        "English": len(load_existing_texts(english_file))
    }

    print("=" * 75)
    print("🚀 AffectFusion: Target-Driven Multi-Token Reddit Scraper")
    print(f"[*] Targets: Bangla={args.target_bangla} | Banglish={args.target_banglish} | English={args.target_english}")
    print(f"[*] Current: Bangla={counts['Bangla']} | Banglish={counts['Banglish']} | English={counts['English']} | Total={sum(counts.values())}")
    print("=" * 75)

    token_idx = 0
    iteration = 0

    # Build queue prioritizing categories that need more rows
    while (counts["Bangla"] < args.target_bangla or
           counts["Banglish"] < args.target_banglish or
           counts["English"] < args.target_english):

        iteration += 1
        needed = []
        if counts["Bangla"] < args.target_bangla:
            needed.append("Bangla")
        if counts["Banglish"] < args.target_banglish:
            needed.append("Banglish")
        if counts["English"] < args.target_english:
            needed.append("English")

        if not needed:
            break

        # Pick query cluster from needed categories
        target_cat = needed[(iteration - 1) % len(needed)]
        pool = QUERY_POOLS[target_cat]
        terms = pool[(iteration - 1) % len(pool)]

        active_token = APIFY_TOKENS[token_idx % len(APIFY_TOKENS)]
        token_idx += 1
        client = ApifyClient(active_token)

        print(f"\n[Iter {iteration}] Searching for {target_cat} keywords: {terms[:4]} (Token #{((token_idx-1)%len(APIFY_TOKENS))+1})...")

        run_input = {
            "searches": terms,
            "type": "comment",
            "maxPostsCount": 150,
            "maxCommentsCount": 800,
            "crawlCommentsPerPost": True,
            "sort": "new"
        }

        try:
            run = client.actor("harshmaur/reddit-scraper").call(
                run_input=run_input,
                max_total_charge_usd=args.budget_per_call
            )

            if run and run.status in ["SUCCEEDED", "STOPPED"]:
                dataset = client.dataset(run.default_dataset_id)
                items = list(dataset.iterate_items())
                print(f"    -> Extracted {len(items)} items from Reddit.")

                job_new_rows = {"Bangla": [], "Banglish": [], "English": []}

                for it in items:
                    txt = it.get("body") or it.get("title") or it.get("selfText") or ""
                    if not txt:
                        continue
                    res = process_raw_text(txt)
                    if res and res["text_only"] not in seen_texts:
                        seen_texts.add(res["text_only"])
                        lang = res["_lang"]
                        target_limit = (
                            args.target_bangla if lang == "Bangla"
                            else args.target_banglish if lang == "Banglish"
                            else args.target_english
                        )
                        if counts[lang] + len(job_new_rows[lang]) < target_limit:
                            job_new_rows[lang].append(res)

                # Save new items
                for lang, rows in job_new_rows.items():
                    if rows:
                        target_file = (
                            bangla_file if lang == "Bangla"
                            else banglish_file if lang == "Banglish"
                            else english_file
                        )
                        append_to_csv(target_file, rows)
                        append_to_csv(all_file, rows)
                        counts[lang] += len(rows)
                        print(f"    [+] Added {len(rows)} {lang} rows -> Total {lang}: {counts[lang]}")

            else:
                print(f"    [!] Run status: {run.status if run else 'Failed'}")

        except Exception as e:
            print(f"    [!] Error during iteration {iteration}: {e}")

        print(f"    [Status] Bangla: {counts['Bangla']}/{args.target_bangla} | Banglish: {counts['Banglish']}/{args.target_banglish} | English: {counts['English']}/{args.target_english}")
        time.sleep(2)

        if iteration > 60:
            print("[*] Reached max iteration safeguard (60 batches). Stopping run.")
            break

    print("\n" + "=" * 75)
    print("🎉 Scraping Round Finished!")
    print(f"📁 Bangla:    {bangla_file}   ({counts['Bangla']} rows)")
    print(f"📁 Banglish:  {banglish_file} ({counts['Banglish']} rows)")
    print(f"📁 English:   {english_file}  ({counts['English']} rows)")
    print(f"📁 Combined:  {all_file}      ({sum(counts.values())} rows)")
    print("=" * 75)

if __name__ == "__main__":
    main()
