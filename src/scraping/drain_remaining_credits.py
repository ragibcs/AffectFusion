#!/usr/bin/env python3
"""
AffectFusion: Final Credit Exhaustion Scraper for Bangla & Banglish
Thesis FYDP: AffectFusion (Daffodil International University)
Authors: MD. RAGIB HASAN (232-15-775), MD. ASADULLA HEL GALIB (232-15-270)

Drains the remaining ~$7.38 across active Apify tokens focusing 100% on
Bengali Unicode & Banglish data to maximize dataset depth.
"""

import os
import re
import csv
import uuid
import time
import emoji
from typing import List, Dict, Any, Optional, Set
from apify_client import ApifyClient

# Active tokens from environment variable or placeholder
env_tokens = os.getenv("APIFY_TOKENS")
if env_tokens:
    ACTIVE_TOKENS = [t.strip() for t in env_tokens.split(",") if t.strip()]
else:
    ACTIVE_TOKENS = [
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

TARGET_SEARCH_BATCHES = [
    {"target": "Bangla", "terms": ["মন খারাপের দিন", "খুব একা লাগে", "কষ্টের কথা", "কান্না থামে না"]},
    {"target": "Bangla", "terms": ["জীবনের সবচেয়ে সেরা আনন্দ", "দারুণ অনুভূতি", "অনেক মজা করলাম", "সবাইকে ধন্যবাদ"]},
    {"target": "Bangla", "terms": ["দেশের বর্তমান অবস্থা", "ছাত্র আন্দোলন জুলাই", "স্বৈরাচার পতন", "নতুন বাংলাদেশ"]},
    {"target": "Bangla", "terms": ["পড়াশোনার চাপ", "বিসিএস প্রিলি", "চাকরির ইন্টারভিউ", "বেকারত্ব কষ্ট"]},
    {"target": "Bangla", "terms": ["সম্পর্ক ভেঙে গেছে", "ভালোবাসার গল্প", "প্রথম প্রেম", "বিয়ে নিয়ে চিন্তা"]},
    {"target": "Bangla", "terms": ["মেডিকেল লাইফ কষ্ট", "ইঞ্জিনিয়ারিং পেরা", "ভার্সিটি ক্যাম্পাস স্মৃতি", "হল লাইফ"]},
    {"target": "Bangla", "terms": ["ভয়ঙ্কর অভিজ্ঞতা", "রাতের ঢাকা", "ভূতুড়ে কাণ্ড", "আতঙ্কের মুহূর্ত"]},
    {"target": "Bangla", "terms": ["জঘন্য মানুষ", "দালালি বন্ধ কর", "লজ্জাজনক কাজ", "ঘৃণা লাগে"]},
    {"target": "Banglish", "terms": ["mon valo nai", "kosto hocche bhai", "bhalo thakish", "miss korbo"]},
    {"target": "Banglish", "terms": ["khub moja hoise", "congrats vai", "onek shundor lagse", "sera bhai"]},
    {"target": "Banglish", "terms": ["dhaka shohor er jam", "rickshawala mama", "bristi r dine cha", "addabaji"]},
    {"target": "Banglish", "terms": ["pera nai chill", "exam sesh", "cgpa niye tension", "assignment submission"]},
    {"target": "Banglish", "terms": ["bideshe study", "ielts score", "visa rejection kosto", "germany jabo"]},
    {"target": "Banglish", "terms": ["breakup hoye gese", "crush re bolte parina", "prothom prem", "maya lage"]},
    {"target": "Banglish", "terms": ["cricket match harlo", "bcs coaching", "job salary kom", "shobai kemon achen"]}
]

def clean_and_anonymize(text: str) -> str:
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
    print("🚀 AffectFusion: Scraping with Remaining Apify Credits")
    print(f"[*] Starting Counts: Bangla={counts['Bangla']} | Banglish={counts['Banglish']} | English={counts['English']}")
    print(f"[*] Total Active Tokens: {len(ACTIVE_TOKENS)}")
    print("=" * 75)

    token_idx = 0

    for i, batch in enumerate(TARGET_SEARCH_BATCHES):
        active_token = ACTIVE_TOKENS[token_idx % len(ACTIVE_TOKENS)]
        token_idx += 1
        client = ApifyClient(active_token)

        print(f"\n[Batch {i+1}/{len(TARGET_SEARCH_BATCHES)}] Target: {batch['target']} | Queries: {batch['terms'][:3]} (Token #{((token_idx-1)%len(ACTIVE_TOKENS))+1})...")

        run_input = {
            "searches": batch["terms"],
            "type": "comment",
            "maxPostsCount": 150,
            "maxCommentsCount": 800,
            "crawlCommentsPerPost": True,
            "sort": "new"
        }

        try:
            run = client.actor("harshmaur/reddit-scraper").call(
                run_input=run_input,
                max_total_charge_usd=0.50
            )

            if run and run.status in ["SUCCEEDED", "STOPPED"]:
                dataset = client.dataset(run.default_dataset_id)
                items = list(dataset.iterate_items())
                print(f"    -> Scraped {len(items)} items from Reddit.")

                job_new_rows = {"Bangla": [], "Banglish": [], "English": []}

                for it in items:
                    txt = it.get("body") or it.get("title") or it.get("selfText") or ""
                    if not txt:
                        continue
                    res = process_raw_text(txt)
                    if res and res["text_only"] not in seen_texts:
                        seen_texts.add(res["text_only"])
                        lang = res["_lang"]
                        job_new_rows[lang].append(res)

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
            print(f"    [!] Error during batch {i+1}: {e}")

        time.sleep(2)

    print("\n" + "=" * 75)
    print("🎉 All Remaining Credit Scraped Successfully!")
    print(f"📁 Bangla:    {bangla_file}   ({counts['Bangla']} rows)")
    print(f"📁 Banglish:  {banglish_file} ({counts['Banglish']} rows)")
    print(f"📁 English:   {english_file}  ({counts['English']} rows)")
    print(f"📁 Combined:  {all_file}      ({sum(counts.values())} total rows)")
    print("=" * 75)

if __name__ == "__main__":
    main()
