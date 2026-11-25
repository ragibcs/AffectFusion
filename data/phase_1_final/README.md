# AffectFusion Phase-I Final Multimodal Benchmark (`N = 30,000`)

This directory contains the locked, zero-leakage **80/10/10** multilingual (`ENG` / `BN` / `BNG`) text-and-emoji emotion benchmark constructed in **FYDP Phase-I**.

## Split Provenance & Zero-Leakage Guarantee

| Split | File(s) | Rows | Real Scraped | DAIR-AI Twitter (Train-Only) | Gemini Flash Aug. (Train-Only) | Human Gold Verified | Emoji Share |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (80%)** | `train_24k.parquet` | **24,000** | 18,872 | 1,600* | 3,528* | 0 (AI `conf >= 3`) | 59.09% |
| **Validation (10%)** | `val_3k.parquet`, `val_3k.csv` | **3,000** | **3,000 (100%)** | **0** | **0** | **2,000 (66.7%)** | 69.97% |
| **Test (10%)** | `test_3k.parquet`, `test_3k.csv` | **3,000** | **3,000 (100%)** | **0** | **0** | **3,000 (100%)** | 65.80% |
| **Total** | — | **30,000** | **24,872** | **1,600** | **3,528** | **5,000 (κ = 0.8087)** | **60.85%** |

*\*1,147 of the 3,528 Train-only Fear paraphrases were derived from DAIR-AI Twitter seeds, yielding 2,747 rows carrying `is_external_train_only = 1`; strictly `0` augmented or external rows appear in `val_3k` or `test_3k`.*

## 9 Emotion Classes (`label_id`: `0`–`8`)
`0: Joy`, `1: Neutral`, `2: Anger`, `3: Sadness`, `4: Surprise`, `5: Sarcasm`, `6: Fear`, `7: Disgust`, `8: Anticipation`.

## 25-Column Unified Dual-Stream Schema
`row_id`, `raw_text`, `text_stream`, `emoji_stream`, `demoji_text`, `baseline_input`, `label`, `label_id`, `language_tag`, `language_id`, `has_emoji`, `emoji_count`, `unique_emoji_count`, `emoji_density`, `text_length`, `word_count`, `exclamation_count`, `question_count`, `uppercase_ratio`, `has_sarcasm_emoji_cue`, `ai_confidence`, `platform`, `is_augmented`, `is_external_train_only`, `is_human_verified`.

## Human Verification Agreement (`../gold_verification/`)
- **Combined 5,000 Gold Rows (`2,500` MD. Ragib Hasan + `2,500` MD. Asadulla Hel Galib):** **83.00% Raw Agreement**, **Cohen's κ = 0.8087** (`ENG`: 0.8129, `BN`: 0.8128, `BNG`: 0.7958).
- **Corrections Applied:** `574` labels corrected in `test_3k` (`100%` Gold), `276` labels corrected in `val_3k` (`66.7%` Gold).
