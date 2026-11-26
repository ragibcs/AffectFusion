# AffectFusion Phase-I Final Multimodal Benchmark (`N = 30,000` · Consensus Gold v4)

This directory contains the locked, zero-leakage **80/10/10** multilingual (`ENG` / `BN` / `BNG`) text-and-emoji emotion benchmark constructed in **FYDP Phase-I** and upgraded via **2-Round Blind-Swap Human-vs-Human Consensus Gold** adjudication (`5,000` double-annotated rows = `10,000` human judgments).

- **Hugging Face Dataset:** [ragibhasan/AffectFusion](https://huggingface.co/datasets/ragibhasan/AffectFusion) · DOI: [`10.57967/hf/11874`](https://doi.org/10.57967/hf/11874)
- **Zenodo Permanent Archive:** [Record #24543680](https://zenodo.org/records/24543680?preview_file=affectfusion_dataset_v1.0.0.zip)

## Split Provenance & Zero-Leakage Guarantee

| Split | File(s) | Rows | Real Scraped | DAIR-AI (Train-Only) | Gemini Flash Aug. (Train-Only) | Consensus Human Gold | Emoji Share |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (80%)** | `train_24k.parquet` | **24,000** | 18,872 | 1,600* | 3,528* | 0 (AI `conf >= 3`) | 59.09% |
| **Validation (10%)** | `val_3k.parquet`, `val_3k.csv` | **3,000** | **3,000 (100%)** | **0** | **0** | **2,000 (66.7%)** | 69.97% |
| **Test (10%)** | `test_3k.parquet`, `test_3k.csv` | **3,000** | **3,000 (100%)** | **0** | **0** | **3,000 (100%)** | 65.80% |
| **Total** | — | **30,000** | **24,872** | **1,600** | **3,528** | **5,000 (κ = 0.7717, α = 0.7717)** | **60.85%** |

*\*`1,600` Train-only DAIR-AI rows (`Fear 1,100` + `Surprise 500`) and `3,528` Train-only Gemini Flash emoji-preserving paraphrases (`Fear +1,384` incl. `1,116` cross-lingual ENG→BN/BNG, `Surprise +1,041`, `Anger +613`, `Disgust +396`, `Anticipation +94`) are quarantined strictly inside `train_24k`; strictly `0` augmented or external rows appear in `val_3k` or `test_3k`.*

## 9 Emotion Classes (`label_id`: `0`–`8`) & Consensus Gold Counts
`0: Joy`, `1: Neutral`, `2: Anger`, `3: Sadness`, `4: Surprise`, `5: Sarcasm`, `6: Fear`, `7: Disgust`, `8: Anticipation`.

- **Consensus `val_3k` (`3,000`):** Anger `431`, Neutral `362`, Joy `348`, Fear `339`, Sarcasm `336`, Sadness `330`, Surprise `299`, Anticipation `291`, Disgust `264`.
- **Consensus `test_3k` (`3,000`):** Anger `470`, Joy `392`, Neutral `385`, Sarcasm `367`, Fear `355`, Sadness `333`, Anticipation `273`, Disgust `218`, Surprise `207`.

## Unified Dual-Stream Schema (25 Core Columns + Consensus Audit Flag)
`row_id`, `raw_text`, `text_stream`, `emoji_stream`, `demoji_text`, `baseline_input`, `label`, `label_id`, `language_tag`, `language_id`, `has_emoji`, `emoji_count`, `unique_emoji_count`, `emoji_density`, `text_length`, `word_count`, `exclamation_count`, `question_count`, `uppercase_ratio`, `has_sarcasm_emoji_cue`, `ai_confidence`, `platform`, `is_augmented`, `is_external_train_only`, `is_human_verified` (+ `is_double_adjudicated` in Consensus Gold `val_3k` and `test_3k`).

## 2-Round Blind-Swap Human-vs-Human Agreement (`../gold_verification/round2_blind_swap/`)
- **Paired Human-vs-Human Audit (`N = 5,000` rows × 2 annotators = `10,000` independent judgments by MD. Ragib Hasan & MD. Asadulla Hel Galib):** **79.80% Unanimous Agreement (`3,990 / 5,000`)**, **Cohen's κ = 0.7717**, **Krippendorff's α = 0.7717**.
- **By Language:** `ENG (1,933)`: `79.20%`, `κ = 0.7651`, `α = 0.7651`; `BN (1,653)`: `82.94%`, `κ = 0.8052`, `α = 0.8052`; `BNG (1,414)`: `76.94%`, `κ = 0.7377`, `α = 0.7378`.
- **By Split & Adjudication:** `test_3k (3,000)`: `78.03%`, `κ = 0.7515` (`472` labels updated vs. Round 1); `val_3k (2,000)`: `82.45%`, `κ = 0.8019` (`264` labels updated vs. Round 1). Across `1,010` disagreements: `526` resolved to Annotator 1, `482` to Annotator 2, `2` to a third class.
