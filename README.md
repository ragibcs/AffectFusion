# AffectFusion: A Cross-Attention Transformer Framework for Explainable Multimodal Emotion Recognition from Text and Emoji Semantics

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()
[![Institution](https://img.shields.io/badge/Institution-Daffodil%20International%20University-red.svg)](https://daffodilvarsity.edu.bd/)

Final Year Design Project (FYDP) / Thesis Project  
**Department of Computer Science and Engineering (CSE)**  
**Daffodil International University (DIU), Dhaka, Bangladesh**

---

## 👥 Authors & Supervision

- **Lead Researchers:**
  - **MD. RAGIB HASAN** (Student ID: `232-15-775`)
  - **MD. ASADULLA HEL GALIB** (Student ID: `232-15-270`)
- **Supervisor:**
  - **Sharmin Akter**, Assistant Professor, Dept. of CSE, Daffodil International University
- **Co-Supervisor:**
  - **Dewan Mamun Raza**, Assistant Professor, Dept. of CSE, Daffodil International University

---

## 📌 Project Overview

Digital communication increasingly relies on emojis to modify, amplify, or contradict the textual sentiment of messages (e.g., sarcasm, irony, emotional emphasis). Conventional text-only NLP models fail to interpret these multimodal signals, while early fusion (simple concatenation) models fail to learn fine-grained token-emoji interactions.

**AffectFusion** introduces a cross-attention transformer framework that:
1. Represents textual tokens and emoji semantics through dedicated contextualized and learned embedding spaces.
2. Employs a multi-head cross-attention fusion mechanism where text representations query emoji key-value representations and vice-versa.
3. Integrates post-hoc explainability (SHAP, LIME, Attention Weight Visualization) for transparent, trustworthy emotion classification.
4. Evaluates performance on real-world, noisy, code-mixed social media datasets across **Bengali (বাংলা)**, **Banglish (Transliterated Bengali)**, and **Regional English**.

---

## 📊 Dataset Structure & Statistics

The repository contains multi-platform social media datasets specifically harvested and preprocessed for multimodal emotion recognition and text-emoji contradiction analysis.

### 1. YouTube Datasets (90,000 Balanced Samples)

Each YouTube dataset contains **30,000 rows** balanced strictly with **15,000 samples with emojis (`has_emoji = 1`)** and **15,000 samples without emojis (`has_emoji = 0`)**:

| Folder / Dataset | Target Script & Language | Total Samples | With Emoji (50%) | Without Emoji (50%) | Format |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 📁 `AffectFusion_Bangla_YouTube/` | Native Bengali Script (বাংলা) | **30,000** | 15,000 | 15,000 | `.csv` + `.xlsx` |
| 📁 `AffectFusion_Banglish_YouTube/` | Banglish (Transliterated Bengali) | **30,000** | 15,000 | 15,000 | `.csv` + `.xlsx` |
| 📁 `AffectFusion_English_YouTube/` | Regional Community English | **30,000** | 15,000 | 15,000 | `.csv` + `.xlsx` |
| **Total YouTube Corpus** | **All 3 Modalities** | **90,000** | **45,000** | **45,000** | — |

---

### 2. Reddit Datasets (10,652 Samples)

Harvested across `r/bangladesh`, `r/kolkata`, `r/Dhaka`, and 50+ emotional keyword search clusters:

| Dataset File | Language / Category | Sample Count | With Emojis |
| :--- | :--- | :---: | :---: |
| 📄 `Bangla_reddit_unlabeled.csv` | Native Bengali (বাংলা) | **2,192 rows** | 319 (14.6%) |
| 📄 `Banglish_reddit_unlabeled.csv` | Banglish (বাংলিশ) | **1,613 rows** | 237 (14.7%) |
| 📄 `English_reddit_unlabeled.csv` | Regional English | **6,847 rows** | 648 (9.5%) |
| 📄 `All_Reddit_Unlabeled.csv` | **Master Combined Reddit Dataset** | **10,652 rows** | **1,204 (11.3%)** |

---

## 📑 20-Column Data Schema Specification

All dataset files (`.csv` and `.xlsx`) strictly adhere to the unified 20-column AffectFusion schema:

| # | Column Name | Type | Description |
| :---: | :--- | :---: | :--- |
| **1** | `ID` | `string` | Unique UUID identifier for every sample. |
| **2** | `Data` | `string` | Cleaned social media post/comment (PII, URLs, handles stripped; emojis preserved). |
| **3** | `text_only` | `string` | Pure textual content with all emojis removed. |
| **4** | `emoji_stream` | `string` | Extracted emojis in order of appearance (e.g. `😍,🎉` or `💀`). |
| **5** | `has_emoji` | `int (0/1)` | Binary flag indicating emoji presence (`1` if emoji count > 0, else `0`). |
| **6** | `emoji_count` | `int` | Total number of emojis present in the sample. |
| **7** | `word_count` | `int` | Total number of whitespace-delimited words in `text_only`. |
| **8-17** | **10 Emotion Columns**<br>`Joy, Sadness, Anger, Fear, Surprise, Disgust, Love, Anticipation, Sarcasm, Neutral` | `int (0/1)` / `empty` | Multi-label binary indicators for target emotional categories aligned with GoEmotions. Blank for unlabeled pools; populated in annotated subsets. |
| **18** | `label_source` | `string` | Annotation source (`pending_manual`, `hybrid_annotated`, `manual_verified`). |
| **19** | `annotator_agreement` | `float` | Inter-annotator confidence score / modality alignment score (e.g., `0.95`, `0.80`). |
| **20** | `sarcasm_or_contradiction_flag` | `int (0/1)` | Binary indicator (`1` if text sentiment and emoji semantics contradict). |

---

## 🛠️ Repository File Tree

```
├── AffectFusion_Bangla_YouTube/
│   ├── Bangla_YouTube_Unlabeled_30000.csv       # 30,000 Bangla YouTube samples (15k emoji / 15k no-emoji)
│   ├── Bangla_YouTube_Unlabeled_30000.xlsx      # Excel version
│   ├── cleaned_candidate_pool.csv               # Cleaned candidate pool
│   └── raw_scraped_comments.csv                 # 160k raw scraped comments pool
│
├── AffectFusion_Banglish_YouTube/
│   ├── Banglish_YouTube_Unlabeled_30000.csv     # 30,000 Banglish YouTube samples (15k emoji / 15k no-emoji)
│   ├── Banglish_YouTube_Unlabeled_30000.xlsx    # Excel version
│   └── cleaned_candidate_pool.csv
│
├── AffectFusion_English_YouTube/
│   ├── English_YouTube_Unlabeled_30000.csv      # 30,000 English YouTube samples (15k emoji / 15k no-emoji)
│   ├── English_YouTube_Unlabeled_30000.xlsx     # Excel version
│   └── cleaned_candidate_pool.csv
│
├── AffectFusion_all_reddit/
│   ├── All_Reddit_Unlabeled.csv                 # 10,652 Master Reddit samples
│   ├── Bangla_reddit_unlabeled.csv              # 2,192 Bangla Reddit samples
│   ├── Banglish_reddit_unlabeled.csv            # 1,613 Banglish Reddit samples
│   └── English_reddit_unlabeled.csv             # 6,847 English Reddit samples
│
├── generate_youtube_30k_final.py                # Multi-threaded YouTube dataset builder
├── multi_token_reddit_scraper.py                # Multi-token rotating Reddit scraper
├── build_full_dataset.py                        # Automated multilingual scraper & hybrid annotator
├── thesis_data_scraper.py                       # Modular multi-platform scraper (YouTube, Reddit, Twitter, FB)
├── SU26D1-775.pdf                               # Thesis Proposal Document
├── SU26D1-775 Title Defense Evaluation Report.pdf
├── requirements.txt                             # Python dependencies
├── .env.example                                 # Environment variables template
└── README.md                                    # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/ragibcs/AffectFusion.git
cd AffectFusion

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and insert your API credentials if running scrapers:
```bash
cp .env.example .env
```

---

## 📖 Citation & Academic Contact

For questions regarding the dataset, preprocessing pipeline, or model architecture:

- **MD. Ragib Hasan** — `hasan2305101775@diu.edu.bd`
- **MD. Asadulla Hel Galib** — `galib2305101270@diu.edu.bd`
- **Department of Computer Science and Engineering, Daffodil International University**

---

*© 2026 Department of Computer Science and Engineering, Daffodil International University.*
