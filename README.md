# 🏆 ARISE — Business Entity Resolution | ML Challenge 2026

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen?style=for-the-badge)
![F0.5 Score](https://img.shields.io/badge/F₀.₅%20Score-0.9725-gold?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**Team ARISE** · ML Challenge 2026 · Business Entity Resolution

*Macro F₀.₅ Score: **0.9725** | Precision: **0.9911** | Recall: **0.9389***

</div>

---

## 📌 About the Hackathon

**Event:** ML Challenge 2026 — Business Entity Resolution  
**Team Name:** ARISE  
**Submission Date:** 26 September 2026  
**Final Submission:** `ARISE_submission.zip`  

### The Challenge

In large-scale commercial platforms, business identity data arrives from **multiple independent sources** — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers.

The task: given business records from **3 independent data sources** with noisy and inconsistent fields, determine which records across sources refer to the **same real-world business entity**.

> **Source 1** is the deduplicated reference source. Find all matching records from Source 2 and Source 3 for each Source 1 entity.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| **Macro F₀.₅ (Validation)** | **0.9725** |
| Precision | 0.9911 |
| Recall | 0.9389 |
| Singleton Accuracy | 99.2% |
| Candidates per Entity | ~11.8 (Top-12) |
| Search Space Reduction | >99.9998% |

---

## 🏗️ Solution Architecture

We adopt a **Two-Stage Cascaded ML Architecture**:

```
Raw Data (3 Sources x 3 Countries)
         |
         v
+-------------------------------------+
|  Stage 1: Candidate Generation      |
|  * anyascii transliteration         |
|  * Domain name normalization        |
|  * Multi-Channel Inverted Index     |
|    - Distinctive Name Tokens        |
|    - Prefix 4-Gram Name Keys        |
|    - Numeric Street Co-occurrence   |
|    - Numeric Compound Keys          |
|  * IDF-weighted Top-12 scoring      |
+-------------------------------------+
               |  ~11.8 candidates/entity
               v
+-------------------------------------+
|  Stage 2: LightGBM Pairwise Scorer  |
|  * 20-dimensional feature vector    |
|  * Name similarity (7 features)     |
|  * Address similarity (7 features)  |
|  * Cross-field features (2)         |
|  * Blocking metadata (4)            |
|  * Calibrated threshold theta=0.68  |
+-------------------------------------+
               |
               v
      matching_results.tsv
```

---

## 🔑 Key Innovations

### 1. Dynamic Open-Set Country Partitioning
Zero cross-country matches exist in training data (verified across 7.6M+ ground truth pairs). We partition all blocking and matching strictly by country — enabling zero-overhead handling of unseen `France` test entities.

### 2. Universal Script Normalization
- **India**: 23.5% non-Latin scripts (Devanagari, Tamil, Kannada, Bengali) → `anyascii` transliteration
- **France**: Diacritic normalization (`é→e`, `à→a`, `è→e`, `î→i`)
- **Domain Names**: 6.2% of names are unspaced internet domains → regex token splitting

### 3. Multi-Channel Inverted Index with IDF Scoring
4 complementary blocking channels ensure >95% recall while reducing the 6.7×10¹² pair space by >99.9998%:
- `N:{Country}:{token}` — distinctive name tokens
- `NP:{Country}:{prefix4}` — 4-gram prefix keys for typo robustness
- `A:{Country}:{num}_{text}` — street number + address token co-occurrence
- `A:{Country}:{num1}_{num2}` — numeric compound keys (street + PIN/ZIP)

### 4. F₀.₅-Calibrated Decision Threshold
Grid search over θ ∈ [0.45, 0.85] on validation candidates → **θ* = 0.68** achieving:
- 2× precision weight alignment with the evaluation metric
- Perfect calibration with the ~6% singleton distribution

---

## 📁 Repository Structure

```
student_resource/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocessing.py      # Unicode transliteration, cleaning, tokenization
│       │   ├── blocking.py           # Multi-channel inverted index (v1)
│       │   ├── blocking_v2.py        # Multi-channel inverted index (v2, final)
│       │   ├── features.py           # 20-dim pairwise feature vector
│       │   ├── model.py              # LightGBM training + threshold calibration
│       │   └── pipeline.py           # End-to-end CLI orchestrator
│       ├── README.md                 # Reproduction instructions
│       └── requirements.txt          # Pinned dependencies
├── dataset/
│   ├── train/                        # Training TSV files + ground truth
│   └── test/                         # Test TSV files
├── model_artifacts/                  # Trained model + vectorizers
├── output/                           # Final submission outputs
│   ├── matching_results.tsv          # Leaderboard submission file
│   └── candidate_pairs.tsv           # Blocking candidate set
├── scratch/                          # EDA and experimental notebooks
├── utils/
│   └── validate_submission.py        # Submission validator (stdlib only)
├── Documentation_template.md         # Filled methodology document
└── README.md                         # Challenge problem statement (official)
```

---

## ⚙️ Feature Engineering (20 Dimensions)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `n_ratio` | Name | Normalized Levenshtein edit distance ratio |
| 2 | `n_tsort` | Name | Token Sort ratio (word-order invariant) |
| 3 | `n_tset` | Name | Token Set ratio (suffix-invariant) |
| 4 | `n_part` | Name | Partial string alignment ratio |
| 5 | `n_len_diff` | Name | Character length disparity |
| 6 | `n_len_ratio` | Name | Character length ratio |
| 7 | `n_jaccard` | Name | Word-level Jaccard IoU |
| 8 | `addr_null` | Address | Binary: target address missing |
| 9 | `a_ratio` | Address | Levenshtein edit distance ratio |
| 10 | `a_tsort` | Address | Token Sort ratio |
| 11 | `a_tset` | Address | **#1 Feature** — Token Set ratio |
| 12 | `a_part` | Address | Partial alignment ratio |
| 13 | `a_jaccard` | Address | Word-level token overlap |
| 14 | `num_common` | Address | Shared numeric token count |
| 15 | `first_num_match` | Address | Binary: leading street number match |
| 16 | `cross1` | Cross | S1 name partial inside candidate address |
| 17 | `cross2` | Cross | Candidate name partial inside S1 address |
| 18 | `is_s2` | Meta | Binary: Source 2 vs Source 3 |
| 19 | `b_score` | Meta | Accumulated IDF blocking score |
| 20 | `b_rank` | Meta | Rank position within candidate list (1-12) |

---

## 🚀 Reproduction

### Prerequisites
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### Run Full Pipeline
```bash
python code/business_entity_resolution/src/pipeline.py --mode all \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --model-dir model_artifacts \
    --output-dir output \
    --top-k 12 \
    --threads 12 \
    --threshold 0.68
```

### Validate Submission
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Expected output: `PASS` ✅

---

## 📊 Model Details

| Parameter | Value |
|-----------|-------|
| Classifier | LightGBM (`LGBMClassifier`) |
| Objective | Binary Logloss |
| Estimators | 300 trees |
| Learning Rate | 0.06 |
| Num Leaves | 63 |
| Min Child Samples | 20 |
| Subsample | 0.8 |
| Colsample by Tree | 0.8 |
| Decision Threshold | **θ* = 0.68** |

---

## 📈 Evaluation Metric

F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

- **Precision-heavy**: False merges penalized 2× more than missed links
- **Macro-averaged**: Per Source-1 entity, then averaged across all entities
- **Singletons included**: Correct empty predictions score 1.0

---

## 🛠️ Tech Stack

- **Python** 3.10+
- **LightGBM** — gradient boosted decision trees
- **RapidFuzz** — fast string similarity (Levenshtein, Token Sort/Set, Partial)
- **anyascii** — Unicode → ASCII transliteration for Indic + European scripts
- **pandas** — data ingestion and manipulation
- **numpy** — numerical feature computation
- **joblib** — model serialization and parallel processing

---

## 📝 Methodology

Full technical write-up in [`Documentation_template.md`](Documentation_template.md).

Key design decisions:
1. **Country isolation** — 0% cross-country matches → partition-first strategy
2. **Compact candidates** — Top-12 per entity balances recall ceiling with scoring speed
3. **Hard negative training** — blocking-stage negatives make the classifier boundary sharp
4. **Threshold calibration** — grid search directly on macro F₀.₅ (not AUC)

---

## 👥 Team ARISE

Built for the **ML Challenge 2026** hackathon.

---

## 📄 License

MIT License. All models are MIT/Apache 2.0 licensed and ≤8B parameters per competition rules.
