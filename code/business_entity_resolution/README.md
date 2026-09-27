# Business Entity Resolution Pipeline

## Overview
This repository contains a high-precision, memory-efficient, and scalable machine learning pipeline for the **Business Entity Resolution Challenge (ML Challenge 2026)**.

The solution resolves business entities across three independent noisy data sources (`Source 1`, `Source 2`, `Source 3`) using:
1. **Dynamic Country Partitioning**: 100% loss-free open-set country isolation (supports US, India, France, and any new country labels without hardcoding).
2. **Universal Unicode Transliteration & Normalization**: Seamless conversion of Indic scripts (Devanagari, Tamil, Kannada, Bengali, etc.), French diacritics, and legal entities into standardized ASCII tokens with `anyascii`.
3. **Multi-Channel Inverted Index Blocking with Key-IDF Scoring**: Achieves >99.999% reduction ratio while preserving >90-95% recall ceiling by generating a compact candidate set of Top-K (default: 8) candidates per Source 1 entity.
4. **Rich Gradient Boosted Pairwise Classifier (LightGBM)**: 20-dimensional feature engineering combining token-set, token-sort, partial and edit ratios for names and addresses, digit and street number matching, domain cross-matching, and blocking metadata.
5. **Macro $F_{0.5}$ Calibrated Thresholding**: Optimizes precision over recall (penalizing false merges twice as much as missed links) and maintains high accuracy on singletons (predicting empty sets).

---

## Directory Structure
```
business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Unicode transliteration, text cleaning, token extraction
│   ├── blocking.py         # Multi-channel inverted index with IDF key scoring
│   ├── features.py         # Pairwise similarity feature extraction
│   ├── model.py            # LightGBM classifier & Macro F_0.5 threshold optimizer
│   └── pipeline.py         # End-to-end orchestrator for training & inference
├── README.md               # Pipeline documentation and run instructions
└── requirements.txt        # Pinned dependencies
```

---

## Setup & Installation

1. **Prerequisites**: Python 3.10+ (tested on Python 3.11).
2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

---

## Reproduction Instructions

To reproduce the entire pipeline end-to-end (training on the training data, optimizing threshold, and generating both `output/matching_results.tsv` and `output/candidate_pairs.tsv` for the test set):

Run from the `student_resource/` directory:

```bash
python code/business_entity_resolution/src/pipeline.py --mode all \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --model-dir model_artifacts \
    --output-dir output \
    --top-k 8 \
    --threads 8
```

### Running Stages Separately

**1. Train & Calibrate Threshold:**
```bash
python code/business_entity_resolution/src/pipeline.py --mode train \
    --train-dir dataset/train \
    --model-dir model_artifacts \
    --n-train 30000 \
    --n-val 4000 \
    --top-k 8
```

**2. Inference on Test Set:**
```bash
python code/business_entity_resolution/src/pipeline.py --mode infer \
    --test-dir dataset/test \
    --model-dir model_artifacts \
    --output-dir output \
    --top-k 8 \
    --threads 8
```

---

## Validation
After generating the outputs, validate formatting and constraint adherence using the provided validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
