# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** ARISE
**Submission Date:** 26 September 2026

---

## 1. Executive Summary
We present a high-precision, scalable Machine Learning architecture for large-scale Business Entity Resolution across three heterogeneous, noisy data sources (`Source 1`, `Source 2`, `Source 3`). Our solution combines **Dynamic Open-Set Country Partitioning**, **Universal Script Normalization** (bridging Latin, Indic, and French variations via AnyASCII and American Soundex phonetic encoding), a **Multi-Channel Inverted Index with IDF Key Scoring** that cuts the comparison space by >99.999% while producing a compact candidate pool of Top-12 candidates per entity, a **LightGBM Pairwise Classifier** calibrated specifically for the macro-averaged $F_{0.5}$ metric, and a **Global 1-to-1 Target Conflict Resolution Engine** that eliminates duplicate target assignments and guarantees zero false-positive target collisions.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) across 12.5M training records and 11.7M test records revealed fundamental domain structures:
1. **Strict Country Isolation**: Across all 7,638,365 true ground truth match pairs in the entire training set, **0.000%** of matches cross country boundaries. Businesses strictly resolve within their respective national boundaries (`US`, `India`, `France`).
2. **Partial Function Bijection (Target Uniqueness)**: In ground truth, **0 targets match multiple Source 1 entities**. Every target record represents a distinct real-world legal record that belongs to at most one Source 1 business entity. Without global conflict resolution, independent scoring produces hundreds of thousands of duplicate match assignments that penalize precision.
3. **Script & Transliteration Discrepancies**: While Source 1 records are in Latin script, Source 2 and Source 3 in India contain non-Latin scripts (Devanagari, Tamil, Kannada, Bengali, Gujarati). Transliterations introduce phonetic drift (`green` vs `grin`, `shiv` vs `siv`, `shree` vs `sri`, `bombay` vs `bambe`). Soundex phonetic blocking maps these transliterations to identical invariant keys.
4. **Initials & Legal Suffixes**: Frequent abbreviations (e.g. `SK`, `RK`, `MK`) must be preserved as 2-letter name tokens rather than discarded by standard length filters.
5. **Singleton Distribution**: ~5.6% of Source 1 entities have zero true matches in Sources 2 and 3. In macro $F_{0.5}$ evaluation, correctly identifying a singleton earns 1.0, while any false positive merge collapses the score to 0.0.

### 2.2 Solution Strategy
We adopt a **Three-Stage Cascaded ML Architecture**:
- **Stage 1 (Candidate Generation / Blocking)**: Sub-linear multi-channel inverted indexing with document frequency thresholding, phonetic Soundex mapping, and key-IDF ranking. Emits a tight candidate set of Top-12 candidates per Source 1 entity.
- **Stage 2 (Pairwise Scoring & Decision Calibration)**: A 20-dimensional gradient-boosted decision tree (LightGBM) trained on hard negatives from the blocking stage. A calibrated decision threshold ($\theta^* = 0.75$) prioritizes precision in accordance with the macro $F_{0.5}$ metric.
- **Stage 3 (Global 1-to-1 Conflict Resolution)**: Post-inference bipartite target assignment: if multiple Source 1 entities predict the same target record $T$, $T$ is assigned exclusively to the entity with the highest prediction probability $\arg\max P(\text{match})$, completely eliminating duplicate false positives.

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Keys Used
To ensure high recall without combinatoric explosion, blocking keys are extracted across 5 complementary channels:
1. **Distinctive Name Tokens (`N:{Country}:{token}`)**: Informative alphanumeric words from the business name after removing legal suffixes (`Inc`, `LLC`, `Pvt Ltd`, `SARL`, `SAS`, `SCI`, `EURL`), preserving 2-letter initials.
2. **Prefix 4-Gram Name Keys (`NP:{Country}:{prefix4}`)**: First 4 characters of words with length $\ge 4$, providing robust candidate retrieval under misspellings and OCR/transcription typos.
3. **Phonetic Soundex Keys (`SX:{Country}:{soundex}`)**: 4-character American Soundex codes on distinctive name words ($\ge 3$ characters), bridging Indic phonetic transliteration drift (`grin` $\leftrightarrow$ `green`, `siv` $\leftrightarrow$ `shiv`).
4. **Dual Numeric Street Co-occurrence (`A:{Country}:{num}_{text}`)**: Primary and secondary street numbers paired with the first 3 non-stopword address tokens, capturing physical locations even under inverted number ordering (e.g., `No. 225 E-7` vs `E-7`).
5. **Numeric Compound Keys (`A:{Country}:{num1}_{num2}`)**: Primary street number paired with postal PIN/ZIP codes or secondary unit numbers.

### 3.2 Key Weighting & Candidate Pruning
- Each key $k$ is weighted by its Inverse Document Frequency:
  $$\text{IDF}(k) = \ln\left(1 + \frac{N}{1 + \text{DF}(k)}\right)$$
- Keys with document frequency exceeding 4,000 occurrences are pruned to prevent posting list bloat.
- Candidates for each Source 1 entity are scored by accumulating shared key IDF weights across the top 12 rarest keys, retaining the **Top-12** highest scoring records.

### 3.3 Candidate Set Statistics & Recall Guarantee
- **Total candidate pairs generated**: Exactly $\le 12$ candidates per Source 1 entity ($\approx 11.1$ candidates/entity average).
- **Search Space Reduction**: Cuts the $6.7 \times 10^{12}$ cross-source pair space down to $\approx 1.9 \times 10^7$ candidate pairs (**>99.9998% reduction ratio**).
- **Recall Upper Bound**: Preserves **>98.9%** true match coverage within the candidate pool.

---

## 4. Matching Model

### 4.1 Feature Engineering (20 Dimensions)
For every candidate pair $(S_1, S_{\text{cand}})$, we compute a comprehensive feature vector:
1. **Name Similarity Features**:
   - `n_ratio`: Normalized Levenshtein edit distance ratio.
   - `n_tsort`: RapidFuzz Token Sort ratio (invariant to word transposition).
   - `n_tset`: RapidFuzz Token Set ratio (invariant to legal suffix additions).
   - `n_part`: Partial string alignment ratio.
   - `n_len_diff` & `n_len_ratio`: Character length disparity metrics.
   - `n_jaccard`: Word-level Jaccard intersection over union.
2. **Address Similarity Features**:
   - `addr_null`: Binary indicator (1.0 if target address is null/missing).
   - `a_ratio`, `a_tsort`, `a_tset`, `a_part`: Address token and edit similarities.
   - `a_jaccard`: Word-level token overlap across addresses.
   - `num_common`: Count of shared numeric tokens (house numbers, postal codes).
   - `first_num_match`: Binary indicator whether the leading street number matches exactly.
3. **Cross-Field Features**:
   - `cross1`: Partial alignment of $S_1$ business name inside candidate address.
   - `cross2`: Partial alignment of candidate business name inside $S_1$ address.
4. **Blocking Metadata**:
   - `is_s2`: Binary flag denoting whether candidate originates from Source 2 vs Source 3.
   - `b_score`: Accumulated IDF blocking score.
   - `b_rank`: Rank position (1 to 12) within the candidate list.

### 4.2 Model Architecture & Hyperparameters
- **Classifier**: LightGBM (`LGBMClassifier`)
- **Objective**: Binary Logloss (`binary`)
- **Trees**: 300 estimators
- **Learning Rate**: 0.06
- **Tree Depth / Complexity**: `num_leaves = 63`, `min_child_samples = 20`
- **Subsampling**: `subsample = 0.8`, `colsample_bytree = 0.8`

### 4.3 Decision Threshold & Global Conflict Resolution
Because the macro $F_{0.5}$ metric penalizes false positives twice as heavily as false negatives:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
We calibrated the optimal decision threshold to **$\theta^* = 0.75$** combined with global target assignment:
$$\text{Assign}(T) = \arg\max_{S_1} P(\text{match}(S_1, T)) \quad \text{subject to } P \ge 0.75$$
This simultaneously maximizes precision (>97.5%) and preserves high recall (>87.3%), while strictly preventing multiple Source 1 entities from claiming the same secondary record.

---

## 5. Results & Validation

- **Macro $F_{0.5}$ Score (Validation - India)**: **0.9226**
- **Macro $F_{0.5}$ Score (Validation - US)**: **0.9636**
- **Precision**: 97.7% - 98.2%
- **Recall**: 87.3% - 91.5%
- **Singleton Accuracy**: 99.1% (correctly identified empty match lists)

---

## 6. Conclusion
By uniting phonetic Soundex indexing, dual-number address co-occurrence, a calibrated LightGBM classifier, and global 1-to-1 conflict resolution, our architecture achieves an exceptional **~0.92+ Macro $F_{0.5}$ score** across diverse jurisdictions while generating an average of only ~11.1 candidates per entity. The end-to-end pipeline processes over 11.7 million records in parallel, scaling seamlessly to production requirements.

---

## Appendix: Code Artefacts
All runnable code is organized inside `code/business_entity_resolution/`:
- `src/preprocessing.py`: Unicode transliteration, text cleaning, 2-letter token preservation, and Soundex.
- `src/blocking.py`: Multi-channel country inverted indexing and Top-K candidate generation.
- `src/features.py`: 20-dimensional pairwise feature vector calculation.
- `src/model.py`: LightGBM training, threshold calibration, and serialization.
- `src/pipeline.py`: Unified command-line orchestrator with global target collision resolution.
- `README.md` & `requirements.txt`: Environment setup and reproduction instructions.

**Reproduction Command:**
```bash
python code/business_entity_resolution/src/pipeline.py --mode infer \
    --test-dir dataset/test \
    --model-dir model_artifacts \
    --output-dir output \
    --top-k 12 \
    --threads 14 \
    --threshold 0.75
```

