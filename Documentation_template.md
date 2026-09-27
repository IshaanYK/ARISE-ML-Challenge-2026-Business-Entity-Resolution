# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** ARISE
**Submission Date:** 26 September 2026

---

## 1. Executive Summary
We present a high-precision, scalable Machine Learning architecture for large-scale Business Entity Resolution across three heterogeneous, noisy data sources (`Source 1`, `Source 2`, `Source 3`). Our solution combines **Dynamic Open-Set Country Partitioning**, **Universal Script Normalization** (bridging Latin, Indic, and French variations), a **Multi-Channel Inverted Index with IDF Key Scoring** that reduces the comparison space by >99.999% while producing a compact candidate pool of Top-12 candidates per entity, and a **LightGBM Pairwise Classifier** calibrated specifically for the macro-averaged $F_{0.5}$ metric (achieving **0.9725** on validation data).

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) across 12.5M training records and 11.7M test records revealed fundamental domain structures:
1. **Strict Country Isolation**: Across all 7,638,365 true ground truth match pairs in the entire training set, **0.000%** of matches cross country boundaries. Businesses strictly resolve within their respective national boundaries (`US`, `India`, `France`).
2. **Script & Transliteration Discrepancies**: While Source 1 records are 100% in Latin script, Source 2 and Source 3 in India contain 23.5% non-Latin scripts (Devanagari, Tamil, Kannada, Bengali, etc.). In France, diacritics and accents (`é`, `à`, `è`, `î`) cause naive string equality to fail.
3. **Domain Name Artifacts**: Approximately 6.2% of business names in Source 2 and Source 3 are formatted as unspaced internet domains (e.g. `maurewilliamscolombier.com`, `wareelvinamd.com`), which mask standard word token boundaries.
4. **Missing vs. DBA Attributes**: 
   - 88.0% of true matches exhibit strong similarity across both name and address.
   - 7.5% of true matches have missing/null addresses in Source 2/3 but share exact/near-exact names.
   - 4.3% represent trade names / DBAs where business names diverge completely, but street addresses and municipal building numbers match almost verbatim.
5. **Singleton Distribution**: 5.58% of Source 1 entities have zero true matches in Sources 2 and 3. In macro $F_{0.5}$ evaluation, correctly identifying a singleton earns a full 1.0, while any false positive merge collapses the score to 0.0.

### 2.2 Solution Strategy
We adopt a **Two-Stage Cascaded ML Architecture**:
- **Stage 1 (Candidate Generation / Blocking)**: Sub-linear multi-channel inverted indexing with document frequency thresholding and key-IDF ranking. Emits a tight candidate set of Top-12 candidates per Source 1 entity, directly addressing the competition's core evaluation criterion favoring compact candidate sets.
- **Stage 2 (Pairwise Scoring & Decision Calibration)**: A 20-dimensional gradient-boosted decision tree (LightGBM) trained on hard negatives from the blocking stage. A calibrated decision threshold ($\theta^* = 0.68$) is chosen via exhaustive macro $F_{0.5}$ validation optimization, prioritizing precision while aligning with the ~6% singleton prior.

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Keys Used
To ensure high recall without combinatoric explosion, blocking keys are extracted across 4 complementary channels after `anyascii` transliteration and domain normalization:
1. **Distinctive Name Tokens (`N:{Country}:{token}`)**: Informative alphanumeric words from the business name after removing legal suffixes (`Inc`, `LLC`, `Pvt Ltd`, `SARL`, `SAS`, `SCI`, `EURL`).
2. **Prefix 4-Gram Name Keys (`NP:{Country}:{prefix4}`)**: First 4 characters of words with length $\ge 5$, providing robust candidate retrieval under misspellings and OCR/transcription typos.
3. **Numeric Street Co-occurrence (`A:{Country}:{num}_{text}`)**: Street or plot numbers paired with the first 3 non-stopword address tokens, capturing physical locations even when trade names differ.
4. **Numeric Compound Keys (`A:{Country}:{num1}_{num2}`)**: Primary street number paired with postal PIN/ZIP codes or secondary unit numbers.

### 3.2 Key Weighting & Candidate Pruning
- Each key $k$ is weighted by its Inverse Document Frequency:
  $$\text{IDF}(k) = \ln\left(1 + \frac{N}{1 + \text{DF}(k)}\right)$$
- Keys with document frequency exceeding 4,000 occurrences are pruned to prevent posting list bloat while retaining high-value business terms.
- Candidates for each Source 1 entity are scored by accumulating shared key IDF weights, retaining the **Top-12** highest scoring records.

### 3.3 Candidate Set Statistics & Recall Guarantee
- **Total candidate pairs generated**: Exactly $\le 12$ candidates per Source 1 entity ($\approx 11.8$ candidates/entity average).
- **Search Space Reduction**: Cuts the $6.7 \times 10^{12}$ cross-source pair space down to $\approx 2.0 \times 10^7$ candidate pairs (**>99.9998% reduction ratio**).
- **Recall Upper Bound**: Preserves **>95.0%** true match coverage within the top-12 candidates.

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
   - `b_rank`: Rank position (1 to 8) within the candidate list.

### 4.2 Model Architecture & Hyperparameters
- **Classifier**: LightGBM (`LGBMClassifier`)
- **Objective**: Binary Logloss (`binary`)
- **Trees**: 300 estimators
- **Learning Rate**: 0.06
- **Tree Depth / Complexity**: `num_leaves = 63`, `min_child_samples = 20`
- **Subsampling**: `subsample = 0.8`, `colsample_bytree = 0.8`

### 4.3 Feature Importance Analysis
Gini gain ranking demonstrates that address token set similarity (`a_tset`), name partial ratio (`n_part`), and address null indicators (`addr_null`) provide the strongest discriminative power:
| Rank | Feature | Importance (Gain) | Description |
|:---:|:---|:---:|:---|
| 1 | `a_tset` | 1,330,165.3 | Address Token Set Ratio |
| 2 | `n_part` | 48,258.8 | Name Partial Ratio |
| 3 | `addr_null` | 42,100.3 | Address Missing / Null Indicator |
| 4 | `a_tsort` | 24,073.5 | Address Token Sort Ratio |
| 5 | `n_tset` | 22,356.2 | Name Token Set Ratio |
| 6 | `b_score` | 15,502.7 | Accumulated IDF Blocking Score |
| 7 | `n_tsort` | 12,585.2 | Name Token Sort Ratio |

### 4.4 Decision Threshold Calibration ($F_{0.5}$ Optimization)
Because the macro $F_{0.5}$ metric penalizes false positives twice as heavily as false negatives:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
We performed grid search calibration across validation thresholds $\theta \in [0.45, 0.85]$ on candidate pairs generated with $K=12$. The optimal threshold evaluated to **$\theta^* = 0.68$**, which achieves an ideal balance: 0.9911 precision and 0.9389 recall, perfectly aligning with the true ~6.0% singleton distribution (6.2% empty rate).

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation Set)**: **0.9725**
- **Precision**: 0.9911
- **Recall**: 0.9389
- **Singleton Accuracy**: 99.2% (correctly predicted empty matched lists)

### 5.1 Error Analysis
1. **False Positives (Wrong Merges)**:
   - Primary occurrence: Co-located businesses inside large industrial estates, tech parks, or shopping complexes (e.g. "Embassy TechVillage, Bangalore" or "Tower A, DLF CyberCity") where physical addresses are identical but company names share generic sector terms ("Solutions", "Tech").
   - Remediation: Enforced calibrated threshold ($\theta=0.68$) and name length disparity penalties.
2. **False Negatives (Missed Links)**:
   - Extreme acronyms or DBA rebrandings where neither the business name nor the address shared any common tokens or numbers.

---

## 6. Conclusion
By pairing country-partitioned multi-channel inverted indexing with a high-precision LightGBM classifier and strict $F_{0.5}$ threshold calibration, our solution achieves an outstanding **0.9725 Macro $F_{0.5}$ score** while maintaining an ultra-compact candidate set of only ~11.8 candidates per entity. The end-to-end pipeline processes over 11.7 million records in minutes, scaling seamlessly to production requirements.

---

## Appendix

### A. Code Artefacts
All runnable code is organized inside `code/business_entity_resolution/`:
- `src/preprocessing.py`: Unicode transliteration, text cleaning, and token extraction.
- `src/blocking.py`: Multi-channel country inverted indexing and Top-K candidate generation.
- `src/features.py`: 20-dimensional pairwise feature vector calculation.
- `src/model.py`: LightGBM training, threshold calibration, and serialization.
- `src/pipeline.py`: Unified command-line orchestrator for training and inference.
- `README.md` & `requirements.txt`: Environment setup and reproduction instructions.

**Reproduction Command:**
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
