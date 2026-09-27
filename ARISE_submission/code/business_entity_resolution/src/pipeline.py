import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import time
import argparse
import concurrent.futures as cf
import numpy as np
import polars as pl

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import normalize_text
from src.blocking import CountryInvertedIndex
from src.features import extract_pair_features
from src.model import EntityResolutionModel


def train_pipeline(
    train_dir: str = "dataset/train",
    model_dir: str = "model_artifacts",
    n_train_entities: int = 25000,
    n_val_entities: int = 3000,
    top_k_candidates: int = 8
):
    """Train the entity matching model and optimize the decision threshold."""
    print("=" * 70)
    print("STARTING ENTITY RESOLUTION TRAINING PIPELINE")
    print(f"Train directory: {train_dir}")
    print(f"Target sample size: {n_train_entities} train entities, {n_val_entities} val entities")
    print("=" * 70)
    t0 = time.time()

    # 1. Read ground truth
    print("\n[Step 1/5] Loading Ground Truth...")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    gt = pl.read_csv(gt_path, separator="\t")
    
    # Stratified split: include both non-singletons and singletons
    non_sing = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
    sing = gt.filter(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == ""))
    
    n_sing_train = int(n_train_entities * 0.06)
    n_nonsing_train = n_train_entities - n_sing_train
    
    n_sing_val = int(n_val_entities * 0.06)
    n_nonsing_val = n_val_entities - n_sing_val
    
    train_gt_df = pl.concat([
        non_sing.head(n_nonsing_train),
        sing.head(n_sing_train)
    ])
    val_gt_df = pl.concat([
        non_sing.slice(n_nonsing_train, n_nonsing_val),
        sing.slice(n_sing_train, n_sing_val)
    ])
    
    train_gt = {}
    val_gt = {}
    all_targets = set()
    
    for r in train_gt_df.iter_rows(named=True):
        s1 = r["source1_entity_id"]
        m = set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set()
        train_gt[s1] = m
        all_targets.update(m)
        
    for r in val_gt_df.iter_rows(named=True):
        s1 = r["source1_entity_id"]
        m = set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set()
        val_gt[s1] = m
        all_targets.update(m)
        
    all_s1_ids = list(train_gt.keys()) + list(val_gt.keys())
    print(f"Sampled {len(train_gt)} train S1 entities, {len(val_gt)} val S1 entities. Total true targets: {len(all_targets)}")

    # 2. Load Source 1 records
    print("\n[Step 2/5] Loading Source 1 records...")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s1_df = pl.read_csv(s1_path, separator="\t").filter(pl.col("entity_id").is_in(all_s1_ids))
    s1_records = {}
    for r in s1_df.iter_rows(named=True):
        s1_records[r["entity_id"]] = {
            "entity_id": r["entity_id"],
            "business_name": r["business_name"],
            "business_address": r["business_address"],
            "country": r["country"],
            "norm_name": normalize_text(r["business_name"]),
            "norm_addr": normalize_text(r["business_address"])
        }

    # 3. Load target pool (all true targets + distractors)
    print("\n[Step 3/5] Loading Target records (Source 2 and Source 3)...")
    s2_targets = [t for t in all_targets if t.startswith("S2-")]
    s3_targets = [t for t in all_targets if t.startswith("S3-")]
    
    s2_df = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t")
    s2_pool = pl.concat([
        s2_df.filter(pl.col("entity_id").is_in(s2_targets)),
        s2_df.filter(~pl.col("entity_id").is_in(s2_targets)).head(70000)
    ])
    
    s3_df = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t")
    s3_pool = pl.concat([
        s3_df.filter(pl.col("entity_id").is_in(s3_targets)),
        s3_df.filter(~pl.col("entity_id").is_in(s3_targets)).head(70000)
    ])
    
    targets_df = pl.concat([s2_pool, s3_pool])
    print(f"Total target pool size: {len(targets_df)}")

    # 4. Build country inverted index & candidate pairs
    print("\n[Step 4/5] Building Inverted Index & Generating Features...")
    country_indices = {}
    for c in targets_df["country"].unique().to_list():
        idx = CountryInvertedIndex()
        idx.build(targets_df.filter(pl.col("country") == c))
        country_indices[c] = idx

    def generate_features_for_subset(gt_dict):
        X, y, pair_info = [], [], {}
        for s1_id, true_set in gt_dict.items():
            s1 = s1_records.get(s1_id)
            if not s1:
                continue
            idx = country_indices.get(s1["country"])
            if not idx:
                pair_info[s1_id] = []
                continue
                
            cands = idx.query_top_k(s1["norm_name"], s1["norm_addr"], s1["country"], top_k=top_k_candidates)
            pair_info[s1_id] = []
            
            for rank, (cid, b_score) in enumerate(cands, 1):
                tgt = idx.target_records.get(cid)
                if not tgt:
                    continue
                feats = extract_pair_features(
                    s1["norm_name"], s1["norm_addr"],
                    tgt["norm_name"], tgt["norm_addr"],
                    cid, b_score, rank
                )
                label = 1 if cid in true_set else 0
                X.append(feats)
                y.append(label)
                pair_info[s1_id].append((cid, feats))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), pair_info

    X_train, y_train, _ = generate_features_for_subset(train_gt)
    X_val, y_val, val_pair_info = generate_features_for_subset(val_gt)
    
    print(f"Generated {len(X_train)} training pairs (Positive: {int(sum(y_train))}, Negative: {len(y_train) - int(sum(y_train))})")
    print(f"Generated {len(X_val)} validation pairs")

    # 5. Train model and calibrate threshold
    print("\n[Step 5/5] Training LightGBM Model & Optimizing Decision Threshold...")
    model = EntityResolutionModel(n_estimators=300, learning_rate=0.06, num_leaves=63)
    model.fit(X_train, y_train)

    # Validation predictions
    val_cand_preds = {}
    for s1_id, cand_list in val_pair_info.items():
        if not cand_list:
            val_cand_preds[s1_id] = []
            continue
        cids = [c[0] for c in cand_list]
        feats = np.array([c[1] for c in cand_list], dtype=np.float32)
        probs = model.predict_proba(feats)
        val_cand_preds[s1_id] = list(zip(cids, probs))

    opt_th = model.optimize_threshold(val_cand_preds, val_gt)
    
    # Save artifacts
    print(f"\nSaving model artifacts to {model_dir}...")
    model.save(model_dir)
    print(f"Training pipeline successfully completed in {time.time()-t0:.2f}s!")
    return model


def inference_country_batch(
    country: str,
    s1_df_country: pl.DataFrame,
    s2_df_country: pl.DataFrame,
    s3_df_country: pl.DataFrame,
    model: EntityResolutionModel,
    top_k: int = 12,
    num_threads: int = 12
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Execute high-speed candidate generation and matching inference for one country."""
    print(f"\n--- Processing Country: {country} ---")
    print(f"S1 count: {len(s1_df_country)}, S2 count: {len(s2_df_country)}, S3 count: {len(s3_df_country)}")
    t0 = time.time()

    # 1. Build country target pool and inverted index
    targets_df = pl.concat([s2_df_country, s3_df_country])
    idx = CountryInvertedIndex(max_df=4000)
    idx.build(targets_df)
    print(f"Indexed {len(targets_df)} targets for {country} in {time.time()-t0:.2f}s")

    # 2. Pre-normalize S1 entities using fast columnar access
    s1_ids = s1_df_country["entity_id"].to_list()
    s1_names = s1_df_country["business_name"].to_list()
    s1_addrs = s1_df_country["business_address"].to_list()
    s1_rows = [
        {
            "entity_id": eid,
            "norm_name": normalize_text(n),
            "norm_addr": normalize_text(a),
            "country": country
        }
        for eid, n, a in zip(s1_ids, s1_names, s1_addrs)
    ]

    candidate_results = {}
    all_scored_matches = []
    threshold = model.threshold

    def process_s1_batch(batch):
        batch_cands = {}
        batch_scored = []
        batch_feats = []
        batch_keys = []  # (s1_id, cid)

        for s1 in batch:
            s1_id = s1['entity_id']
            cands = idx.query_top_k(s1['norm_name'], s1['norm_addr'], country, top_k=top_k)
            cand_ids = [c[0] for c in cands]
            batch_cands[s1_id] = cand_ids
            
            for rank, (cid, b_score) in enumerate(cands, 1):
                tgt = idx.target_records.get(cid)
                if tgt:
                    feats = extract_pair_features(
                        s1['norm_name'], s1['norm_addr'],
                        tgt['norm_name'], tgt['norm_addr'],
                        cid, b_score, rank
                    )
                    batch_feats.append(feats)
                    batch_keys.append((s1_id, cid))
                    
        # Batch model prediction
        if batch_feats:
            probs = model.predict_proba(np.array(batch_feats, dtype=np.float32))
            for (s1_id, cid), prob in zip(batch_keys, probs):
                if prob >= threshold:
                    batch_scored.append((s1_id, cid, float(prob)))
                    
        return batch_cands, batch_scored

    # Process in chunks of 5000 entities
    chunk_size = 5000
    chunks = [s1_rows[i:i+chunk_size] for i in range(0, len(s1_rows), chunk_size)]
    
    print(f"Running inference across {len(chunks)} chunks with {num_threads} worker threads...", flush=True)
    with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for chunk_idx, (b_cands, b_scored) in enumerate(executor.map(process_s1_batch, chunks), 1):
            candidate_results.update(b_cands)
            all_scored_matches.extend(b_scored)
            if chunk_idx % 10 == 0 or chunk_idx == len(chunks):
                print(f"[{country}] Processed {min(chunk_idx * chunk_size, len(s1_rows))}/{len(s1_rows)} entities ({time.time()-t0:.1f}s)...", flush=True)

    print(f"[{country}] Resolving global 1-to-1 target collisions across {len(all_scored_matches)} candidate matches...", flush=True)
    best_target_assignment = {}  # cid -> (s1_id, prob)
    for s1_id, cid, prob in all_scored_matches:
        if cid not in best_target_assignment or prob > best_target_assignment[cid][1]:
            best_target_assignment[cid] = (s1_id, prob)

    matched_results = {}
    for cid, (s1_id, prob) in best_target_assignment.items():
        matched_results.setdefault(s1_id, []).append((cid, prob))

    # Sort each entity's matches by probability descending
    for s1_id in matched_results:
        matched_results[s1_id].sort(key=lambda x: -x[1])
        matched_results[s1_id] = [cid for cid, _ in matched_results[s1_id]]

    print(f"Completed {country} in {time.time()-t0:.2f}s! Total matched entities: {len(matched_results)}/{len(s1_rows)}", flush=True)
    return candidate_results, matched_results


def run_full_inference(
    test_dir: str = "dataset/test",
    model_dir: str = "model_artifacts",
    output_dir: str = "output",
    top_k: int = 12,
    num_threads: int = 12,
    threshold: float = None
):
    """Run end-to-end inference on the full test set and generate submission files."""
    print("=" * 70)
    print("STARTING TEST SET INFERENCE PIPELINE")
    print(f"Test directory: {test_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Candidate Top-K: {top_k}")
    print(f"Worker Threads: {num_threads}")
    print("=" * 70)
    t_start = time.time()

    # 1. Load trained model
    model = EntityResolutionModel()
    model.load(model_dir)
    if threshold is not None:
        model.threshold = threshold
    print(f"Loaded model with calibrated decision threshold: {model.threshold:.2f}")

    # 2. Load test source files
    print("\nLoading test source files...")
    t0 = time.time()
    s1_test = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t")
    s2_test = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t")
    s3_test = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t")
    print(f"Loaded test files in {time.time()-t0:.2f}s: S1={len(s1_test)}, S2={len(s2_test)}, S3={len(s3_test)}")

    all_candidates = {}
    all_matches = {}

    # 3. Dynamic country partitioning (open set: handles France, US, India, or any new country)
    countries = s1_test["country"].unique().to_list()
    print(f"\nUnique countries discovered in test set: {countries}")

    for country in countries:
        s1_c = s1_test.filter(pl.col("country") == country)
        s2_c = s2_test.filter(pl.col("country") == country)
        s3_c = s3_test.filter(pl.col("country") == country)
        
        cands_c, matches_c = inference_country_batch(
            country, s1_c, s2_c, s3_c, model, top_k=top_k, num_threads=num_threads
        )
        all_candidates.update(cands_c)
        all_matches.update(matches_c)

    # 4. Write output files strictly following challenge TSV specification
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print("\nWriting submission files...")
    # Preserve exact order of S1 test file
    s1_ordered_ids = s1_test["entity_id"].to_list()

    with open(matching_path, "w", encoding="utf-8") as f_match, \
         open(candidate_path, "w", encoding="utf-8") as f_cand:
        
        # Headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in s1_ordered_ids:
            cands = all_candidates.get(s1_id, [])
            matches = all_matches.get(s1_id, [])
            
            # Enforce that matches must be a strict subset of candidates
            cand_set = set(cands)
            valid_matches = [m for m in matches if m in cand_set]
            
            cand_str = ",".join(cands)
            match_str = ",".join(valid_matches)
            
            f_cand.write(f"{s1_id}\t{cand_str}\n")
            f_match.write(f"{s1_id}\t{match_str}\n")

    print(f"Successfully generated:")
    print(f"  - {matching_path}")
    print(f"  - {candidate_path}")
    print(f"\nTotal inference runtime: {time.time()-t_start:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution End-to-End Pipeline")
    parser.add_argument("--mode", choices=["train", "infer", "all"], default="all", help="Pipeline execution mode")
    parser.add_argument("--train-dir", default="dataset/train", help="Directory with train files")
    parser.add_argument("--test-dir", default="dataset/test", help="Directory with test files")
    parser.add_argument("--model-dir", default="model_artifacts", help="Directory for model artifacts")
    parser.add_argument("--output-dir", default="output", help="Directory for submission output TSVs")
    parser.add_argument("--n-train", type=int, default=30000, help="Number of training S1 entities")
    parser.add_argument("--n-val", type=int, default=4000, help="Number of validation S1 entities")
    parser.add_argument("--top-k", type=int, default=12, help="Top-K candidates per S1 entity")
    parser.add_argument("--threads", type=int, default=12, help="Number of parallel worker threads")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override")
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        train_pipeline(
            train_dir=args.train_dir,
            model_dir=args.model_dir,
            n_train_entities=args.n_train,
            n_val_entities=args.n_val,
            top_k_candidates=args.top_k
        )

    if args.mode in ["infer", "all"]:
        run_full_inference(
            test_dir=args.test_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            top_k=args.top_k,
            num_threads=args.threads,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()
