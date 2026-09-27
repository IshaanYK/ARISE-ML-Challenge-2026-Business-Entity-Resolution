import json
import os
import numpy as np
import lightgbm as lgb
from src.features import FEATURE_NAMES


class EntityResolutionModel:
    """Gradient Boosted Decision Tree classifier optimized for macro F_0.5 matching."""
    def __init__(
        self,
        n_estimators: int = 800,
        learning_rate: float = 0.03,
        num_leaves: int = 127,
        threshold: float = 0.90
    ):
        self.clf = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,       # finer splits on rare positives
            max_bin=255,                # cap per-feature histogram memory
            reg_alpha=0.05,
            reg_lambda=0.05,
            scale_pos_weight=4,         # precision-first: F0.5 punishes FP 2x harder than FN
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier on candidate pair feature vectors."""
        self.clf.fit(X, y)
        self.booster = self.clf.booster_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Output match probabilities for candidate pairs directly from booster."""
        if hasattr(self, 'booster') and self.booster is not None:
            return self.booster.predict(X)
        return self.clf.predict_proba(X)[:, 1]

    def optimize_threshold(self, val_candidates: dict[str, list[tuple[str, float]]], val_gt: dict[str, set[str]]) -> float:
        """Find the decision threshold that strictly maximizes Macro F_0.5 on validation data."""
        best_th = self.threshold
        best_score = -1.0
        
        # Grid search over candidate thresholds
        for th in np.arange(0.50, 0.98, 0.01):  # wider + finer search for precision-heavy regime
            scores = []
            for s1_id, true_set in val_gt.items():
                cands = val_candidates.get(s1_id, [])
                pred_set = {cid for cid, prob in cands if prob >= th}
                
                # Singletons
                if len(true_set) == 0:
                    score = 1.0 if len(pred_set) == 0 else 0.0
                elif len(pred_set) == 0:
                    score = 0.0
                else:
                    tp = len(pred_set & true_set)
                    if tp == 0:
                        score = 0.0
                    else:
                        p = tp / len(pred_set)
                        r = tp / len(true_set)
                        score = (1.25 * p * r) / (0.25 * p + r)
                scores.append(score)
                
            macro_f05 = float(np.mean(scores))
            if macro_f05 > best_score:
                best_score = macro_f05
                best_th = float(th)
                
        self.threshold = best_th
        print(f"Optimal Decision Threshold calibrated to: {self.threshold:.2f} (Macro F_0.5: {best_score:.4f})")
        return self.threshold

    def save(self, model_dir: str):
        """Save model booster and threshold to directory."""
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.txt")
        self.clf.booster_.save_model(model_path)
        
        meta = {
            "threshold": self.threshold,
            "feature_names": FEATURE_NAMES
        }
        with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, model_dir: str):
        """Load trained booster and threshold from directory."""
        model_path = os.path.join(model_dir, "model.txt")
        booster = lgb.Booster(model_file=model_path)
        self.booster = booster
        self.clf._Booster = booster
        self.clf.fitted_ = True
        
        with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.threshold = meta["threshold"]
