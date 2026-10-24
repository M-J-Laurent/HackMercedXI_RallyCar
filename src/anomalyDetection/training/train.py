"""Train the anomaly classifier on the synthetic dataset (split by track, never by row).

    cd src
    python -m anomalyDetection.training.train --backend lightgbm
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

from anomalyDetection.schema import LABELS, TRACK_ID
from anomalyDetection.training.evaluate import evaluate
from anomalyDetection.training.features import build_table

REPO = Path(__file__).resolve().parents[3]


def make_model(backend: str, seed: int):
    if backend == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=8, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss",
                             early_stopping_rounds=30, random_state=seed)


def fit(backend, model, X, y, w, Xv, yv):
    if backend == "lightgbm":
        import lightgbm as lgb
        model.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(30, verbose=False)])
    else:
        model.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], verbose=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=REPO / "dat" / "synthetic")
    p.add_argument("--model-dir", type=Path, default=REPO / "models")
    p.add_argument("--backend", choices=["lightgbm", "xgboost"], default="lightgbm")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    runs, refs = pd.read_csv(a.data / "runs.csv"), pd.read_csv(a.data / "references.csv")
    X, y, meta = build_table(runs, refs)
    groups = meta[TRACK_ID]

    # 70 / 15 / 15 by track so no stage appears in more than one split
    rest, test = next(GroupShuffleSplit(1, test_size=0.15, random_state=a.seed).split(X, y, groups))
    tr, va = next(GroupShuffleSplit(1, test_size=0.18, random_state=a.seed).split(X.iloc[rest], y[rest], groups.iloc[rest]))
    tr, va = rest[tr], rest[va]

    missing = set(range(len(LABELS))) - set(y[tr])
    if missing:
        raise SystemExit(f"train split has no samples of {[LABELS[i] for i in missing]}; "
                         "generate more tracks/runs or change --seed")
    w = compute_sample_weight("balanced", y[tr])
    model = make_model(a.backend, a.seed)
    fit(a.backend, model, X.iloc[tr], y[tr], w, X.iloc[va], y[va])

    a.model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "labels": LABELS, "feature_names": list(X.columns), "backend": a.backend}
    joblib.dump(bundle, a.model_dir / "anomaly_model.joblib")
    split = {k: sorted(groups.iloc[i].unique().tolist()) for k, i in (("train", tr), ("val", va), ("test", test))}
    (a.model_dir / "split.json").write_text(json.dumps(split, indent=1))
    print(f"saved model -> {a.model_dir}\n")

    faults = json.loads((a.data / "faults.json").read_text())
    evaluate(bundle, X.iloc[test].reset_index(drop=True), y[test], meta.iloc[test].reset_index(drop=True), faults)


if __name__ == "__main__":
    main()
