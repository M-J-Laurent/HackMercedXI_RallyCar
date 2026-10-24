"""Evaluate a trained model on held-out synthetic tracks.

    cd src
    python -m anomalyDetection.training.evaluate
For real recorded laps use infer.py (there are no ground-truth labels for those).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from anomalyDetection.schema import DIST, LABELS, NORMAL, RUN_ID
from anomalyDetection.training.features import build_table

REPO = Path(__file__).resolve().parents[3]


def segment_detection(meta: pd.DataFrame, pred: np.ndarray, faults: dict) -> pd.DataFrame:
    """Per injected fault: was it flagged at all, and was it flagged as the right type?"""
    rows = []
    for rid, g in meta.groupby(RUN_ID):
        p = np.array(LABELS)[pred[g.index]]
        d = g[DIST].to_numpy()
        for f in faults[rid]:
            m = (d >= f["start_m"]) & (d <= f["end_m"])
            if m.any():
                rows.append({"label": f["label"], "detected": (p[m] != NORMAL).mean() >= 0.5,
                             "correct_type": (p[m] == f["label"]).mean() >= 0.5})
    return pd.DataFrame(rows).groupby("label")[["detected", "correct_type"]].mean().round(3)


def evaluate(bundle, X, y, meta, faults=None) -> None:
    pred = bundle["model"].predict_proba(X[bundle["feature_names"]]).argmax(axis=1)
    print(classification_report(y, pred, labels=range(len(LABELS)), target_names=LABELS, digits=3, zero_division=0))
    print("confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(y, pred, labels=range(len(LABELS))), index=LABELS, columns=LABELS))
    normal = y == LABELS.index(NORMAL)
    print(f"\nfalse-alarm rate on normal samples: {(pred[normal] != LABELS.index(NORMAL)).mean():.3%}")
    if faults:
        print("\nper-fault detection (>=50% of the fault span):")
        print(segment_detection(meta, pred, faults).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=REPO / "dat" / "synthetic")
    p.add_argument("--model-dir", type=Path, default=REPO / "models")
    a = p.parse_args()
    bundle = joblib.load(a.model_dir / "anomaly_model.joblib")
    test_tracks = set(json.loads((a.model_dir / "split.json").read_text())["test"])
    runs, refs = pd.read_csv(a.data / "runs.csv"), pd.read_csv(a.data / "references.csv")
    X, y, meta = build_table(runs, refs, test_tracks)
    evaluate(bundle, X, y, meta, json.loads((a.data / "faults.json").read_text()))


if __name__ == "__main__":
    main()
