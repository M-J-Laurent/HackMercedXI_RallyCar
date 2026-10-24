"""Run the trained model on a post-race lap and emit anomaly segments for the LLM.

    cd src
    python -m anomalyDetection.training.infer --run ../dat/real/lap.csv --ref ../dat/real/ref.csv

Output is JSON: [{label, start_m, end_m, confidence, evidence{...}}, ...]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from anomalyDetection.schema import DIST, LABELS, NORMAL
from anomalyDetection.training.features import build_features

REPO = Path(__file__).resolve().parents[3]


def detect(run: pd.DataFrame, ref: pd.DataFrame, bundle: dict,
           min_len_m: float = 10.0, min_conf: float = 0.5) -> list[dict]:
    X, meta = build_features(run, ref)
    proba = bundle["model"].predict_proba(X[bundle["feature_names"]])
    pred = proba.argmax(axis=1)
    d = meta[DIST].to_numpy()
    step = d[1] - d[0]
    normal = LABELS.index(NORMAL)

    segments, i = [], 0
    while i < len(pred):
        j = i
        while j + 1 < len(pred) and pred[j + 1] == pred[i]:
            j += 1
        conf = float(proba[i:j + 1, pred[i]].mean())
        if pred[i] != normal and (j - i + 1) * step >= min_len_m and conf >= min_conf:
            sl = slice(i, j + 1)
            segments.append({
                "label": LABELS[pred[i]], "start_m": float(d[i]), "end_m": float(d[j]),
                "confidence": round(conf, 3),
                "evidence": {  # run minus reference over the segment, for the LLM to cite
                    "speed_delta_ms": round(float(X["d_speed"].iloc[sl].mean()), 2),
                    "throttle_delta": round(float(X["d_throttle"].iloc[sl].mean()), 3),
                    "brake_delta": round(float(X["d_brake"].iloc[sl].mean()), 3),
                    "steer_delta": round(float(X["d_steer"].iloc[sl].mean()), 3),
                    "gear_delta": round(float(X["d_gear"].iloc[sl].mean()), 2),
                },
            })
        i = j + 1
    return segments


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="telemetry CSV of the lap to analyse")
    p.add_argument("--ref", required=True, help="reference CSV for the same stage")
    p.add_argument("--model-dir", type=Path, default=REPO / "models")
    a = p.parse_args()
    bundle = joblib.load(a.model_dir / "anomaly_model.joblib")
    print(json.dumps(detect(pd.read_csv(a.run), pd.read_csv(a.ref), bundle), indent=2))


if __name__ == "__main__":
    main()
