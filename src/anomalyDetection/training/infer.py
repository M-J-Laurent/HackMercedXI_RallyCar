"""Run the trained model on a post-race lap and emit anomaly segments for the LLM.

    cd src
    python -m anomalyDetection.training.infer --run ../dat/real/lap.csv --ref ../dat/real/ref.csv

Detection is the model's main output. `suggested_type` is a low-trust hint; the
LLM is expected to diagnose from `context` (driver vs reference traces around
the segment). Output is JSON:
    [{start_m, end_m, confidence, suggested_type, suggested_type_confidence,
      evidence{...}, context{step_m, columns, driver[[...]], reference[[...]]}}, ...]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from anomalyDetection.schema import DIST, LABELS, NORMAL
from anomalyDetection.training.features import build_features, resample_to_grid

REPO = Path(__file__).resolve().parents[3]

CONTEXT_CHANNELS = ["speed", "throttle", "brake", "steer", "gear", "rpm", "g_lat"]


def _context(r: pd.DataFrame, q: pd.DataFrame, i: int, j: int, pad_m: float, step_m: float) -> dict:
    """Driver and reference traces from pad_m before the segment to pad_m after it."""
    grid = r[DIST].iloc[1] - r[DIST].iloc[0]
    pad, every = int(round(pad_m / grid)), max(int(round(step_m / grid)), 1)
    idx = np.arange(max(i - pad, 0), min(j + pad, len(r) - 1) + 1, every)
    cols = [DIST, *CONTEXT_CHANNELS]
    rows = lambda d: d.iloc[idx][cols].round(2).values.tolist()
    return {"step_m": float(every * grid), "columns": cols, "driver": rows(r), "reference": rows(q)}


def detect(run: pd.DataFrame, ref: pd.DataFrame, bundle: dict, threshold: float = 0.5,
           min_len_m: float = 10.0, max_gap_m: float = 4.0,
           context_pad_m: float = 100.0, context_step_m: float = 10.0) -> list[dict]:
    X, meta = build_features(run, ref)
    r, q = resample_to_grid(run).iloc[:len(X)], resample_to_grid(ref).iloc[:len(X)]
    proba = bundle["model"].predict_proba(X[bundle["feature_names"]])
    normal = LABELS.index(NORMAL)
    p_anom = 1.0 - proba[:, normal]
    mask = p_anom >= threshold
    d = meta[DIST].to_numpy()
    step = d[1] - d[0]
    max_gap = int(round(max_gap_m / step))

    segments, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while True:  # extend through short gaps so one mistake isn't split in two
            nxt = next((k for k in range(j + 1, min(j + 2 + max_gap, n)) if mask[k]), None)
            if nxt is None:
                break
            j = nxt
        if (j - i + 1) * step >= min_len_m:
            sl = slice(i, j + 1)
            mass = proba[sl].mean(axis=0)
            mass[normal] = 0.0
            typ = int(mass.argmax())
            segments.append({
                "start_m": float(d[i]), "end_m": float(d[j]),
                "confidence": round(float(p_anom[sl].mean()), 3),
                "suggested_type": LABELS[typ],  # may be wrong: verify against context
                "suggested_type_confidence": round(float(mass[typ] / max(mass.sum(), 1e-9)), 3),
                "evidence": {  # run minus reference over the segment
                    "speed_delta_ms": round(float(X["d_speed"].iloc[sl].mean()), 2),
                    "throttle_delta": round(float(X["d_throttle"].iloc[sl].mean()), 3),
                    "brake_delta": round(float(X["d_brake"].iloc[sl].mean()), 3),
                    "steer_delta": round(float(X["d_steer"].iloc[sl].mean()), 3),
                    "gear_delta": round(float(X["d_gear"].iloc[sl].mean()), 2),
                },
                "context": _context(r, q, i, j, context_pad_m, context_step_m),
            })
        i = j + 1
    return segments


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="telemetry CSV of the lap to analyse")
    p.add_argument("--ref", required=True, help="reference CSV for the same stage")
    p.add_argument("--model-dir", type=Path, default=REPO / "models")
    p.add_argument("--threshold", type=float, default=0.5, help="min P(anomalous) to flag a sample")
    a = p.parse_args()
    bundle = joblib.load(a.model_dir / "anomaly_model.joblib")
    print(json.dumps(detect(pd.read_csv(a.run), pd.read_csv(a.ref), bundle, a.threshold)))


if __name__ == "__main__":
    main()
