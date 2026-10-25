"""Run the trained model on a post-race lap and emit anomaly segments for the LLM.

    cd src
    python -m anomalyDetection.training.infer --run ../dat/real/lap.csv --ref ../dat/real/ref.csv

Two sources feed the same output:
    model  flags where something went wrong (its `suggested_type` is a low-trust hint)
    rules  obvious events (engine stall after a hard stop = crash or spin), see rules.py;
           these set `suggested_type` themselves and carry a `rule` block
The LLM is expected to diagnose from `context` (driver vs reference traces around the segment).
Output is JSON:
    [{start_m, end_m, source, confidence, suggested_type, suggested_type_confidence,
      evidence{...}, rule{...}?, context{step_m, columns, driver[[...]], reference[[...]]}}, ...]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from anomalyDetection.schema import DIST, LABELS, NORMAL
from anomalyDetection.stage_map import load_stage_map, locate
from anomalyDetection.training.features import build_features, resample_to_grid
from anomalyDetection.training.rules import find_stall_events

REPO = Path(__file__).resolve().parents[3]

MIN_LOST_S = 0.2            # reporting mode: ignore model flags that cost less time than this
LOSS_RATE_PER_STEP = 0.02   # s per 2 m step: about 20% slower than the reference at 20 m/s
CONTEXT_CHANNELS = ["speed", "throttle", "brake", "steer", "gear", "rpm", "g_lat"]


def _context(r: pd.DataFrame, q: pd.DataFrame, i: int, j: int, pad_m: float, step_m: float) -> dict:
    """Driver and reference traces from pad_m before the segment to pad_m after it."""
    grid = r[DIST].iloc[1] - r[DIST].iloc[0]
    pad, every = int(round(pad_m / grid)), max(int(round(step_m / grid)), 1)
    idx = np.arange(max(i - pad, 0), min(j + pad, len(r) - 1) + 1, every)
    cols = [DIST, *CONTEXT_CHANNELS, *(c for c in ("slip_angle_deg",) if c in r.columns)]
    rows = lambda d: d.iloc[idx][cols].round(2).values.tolist()
    return {"step_m": float(every * grid), "columns": cols, "driver": rows(r), "reference": rows(q)}


def _stretch(n: float, unit: str) -> str:
    return f"{abs(n):.0f} {unit}"


def facts_for(r: pd.DataFrame, q: pd.DataFrame, i: int, j: int) -> list[str]:
    """Plain-sentence measurements over [i, j] at full 2 m resolution: the driver vs the reference.
    Differences are already worked out so a small LLM does not have to do arithmetic."""
    d = r[DIST].to_numpy()
    sl = slice(i, j + 1)
    grid = float(d[1] - d[0])
    out = []

    v0, rv0 = r["speed"].iloc[i] * 3.6, q["speed"].iloc[i] * 3.6
    if abs(v0 - rv0) >= 3:
        out.append(f"At {d[i]:.0f} m you were going {v0:.0f} km/h, {_stretch(v0 - rv0, 'km/h')} {'faster' if v0 > rv0 else 'slower'} than the reference ({rv0:.0f} km/h).")
    else:
        out.append(f"At {d[i]:.0f} m you were going {v0:.0f} km/h, about the same as the reference ({rv0:.0f} km/h).")

    k = i + int(np.argmin(r["speed"].to_numpy()[sl]))
    rk = i + int(np.argmin(q["speed"].to_numpy()[sl]))
    v, rv = r["speed"].iloc[k] * 3.6, q["speed"].iloc[rk] * 3.6
    out.append(f"Your slowest speed in this stretch was {v:.0f} km/h at {d[k]:.0f} m; the reference's slowest was {rv:.0f} km/h"
               + (f", so you were {_stretch(rv - v, 'km/h')} lower." if rv - v >= 3 else "."))

    lo = max(i - int(50 / grid), 0)
    first = lambda df: next((float(d[x]) for x in range(lo, j + 1) if df["brake"].iloc[x] > 0.3), None)
    yb, rb = first(r), first(q)
    if yb is not None and rb is not None:
        diff = yb - rb
        out.append(f"You first braked hard at {yb:.0f} m; the reference first braked hard at {rb:.0f} m"
                   + (f" ({_stretch(diff, 'm')} {'later' if diff > 0 else 'earlier'} for you)." if abs(diff) >= 6 else " (about the same place)."))
    elif yb is not None:
        out.append(f"You braked hard at {yb:.0f} m; the reference did not brake hard here.")
    elif rb is not None:
        out.append(f"You did not brake hard here; the reference braked hard at {rb:.0f} m.")

    off = lambda df: int((df["throttle"].to_numpy()[sl] < 0.2).sum() * grid)
    if off(r) - off(q) >= 10:
        out.append(f"You were off the throttle for {off(r)} m of this stretch; the reference for {off(q)} m.")

    if "slip_angle_deg" in r.columns and "slip_angle_deg" in q.columns:
        sy, sr = float(np.abs(r["slip_angle_deg"].to_numpy()[sl]).max()), float(np.abs(q["slip_angle_deg"].to_numpy()[sl]).max())
        if sy >= 30 and sy - sr >= 10:
            out.append(f"Your largest slide angle was {sy:.0f} degrees; the reference's was {sr:.0f} degrees.")
    return out


def score_run(run: pd.DataFrame, ref: pd.DataFrame, bundle: dict):
    """The expensive part of detect(): features + model. Reuse it to try several thresholds."""
    X, meta = build_features(run, ref, bundle.get("stage_map"))
    r, q = resample_to_grid(run).iloc[:len(X)], resample_to_grid(ref).iloc[:len(X)]
    missing = [c for c in bundle["feature_names"] if c not in X.columns]
    if missing:  # e.g. a lap exported without slip angle: those features read as 0
        X = pd.concat([X, pd.DataFrame(0.0, index=X.index, columns=missing, dtype="float32")], axis=1)
    return X, meta, r, q, bundle["model"].predict_proba(X[bundle["feature_names"]])


def detect(run: pd.DataFrame, ref: pd.DataFrame, bundle: dict, threshold: float = 0.3,
           min_len_m: float = 4.0, max_gap_m: float = 4.0,
           context_pad_m: float = 100.0, context_step_m: float = 10.0, scored=None,
           extend_threshold: float = 0.3, ignore_types: tuple = ("wrong_gear",),
           use_rules: bool = True, stage_map: dict | None = None, expand: bool = True,
           loss_rate: float = LOSS_RATE_PER_STEP, max_back_m: float = 60.0, max_fwd_m: float = 300.0,
           min_lost_s: float = MIN_LOST_S) -> list[dict]:
    """Model segments start where P(anomalous) >= threshold and grow while it stays >= extend_threshold.

    The model fires in narrow spikes, but a mistake costs time until the driver is back up to speed. With
    `expand`, each flagged spot grows backwards and forwards while the lap is still losing time faster than
    its usual pace (X["d_dt"], seconds per grid step), and `time_lost_s` is the excess time over that whole
    span. Without it, segments are just the flagged core (used to score detection). In that reporting mode a
    model-only segment that cost less than `min_lost_s` is dropped (a coach has nothing to say about it);
    rule events (spins, crashes) are always kept.
    """
    X, meta, r, q, proba = scored if scored is not None else score_run(run, ref, bundle)
    stage_map = stage_map or bundle.get("stage_map")  # corner info for the LLM (the model may or may not use it)
    normal = LABELS.index(NORMAL)
    skip = [LABELS.index(t) for t in ignore_types]  # classes that count as normal (too noisy on real laps)
    p_anom = 1.0 - proba[:, normal] - proba[:, skip].sum(axis=1)
    core = p_anom >= threshold
    mask = p_anom >= min(extend_threshold, threshold)
    d = meta[DIST].to_numpy()
    step = d[1] - d[0]
    max_gap = int(round(max_gap_m / step))

    found, i, n = [], 0, len(mask)  # (first, last, rule event or None)
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
        if (j - i + 1) * step >= min_len_m and core[i:j + 1].any():
            found.append([i, j, None, True])
        i = j + 1
    if use_rules and run is not None:
        for ev in find_stall_events(run):
            a, b = int(np.searchsorted(d, ev["start_m"])), min(int(np.searchsorted(d, ev["end_m"])), n - 1)
            found.append([a, b, ev, False])

    # a rule event and the model segments it overlaps are one mistake
    found.sort(key=lambda f: f[0])
    merged: list[list] = []
    for a, b, ev, from_model in found:
        if merged and a <= merged[-1][1] + max_gap:
            m = merged[-1]
            m[1] = max(m[1], b)
            m[2] = m[2] or ev
            m[3] = m[3] or from_model
            m[4] = m[4] or (ev is not None)
        else:
            merged.append([a, b, ev, from_model, ev is not None])

    # grow each flagged spot to the stretch where time was actually being lost
    if expand and merged:
        excess = X["d_dt"].to_numpy(dtype=float)  # seconds lost per grid step beyond the lap's usual pace
        smooth = pd.Series(excess).rolling(5, center=True, min_periods=1).mean().to_numpy()
        back, fwd = int(max_back_m / step), int(max_fwd_m / step)
        bridge_back, bridge_fwd = int(10 / step), int(30 / step)  # the loss often starts a few metres after the flag
        grown = []
        for i, j, ev, from_model, from_rule in merged:
            lo, hi = i, j
            k, misses = i - 1, 0
            while k >= max(i - back, 0) and misses <= bridge_back:   # tolerate short dips back to normal
                misses = 0 if smooth[k] > loss_rate else misses + 1
                if smooth[k] > loss_rate:
                    lo = k
                k -= 1
            k, misses = j + 1, 0
            while k <= min(j + fwd, n - 1) and misses <= bridge_fwd:
                misses = 0 if smooth[k] > loss_rate else misses + 1
                if smooth[k] > loss_rate:
                    hi = k
                k += 1
            grown.append([lo, hi, ev, from_model, from_rule, i, j])
        merged = []
        for lo, hi, ev, from_model, from_rule, ci, cj in grown:   # spots whose growth overlaps are one mistake
            if merged and lo <= merged[-1][1] + 1:
                m = merged[-1]
                m[1], m[2] = max(m[1], hi), m[2] or ev
                m[3], m[4] = m[3] or from_model, m[4] or from_rule
                m[6] = max(m[6], cj)
            else:
                merged.append([lo, hi, ev, from_model, from_rule, ci, cj])
    else:
        merged = [[i, j, ev, fm, fr, i, j] for i, j, ev, fm, fr in merged]

    segments = []
    for i, j, ev, from_model, from_rule, ci, cj in merged:
        sl = slice(i, j + 1)
        core_sl = slice(ci, cj + 1)
        mass = proba[core_sl].mean(axis=0)
        mass[normal] = 0.0
        mass[skip] = 0.0
        typ = int(mass.argmax())
        if ev:
            label, type_conf, conf = ev["label"], 1.0, 1.0
        else:
            label, type_conf, conf = LABELS[typ], float(mass[typ] / max(mass.sum(), 1e-9)), float(p_anom[core_sl].max())
        seg = {
            "start_m": float(d[i]), "end_m": float(d[j]),
            "flagged_start_m": float(d[ci]), "flagged_end_m": float(d[cj]),   # where the model itself fired
            "source": "rule+model" if from_rule and from_model else "rule" if from_rule else "model",
            "confidence": round(conf, 3),  # peak P(anomalous) in the segment; 1.0 when a rule fired
            "suggested_type": label,  # a hint from the model; from a rule it is much more reliable
            "suggested_type_confidence": round(type_conf, 3),
            "evidence": {  # run minus reference over the segment
                "speed_delta_ms": round(float((X["d_speed"] + X["d_speed_trend"]).iloc[sl].mean()), 2),
                "local_speed_delta_ms": round(float(X["d_speed"].iloc[sl].mean()), 2),  # minus overall pace gap
                "throttle_delta": round(float(X["d_throttle"].iloc[sl].mean()), 3),
                "brake_delta": round(float(X["d_brake"].iloc[sl].mean()), 3),
                "steer_delta": round(float(X["d_steer"].iloc[sl].mean()), 3),
                "gear_delta": round(float(X["d_gear"].iloc[sl].mean()), 2),
                # time this mistake cost: excess seconds over the whole stretch, beyond the lap's usual pace
                "time_lost_s": round(float(X["d_dt"].iloc[sl].sum()), 1),
            },
            "facts": facts_for(r, q, i, j),
            "context": _context(r, q, i, j, context_pad_m, context_step_m),
        }
        if ev:
            seg["rule"] = ev["rule"]
        if stage_map:
            seg["corner"] = locate(stage_map, seg["start_m"], seg["end_m"])
        segments.append(seg)
    if expand:
        segments = [x for x in segments if x["source"] != "model" or x["evidence"]["time_lost_s"] >= min_lost_s]
    return segments


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="telemetry CSV of the lap to analyse")
    p.add_argument("--ref", required=True, help="reference CSV for the same stage")
    p.add_argument("--model-dir", type=Path, default=REPO / "models")
    p.add_argument("--threshold", type=float, default=0.3, help="min P(anomalous) to flag a sample")
    p.add_argument("--stage-map", type=Path, default=REPO / "dat" / "real" / "stage1_map.json",
                   help="stage map JSON; adds `corner` info to each segment (skipped if the file is missing)")
    p.add_argument("--no-rules", action="store_true", help="model only")
    a = p.parse_args()
    bundle = joblib.load(a.model_dir / "anomaly_model.joblib")
    print(json.dumps(detect(pd.read_csv(a.run), pd.read_csv(a.ref), bundle, a.threshold, use_rules=not a.no_rules,
                      stage_map=load_stage_map(a.stage_map) if a.stage_map.exists() else None)))


if __name__ == "__main__":
    main()
