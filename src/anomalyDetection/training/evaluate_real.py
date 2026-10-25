"""Score a model on real recorded laps against the driver's annotated mistakes.

    cd src
    python -m anomalyDetection.training.evaluate_real --model-dir ../models/real_a --holdout 2400 4800

Reports, per detection threshold:
    recall      annotated mistakes inside the holdout range that a flagged segment overlaps (+-40 m)
    flagged %   share of the holdout metres flagged on laps that should be mostly clean
                (clean_01, away from its annotated slide) and on the sloppy laps away from annotations
The annotations are approximate and incomplete, so treat flagged % as an upper bound on false alarms.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from anomalyDetection.training.infer import detect, score_run

REPO = Path(__file__).resolve().parents[3]
SLACK_M = 40.0
FAR_M = 150.0


def overlaps(segs, a, b):
    return any(s["start_m"] <= b + SLACK_M and s["end_m"] >= a - SLACK_M for s in segs)


def score(segs, spots, lo, hi, length_cut):
    """(recall hits, total spots, flagged metres away from spots, metres away from spots) in [lo, hi)."""
    inside = [(a, b) for a, b in spots if lo <= a and b < hi]
    hits = sum(overlaps(segs, a, b) for a, b in inside)
    far_total = flagged = 0.0
    for m in range(int(lo), int(hi), 10):
        if all(abs(m - (a + b) / 2) > (b - a) / 2 + FAR_M for a, b in spots):
            far_total += 10
            flagged += 10 * any(s["start_m"] <= m <= s["end_m"] for s in segs)
    return hits, len(inside), flagged, far_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=REPO / "models")
    p.add_argument("--real-dir", type=Path, default=REPO / "dat" / "real")
    p.add_argument("--ref", default="stage1_clean_02.csv")
    p.add_argument("--holdout", nargs=2, type=float, default=(0, 4800), metavar=("START_M", "END_M"))
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    a = p.parse_args()

    bundle = joblib.load(a.model_dir / "anomaly_model.joblib")
    notes = {k: v for k, v in json.loads((a.real_dir / "annotations.json").read_text()).items() if k != "_note"}
    ref = pd.read_csv(a.real_dir / a.ref)
    lo, hi = a.holdout
    print(f"model {a.model_dir.name}, scoring metres {lo:.0f}-{min(hi, 4800):.0f}\n")
    print(f"{'thr':>4} {'lap':<22} {'recall':>8} {'flagged % (away from notes)':>28}")
    laps = {f: pd.read_csv(a.real_dir / f) for f in notes}
    scored = {f: score_run(laps[f], ref, bundle) for f in notes}  # once per lap, not per threshold
    for thr in a.thresholds:
        tot = [0, 0, 0.0, 0.0]
        for f, spots in notes.items():
            segs = detect(laps[f], ref, bundle, threshold=thr, context_pad_m=0, scored=scored[f])
            h, n, fl, ft = score(segs, spots, lo, hi, 0)
            tot = [tot[0] + h, tot[1] + n, tot[2] + fl, tot[3] + ft]
            print(f"{thr:>4} {f:<22} {h:>3}/{n:<4} {100 * fl / max(ft, 1):>27.1f}%")
        print(f"{thr:>4} {'ALL':<22} {tot[0]:>3}/{tot[1]:<4} {100 * tot[2] / max(tot[3], 1):>27.1f}%\n")


if __name__ == "__main__":
    main()
