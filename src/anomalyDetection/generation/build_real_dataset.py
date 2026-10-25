"""Build a labelled training set from real recorded laps (see real_augment.py).

    cd src
    python -m anomalyDetection.generation.build_real_dataset --runs 300
    python -m anomalyDetection.generation.build_real_dataset --range 0 2400 --out ../dat/real_aug_a   # half the stage

Writes references.csv / runs.csv / faults.json to dat/real_aug/ in the same format as
build_dataset.py, so training/train.py works on it unchanged (use --split run).
--range crops the stage so a model can be tested on the half it never saw.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anomalyDetection.generation.real_augment import load_lap, make_real_run
from anomalyDetection.schema import EXTRA_CHANNELS, RUN_ID, TELEMETRY_COLUMNS, TRACK_ID

REPO = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--real-dir", type=Path, default=REPO / "dat" / "real")
    p.add_argument("--ref", default="stage1_clean_02.csv", help="reference lap file in --real-dir")
    p.add_argument("--clean", nargs="+", default=["stage1_clean_01.csv", "stage1_clean_02.csv"],
                   help="clean laps used as the base for normal variants")
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--max-faults", type=int, default=4)
    p.add_argument("--clean-frac", type=float, default=0.25)
    p.add_argument("--range", nargs=2, type=float, metavar=("START_M", "END_M"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=REPO / "dat" / "real_aug")
    a = p.parse_args()

    crop = tuple(a.range) if a.range else None
    rng = np.random.default_rng(a.seed)
    ref = load_lap(a.real_dir / a.ref, crop)
    laps = [load_lap(a.real_dir / f, crop) for f in a.clean]

    runs, faults = [], {}
    for r in range(a.runs):
        n_faults = 0 if rng.random() < a.clean_frac else int(rng.integers(1, a.max_faults + 1))
        df, meta = make_real_run(laps, rng, n_faults)
        rid = f"real_r{r:03d}"
        df.insert(0, RUN_ID, rid)
        df.insert(0, TRACK_ID, "real")
        runs.append(df)
        faults[rid] = meta

    refdf = ref[TELEMETRY_COLUMNS + [c for c in EXTRA_CHANNELS if c in ref.columns]].copy()
    refdf.insert(0, TRACK_ID, "real")
    a.out.mkdir(parents=True, exist_ok=True)
    refdf.round(4).to_csv(a.out / "references.csv", index=False)
    runs = pd.concat(runs)
    runs.round(4).to_csv(a.out / "runs.csv", index=False)
    (a.out / "faults.json").write_text(json.dumps(faults, indent=1))
    (a.out / "meta.json").write_text(json.dumps({"offset_m": crop[0] if crop else 0.0}))  # cropped laps are re-zeroed
    print(f"label counts:\n{runs['label'].value_counts().to_string()}\n-> {a.out}")


if __name__ == "__main__":
    main()
