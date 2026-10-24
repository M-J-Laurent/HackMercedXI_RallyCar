"""Build the labelled synthetic training set.

    cd src
    python -m anomalyDetection.generation.build_dataset --tracks 24 --runs-per-track 10

Writes to dat/synthetic/:
    references.csv   noise-free optimal run per track (track_id + schema columns)
    runs.csv         flawed / clean runs (track_id, run_id, schema columns, label)
    faults.json      run_id -> injected faults, for segment-level evaluation
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anomalyDetection.generation.createTelemetrydat import generate_reference
from anomalyDetection.generation.inject_faults import make_flawed_run
from anomalyDetection.generation.physics import SURFACE_MU, CarParams
from anomalyDetection.generation.track import synthetic_track
from anomalyDetection.schema import RUN_ID, TRACK_ID

REPO = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tracks", type=int, default=24)
    p.add_argument("--runs-per-track", type=int, default=10)
    p.add_argument("--length", type=float, default=4000.0)
    p.add_argument("--max-faults", type=int, default=4)
    p.add_argument("--clean-frac", type=float, default=0.2, help="share of runs with no faults")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=REPO / "dat" / "synthetic")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    car = CarParams()
    surfaces = list(SURFACE_MU)
    refs, runs, faults = [], [], {}

    for t in range(a.tracks):
        tid = f"trk{t:03d}"
        surface = surfaces[t % len(surfaces)]
        mu = SURFACE_MU[surface]
        track = synthetic_track(a.seed * 10_000 + t, a.length, surface)

        ref = generate_reference(track, car, mu)
        ref.insert(0, TRACK_ID, tid)
        refs.append(ref)

        for r in range(a.runs_per_track):
            n_faults = 0 if rng.random() < a.clean_frac else int(rng.integers(1, a.max_faults + 1))
            df, meta = make_flawed_run(track, car, mu, rng, n_faults, pace=rng.uniform(0.93, 0.99))
            rid = f"{tid}_r{r:02d}"
            df.insert(0, RUN_ID, rid)
            df.insert(0, TRACK_ID, tid)
            runs.append(df)
            faults[rid] = meta
        print(f"{tid} ({surface}, {track.length:.0f} m)")

    a.out.mkdir(parents=True, exist_ok=True)
    pd.concat(refs).to_csv(a.out / "references.csv", index=False)
    runs = pd.concat(runs)
    runs.to_csv(a.out / "runs.csv", index=False)
    (a.out / "faults.json").write_text(json.dumps(faults, indent=1))
    print(f"\nlabel counts:\n{runs['label'].value_counts().to_string()}\n-> {a.out}")


if __name__ == "__main__":
    main()
