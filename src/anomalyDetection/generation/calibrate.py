"""Fit surface friction and engine power so the simulated speed profile matches a recorded lap.

Use the fastest, cleanest recorded lap you have: any mistakes in it get baked into mu / power.

    cd src
    python -m anomalyDetection.generation.calibrate --recorded ../dat/real/lap.csv --surface gravel
Then pass the printed values to createTelemetrydat.py (--mu / --power).
"""
import argparse

import numpy as np
import pandas as pd

from anomalyDetection.generation.physics import SURFACE_MU, CarParams, speed_profile
from anomalyDetection.generation.track import from_recorded
from anomalyDetection.schema import DIST


def fit(df: pd.DataFrame, surface: str, flip_handedness: bool = False) -> tuple[float, float, float]:
    track = from_recorded(df, surface, flip_handedness=flip_handedness)
    v_rec = np.interp(track.s, df[DIST], df["speed"])

    def rmse(mu, power):
        v = speed_profile(track, CarParams(power_w=power), mu, v_start=float(v_rec[0]))
        return float(np.sqrt(np.mean((v - v_rec) ** 2)))

    mus = np.linspace(0.3, 1.3, 11)
    powers = np.linspace(120e3, 380e3, 8)
    best = min(((rmse(m, p), m, p) for m in mus for p in powers))
    _, m0, p0 = best
    for m in np.linspace(m0 - 0.1, m0 + 0.1, 9):  # refine around the coarse optimum
        for p in np.linspace(p0 - 35e3, p0 + 35e3, 7):
            best = min(best, (rmse(m, p), m, p))
    return best[1], best[2], best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--recorded", required=True)
    ap.add_argument("--surface", default="gravel", choices=list(SURFACE_MU))
    ap.add_argument("--flip-handedness", action="store_true")
    a = ap.parse_args()
    mu, power, err = fit(pd.read_csv(a.recorded), a.surface, a.flip_handedness)
    print(f"mu={mu:.3f}  power_w={power:.0f}  speed RMSE={err:.2f} m/s")
    print(f"--mu {mu:.3f} --power {power:.0f}")


if __name__ == "__main__":
    main()
