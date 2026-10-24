"""Generate a 'perfect' (reference) run, or a human-ish clean run, for a track.

    cd src
    python -m anomalyDetection.generation.createTelemetrydat --seed 1 --out ../dat/synthetic/ref.csv
    python -m anomalyDetection.generation.createTelemetrydat --recorded ../dat/real/lap.csv --surface gravel --out ref.csv
"""
import argparse

import numpy as np
import pandas as pd

from anomalyDetection.generation.physics import (
    SURFACE_MU, CarParams, derive_channels, speed_profile,
)
from anomalyDetection.generation.track import Track, from_recorded, synthetic_track


def smooth_noise(n: int, rng: np.random.Generator, ds: float, corr_m: float = 40.0) -> np.ndarray:
    """Unit-variance noise correlated over ~corr_m metres."""
    w = max(int(corr_m / ds), 1)
    z = rng.standard_normal(n + 2 * w)
    s = np.convolve(z, np.ones(w) / w, mode="same")[w:w + n]
    return s / (s.std() + 1e-9)


def perturb_profile(v: np.ndarray, rng: np.random.Generator, ds: float,
                    pace: float = 1.0, jitter: float = 0.01) -> np.ndarray:
    """Human-ish pace: a bit slower than optimal with slow drift. Never faster than optimal."""
    return np.minimum(v * pace * (1 + jitter * smooth_noise(len(v), rng, ds)), v)


def add_channel_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sensor / inputs noise so the model doesn't learn 'noisy == anomalous'."""
    n = len(df)
    df = df.copy()
    df["steer"] = np.clip(df["steer"] + 0.01 * rng.standard_normal(n), -1, 1)
    df["throttle"] = np.clip(df["throttle"] + 0.02 * rng.standard_normal(n), 0, 1)
    df["brake"] = np.clip(df["brake"] * (1 + 0.03 * rng.standard_normal(n)), 0, 1)
    df["g_lat"] += 0.01 * rng.standard_normal(n)
    df["g_long"] += 0.01 * rng.standard_normal(n)
    for w in ("fl", "fr", "rl", "rr"):
        df[f"wheel_speed_{w}"] += 0.05 * rng.standard_normal(n)
    return df


def generate_reference(track: Track, car: CarParams, mu: float) -> pd.DataFrame:
    """Noise-free time-optimal run."""
    return derive_channels(track, speed_profile(track, car, mu), car, mu)


def generate_clean_run(track: Track, car: CarParams, mu: float,
                       rng: np.random.Generator, pace: float = 0.97) -> pd.DataFrame:
    v = perturb_profile(speed_profile(track, car, mu), rng, track.ds, pace)
    return add_channel_noise(derive_channels(track, v, car, mu), rng)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--recorded", help="recorded lap CSV (schema.py columns); else a synthetic track is used")
    p.add_argument("--flip-handedness", action="store_true", help="mirror recorded x/z if left turns read negative")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--length", type=float, default=4000.0)
    p.add_argument("--surface", default="gravel", choices=list(SURFACE_MU))
    p.add_argument("--mu", type=float, help="override surface friction (use calibrate.py)")
    p.add_argument("--power", type=float, help="override engine power in W (use calibrate.py)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    car = CarParams(power_w=a.power) if a.power else CarParams()
    mu = a.mu or SURFACE_MU[a.surface]
    if a.recorded:
        track = from_recorded(pd.read_csv(a.recorded), a.surface, flip_handedness=a.flip_handedness)
    else:
        track = synthetic_track(a.seed, a.length, a.surface)
    df = generate_reference(track, car, mu)
    df.to_csv(a.out, index=False)
    print(f"{track.name}: {track.length:.0f} m, lap {df['time'].iloc[-1]:.1f} s, mu={mu} -> {a.out}")


if __name__ == "__main__":
    main()
