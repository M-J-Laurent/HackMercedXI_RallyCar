"""Inject labelled driving mistakes into an otherwise good run.

Two phases, because some mistakes change the speed profile (braking points,
throttle timing) and others only change channels (slip, gear):
    1. inject_profile_faults  -> edits the speed profile
    2. apply_channel_faults   -> edits derived channels (after derive_channels)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomalyDetection.generation.createTelemetrydat import (
    add_channel_noise, perturb_profile,
)
from anomalyDetection.generation.physics import (
    CarParams, derive_channels, gentle_braking_from, rpm_for, speed_profile,
)
from anomalyDetection.generation.track import Track
from anomalyDetection.schema import DIST, FAULT_LABELS, LABEL_COL, NORMAL


@dataclass
class Fault:
    label: str
    i0: int
    i1: int
    magnitude: float       # metres for timing faults, fraction otherwise
    sign: float = 1.0      # corner direction (+ left)
    gear_offset: int = 0


@dataclass
class Corner:
    brake_start: int
    apex: int
    accel_end: int


def bump(m: int) -> np.ndarray:
    return np.sin(np.pi * np.linspace(0, 1, max(m, 2))) ** 2


def find_corners(track: Track, v_opt: np.ndarray) -> list[Corner]:
    """Corners = braking zones that end in a speed minimum. Use the noise-free profile."""
    ds, n = track.ds, len(v_opt)
    out = []
    for i in range(5, n - 10):
        if not (v_opt[i - 1] > v_opt[i] + 1e-6 and v_opt[i + 1] >= v_opt[i]):
            continue
        bs = i
        while bs > 0 and v_opt[bs - 1] > v_opt[bs] + 1e-6:
            bs -= 1
        if v_opt[bs] < 1.15 * v_opt[i] or (i - bs) * ds < 12:
            continue
        ae = i
        while ae < n - 1 and abs(v_opt[ae + 1] - v_opt[ae]) <= 1e-6 and (ae - i) * ds < 60:
            ae += 1  # skip the minimum-speed plateau
        while ae < n - 1 and v_opt[ae + 1] > v_opt[ae] + 1e-6 and (ae - i) * ds < 120:
            ae += 1
        if (ae - i) * ds < 20:
            continue
        out.append(Corner(bs, i, ae))
    return out


def inject_profile_faults(track: Track, v_opt: np.ndarray, v: np.ndarray, car: CarParams,
                          mu: float, rng: np.random.Generator,
                          n_faults: int) -> tuple[np.ndarray, list[Fault]]:
    ds, s, n = track.ds, track.s, len(v)
    out, faults, used = v.copy(), [], []
    corners = find_corners(track, v_opt)

    for ci in rng.permutation(len(corners)):
        if len(faults) >= n_faults:
            break
        c = corners[ci]
        lo, hi = max(c.brake_start - int(40 / ds), 0), min(c.accel_end + int(15 / ds), n - 1)
        if any(lo <= u1 and hi >= u0 for u0, u1 in used):
            continue
        label = FAULT_LABELS[rng.integers(len(FAULT_LABELS))]
        sign = float(np.sign(track.kappa[c.apex])) or 1.0
        f = None

        if label == "braked_too_early":
            env, j0 = gentle_braking_from(out, track, car, mu, c.apex, c.brake_start,
                                              rng.uniform(0.4, 0.7))
            if (c.brake_start - j0) * ds >= 8:
                out = env
                f = Fault(label, j0, c.apex, (c.brake_start - j0) * ds, sign)

        elif label == "braked_too_late":
            d = rng.uniform(8, 22)
            base = out.copy()
            seg = np.arange(c.brake_start, c.apex + 1)
            shifted = np.interp(s[seg] - d, s, base)
            cap = np.maximum(base[seg], 1.3 * base[c.apex])
            out[seg] = np.minimum(np.maximum(shifted, base[seg]), cap)
            excess = out[c.apex] - base[c.apex]
            rec = np.arange(c.apex + 1, min(n, c.apex + 1 + int(60 / ds)))
            out[rec] = base[rec] + excess * np.exp(-(s[rec] - s[c.apex]) / 20)
            if excess > 0.3:
                f = Fault(label, c.brake_start, min(n - 1, c.apex + int(40 / ds)), d, sign)
            else:
                out = base

        elif label == "lifted_mid_corner":
            i1 = min(n - 1, c.apex + int(rng.uniform(25, 45) / ds))
            depth = rng.uniform(0.07, 0.14)
            out[c.apex:i1 + 1] *= 1 - depth * bump(i1 - c.apex + 1)
            f = Fault(label, c.apex, i1, depth, sign)

        elif label == "late_throttle":
            d = rng.uniform(15, 45)
            i1 = min(n - 1, c.accel_end + int(d / ds))
            idx = np.arange(c.apex, i1 + 1)
            shifted = np.interp(np.maximum(s[idx] - d, s[c.apex]), s, out)
            out[idx] = np.minimum(out[idx], shifted)
            f = Fault(label, c.apex, i1, d, sign)

        elif label == "oversteer":
            i0, i1 = c.apex - int(5 / ds), min(n - 1, c.apex + int(25 / ds))
            out[i0:i1 + 1] *= 1 - 0.05 * bump(i1 - i0 + 1)
            f = Fault(label, i0, i1, 0.10, sign)

        elif label == "understeer":
            i0, i1 = max(c.apex - int(25 / ds), 0), min(n - 1, c.apex + int(5 / ds))
            out[i0:i1 + 1] *= 1 - 0.06 * bump(i1 - i0 + 1)
            f = Fault(label, i0, i1, 0.5, sign)

        elif label == "wrong_gear":
            f = Fault(label, max(c.apex - int(20 / ds), 0), min(n - 1, c.apex + int(30 / ds)),
                      1.0, sign, int(rng.choice([-1, 1])))

        if f is not None:
            faults.append(f)
            used.append((min(lo, f.i0), max(hi, f.i1)))
    return out, faults


def apply_channel_faults(df: pd.DataFrame, faults: list[Fault], car: CarParams) -> pd.DataFrame:
    df = df.copy()
    ng = len(car.gear_ratios)
    for f in faults:
        sl = slice(f.i0, f.i1 + 1)
        w = bump(f.i1 - f.i0 + 1)
        if f.label == "oversteer":  # rear steps out: rear wheels spin, driver counter-steers
            for c in ("wheel_speed_rl", "wheel_speed_rr"):
                df.loc[df.index[sl], c] *= 1 + f.magnitude * w
            df.loc[df.index[sl], "steer"] = np.clip(df["steer"].iloc[sl] - f.sign * 0.5 * w, -1, 1)
            df.loc[df.index[sl], "g_lat"] *= 1 - 0.15 * w
        elif f.label == "understeer":  # fronts wash wide: more lock, less lateral g
            df.loc[df.index[sl], "steer"] = np.clip(df["steer"].iloc[sl] * (1 + 0.5 * w), -1, 1)
            df.loc[df.index[sl], "g_lat"] *= 1 - 0.3 * w
            for c in ("wheel_speed_fl", "wheel_speed_fr"):
                df.loc[df.index[sl], c] *= 1 - 0.05 * w
        elif f.label == "lifted_mid_corner":
            df.loc[df.index[sl], "throttle"] *= 1 - np.clip(1.6 * w, 0, 1)
        elif f.label == "wrong_gear":
            gear = np.clip(df["gear"].iloc[sl].to_numpy() + f.gear_offset, 1, ng)
            df.loc[df.index[sl], "gear"] = gear
            df.loc[df.index[sl], "rpm"] = rpm_for(df["speed"].iloc[sl].to_numpy(), gear, car)
    return df


def make_flawed_run(track: Track, car: CarParams, mu: float, rng: np.random.Generator,
                    n_faults: int, pace: float) -> tuple[pd.DataFrame, list[dict]]:
    """A human-ish run with `n_faults` labelled mistakes (0 gives an all-normal run)."""
    v_opt = speed_profile(track, car, mu)
    v = perturb_profile(v_opt, rng, track.ds, pace)
    v, faults = inject_profile_faults(track, v_opt, v, car, mu, rng, n_faults)
    df = add_channel_noise(derive_channels(track, v, car, mu), rng)
    df = apply_channel_faults(df, faults, car)

    labels = np.full(len(df), NORMAL, dtype=object)
    for f in faults:
        labels[f.i0:f.i1 + 1] = f.label
    df[LABEL_COL] = labels
    meta = [{"label": f.label, "start_m": float(track.s[f.i0]), "end_m": float(track.s[f.i1]),
             "magnitude": float(f.magnitude)} for f in faults]
    return df, meta
