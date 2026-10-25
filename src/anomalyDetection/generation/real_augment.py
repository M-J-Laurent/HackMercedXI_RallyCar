"""Training data built from real recorded laps instead of simulated physics.

    normal variants : real clean laps, blended and pace-warped so they cover the natural
                      lap-to-lap spread (the thing the simulated data got wrong)
    faulty variants : the same, plus labelled mistakes edited into the real channels

Speed edits are applied coherently (wheel speeds, rpm, g_lat, g_long follow) so an injected
slowdown does not give itself away through inconsistent channels.
"""
import numpy as np
import pandas as pd

from anomalyDetection.generation.createTelemetrydat import smooth_noise
from anomalyDetection.generation.physics import G, CarParams
from anomalyDetection.resample import resample_to_grid
from anomalyDetection.schema import (
    CHANNELS, DIST, FAULT_LABELS, GRID_STEP_M, LABEL_COL, NORMAL, TELEMETRY_COLUMNS, WHEELS,
)

DS = GRID_STEP_M
WHEEL_COLS = [f"wheel_speed_{w}" for w in WHEELS]
BLENDED = [c for c in CHANNELS if c not in ("gear",)]


def load_lap(path, crop: tuple[float, float] | None = None) -> pd.DataFrame:
    """Recorded lap on the 2 m grid, optionally cropped to [a, b) metres and re-zeroed."""
    df = resample_to_grid(pd.read_csv(path))
    if crop:
        a, b = crop
        df = df[(df[DIST] >= a) & (df[DIST] < b)].copy()
        df[DIST] -= a
    return df.reset_index(drop=True)


def bump(m: int) -> np.ndarray:
    return np.sin(np.pi * np.linspace(0, 1, m)) ** 2


def smoothstep(m: int) -> np.ndarray:
    u = np.linspace(0, 1, m)
    return u * u * (3 - 2 * u)


def apply_speed_ratio(df: pd.DataFrame, ratio: np.ndarray) -> pd.DataFrame:
    """Scale speed by a per-sample ratio and keep the dependent channels consistent."""
    out = df.copy()
    v0, v1 = df["speed"].to_numpy(), df["speed"].to_numpy() * ratio
    a0, a1 = np.gradient(0.5 * v0 * v0, DS), np.gradient(0.5 * v1 * v1, DS)
    out["speed"] = v1
    for c in WHEEL_COLS:
        out[c] = df[c] * ratio
    out["rpm"] = np.maximum(df["rpm"] * ratio, df["rpm"].quantile(0.01))
    out["g_lat"] = df["g_lat"] * ratio**2
    out["g_long"] = df["g_long"] + (a1 - a0) / G
    # time follows the new speed; per-step cap keeps near-stops from adding absurd amounts
    va0, va1 = 0.5 * (v0[1:] + v0[:-1]), 0.5 * (v1[1:] + v1[:-1])
    extra = np.clip(DS / np.maximum(va1, 0.3) - DS / np.maximum(va0, 0.3), -0.5, 0.6)
    dt = np.maximum(np.diff(df["time"].to_numpy()) + extra, 0.001)
    out["time"] = df["time"].iloc[0] + np.r_[0.0, np.cumsum(dt)]
    return out


def blend(a: pd.DataFrame, b: pd.DataFrame, alpha: np.ndarray) -> pd.DataFrame:
    n = min(len(a), len(b))
    a, b, alpha = a.iloc[:n].reset_index(drop=True), b.iloc[:n].reset_index(drop=True), alpha[:n]
    out = a.copy()
    for c in BLENDED:
        out[c] = a[c] * (1 - alpha) + b[c] * alpha
    out["gear"] = np.where(alpha < 0.5, a["gear"], b["gear"])
    return out


def shift_distance(df: pd.DataFrame, delta_m: float) -> pd.DataFrame:
    """Same lap but sampled delta_m further along: a different line / timing offset."""
    out, s = df.copy(), df[DIST].to_numpy()
    for c in CHANNELS:
        out[c] = np.interp(s + delta_m, s, df[c].to_numpy())
    return out


def make_normal_variant(laps: list[pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    """A plausible clean lap: blend of real clean laps, then pace drift and a small offset."""
    n = min(len(l) for l in laps)
    if len(laps) > 1 and rng.random() < 0.6:
        i, j = rng.choice(len(laps), 2, replace=False)
        alpha = 1 / (1 + np.exp(-1.5 * smooth_noise(n, rng, DS, rng.uniform(100, 300))))
        base = blend(laps[i], laps[j], alpha)
    else:
        base = laps[rng.integers(len(laps))].iloc[:n].reset_index(drop=True)
    ratio = 1 + rng.uniform(-0.06, 0.03) + 0.04 * smooth_noise(n, rng, DS, rng.uniform(80, 300))
    base = apply_speed_ratio(base, np.clip(ratio, 0.8, 1.04))
    return shift_distance(base, rng.uniform(-6, 6))


def find_dips(lap: pd.DataFrame) -> list[int]:
    """Where the real lap slows for a corner: a place a mistake plausibly happens."""
    v = lap["speed"].rolling(15, center=True, min_periods=1).mean().to_numpy()
    n, w = len(v), 40
    return [i for i in range(w, n - w)
            if v[i] <= v[i - w:i + w + 1].min() and v[max(i - 75, 0):i + 76].max() - v[i] >= 5]


def inject_real_faults(lap: pd.DataFrame, rng: np.random.Generator, n_faults: int):
    """Edit labelled mistakes into a real lap. Returns (lap, labels, faults metadata)."""
    n = len(lap)
    dips = find_dips(lap)
    ratio, edits, used = np.ones(n), [], []
    m = lambda metres: int(metres / DS)

    for _ in range(n_faults * 4):
        if len(edits) >= n_faults:
            break
        apex = int(rng.choice(dips)) if dips and rng.random() < 0.7 else int(rng.integers(40, n - 40))
        label = FAULT_LABELS[rng.integers(len(FAULT_LABELS))]
        if label == "braked_too_early":
            i0, i1 = apex - m(rng.uniform(40, 110)), apex
        elif label == "braked_too_late":
            i0, i1 = apex - m(30), apex + m(45)
        elif label == "lifted_mid_corner":
            i0, i1 = apex, apex + m(rng.uniform(25, 50))
        elif label == "late_throttle":
            i0, i1 = apex, apex + m(rng.uniform(35, 70))
        elif label == "oversteer":
            i0, i1 = apex - m(10), apex + m(40)
        elif label == "understeer":
            i0, i1 = apex - m(40), apex + m(10)
        elif label == "spinout":
            i0, i1 = apex - m(15), apex + m(25)
        elif label == "reversing":
            i0, i1 = apex - m(8), apex + m(12)
        else:  # wrong_gear
            i0, i1 = apex - m(30), apex + m(40)
        if i0 < 0 or i1 >= n or any(i0 - 20 <= u1 and i1 + 20 >= u0 for u0, u1 in used):
            continue

        w = bump(i1 - i0 + 1)
        e = {"label": label, "i0": i0, "i1": i1, "w": w}
        sl = slice(i0, i1 + 1)
        if label == "braked_too_early":
            d = rng.uniform(0.12, 0.4)
            ramp = smoothstep(i1 - i0 + 1)
            ratio[sl] *= 1 - d * ramp
            e.update(depth=d, brake=rng.uniform(0.3, 0.9) * ramp)
        elif label == "braked_too_late":
            up, dep = rng.uniform(0.05, 0.15), rng.uniform(0.2, 0.5)
            pre, post = apex - i0 + 1, i1 - apex + 1
            ratio[i0:apex + 1] *= 1 + up * smoothstep(pre)
            ratio[apex:i1 + 1] *= 1 - dep * bump(post)
            e.update(depth=dep, brake=None, apex=apex, post=bump(post) * rng.uniform(0.5, 1.0))
        elif label == "lifted_mid_corner":
            d = rng.uniform(0.07, 0.15)
            ratio[sl] *= 1 - d * w
            e.update(depth=d)
        elif label == "late_throttle":
            d = rng.uniform(0.08, 0.25)
            ratio[sl] *= 1 - d * w
            e.update(depth=d)
        elif label == "oversteer":
            d = rng.uniform(0.08, 0.3)
            ratio[sl] *= 1 - d * w
            e.update(depth=d, slip=rng.uniform(0.08, 0.25), counter=rng.uniform(0.3, 0.7))
        elif label == "understeer":
            d = rng.uniform(0.05, 0.15)
            ratio[sl] *= 1 - d * w
            e.update(depth=d)
        elif label == "spinout":
            d = rng.uniform(0.85, 0.97)
            ratio[sl] *= 1 - d * w
            e.update(depth=d, amp=rng.uniform(0.4, 0.7), g=rng.uniform(0.8, 1.4),
                     freq=int(rng.choice([1, 2])), gap=rng.uniform(2, 9))
        elif label == "reversing":
            d = rng.uniform(0.9, 0.99)
            ratio[sl] *= 1 - d * w
            e.update(depth=d, gap=rng.uniform(2, 8))
        else:
            e.update(depth=1.0, offset=int(rng.choice([-1, 1])))
        steer =lap["steer"].to_numpy()[max(apex - m(10), 0):apex + m(10) + 1]
        e["sign"] = float(np.sign(steer.mean())) or 1.0
        edits.append(e)
        used.append((i0, i1))

    out = apply_speed_ratio(lap, ratio)
    ratios = np.asarray(CarParams().gear_ratios)
    for e in edits:
        sl, w, lab = slice(e["i0"], e["i1"] + 1), e["w"], e["label"]
        col = lambda c: out.columns.get_loc(c)
        if lab == "braked_too_early":
            out.iloc[sl, col("brake")] = np.maximum(out["brake"].iloc[sl], e["brake"])
        elif lab == "braked_too_late":
            a = e["apex"]
            out.iloc[a:e["i1"] + 1, col("brake")] = np.maximum(out["brake"].iloc[a:e["i1"] + 1], e["post"])
        elif lab in ("lifted_mid_corner", "late_throttle"):
            gain = 1.6 if lab == "lifted_mid_corner" else 1.5
            out.iloc[sl, col("throttle")] = out["throttle"].iloc[sl] * (1 - np.clip(gain * w, 0, 1))
        elif lab == "oversteer":
            for c in ("wheel_speed_rl", "wheel_speed_rr"):
                out.iloc[sl, col(c)] = out[c].iloc[sl] * (1 + e["slip"] * w)
            out.iloc[sl, col("steer")] = np.clip(out["steer"].iloc[sl] - e["sign"] * e["counter"] * w, -1, 1)
            out.iloc[sl, col("g_lat")] = out["g_lat"].iloc[sl] * (1 - 0.2 * w)
        elif lab == "understeer":
            out.iloc[sl, col("steer")] = np.clip(out["steer"].iloc[sl] * (1 + 0.4 * w), -1, 1)
            out.iloc[sl, col("g_lat")] = out["g_lat"].iloc[sl] * (1 - 0.3 * w)
            for c in ("wheel_speed_fl", "wheel_speed_fr"):
                out.iloc[sl, col(c)] = out[c].iloc[sl] * (1 - 0.05 * w)
        elif lab in ("spinout", "reversing"):
            mid = (e["i0"] + e["i1"]) // 2
            if lab == "spinout":  # wild steering and lateral g while the car rotates
                t = np.linspace(0, 1, len(w))
                out.iloc[sl, col("steer")] = np.clip(out["steer"].iloc[sl] + e["amp"] * np.sin(2 * np.pi * e["freq"] * t) * w, -1, 1)
                out.iloc[sl, col("g_lat")] = out["g_lat"].iloc[sl] + e["g"] * np.sin(2 * np.pi * e["freq"] * t + 1) * w
            low = w > (0.7 if lab == "spinout" else 0.6)
            out.iloc[sl, col("gear")] = np.where(low, 1 if lab == "spinout" else 0, out["gear"].iloc[sl])
            out.iloc[mid:, col("time")] = out["time"].iloc[mid:] + e["gap"]  # seconds lost recovering
        elif lab == "wrong_gear":
            g0 = out["gear"].iloc[sl].to_numpy()
            g1 = np.clip(g0 + e["offset"], 1, len(ratios))
            ok = g0 >= 1
            scale = np.where(ok, ratios[np.clip(g1, 1, len(ratios)).astype(int) - 1]
                             / ratios[np.clip(g0, 1, len(ratios)).astype(int) - 1], 1.0)
            out.iloc[sl, col("gear")] = np.where(ok, g1, g0)
            out.iloc[sl, col("rpm")] = out["rpm"].iloc[sl] * scale

    labels = np.full(n, NORMAL, dtype=object)
    for e in edits:
        labels[e["i0"]:e["i1"] + 1] = e["label"]
    s = lap[DIST].to_numpy()
    meta = [{"label": e["label"], "start_m": float(s[e["i0"]]), "end_m": float(s[e["i1"]]),
             "magnitude": float(e["depth"])} for e in edits]
    return out, labels, meta


def make_real_run(clean_laps: list[pd.DataFrame], rng: np.random.Generator, n_faults: int):
    """One training run: natural-variation clean lap with n_faults labelled mistakes."""
    lap = make_normal_variant(clean_laps, rng)
    out, labels, meta = inject_real_faults(lap, rng, n_faults)
    out = out[TELEMETRY_COLUMNS].copy()
    out[LABEL_COL] = labels
    return out, meta
