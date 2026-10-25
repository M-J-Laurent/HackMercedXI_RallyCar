"""Turn (run, reference) telemetry pairs into per-sample features on a shared distance grid."""
import numpy as np
import pandas as pd

from anomalyDetection.resample import resample_to_grid
from anomalyDetection.schema import DIST, LABEL_COL, LABELS, RUN_ID, TRACK_ID

DELTA_CHANNELS = ["speed", "throttle", "brake", "steer", "g_lat", "g_long", "rpm", "gear"]
RAW_CHANNELS = ["speed", "throttle", "brake", "steer", "g_lat", "g_long"]
WINDOWED = ["d_speed", "d_throttle", "d_brake", "d_steer", "d_g_lat", "d_gear", "d_rpm", "d_rear_front"]
WINDOWS = (5, 15)       # samples (x GRID_STEP_M metres)
SHIFTS = (-10, -5, 5, 10)
PACE_WINDOW = 301       # samples (~600 m): slower-than-this offsets count as overall pace, not mistakes


def _rear_front(d: pd.DataFrame) -> np.ndarray:
    return ((d.wheel_speed_rl + d.wheel_speed_rr)
            / np.maximum(d.wheel_speed_fl + d.wheel_speed_fr, 1.0)).to_numpy()


def build_features(run: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (features, meta). meta holds lap_distance (and nothing label-related)."""
    r, q = resample_to_grid(run), resample_to_grid(ref)
    n = min(len(r), len(q))
    r, q = r.iloc[:n].reset_index(drop=True), q.iloc[:n].reset_index(drop=True)

    f = pd.DataFrame(index=range(n))
    for c in DELTA_CHANNELS:
        f[f"d_{c}"] = r[c] - q[c]
    # A generally slower (or faster) lap than the reference is pace, not a mistake: keep the
    # slow trend as its own feature and make the deltas local.
    trend = f["d_speed"].rolling(PACE_WINDOW, center=True, min_periods=1).median()
    f["d_speed_trend"] = trend
    f["d_speed"] = f["d_speed"] - trend
    f["d_rpm"] = f["d_rpm"] - f["d_rpm"].rolling(PACE_WINDOW, center=True, min_periods=1).median()
    for c in RAW_CHANNELS:
        f[c] = r[c]
        f[f"ref_{c}"] = q[c]
    f["speed_ratio"] = r.speed / np.maximum(q.speed + trend, 1.0)
    f["rear_front"] = _rear_front(r)
    f["d_rear_front"] = f["rear_front"] - _rear_front(q)
    f["front_lr"] = (r.wheel_speed_fr - r.wheel_speed_fl) / np.maximum(r.speed, 1.0)
    f["d_front_lr"] = f["front_lr"] - (q.wheel_speed_fr - q.wheel_speed_fl) / np.maximum(q.speed, 1.0)

    for c in WINDOWED:
        for w in WINDOWS:
            roll = f[c].rolling(w, center=True, min_periods=1)
            f[f"{c}_m{w}"] = roll.mean()
            f[f"{c}_s{w}"] = roll.std().fillna(0)
        for sh in SHIFTS:  # look ahead / behind so timing faults are visible
            f[f"{c}_sh{sh}"] = f[c].shift(sh)
    f = f.fillna(0).astype("float32")
    return f, pd.DataFrame({DIST: r[DIST].to_numpy()})


def build_table(runs: pd.DataFrame, refs: pd.DataFrame, track_ids=None):
    """Features + integer labels + meta for every run in a runs.csv-style frame."""
    label_idx = {l: i for i, l in enumerate(LABELS)}
    ref_by_track = {t: g for t, g in refs.groupby(TRACK_ID)}
    Xs, ys, metas = [], [], []
    for (tid, rid), g in runs.groupby([TRACK_ID, RUN_ID], sort=False):
        if track_ids is not None and tid not in track_ids:
            continue
        X, meta = build_features(g, ref_by_track[tid])
        # runs are generated on the same grid, so labels line up by position
        y = g[LABEL_COL].map(label_idx).to_numpy()[:len(X)]
        meta[TRACK_ID], meta[RUN_ID] = tid, rid
        Xs.append(X), ys.append(y), metas.append(meta)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), pd.concat(metas, ignore_index=True)
