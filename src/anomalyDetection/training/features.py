"""Turn (run, reference) telemetry pairs into per-sample features on a shared distance grid."""
import numpy as np
import pandas as pd

from anomalyDetection.resample import resample_to_grid
from anomalyDetection.schema import DIST, EXTRA_CHANNELS, LABEL_COL, LABELS, RUN_ID, TRACK_ID
from anomalyDetection.stage_map import map_features

DELTA_CHANNELS = ["speed", "throttle", "brake", "steer", "g_lat", "g_long", "rpm", "gear"]
RAW_CHANNELS = ["speed", "throttle", "brake", "steer", "g_lat", "g_long"]
WINDOWED = ["d_speed", "d_throttle", "d_brake", "d_steer", "d_g_lat", "d_gear", "d_rpm", "d_rear_front",
             "d_dt", "gear_zero"]
TIME_SUM_WINDOWS = (5, 15, 50)   # samples: seconds lost over ~10 / 30 / 100 m
WINDOWS = (5, 15)       # samples (x GRID_STEP_M metres)
SHIFTS = (-10, -5, 5, 10)
PACE_WINDOW = 301       # samples (~600 m): slower-than-this offsets count as overall pace, not mistakes


def _rear_front(d: pd.DataFrame) -> np.ndarray:
    return ((d.wheel_speed_rl + d.wheel_speed_rr)
            / np.maximum(d.wheel_speed_fl + d.wheel_speed_fr, 1.0)).to_numpy()


def build_features(run: pd.DataFrame, ref: pd.DataFrame, stage_map: dict | None = None,
                   dist_offset: float = 0.0, use_extras: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (features, meta). meta holds lap_distance (and nothing label-related).

    With a stage_map, corner context is added. dist_offset converts a cropped lap's distance back
    to real stage distance (cropped training data is re-zeroed).
    """
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

    # Time lost against the reference per 2 m step. Reversing leaves almost no other trace (the
    # converter drops reverse rows), but it shows up here as seconds passing with no distance.
    dt_r, dt_q = np.diff(r["time"].to_numpy(), prepend=r["time"].iloc[0]), np.diff(q["time"].to_numpy(), prepend=q["time"].iloc[0])
    d_dt = pd.Series(dt_r - dt_q)
    f["dt"] = dt_r
    f["d_dt"] = d_dt - d_dt.rolling(PACE_WINDOW, center=True, min_periods=1).median()
    for w in TIME_SUM_WINDOWS:
        f[f"d_dt_sum{w}"] = f["d_dt"].rolling(w, center=True, min_periods=1).sum()
    f["gear_zero"] = (r["gear"] == 0).astype(float)

    windowed = list(WINDOWED)
    if use_extras and all(c in r.columns and c in q.columns for c in EXTRA_CHANNELS):
        # slide angle, rotation and wheelspin: how far from the reference's driving, and how violent
        f["slip"], f["abs_slip"] = r["slip_angle_deg"], r["slip_angle_deg"].abs()
        f["d_slip"] = r["slip_angle_deg"] - q["slip_angle_deg"]
        f["d_abs_slip"] = f["abs_slip"] - q["slip_angle_deg"].abs()
        f["yaw"], f["d_yaw"] = r["yaw_rate_dps"], r["yaw_rate_dps"] - q["yaw_rate_dps"]
        f["ws_rear"], f["ws_front"] = r["wheelspin_rear"], r["wheelspin_front"]
        f["d_ws_rear"] = r["wheelspin_rear"] - q["wheelspin_rear"]
        f["d_ws_front"] = r["wheelspin_front"] - q["wheelspin_front"]
        f["slip_per_g"] = r["slip_angle_deg"] / (r["g_lat"].abs() + 0.3) * np.sign(r["g_lat"])  # slide beyond what the corner needs
        windowed += ["abs_slip", "d_slip", "d_abs_slip", "d_yaw", "d_ws_rear", "d_ws_front", "slip_per_g"]

    extra = {}
    for c in windowed:
        for w in WINDOWS:
            roll = f[c].rolling(w, center=True, min_periods=1)
            extra[f"{c}_m{w}"] = roll.mean()
            extra[f"{c}_s{w}"] = roll.std().fillna(0)
        for sh in SHIFTS:  # look ahead / behind so timing faults are visible
            extra[f"{c}_sh{sh}"] = f[c].shift(sh)
    f = pd.concat([f, pd.DataFrame(extra)], axis=1)
    if stage_map is not None:
        f = pd.concat([f, map_features(r[DIST].to_numpy() + dist_offset, stage_map)], axis=1)
    f = f.fillna(0).astype("float32")
    return f, pd.DataFrame({DIST: r[DIST].to_numpy()})


def build_table(runs: pd.DataFrame, refs: pd.DataFrame, track_ids=None, stage_map=None, dist_offset=0.0,
                use_extras: bool = True):
    """Features + integer labels + meta for every run in a runs.csv-style frame."""
    label_idx = {l: i for i, l in enumerate(LABELS)}
    ref_by_track = {t: g for t, g in refs.groupby(TRACK_ID)}
    Xs, ys, metas = [], [], []
    for (tid, rid), g in runs.groupby([TRACK_ID, RUN_ID], sort=False):
        if track_ids is not None and tid not in track_ids:
            continue
        X, meta = build_features(g, ref_by_track[tid], stage_map, dist_offset, use_extras)
        # runs are generated on the same grid, so labels line up by position
        y = g[LABEL_COL].map(label_idx).to_numpy()[:len(X)]
        meta[TRACK_ID], meta[RUN_ID] = tid, rid
        Xs.append(X), ys.append(y), metas.append(meta)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), pd.concat(metas, ignore_index=True)
