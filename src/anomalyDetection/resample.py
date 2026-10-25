"""Shared by generation/ and training/: put a lap on a fixed distance grid."""
import numpy as np
import pandas as pd

from anomalyDetection.schema import CHANNELS, DIST, EXTRA_CHANNELS, GRID_STEP_M


def resample_to_grid(df: pd.DataFrame, step: float = GRID_STEP_M) -> pd.DataFrame:
    """Fixed distance grid so runs of different pace line up sample-for-sample."""
    d = df[DIST].to_numpy()
    s = np.arange(0, d.max(), step)
    out = {DIST: s}
    for c in CHANNELS:
        out[c] = np.interp(s, d, df[c].to_numpy())
    out["gear"] = np.rint(out["gear"])
    for c in EXTRA_CHANNELS:  # optional channels, only if the lap has them
        if c in df.columns:
            y = df[c].to_numpy()
            if c == "slip_angle_deg":  # unwrap so interpolating across +-180 does not invent a 0
                y = np.degrees(np.unwrap(np.radians(y)))
            out[c] = np.interp(s, d, y)
            if c == "slip_angle_deg":
                out[c] = (out[c] + 180) % 360 - 180
    return pd.DataFrame(out)
