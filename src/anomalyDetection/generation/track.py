"""Track geometry: either a synthetic stage or one rebuilt from a recorded lap."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomalyDetection.schema import DIST, GRID_STEP_M


@dataclass
class Track:
    name: str
    surface: str
    s: np.ndarray      # distance along stage, m, uniform grid
    x: np.ndarray
    z: np.ndarray
    kappa: np.ndarray  # signed curvature 1/m, positive = left turn

    @property
    def ds(self) -> float:
        return float(self.s[1] - self.s[0])

    @property
    def length(self) -> float:
        return float(self.s[-1])


def _smooth(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return a
    pad = np.pad(a, (w, w), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="same")[w:-w]


def synthetic_track(seed: int, length_m: float = 4000.0, surface: str = "gravel",
                    ds: float = GRID_STEP_M) -> Track:
    """Random stage: straights joined by corners from fast sweepers to hairpins."""
    rng = np.random.default_rng(seed)
    n = int(length_m / ds)
    kappa = np.zeros(n)
    i = 0
    while i < n:
        i += int(rng.uniform(40, 260) / ds)
        corner_len = int(rng.uniform(30, 160) / ds)
        radius = rng.uniform(10, 20) if rng.random() < 0.12 else rng.uniform(20, 180)
        j = min(n, i + corner_len)
        if j - i < 4:
            break
        u = np.linspace(0, 1, j - i)
        ramp = np.clip(np.minimum(u, 1 - u) / 0.25, 0, 1)  # ease in / out
        kappa[i:j] = rng.choice([-1, 1]) * ramp / radius
        i = j
    kappa = _smooth(kappa, 5)
    theta = np.cumsum(kappa) * ds
    x = np.cumsum(np.cos(theta)) * ds
    z = np.cumsum(np.sin(theta)) * ds
    return Track(f"synthetic_{seed}", surface, np.arange(n) * ds, x, z, kappa)


def from_recorded(df: pd.DataFrame, surface: str, name: str = "recorded",
                  ds: float = GRID_STEP_M, smooth_m: float = 20.0,
                  flip_handedness: bool = False) -> Track:
    """Rebuild the centreline from a recorded lap's x/z.

    If left turns come out as negative curvature, the game's axes are mirrored
    relative to ours: pass flip_handedness=True.
    """
    d = df[DIST].to_numpy()
    s = np.arange(0, d.max(), ds)
    w = max(int(smooth_m / ds), 1)
    x = _smooth(np.interp(s, d, df["x"].to_numpy()), w)
    z = _smooth(np.interp(s, d, df["z"].to_numpy()), w)
    dx, dz = np.gradient(x, ds), np.gradient(z, ds)
    ddx, ddz = np.gradient(dx, ds), np.gradient(dz, ds)
    kappa = (dx * ddz - dz * ddx) / np.maximum(dx**2 + dz**2, 1e-9) ** 1.5
    kappa = _smooth(kappa, w)
    if flip_handedness:
        kappa = -kappa
    return Track(name, surface, s, x, z, kappa)
