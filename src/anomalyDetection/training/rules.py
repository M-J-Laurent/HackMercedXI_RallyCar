"""Physics-level mistakes that are obvious enough to need no model.

An engine stall (rpm collapsing to ~0 while the car was moving) happened only at real
crashes and spins in our recorded laps, never on a clean lap, so it is a reliable marker.
A violent deceleration with no braking at the same moment means the car hit something.

Works on the raw recorded lap (60 Hz), before any resampling.
"""
import numpy as np
import pandas as pd

from anomalyDetection.schema import DIST

STALL_RPM = 600.0          # engine effectively off
MIN_DISTANCE_M = 30.0      # ignore the start line
COLLISION_G = 4.0          # decel over 0.25 s that brakes cannot produce
MERGE_S = 3.0              # stall rows closer than this are one event
LEAD_S, TAIL_S = 1.5, 3.0  # report from just before the hit until the car is moving again
WINDOW = 15                # samples (~0.25 s) for deceleration


def find_stall_events(run: pd.DataFrame) -> list[dict]:
    t, v, rpm = run["time"].to_numpy(), run["speed"].to_numpy(), run["rpm"].to_numpy()
    dist, brake = run[DIST].to_numpy(), run["brake"].to_numpy()
    n = len(run)
    rate = max(int(round(1 / max(np.median(np.diff(t)), 1e-3))), 1)  # samples per second

    groups: list[list[int]] = []
    for i in np.flatnonzero((rpm < STALL_RPM) & (dist > MIN_DISTANCE_M)):
        if groups and t[i] - t[groups[-1][1]] < MERGE_S:
            groups[-1][1] = i
        else:
            groups.append([i, i])

    events = []
    for a, b in groups:
        w0, w1 = max(a - int(LEAD_S * rate), 0), min(a + WINDOW, n - WINDOW - 1)
        idx = np.arange(w0, max(w1, w0) + 1)
        decel_g = -(v[idx + WINDOW] - v[idx]) / np.maximum(t[idx + WINDOW] - t[idx], 1e-3) / 9.81
        peak, brake_max = float(decel_g.max()), float(brake[w0:a + WINDOW].max())
        end = min(b + int(TAIL_S * rate), n - 1)
        events.append({
            "label": "collision" if peak >= COLLISION_G and brake_max < 0.3 else "spinout",
            "start_m": float(dist[w0]), "end_m": float(dist[end]),
            "rule": {
                "name": "engine_stall",
                "stall_s": round(float(t[b] - t[a]), 1),
                "speed_before_kmh": round(float(v[w0] * 3.6)),
                "speed_min_kmh": round(float(v[w0:end + 1].min() * 3.6)),
                "peak_decel_g": round(peak, 1),
                "max_brake": round(brake_max, 2),
            },
        })
    return events
