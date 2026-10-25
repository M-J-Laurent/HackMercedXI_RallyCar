"""A pseudo map of one stage, built from clean recorded laps: where the corners are.

    cd src
    python -m anomalyDetection.stage_map --clean ../dat/real/stage1_clean_01.csv ../dat/real/stage1_clean_02.csv \\
        --ref ../dat/real/stage1_clean_02.csv --flip-handedness --out ../dat/real/stage1_map.json

Corners come from the curvature of the recorded x/z path. Corners from several clean laps are merged
(same direction, apex within 25 m); a corner is kept if the reference lap has it or two laps agree.
Each corner also records the speed the clean laps carried through it, which is the natural variation
the reference alone cannot show.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anomalyDetection.generation.track import from_recorded
from anomalyDetection.resample import resample_to_grid
from anomalyDetection.schema import DIST

MAX_RADIUS_M = 120.0   # tighter than this counts as a corner
MIN_LENGTH_M = 24.0
MERGE_APEX_M = 25.0
APPROACH_M = 100.0     # the braking zone before a corner starts
RUNOUT_M = 40.0        # exit phase continues this far past the corner end
APEX_ZONE_M = 10.0


def _kind(radius: float) -> str:
    return "hairpin" if radius < 20 else "tight" if radius < 40 else "medium" if radius < 80 else "fast"


def corners_from_lap(df: pd.DataFrame, flip_handedness: bool, surface: str = "gravel") -> list[dict]:
    t = from_recorded(df, surface, flip_handedness=flip_handedness, smooth_m=30)
    k = t.kappa
    on = np.abs(k) > 1 / MAX_RADIUS_M
    out, i = [], 0
    while i < len(k):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(k) and (on[j + 1] or on[j + 1:j + 11].any()):  # bridge gaps up to 20 m
            j += 1
        if (j - i) * t.ds >= MIN_LENGTH_M:
            seg = k[i:j + 1]
            apex = i + int(np.argmax(np.abs(seg)))
            out.append({"start_m": float(t.s[i]), "end_m": float(t.s[j]), "apex_m": float(t.s[apex]),
                        "dir": float(np.sign(seg.mean())), "radius_m": float(1 / np.abs(seg).max())})
        i = j + 1
    return out


def _apex_speed_kmh(lap: pd.DataFrame, apex_m: float) -> float:
    m = (lap[DIST] >= apex_m - 20) & (lap[DIST] <= apex_m + 20)
    return float(lap.loc[m, "speed"].min() * 3.6) if m.any() else float("nan")


def build_stage_map(clean_paths: list[Path], ref_path: Path, flip_handedness: bool, name: str = "stage") -> dict:
    laps = {str(p): pd.read_csv(p) for p in clean_paths}
    ref_key = str(ref_path)
    laps.setdefault(ref_key, pd.read_csv(ref_path))
    found = []  # (apex, lap key, corner)
    for key, df in laps.items():
        for c in corners_from_lap(df, flip_handedness):
            found.append((c["apex_m"], key, c))
    found.sort(key=lambda f: f[0])

    clusters: list[list] = []
    for apex, key, c in found:
        for cl in clusters:
            if cl[0][2]["dir"] == c["dir"] and abs(np.median([x[0] for x in cl]) - apex) <= MERGE_APEX_M \
                    and all(x[1] != key for x in cl):
                cl.append((apex, key, c))
                break
        else:
            clusters.append([(apex, key, c)])

    grid = {k: resample_to_grid(df) for k, df in laps.items()}
    corners = []
    for cl in clusters:
        keys = {x[1] for x in cl}
        if ref_key not in keys and len(keys) < 2:
            continue
        cs = [x[2] for x in cl]
        apex = float(np.median([c["apex_m"] for c in cs]))
        speeds = [_apex_speed_kmh(grid[k], apex) for k in laps]
        radius = float(np.median([c["radius_m"] for c in cs]))
        corners.append({
            "apex_m": round(apex, 1),
            "start_m": round(float(np.median([c["start_m"] for c in cs])), 1),
            "end_m": round(float(np.median([c["end_m"] for c in cs])), 1),
            "direction": "left" if cs[0]["dir"] > 0 else "right",
            "radius_m": round(radius, 1), "kind": _kind(radius),
            "support": len(keys),
            "ref_apex_speed_kmh": round(_apex_speed_kmh(grid[ref_key], apex)),
            "clean_apex_speed_kmh": [round(min(speeds)), round(max(speeds))],
        })
    corners.sort(key=lambda c: c["apex_m"])
    for n, c in enumerate(corners, 1):
        c["number"] = n
    return {"name": name, "length_m": round(float(max(g[DIST].max() for g in grid.values())), 1),
            "flip_handedness": flip_handedness, "clean_laps": len(laps), "corners": corners}


def load_stage_map(path) -> dict:
    return json.loads(Path(path).read_text())


def phase_at(c: dict, x: float) -> str | None:
    if x < c["start_m"] - APPROACH_M or x > c["end_m"] + RUNOUT_M:
        return None
    if x < c["start_m"]:
        return "approach"
    if x < c["apex_m"] - APEX_ZONE_M:
        return "entry"
    if x <= c["apex_m"] + APEX_ZONE_M:
        return "apex"
    return "exit"


def locate(stage_map: dict, start_m: float, end_m: float) -> dict:
    """Which corner (and part of it) a segment covers, or where on the straights it is."""
    cs = stage_map["corners"]
    best, best_overlap, touched = None, 0.0, []
    for c in cs:
        lo, hi = c["start_m"] - APPROACH_M, c["end_m"] + RUNOUT_M
        overlap = min(hi, end_m) - max(lo, start_m)
        if overlap >= max(10.0, 0.3 * (end_m - start_m)):  # a graze of the approach zone is still the straight
            touched.append(c["number"])
            if overlap > best_overlap:
                best, best_overlap = c, overlap
    mid = (start_m + end_m) / 2
    if best is None:
        before = [c for c in cs if c["apex_m"] <= mid]
        after = [c for c in cs if c["apex_m"] > mid]
        return {"on_straight": True,
                "after_corner": before[-1]["number"] if before else None,
                "before_corner": after[0]["number"] if after else None,
                "to_next_corner_m": round(after[0]["start_m"] - mid) if after else None}
    c = best
    return {
        "number": c["number"], "direction": c["direction"], "radius_m": c["radius_m"], "kind": c["kind"],
        "apex_m": c["apex_m"],
        "phase_at_start": phase_at(c, start_m) or ("approach" if start_m < c["start_m"] else "exit"),
        "phase_at_end": phase_at(c, end_m) or ("approach" if end_m < c["start_m"] else "exit"),
        "offset_from_apex_m": round(mid - c["apex_m"]),
        "ref_apex_speed_kmh": c["ref_apex_speed_kmh"],
        "clean_apex_speed_kmh": c["clean_apex_speed_kmh"],
        **({"also_touches": [n for n in touched if n != c["number"]]} if len(touched) > 1 else {}),
    }


MAP_FEATURES = ["map_phase", "map_apex_off", "map_log_radius", "map_dir", "map_ref_apex_speed",
                "map_apex_band", "map_to_next_apex", "map_since_prev_apex"]


def map_features(distance: np.ndarray, stage_map: dict) -> pd.DataFrame:
    """Per-sample map context for the model: only relative things, never absolute distance."""
    cs = stage_map["corners"]
    apex = np.array([c["apex_m"] for c in cs])
    d = np.asarray(distance, dtype=float)
    near = np.abs(d[:, None] - apex[None, :]).argmin(axis=1)
    codes = {None: 0, "approach": 1, "entry": 2, "apex": 3, "exit": 4}
    phase = np.array([codes[phase_at(cs[k], x)] for k, x in zip(near, d)], dtype=float)
    off = np.clip(d - apex[near], -150, 150)
    nxt = np.array([(apex[apex > x].min() - x) if (apex > x).any() else 300.0 for x in d]).clip(0, 300)
    prev = np.array([(x - apex[apex <= x].max()) if (apex <= x).any() else 300.0 for x in d]).clip(0, 300)
    in_corner = phase > 0
    return pd.DataFrame({
        "map_phase": phase,
        "map_apex_off": np.where(in_corner, off, 0.0),
        "map_log_radius": np.where(in_corner, np.log([cs[k]["radius_m"] for k in near]), 0.0),
        "map_dir": np.where(in_corner, [1.0 if cs[k]["direction"] == "left" else -1.0 for k in near], 0.0),
        "map_ref_apex_speed": np.where(in_corner, [cs[k]["ref_apex_speed_kmh"] for k in near], 0.0),
        "map_apex_band": np.where(in_corner, [cs[k]["clean_apex_speed_kmh"][1] - cs[k]["clean_apex_speed_kmh"][0] for k in near], 0.0),
        "map_to_next_apex": nxt, "map_since_prev_apex": prev,
    }).astype("float32")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--clean", nargs="+", required=True, type=Path)
    p.add_argument("--ref", required=True, type=Path)
    p.add_argument("--flip-handedness", action="store_true")
    p.add_argument("--name", default="stage")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    m = build_stage_map(a.clean, a.ref, a.flip_handedness, a.name)
    a.out.write_text(json.dumps(m, indent=1))
    print(f"{len(m['corners'])} corners over {m['length_m']:.0f} m from {m['clean_laps']} clean laps -> {a.out}")
    for c in m["corners"]:
        print(f"  #{c['number']:>2} {c['direction']:>5} {c['kind']:<7} r={c['radius_m']:>5.0f} m  apex {c['apex_m']:>6.0f} m  "
              f"ref {c['ref_apex_speed_kmh']} km/h, clean laps {c['clean_apex_speed_kmh']}  (support {c['support']})")


if __name__ == "__main__":
    main()
