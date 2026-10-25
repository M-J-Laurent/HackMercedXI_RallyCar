"""
Bundle lap data for the static site into web/data/bundle.js (loaded with a <script> tag,
so the site works on GitHub Pages AND when index.html is opened straight from disk).

    python web/build_data.py                       # uses dat/real/runs.csv, reference = row marked REFERENCE
    python web/build_data.py --ref stage1_clean_02.csv
    python web/build_data.py --infer               # also run Jay's model on every lap first (needs models/)

Where mistake segments come from, per lap (first match wins):
  1. web/data/analysis/<lap>.json   - output of infer.py, optionally with a "tip" per segment from the LLM
  2. --infer                        - runs infer.py now and saves its output to (1)
  3. dat/real/annotations.json      - the driver's notes (marked as such in the UI)

Standard library only.
"""

import argparse
import bisect
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REAL = os.path.join(REPO, "dat", "real")
OUT = os.path.join(HERE, "data")
ANALYSIS = os.path.join(OUT, "analysis")
CAL = os.path.join(REPO, "src", "Telementry_Collection", "telemetry", "calibration.json")

GRID_M = 5.0
# column in the lap CSV -> short key in the bundle
CHANNELS = {
    "time": "t", "speed": "v", "throttle": "thr", "brake": "brk", "steer": "str",
    "gear": "gear", "rpm": "rpm", "g_lat": "glat", "g_long": "glong",
    "slip_angle_deg": "slip", "x": "x", "z": "z",
}
STEP_ONLY = {"gear"}  # not interpolated


def load_lap(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols = [c for c in CHANNELS if c in r.fieldnames]
        rows = [{c: float(row[c]) for c in ["lap_distance", *cols]} for row in r]
    return rows, cols


def resample(rows, cols):
    d = [r["lap_distance"] for r in rows]
    n = int(d[-1] // GRID_M) + 1
    out = {"d": [round(i * GRID_M, 1) for i in range(n)]}
    for c in cols:
        key, vals = CHANNELS[c], []
        for i in range(n):
            x = i * GRID_M
            j = bisect.bisect_left(d, x)
            if j <= 0:
                v = rows[0][c]
            elif j >= len(d):
                v = rows[-1][c]
            elif c in STEP_ONLY:
                v = rows[j][c]
            else:
                a, b = rows[j - 1], rows[j]
                w = (x - d[j - 1]) / (d[j] - d[j - 1])
                v = a[c] + (b[c] - a[c]) * w
            if c == "speed":
                v *= 3.6  # km/h for display
            vals.append(round(v, 3 if c in ("throttle", "brake", "steer") else 2))
        out[key] = vals
    return out


def run_infer(lap, ref):
    src = os.path.join(REPO, "src")
    cmd = [sys.executable, "-m", "anomalyDetection.training.infer",
           "--run", os.path.join(REAL, lap), "--ref", os.path.join(REAL, ref)]
    res = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  infer failed for {lap}: {res.stderr.strip().splitlines()[-1] if res.stderr else '?'}")
        return None
    segs = json.loads(res.stdout)
    for s in segs:
        s.pop("context", None)  # big; the site has the full traces anyway
    os.makedirs(ANALYSIS, exist_ok=True)
    with open(os.path.join(ANALYSIS, lap.replace(".csv", ".json")), "w") as f:
        json.dump(segs, f, indent=1)
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="reference lap file name in dat/real (default: row whose type mentions REFERENCE)")
    ap.add_argument("--infer", action="store_true", help="run infer.py on every lap (needs models/)")
    a = ap.parse_args()

    with open(os.path.join(REAL, "runs.csv"), newline="") as f:
        runs = [r for r in csv.DictReader(f) if r.get("file")]
    ref = a.ref or next((r["file"] for r in runs if "REFERENCE" in r.get("type", "").upper()), runs[0]["file"])

    notes = {}
    ann_path = os.path.join(REAL, "annotations.json")
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            notes = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    stage_map = None
    map_path = os.path.join(REAL, "stage1_map.json")
    if os.path.exists(map_path):
        with open(map_path) as f:
            stage_map = json.load(f)

    # which way up to draw the map so that left turns look like left turns
    map_flip = False
    if os.path.exists(CAL):
        with open(CAL) as f:
            map_flip = json.load(f).get("xz_heading_sign_for_left_turn", 1) < 0

    laps = []
    for r in runs:
        path = os.path.join(REAL, r["file"])
        if not os.path.exists(path):
            print(f"skip {r['file']} (missing)")
            continue
        rows, cols = load_lap(path)
        lap = {"file": r["file"], "name": r["file"].replace(".csv", ""), "stage": r.get("stage", ""),
               "car": r.get("car", ""), "type": r.get("type", ""), "notes": r.get("notes", ""),
               "is_reference": r["file"] == ref, "trace": resample(rows, cols)}
        if r["file"] != ref:
            apath = os.path.join(ANALYSIS, r["file"].replace(".csv", ".json"))
            segs = None
            if os.path.exists(apath):
                with open(apath) as f:
                    segs = json.load(f)
                lap["segments_source"] = "model"
            elif a.infer:
                segs = run_infer(r["file"], ref)
                lap["segments_source"] = "model" if segs is not None else None
            if segs is None and r["file"] in notes:
                segs = [{"start_m": lo, "end_m": hi} for lo, hi in notes[r["file"]]]
                lap["segments_source"] = "driver_notes"
            lap["segments"] = segs or []
        laps.append(lap)
        print(f"{r['file']}: {len(lap['trace']['d'])} points, "
              f"{len(lap.get('segments', []))} segments ({lap.get('segments_source') or 'reference'})")

    bundle = {"reference": ref, "grid_m": GRID_M, "map_flip": map_flip, "stage_map": stage_map, "laps": laps}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bundle.js"), "w") as f:
        f.write("// generated by web/build_data.py - do not edit by hand\n")
        f.write("window.RALLY_DATA = ")
        json.dump(bundle, f, separators=(",", ":"))
        f.write(";\n")
    print(f"wrote {os.path.join(OUT, 'bundle.js')} ({os.path.getsize(os.path.join(OUT, 'bundle.js')) // 1024} KB)")


if __name__ == "__main__":
    main()
