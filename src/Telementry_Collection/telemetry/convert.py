"""
Step 2: Convert raw logs into the team's telemetry CSV contract (section 3a).

Reads dat/raw/raw_*.csv, writes dat/real/<name>.csv with exactly these columns:
    lap_distance, time, x, z, speed, gear, rpm, throttle, brake, steer,
    g_lat, g_long, wheel_speed_fl, wheel_speed_fr, wheel_speed_rl, wheel_speed_rr

Cleaning rules:
  - only the part after the last restart is kept (lap distance jumping back > 20 m)
  - rows before the car first moves (speed < 0.5 m/s) are dropped
  - rows with negative lap distance are dropped
  - reverse-gear rows are dropped, neutral -> 0
  - lap_distance is made STRICTLY increasing (rows that don't move forward are dropped)
  - steer / g_lat signs and rpm scale come from calibration.json (run check_signs.py first)
  - throttle / brake clipped to 0..1, steer to -1..1

Usage:
    python telemetry/convert.py dat/raw/raw_20261024_153000.csv --name stage1_ref_01
    python telemetry/convert.py --all          # convert every raw file not converted yet
    python telemetry/convert.py FILE --every 2 # optional downsample (keep every 2nd row)
"""

import argparse
import csv
import glob
import os
import sys

from dr2_fields import IDX, MIN_FLOATS, RAW_DIR, REAL_DIR, load_calibration

COLUMNS = [
    "lap_distance", "time", "x", "z", "speed", "gear", "rpm", "throttle", "brake", "steer",
    "g_lat", "g_long", "wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr",
]

RESTART_DROP_M = 20.0
MOVING_MS = 0.5


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def read_raw(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        rows = []
        for line in r:
            try:
                vals = [float(v) for v in line]
            except ValueError:
                continue
            if len(vals) >= MIN_FLOATS:
                rows.append(vals)
    return rows


def convert_rows(raw, cal, every=1):
    warnings = []
    d = IDX["lap_distance"]

    # keep only the segment after the last restart
    start = 0
    for i in range(1, len(raw)):
        if raw[i][d] < raw[i - 1][d] - RESTART_DROP_M:
            start = i
    if start:
        warnings.append(f"restart detected; kept only rows after raw row {start}")
    raw = raw[start:]

    # cut at the finish line: DR2 resets the stage timer to 0 after the finish
    t = IDX["lap_time"]
    for i in range(1, len(raw)):
        if raw[i - 1][t] > 1.0 and raw[i][t] < raw[i - 1][t] - 0.5:
            warnings.append(f"stage finish detected at {raw[i - 1][t]:.2f} s; dropped {len(raw) - i} rows after it")
            raw = raw[:i]
            break

    # drop the stationary part at the start
    first_move = next((i for i, f in enumerate(raw) if f[IDX["speed"]] > MOVING_MS), None)
    if first_move is None:
        return [], warnings + ["car never moved; nothing to write"]
    raw = raw[first_move:]

    reverse = set(cal["reverse_gear_values"])
    out = []
    last_d = None
    dropped_rev = 0
    for k, f in enumerate(raw):
        if every > 1 and k % every:
            continue
        dist = f[d]
        if dist < 0:
            continue
        gear = round(f[IDX["gear"]])
        if gear in reverse or gear < 0:
            dropped_rev += 1
            continue
        if last_d is not None and dist <= last_d:
            continue  # enforce strictly increasing distance
        last_d = dist
        out.append({
            "lap_distance": dist,
            "time": f[IDX["lap_time"]],
            "x": f[IDX["x"]],
            "z": f[IDX["z"]],
            "speed": f[IDX["speed"]],
            "gear": gear,
            "rpm": f[IDX["rpm"]] * cal["rpm_scale"],
            "throttle": clip(f[IDX["throttle"]], 0.0, 1.0),
            "brake": clip(f[IDX["brake"]], 0.0, 1.0),
            "steer": clip(f[IDX["steer"]] * cal["steer_sign"], -1.0, 1.0),
            "g_lat": f[IDX["g_lat"]] * cal["glat_sign"],
            "g_long": f[IDX["g_long"]],
            "wheel_speed_fl": f[IDX["wheel_speed_fl"]],
            "wheel_speed_fr": f[IDX["wheel_speed_fr"]],
            "wheel_speed_rl": f[IDX["wheel_speed_rl"]],
            "wheel_speed_rr": f[IDX["wheel_speed_rr"]],
        })
    if dropped_rev:
        warnings.append(f"dropped {dropped_rev} reverse-gear rows")
    return out, warnings


def validate(rows):
    problems = []
    for i in range(1, len(rows)):
        if rows[i]["lap_distance"] <= rows[i - 1]["lap_distance"]:
            problems.append(f"lap_distance not strictly increasing at row {i}")
            break
    for i in range(1, len(rows)):
        if rows[i]["time"] < rows[i - 1]["time"]:
            problems.append(f"time goes backwards at row {i}")
            break
    for c in ("throttle", "brake"):
        if any(not 0 <= r[c] <= 1 for r in rows):
            problems.append(f"{c} outside 0..1")
    if any(not -1 <= r["steer"] <= 1 for r in rows):
        problems.append("steer outside -1..1")
    return problems


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r[k] if k == "gear" else f"{round(r[k], 5) + 0.0:.5f}") for k in COLUMNS})


def convert_file(src, name, every, cal):
    raw = read_raw(src)
    rows, warnings = convert_rows(raw, cal, every)
    print(f"\n{os.path.basename(src)}: {len(raw)} raw rows -> {len(rows)} clean rows")
    for w in warnings:
        print(f"  note: {w}")
    if not rows:
        print("  skipped (no usable rows)")
        return None
    problems = validate(rows)
    for p in problems:
        print(f"  PROBLEM: {p}")
    dst = os.path.join(REAL_DIR, f"{name}.csv")
    write_csv(rows, dst)
    dur = rows[-1]["time"] - rows[0]["time"]
    gears = sorted({r["gear"] for r in rows})
    print(f"  distance {rows[0]['lap_distance']:.0f} -> {rows[-1]['lap_distance']:.0f} m, "
          f"duration {dur:.1f} s, top speed {max(r['speed'] for r in rows) * 3.6:.0f} km/h, gears {gears}")
    print(f"  wrote {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser(description="Raw DR2 log -> contract CSV in dat/real/")
    ap.add_argument("files", nargs="*", help="raw CSV(s) from dat/raw/")
    ap.add_argument("--name", help="output name (no .csv), e.g. stage1_ref_01. Single file only.")
    ap.add_argument("--all", action="store_true", help="convert every raw file without an output yet")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth row (downsample)")
    args = ap.parse_args()

    cal = load_calibration()
    if not cal.get("verified"):
        print("WARNING: calibration.json not verified yet. Steer/g_lat signs and rpm scale are guesses.\n"
              "         Run check_signs.py on a left-turn recording first.")

    files = list(args.files)
    if args.all:
        files += sorted(glob.glob(os.path.join(RAW_DIR, "raw_*.csv")))
    if not files:
        ap.print_help()
        sys.exit(1)
    if args.name and len(files) > 1:
        sys.exit("--name only works with a single file")

    for src in files:
        name = args.name or os.path.splitext(os.path.basename(src))[0].replace("raw_", "run_", 1)
        if args.all and not args.name and os.path.exists(os.path.join(REAL_DIR, name + ".csv")):
            continue
        convert_file(src, name, args.every, cal)


if __name__ == "__main__":
    main()
