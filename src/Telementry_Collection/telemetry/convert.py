"""
Step 2: Convert raw logs into the team's telemetry CSV (section 3a) PLUS every other value DR2 sends.

Reads dat/raw/raw_*.csv, writes dat/real/<name>.csv.

Columns:
  1) The 16 contract columns, first and in this exact order (Jay's model reads these by name):
       lap_distance, time, x, z, speed, gear, rpm, throttle, brake, steer,
       g_lat, g_long, wheel_speed_fl, wheel_speed_fr, wheel_speed_rl, wheel_speed_rr
  2) Every other value in the 66-float DR2 packet, named (velocity, orientation, suspension,
     clutch, sectors, brake temps, track length, ...). Raw versions of steer/gear/rpm are kept
     as steer_raw / gear_raw / rpm_raw.
  3) Derived values:
       speed_kmh        speed in km/h
       slip_angle_deg   angle between where the nose points and where the car travels.
                        positive = nose points LEFT of travel direction.
                        LEFT turn + big positive = oversteer; RIGHT turn + big negative = oversteer.
                        0 below 3 m/s.
       yaw_rate_dps     rotation speed in deg/s, positive = rotating LEFT
       wheelspin_front  front wheel speed / car speed - 1 (positive = spinning, negative = locking)
       wheelspin_rear   rear  wheel speed / car speed - 1

Cleaning rules:
  - only the part after the last restart is kept (lap distance jumping back > 20 m)
  - the run is cut at the stage finish (DR2 resets the stage timer to 0 after the finish)
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
    python telemetry/convert.py FILE --slim    # contract columns only (old 16-column format)
"""

import argparse
import csv
import glob
import math
import os
import sys

from dr2_fields import IDX, MIN_FLOATS, RAW_DIR, REAL_DIR, load_calibration

COLUMNS = [
    "lap_distance", "time", "x", "z", "speed", "gear", "rpm", "throttle", "brake", "steer",
    "g_lat", "g_long", "wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr",
]

# Codemasters "extradata 3" layout: a name for every one of the 66 floats.
RAW_NAMES = [
    "total_time", "lap_time", "lap_distance", "total_distance",
    "pos_x", "pos_y", "pos_z", "speed_ms",
    "vel_x", "vel_y", "vel_z",
    "vec_a_x", "vec_a_y", "vec_a_z",          # car orientation vector A
    "vec_b_x", "vec_b_y", "vec_b_z",          # car orientation vector B
    "susp_pos_rl", "susp_pos_rr", "susp_pos_fl", "susp_pos_fr",
    "susp_vel_rl", "susp_vel_rr", "susp_vel_fl", "susp_vel_fr",
    "wheel_speed_rl", "wheel_speed_rr", "wheel_speed_fl", "wheel_speed_fr",
    "throttle", "steer_raw", "brake", "clutch", "gear_raw",
    "g_lat_raw", "g_long", "lap", "rpm_raw",
    "sli_pro_support", "race_position", "kers_level", "kers_max", "drs",
    "traction_control", "abs", "fuel_in_tank", "fuel_capacity", "in_pits",
    "sector", "sector1_time", "sector2_time",
    "brake_temp_rl", "brake_temp_rr", "brake_temp_fl", "brake_temp_fr",
    "tyre_pressure_rl", "tyre_pressure_rr", "tyre_pressure_fl", "tyre_pressure_fr",
    "laps_completed", "total_laps", "track_length", "last_lap_time",
    "max_rpm_raw", "idle_rpm_raw", "max_gears",
]
# raw values already present in the contract columns under another name
ALREADY_IN_CONTRACT = {"lap_distance", "lap_time", "pos_x", "pos_z", "speed_ms", "throttle", "brake",
                       "g_long", "wheel_speed_rl", "wheel_speed_rr", "wheel_speed_fl", "wheel_speed_fr"}
EXTRA_RAW = [n for n in RAW_NAMES if n not in ALREADY_IN_CONTRACT]
DERIVED = ["speed_kmh", "slip_angle_deg", "yaw_rate_dps", "wheelspin_front", "wheelspin_rear"]
FULL_COLUMNS = COLUMNS + EXTRA_RAW + DERIVED

RESTART_DROP_M = 20.0
MOVING_MS = 0.5


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


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


def pick_forward_vector(raw):
    """Which orientation vector points forward: the one most aligned with velocity at speed.
    Returns (x_index, z_index, sign, alignment) or None."""
    best = None
    for xi, zi in ((11, 13), (14, 16)):
        dots = []
        for f in raw:
            if len(f) <= 16 or f[7] < 10:
                continue
            nv, nf = math.hypot(f[8], f[10]), math.hypot(f[xi], f[zi])
            if nv > 1e-6 and nf > 1e-6:
                dots.append((f[8] * f[xi] + f[10] * f[zi]) / (nv * nf))
        if dots:
            mean = sum(dots) / len(dots)
            if best is None or abs(mean) > best[3]:
                best = (xi, zi, 1 if mean >= 0 else -1, abs(mean))
    return best


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

    full = all(len(f) >= len(RAW_NAMES) for f in raw)
    fwd = pick_forward_vector(raw) if full else None
    xz_sign = cal.get("xz_heading_sign_for_left_turn", 1)
    if full and fwd is None:
        warnings.append("could not find the forward vector; slip_angle/yaw_rate set to 0")

    reverse = set(cal["reverse_gear_values"])
    out = []
    last_d = None
    dropped_rev = 0
    prev_h = prev_t = None
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
        spd = f[IDX["speed"]]
        row = {
            "lap_distance": dist,
            "time": f[IDX["lap_time"]],
            "x": f[IDX["x"]],
            "z": f[IDX["z"]],
            "speed": spd,
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
        }
        if full:
            for i, name in enumerate(RAW_NAMES):
                if name not in ALREADY_IN_CONTRACT:
                    row[name] = f[i]
            row["speed_kmh"] = spd * 3.6
            front = (row["wheel_speed_fl"] + row["wheel_speed_fr"]) / 2
            rear = (row["wheel_speed_rl"] + row["wheel_speed_rr"]) / 2
            row["wheelspin_front"] = front / spd - 1 if spd > 3 else 0.0
            row["wheelspin_rear"] = rear / spd - 1 if spd > 3 else 0.0
            slip = yaw = 0.0
            if fwd:
                xi, zi, s, _ = fwd
                h_f = math.atan2(f[zi] * s, f[xi] * s)
                if spd > 3 and math.hypot(f[8], f[10]) > 1e-6:
                    slip = math.degrees(wrap(h_f - math.atan2(f[10], f[8]))) * xz_sign
                tt = row["time"]
                if prev_h is not None and tt > prev_t:
                    yaw = math.degrees(wrap(h_f - prev_h)) / (tt - prev_t) * xz_sign
                prev_h, prev_t = h_f, tt
            row["slip_angle_deg"] = slip
            row["yaw_rate_dps"] = yaw
        out.append(row)
    if dropped_rev:
        warnings.append(f"dropped {dropped_rev} reverse-gear rows")
    if not full:
        warnings.append("packets have fewer than 66 values; wrote contract columns only")
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


def write_csv(rows, path, slim=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = COLUMNS if slim or "slip_angle_deg" not in rows[0] else FULL_COLUMNS
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[k] if k == "gear" else f"{round(r[k], 5) + 0.0:.5f}" for k in cols])
    return cols


def convert_file(src, name, every, cal, slim=False):
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
    cols = write_csv(rows, dst, slim)
    dur = rows[-1]["time"] - rows[0]["time"]
    gears = sorted({r["gear"] for r in rows})
    print(f"  distance {rows[0]['lap_distance']:.0f} -> {rows[-1]['lap_distance']:.0f} m, "
          f"duration {dur:.1f} s, top speed {max(r['speed'] for r in rows) * 3.6:.0f} km/h, gears {gears}")
    if "slip_angle_deg" in rows[0] and not slim:
        slips = sorted(abs(r["slip_angle_deg"]) for r in rows if r["speed"] > 8)
        if slips:
            print(f"  slide angle above 30 km/h: median {slips[len(slips) // 2]:.1f} deg, max {slips[-1]:.0f} deg")
    print(f"  wrote {dst}  ({len(cols)} columns)")
    return dst


def main():
    ap = argparse.ArgumentParser(description="Raw DR2 log -> contract CSV (+ all extra values) in dat/real/")
    ap.add_argument("files", nargs="*", help="raw CSV(s) from dat/raw/")
    ap.add_argument("--name", help="output name (no .csv), e.g. stage1_ref_01. Single file only.")
    ap.add_argument("--all", action="store_true", help="convert every raw file without an output yet")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth row (downsample)")
    ap.add_argument("--slim", action="store_true", help="write only the 16 contract columns")
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
        convert_file(src, name, args.every, cal, args.slim)


if __name__ == "__main__":
    main()
