"""
Step 0 / sanity check: figure out sign conventions from a KNOWN LEFT TURN.

How to record the test:
  1. Start logger.py.
  2. Start any stage. As soon as you can move, hold the steering LEFT and drive
     slow circles (or take a long left corner) for ~5-10 seconds.
  3. Ctrl+C the logger.
  4. python telemetry/check_signs.py dat/raw/raw_XXXX.csv --turn left

What it does:
  - finds the part of the recording where you were steering hard while moving
  - prints the RAW sign of steer and g_lat there
  - works out the multipliers so that LEFT = positive (the team contract)
  - checks wheel-speed left/right order (in a left turn the RIGHT wheels spin faster)
  - guesses the rpm scale (raw max under ~1500 means rpm is stored / 10)
  - writes everything to telemetry/calibration.json, which convert.py uses

Add --dump to print min / max / mean of every float index (to verify the index map).
"""

import argparse
import math
import statistics as st

from dr2_fields import IDX, CALIBRATION_PATH, DEFAULT_CALIBRATION, load_calibration, save_calibration
from convert import read_raw

NAME_OF = {v: k for k, v in IDX.items()}


def sign(v):
    return 1 if v >= 0 else -1


def dump(raw):
    n = len(raw[0])
    print(f"\n{'idx':>3}  {'name':<16}{'min':>12}{'max':>12}{'mean':>12}")
    for i in range(n):
        col = [r[i] for r in raw]
        print(f"{i:>3}  {NAME_OF.get(i, ''):<16}{min(col):>12.4g}{max(col):>12.4g}{st.fmean(col):>12.4g}")


def main():
    ap = argparse.ArgumentParser(description="Derive steer/g_lat signs from a known turn")
    ap.add_argument("file", help="raw CSV from dat/raw/")
    ap.add_argument("--turn", choices=["left", "right"], default="left",
                    help="which way you were turning in the test")
    ap.add_argument("--dump", action="store_true", help="print stats for every float index")
    args = ap.parse_args()

    raw = read_raw(args.file)
    if not raw:
        raise SystemExit("No rows in file.")
    if args.dump:
        dump(raw)

    moving = [r for r in raw if r[IDX["speed"]] > 2.0]
    if len(moving) < 30:
        raise SystemExit("Not enough rows with the car moving (> 2 m/s). Record a longer test.")

    # turning window: hardest-steering 25% of moving rows
    by_steer = sorted(moving, key=lambda r: abs(r[IDX["steer"]]), reverse=True)
    window = by_steer[: max(30, len(moving) // 4)]
    steer_raw = st.median(r[IDX["steer"]] for r in window)
    glat_raw = st.median(r[IDX["g_lat"]] for r in window)

    want = 1 if args.turn == "left" else -1          # contract: left = positive
    steer_sign = want * sign(steer_raw)
    glat_sign = want * sign(glat_raw)

    # wheel left/right check (outside wheels spin faster)
    left = st.fmean(r[IDX["wheel_speed_fl"]] + r[IDX["wheel_speed_rl"]] for r in window) / 2
    right = st.fmean(r[IDX["wheel_speed_fr"]] + r[IDX["wheel_speed_rr"]] for r in window) / 2
    outside_faster = (right > left) if args.turn == "left" else (left > right)

    # rpm scale
    rpm_max = max(r[IDX["rpm"]] for r in moving)
    rpm_scale = 10 if rpm_max < 1500 else 1

    # heading change from x/z positions (tells us the world-axis handedness)
    turn_rows = [r for r in moving if abs(r[IDX["steer"]]) >= abs(window[-1][IDX["steer"]])]
    dh = 0.0
    prev = None
    for a, b in zip(turn_rows, turn_rows[1:]):
        dx, dz = b[IDX["x"]] - a[IDX["x"]], b[IDX["z"]] - a[IDX["z"]]
        if math.hypot(dx, dz) < 0.05:
            continue
        h = math.atan2(dz, dx)
        if prev is not None:
            step = (h - prev + math.pi) % (2 * math.pi) - math.pi
            if abs(step) < 0.5:
                dh += step
        prev = h

    gears = sorted({round(r[IDX["gear"]]) for r in raw})

    print(f"\nTest turn: {args.turn.upper()}   ({len(window)} rows in the turning window)")
    print(f"  raw steer  (median) = {steer_raw:+.3f}  -> steer_sign = {steer_sign:+d}"
          f"   {'(already left=positive)' if steer_sign == 1 else '(FLIP needed)'}")
    print(f"  raw g_lat  (median) = {glat_raw:+.3f}  -> glat_sign  = {glat_sign:+d}"
          f"   {'(already left=positive)' if glat_sign == 1 else '(FLIP needed)'}")
    print(f"  wheel speeds: left side {left:.2f} m/s, right side {right:.2f} m/s -> "
          + ("OK, left/right order looks right" if outside_faster
             else "SUSPICIOUS: inside wheels faster. Left/right wheel indices may be swapped in dr2_fields.py"))
    print(f"  raw rpm max = {rpm_max:.0f} -> rpm_scale = {rpm_scale}")
    print(f"  raw gear values seen: {gears}")
    print(f"  x/z heading change during turn: {math.degrees(dh):+.0f} deg "
          f"(sign for a {args.turn} turn in DR2 world coords)")
    if abs(steer_raw) < 0.1:
        print("  WARNING: steering in the window was tiny. Hold the wheel further left and re-record.")
    if abs(glat_raw) < 0.1:
        print("  WARNING: lateral g was tiny. Turn faster / tighter and re-record.")

    cal = load_calibration()
    cal.update({
        "steer_sign": steer_sign,
        "glat_sign": glat_sign,
        "rpm_scale": rpm_scale,
        "wheel_left_right_ok": outside_faster,
        "xz_heading_sign_for_left_turn": sign(dh) * (1 if args.turn == "left" else -1),
        "raw_gear_values_seen": gears,
        "reverse_gear_values": cal.get("reverse_gear_values", DEFAULT_CALIBRATION["reverse_gear_values"]),
        "verified": abs(steer_raw) >= 0.1 and abs(glat_raw) >= 0.1,
        "source_file": args.file,
    })
    save_calibration(cal)
    print(f"\nSaved {CALIBRATION_PATH}")
    print("Now re-run convert.py on your runs so they use these signs.")


if __name__ == "__main__":
    main()
