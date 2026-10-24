"""
DiRT Rally 2.0 UDP packet layout (extradata="3") and shared settings.

Each packet is a flat array of little-endian 32-bit floats (~66 of them, 264 bytes).
The index map below is the commonly cited Codemasters layout. It is NOT verified yet:
run check_signs.py with --dump on a recorded run to confirm each index.

If an index turns out wrong, fix it HERE and re-run convert.py on your raw files.
You never need to re-drive, because logger.py saves every float untouched.
"""

import json
import os

# ---- Network (matches hardware_settings_config.xml) ----
UDP_IP = "127.0.0.1"
UDP_PORT = 20777

# ---- Float index map (unverified guesses) ----
IDX = {
    "total_time": 0,      # seconds since session start
    "lap_time": 1,        # seconds since stage start (used as `time`)
    "lap_distance": 2,    # metres from stage start
    "total_distance": 3,
    "x": 4,
    "y": 5,               # height, dropped in the output
    "z": 6,
    "speed": 7,           # m/s
    "wheel_speed_rl": 25, # wheel order may be RL, RR, FL, FR (check_signs.py tests left/right)
    "wheel_speed_rr": 26,
    "wheel_speed_fl": 27,
    "wheel_speed_fr": 28,
    "throttle": 29,       # 0..1
    "steer": 30,          # -1..1, sign to be verified
    "brake": 31,          # 0..1
    "clutch": 32,
    "gear": 33,           # 0 = neutral, 10 (or -1) may be reverse
    "g_lat": 34,          # sign to be verified
    "g_long": 35,
    "lap": 36,
    "rpm": 37,            # may be rpm / 10
}

MIN_FLOATS = max(IDX.values()) + 1

# ---- Folders (relative to the HAckMerced folder) ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "dat", "raw")
REAL_DIR = os.path.join(ROOT, "dat", "real")
CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

# ---- Calibration (written by check_signs.py, read by convert.py) ----
DEFAULT_CALIBRATION = {
    "steer_sign": 1,              # multiply raw steer by this so that LEFT = positive
    "glat_sign": 1,               # multiply raw g_lat by this so that turning LEFT = positive
    "rpm_scale": 10,              # multiply raw rpm by this
    "reverse_gear_values": [10, -1],
    "verified": False,
}


def load_calibration():
    cal = dict(DEFAULT_CALIBRATION)
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            cal.update(json.load(f))
    return cal


def save_calibration(cal):
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(cal, f, indent=2)
