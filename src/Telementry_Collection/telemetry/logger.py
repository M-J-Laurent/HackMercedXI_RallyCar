"""
Step 1: Raw UDP logger for DiRT Rally 2.0.

Listens on 127.0.0.1:20777 and saves EVERY float of EVERY packet to dat/raw/.
Nothing is interpreted here, so a wrong index map never costs you a re-drive.

A new file starts automatically when:
  - lap distance jumps backwards by more than 20 m (restart / new stage), or
  - the session clock goes backwards, or
  - no packets arrive for 3 seconds (menu, pause, game closed).
Files with under ~2 seconds of data are thrown away (menu noise).

Usage (run BEFORE starting the stage, keep it running while you drive):
    python telemetry/logger.py
    python telemetry/logger.py --quiet        # no live readout
Stop with Ctrl+C.
"""

import argparse
import csv
import os
import socket
import struct
import sys
import time
from datetime import datetime

from dr2_fields import IDX, MIN_FLOATS, RAW_DIR, UDP_IP, UDP_PORT

RESTART_DROP_M = 20.0     # lap distance going back this much = new run
SILENCE_S = 3.0           # no packets this long = run ended
MIN_ROWS = 120            # ~2 s at 60 Hz; shorter files are deleted


class RunWriter:
    def __init__(self, n_floats):
        os.makedirs(RAW_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(RAW_DIR, f"raw_{stamp}.csv")
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow([f"f{i}" for i in range(n_floats)])
        self.rows = 0
        print(f"\n[new run] writing {self.path}")

    def write(self, floats):
        self.w.writerow([f"{v:.6g}" for v in floats])
        self.rows += 1

    def close(self, reason):
        self.f.close()
        if self.rows < MIN_ROWS:
            os.remove(self.path)
            print(f"\n[discarded] {os.path.basename(self.path)} ({self.rows} rows, {reason})")
        else:
            print(f"\n[saved] {os.path.basename(self.path)}  {self.rows} rows  ({reason})")


def live_line(f, rows):
    def g(name):
        i = IDX[name]
        return f[i] if i < len(f) else float("nan")
    return (f"dist {g('lap_distance'):8.1f} m | {g('speed') * 3.6:6.1f} km/h | gear {g('gear'):4.0f} | "
            f"thr {g('throttle'):4.2f} brk {g('brake'):4.2f} | steer {g('steer'):+5.2f} | "
            f"g_lat {g('g_lat'):+5.2f} | rpm(raw) {g('rpm'):6.0f} | rows {rows}")


def main():
    ap = argparse.ArgumentParser(description="Log raw DR2 UDP telemetry to dat/raw/")
    ap.add_argument("--ip", default=UDP_IP)
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--quiet", action="store_true", help="no live readout")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((args.ip, args.port))
    except OSError as e:
        sys.exit(f"Could not bind {args.ip}:{args.port} ({e}). Is another logger already running?")
    sock.settimeout(0.5)  # lets Ctrl+C work on Windows
    print(f"Listening on {args.ip}:{args.port}. Start a stage in DR2. Ctrl+C to stop.")

    run = None
    last_dist = None
    last_clock = None
    last_packet = None
    last_print = 0.0
    warned_size = False

    try:
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                if run and last_packet and time.time() - last_packet > SILENCE_S:
                    run.close("no packets for 3 s")
                    run = None
                continue

            n = len(data) // 4
            floats = struct.unpack(f"<{n}f", data[: n * 4])
            if n < MIN_FLOATS and not warned_size:
                print(f"\nWARNING: packet has only {n} floats (need {MIN_FLOATS}). "
                      f"Check extradata=\"3\" in hardware_settings_config.xml.")
                warned_size = True
                continue

            dist = floats[IDX["lap_distance"]]
            clock = floats[IDX["total_time"]]
            if run and last_dist is not None:
                if dist < last_dist - RESTART_DROP_M:
                    run.close("lap distance went backwards (restart?)")
                    run = None
                elif clock < last_clock - 1.0:
                    run.close("session clock reset")
                    run = None

            if run is None:
                run = RunWriter(n)
            run.write(floats)
            last_dist, last_clock, last_packet = dist, clock, time.time()

            if not args.quiet and last_packet - last_print > 0.25:
                sys.stdout.write("\r" + live_line(floats, run.rows))
                sys.stdout.flush()
                last_print = last_packet
    except KeyboardInterrupt:
        pass
    finally:
        if run:
            run.close("stopped")
        sock.close()
        print("Logger stopped.")


if __name__ == "__main__":
    main()
