# Rally Coach: telemetry capture (Teammate A)

Turns DiRT Rally 2.0 UDP telemetry into CSVs that match the team contract (section 3a).
Pure Python standard library, so no `pip install` is needed. Python 3.8+.

```
HAckMerced/
  telemetry/
    dr2_fields.py     index map + IP/port (127.0.0.1:20777). Fix wrong indexes HERE.
    logger.py         step 1: UDP -> dat/raw/raw_<timestamp>.csv (every float, untouched)
    check_signs.py    sign test from a known left turn -> telemetry/calibration.json
    convert.py        step 2: raw -> dat/real/<name>.csv (contract format, cleaned)
  dat/raw/            raw logs (created automatically)
  dat/real/           files for Jay (created automatically)
```

Game config (already done):
`<udp enabled="true" extradata="3" ip="127.0.0.1" port="20777" delay="1" />`

## 1. Check that packets arrive

Open a terminal in the `HAckMerced` folder:

```
python telemetry\logger.py
```

Start a stage. The terminal should show a live line (distance, km/h, gear, throttle, steer...).
Watch it while you drive. If throttle doesn't go to 1.00 when you floor it, or gear doesn't
step when you shift, an index in `dr2_fields.py` is wrong.
Stop with **Ctrl+C**.

## 2. Sign test (do this once)

1. Run the logger, start a stage, then **hold steering LEFT and drive slow circles / a long
   left corner for 5-10 s**. Ctrl+C.
2. Run:
   ```
   python telemetry\check_signs.py dat\raw\raw_XXXX.csv --turn left --dump
   ```
   It prints the raw steer / g_lat signs, whether they need flipping, a left/right wheel-speed
   check, and the rpm scale, then saves `telemetry\calibration.json`.
   `--dump` prints min/max/mean of every float index so you can double-check the index map.
3. Delete that test file from `dat\raw` afterwards so it doesn't get converted as a run.

## 3. Record runs

Use the **same stage and same car** for everything.

- Several **clean** laps -> the fastest becomes the reference.
- Several **sloppy** laps with mistakes on purpose. Write down roughly where (in metres;
  the live readout shows distance) and what you did.

The logger starts a new file automatically on restarts and when packets stop (menus),
so you can leave it running for the whole session.

## 4. Convert for Jay

```
python telemetry\convert.py dat\raw\raw_20261024_153000.csv --name stage1_ref_01
python telemetry\convert.py dat\raw\raw_20261024_154500.csv --name stage1_sloppy_01
python telemetry\convert.py --all        # anything not converted yet, auto-named run_<timestamp>
```

It prints rows kept, distance covered, duration, top speed and gears, and flags any contract
problem. Output goes to `dat\real\`. If you later fix an index or sign, just run convert again.

## 5. Log what each run is

Keep `dat\real\runs.csv` (the answer key for Jay):

```
file,stage,car,type,notes
stage1_ref_01.csv,<stage>,<car>,clean,fastest; reference
stage1_sloppy_01.csv,<stage>,<car>,sloppy,braked early ~2600 m; lifted mid-corner ~3100 m
```

Then copy `dat\real\` into the team repo (`dat/real/`) and push.

## Known quirks to watch

- Wheel order may be RL, RR, FL, FR (current guess). check_signs.py tests left vs right only.
- rpm may be stored /10 (auto-detected by check_signs.py).
- Gear 10 (or -1) may be reverse; those rows are dropped.
- If the logger says "Could not bind", another logger (or SimHub / another telemetry app)
  is already using port 20777. Close it.
