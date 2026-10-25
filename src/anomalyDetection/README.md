# Anomaly detection

Generation and training are separate: `generation/` makes data, `training/` learns from it.
`schema.py` is the contract between them and the telemetry-handling code.

```
schema.py                  columns, units, labels (edit with the whole team)
generation/
  track.py                 synthetic stages, or a centreline rebuilt from a recorded lap
  physics.py               friction-circle speed profile -> telemetry channels
  createTelemetrydat.py    one reference ("perfect") run for a track
  inject_faults.py         labelled mistakes (early/late braking, slip, wrong gear, ...)
  build_dataset.py         -> dat/synthetic/{references,runs}.csv + faults.json
  calibrate.py             fit mu / power to a recorded lap so the sim matches the game
training/
  features.py              run vs reference deltas on a 2 m distance grid
  train.py                 LightGBM (default) or XGBoost, split by track
  evaluate.py              held-out metrics + per-fault detection
  infer.py                 lap CSV + reference CSV -> JSON anomaly segments for the LLM
```

## Setup (one time)

From the repo root:

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run (from `src/`, venv active)

```
python -m anomalyDetection.generation.build_dataset --tracks 24 --runs-per-track 10
python -m anomalyDetection.training.train            # add --backend xgboost to compare
python -m anomalyDetection.training.evaluate
python -m anomalyDetection.training.infer --run lap.csv --ref ref.csv
```

## Training on real laps (what actually works)

The simulated data does not look like real DR2 laps; a model trained only on it flags ~90% of any real lap.
`generation/real_augment.py` builds training data from the recorded clean laps instead: it blends and pace-warps
them to cover natural lap-to-lap variation, then edits labelled mistakes into the real channels.
Features remove the slow pace offset between a lap and the reference, so being generally slower is not a mistake.

```
python -m anomalyDetection.generation.build_real_dataset --runs 400          # -> dat/real_aug/
python -m anomalyDetection.training.train --data ../dat/real_aug --split-by run_id
python -m anomalyDetection.training.evaluate_real --thresholds 0.5 0.7 0.9   # vs dat/real/annotations.json
```

Honest testing: build with `--range 0 2400` and `--range 2400 4800` into separate `--out` folders, train on each,
and score each on the half it never saw (`evaluate_real --holdout`). The in-sample accuracy printed by `train.py`
on this data is meaningless because every variant comes from the same two laps.

The real-data CSVs carry extra columns (`slip_angle_deg`, `yaw_rate_dps`, `wheelspin_front/rear`; see `schema.EXTRA_CHANNELS`).
When present they become features, and `real_augment.py` edits them when injecting oversteer, understeer, spinouts and
reversing (so a spin shows 100-180 degrees of slip, as in the real laps). On the half-stage test this halved false flags
with the same recall; `train.py --no-extras` turns them off. A lap without them still works (those features read 0).

`training/rules.py` adds a rule layer on top of the model: an engine stall (rpm under 600) after a hard stop marks
a crash or spin, and a 4 g deceleration with no braking marks a collision. These events are merged into the
model's segments in `infer.py` (`source: rule`, `rule+model` or `model`). The 600 rpm and 4 g thresholds came from
looking at the five recorded laps, so they need rechecking on new data.

`stage_map.py` builds a pseudo map of the stage from the clean laps (corner number, direction, radius, apex, and the
speed range the clean laps carried through each corner). `infer.py` attaches it to each segment as `corner`:

```
python -m anomalyDetection.stage_map --clean ../dat/real/stage1_clean_01.csv ../dat/real/stage1_clean_02.csv \
    --ref ../dat/real/stage1_clean_02.csv --flip-handedness --out ../dat/real/stage1_map.json
```

It can also be fed to the model as features (`train.py --stage-map`), but on the half-stage test that gave no gain
(same recall, slightly more false flags), so the default model does not use it; the map is for describing, not detecting.

## Using real recorded laps (calibrating the simulator)

1. Telemetry CSVs must follow `schema.py` (units and sign conventions are documented there).
2. Calibrate: `python -m anomalyDetection.generation.calibrate --recorded ../dat/real/lap.csv --surface gravel`
   If left turns come out negative, add `--flip-handedness`.
3. Make a reference for that stage with the printed `--mu/--power`:
   `python -m anomalyDetection.generation.createTelemetrydat --recorded ../dat/real/lap.csv --mu .. --power .. --out ref.csv`
4. Run `infer.py` with the lap to analyse and that reference.

## Known limits

- Physics is an approximation of DiRT Rally 2.0, not the real engine. Calibrate before trusting absolute numbers.
- Synthetic accuracy is high because faults are injected by known rules. The real test is recorded laps.
- `braked_too_late` and `wrong_gear` are the weakest classes.
