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

## Using real recorded laps

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
