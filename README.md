# Rally Coach

A post-stage coach for **DiRT Rally 2.0**. It compares your lap with a reference lap of the same stage, finds the
mistakes that cost you time (late braking, slides, spins, crashes), shows them on a map and on linked telemetry
charts, and has a small AI model that runs **on your own computer** write a short tip for each one.

Built at HackMerced XI. Everything runs locally: no account, no cloud, and your laps never leave your machine.

## Quick start

You need **Python 3.10 or newer** (3.13 was used here). From this folder:

```
python -m venv .venv
.venv\Scripts\activate              # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
python server/app.py --open
```

Your browser opens at <http://127.0.0.1:8000>. Pick a lap from the **Lap** list. It is analysed straight away and
the mistakes appear as cards, on the map (numbers are corners) and as shaded bands on the charts.
Click a card to zoom in on it. Drag across any chart to zoom.

Press **Ctrl+C** in the terminal to stop. If port 8000 is busy, use `--port 8001`.

### Turn on the AI coach (optional, one time)

Without it you still get every mistake, the time it cost, and the measurements behind it. The AI coach adds a
two-sentence tip per mistake. In the **AI coach** card at the top of the page:

1. **Install Ollama**: downloads the official installer from ollama.com, checks its digital signature and installs
   it for your account (no administrator prompt). It is a large download (about 1.7 GB).
2. **Download AI coach**: downloads the language model (Qwen 2.5 3B, about 1.9 GB). You can pick a smaller or
   larger one in the list. Both downloads show progress and can be cancelled.
3. Tips then appear on each card. **Rewrite tips** asks again.

Everything stays on your computer. On Linux, installing Ollama needs administrator rights, so the card shows the
one command to run instead. If you would rather not use it at all, just ignore the card.

## Using your own laps

**On the page:** click **Lap CSV** (top right) and choose a lap file. It is compared with the reference lap and analysed.
The file needs the columns listed in [CONTEXT.md](CONTEXT.md) section 3a (`lap_distance`, `time`, `x`, `z`, `speed`,
`gear`, `rpm`, `throttle`, `brake`, `steer`, `g_lat`, `g_long` and the four `wheel_speed_*`). The optional extra
columns `slip_angle_deg`, `yaw_rate_dps`, `wheelspin_front` and `wheelspin_rear` make detection more accurate.

**Recording laps from the game:** use the tools in [src/Telementry_Collection](src/Telementry_Collection/README.md).
In short:

1. In DiRT Rally 2.0's `hardware_settings_config.xml`, set
   `<udp enabled="true" extradata="3" ip="127.0.0.1" port="20777" delay="1" />`.
2. Run `python telemetry\logger.py` (in that folder), drive a stage, press Ctrl+C.
3. Run `python telemetry\convert.py <raw file> --name stage1_mylap` to make a clean CSV.
4. Copy the CSV into **`dat/real/`** and add a line for it to `dat/real/runs.csv`.
   Mark your best lap's `type` as `clean (REFERENCE)`: every other lap is compared against it.

The stage map (corner numbers, apex speeds) comes from the clean laps:

```
cd src
python -m anomalyDetection.stage_map --clean ../dat/real/stage1_clean_01.csv ../dat/real/stage1_clean_02.csv ^
    --ref ../dat/real/stage1_clean_02.csv --flip-handedness --out ../dat/real/stage1_map.json
```

(`--flip-handedness` is needed for this telemetry's coordinate system. Use `\` instead of `^` outside Windows.)

> **One stage at a time.** The reference lap and corner map in `dat/real/` are for a single stage
> (Te Awanga Sprint Forward, Subaru Impreza 1995). To coach a different stage, record clean laps of it and rebuild
> the map and the training data (see below).

## How it works

```
DiRT Rally 2.0 --UDP--> capture & clean ---> lap CSV
                                               |
        reference lap ---------------------> compare on a shared 2 m distance grid
                                               |
                   LightGBM model + rules (stall / crash) ---> mistakes + time lost + corner
                                               |
                      small local LLM (Ollama) ---> a tip per mistake ---> website
```

- **Compare:** your lap and the reference are resampled to the same 2 m grid, so every metre lines up. The lap's
  overall pace gap is removed, so being generally slower is not counted as a mistake.
- **Detect:** a LightGBM classifier trained on real laps (with realistic mistakes edited in) flags where you went
  wrong. A rule layer adds the obvious cases: the engine stalling after a hard stop means a crash or a spin.
- **Cost:** each mistake is measured over its whole effect, from where time starts being lost until you are back
  up to pace, not just the moment it was flagged.
- **Explain:** the LLM is given measurements worked out from your telemetry (speeds, braking points, slide angle)
  and must write two sentences. Tips that invent numbers, skip the advice, or drift into setup talk are rejected and
  retried; if none is reliable the card shows the plain measurements instead. Nothing on the page is pre-written.

## What to expect (honest numbers)

- Detection was tested on real laps against the driver's own notes: it found **10 of 14** annotated mistakes and
  flagged about **0.5%** of the rest of the stage. That is a small test on one stage, so treat it as a guide.
- It misses mistakes at places where the reference lap is itself slow, and some small ones.
- The **mistake type** shown on each card (understeer, braked too late, ...) is the model's guess and is often wrong.
  The location and the time lost are far more reliable. Spins and crashes found by the rule layer are the most reliable.
- The **3B AI coach** writes grounded first sentences but its advice can be generic or occasionally off. The 7B
  option (4.7 GB) is noticeably better if your computer can run it.

## Folder map

| Folder | What is in it |
|---|---|
| `server/` | the local server: serves the site, runs the model, talks to the LLM ([server/README.md](server/README.md)) |
| `web/` | the website (plain HTML / CSS / JS, no build step) ([web/README.md](web/README.md)) |
| `src/anomalyDetection/` | detection: data generation, training, evaluation, rules, stage map ([README](src/anomalyDetection/README.md)) |
| `src/Telementry_Collection/` | capture and clean DiRT Rally 2.0 telemetry |
| `dat/real/` | recorded laps, `runs.csv` (which lap is which), the stage map, driver notes used only for scoring |
| `models/` | the trained anomaly model (`anomaly_model.joblib`, 14 MB) |
| `CONTEXT.md` | the data formats between the pieces (for contributors) |

## For contributors

Retrain the detection model (after adding laps, or for a new stage). From `src/`:

```
python -m anomalyDetection.generation.build_real_dataset --runs 400
python -m anomalyDetection.training.train --data ../dat/real_aug --split-by run_id
python -m anomalyDetection.training.evaluate_real --thresholds 0.3      # score against dat/real/annotations.json
```

The static site can also be rebuilt without the server: `python web/build_data.py --infer` writes
`web/data/bundle.js` from the model's real output. Open `web/index.html` directly, or host `web/` anywhere; in that
mode there is no live analysis and no AI coach.

## Troubleshooting

| Problem | Fix |
|---|---|
| "The anomaly model has not been trained yet" | run the three training commands above (or pull `models/anomaly_model.joblib`) |
| The AI coach card says Ollama is not running | click **Start Ollama**, or start the Ollama app |
| Tips never appear | open the AI coach card: the model may still need downloading |
| "missing columns" when uploading a lap | your CSV needs the columns from section 3a of [CONTEXT.md](CONTEXT.md) |
| "lap_distance must be strictly increasing" | the file is not one clean lap; re-run `convert.py` on a single run |
| Left turns look like right turns on the map | the telemetry axes are mirrored: add `--flip-handedness` |
| Install button fails | run the installer from <https://ollama.com/download> yourself, then click **Check again** |

## Team

- **Telemetry capture:** UDP logger, cleaning, calibration and the recorded laps
- **Detection:** data generation, model, rules and stage map
- **Coaching and interface:** the website and the AI coach
