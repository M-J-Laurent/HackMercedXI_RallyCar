# Rally Coach server

One local process that connects the three layers. The site works without it (static mode); with it, every lap is
analysed live and an on-device LLM writes the tips.

```
browser (web/)  <->  server/app.py  <->  anomaly model + rules + stage map   (analysis.py, src/anomalyDetection)
                                    <->  Ollama, a local LLM runtime           (llm.py)
```

## Run it

From the repo root, with the virtual environment active (`pip install -r requirements.txt`):

```
python server/app.py --open        # http://127.0.0.1:8000
```

No extra dependencies: the server is standard library plus numpy / pandas / lightgbm, which the model already needs.
It only listens on 127.0.0.1 and refuses cross-origin POSTs, so other websites can't drive it.

Needs `models/anomaly_model.joblib` (14 MB, committed). If it is missing, train it:

```
cd src
python -m anomalyDetection.generation.build_real_dataset --runs 400
python -m anomalyDetection.training.train --data ../dat/real_aug --split-by run_id
```

## The AI coach (LLM) and why it isn't in git

A usable model is 2 to 5 GB. GitHub rejects files over 100 MB and Git LFS's free tier is about 1 GB, so the model
is **downloaded on demand** through [Ollama](https://ollama.com/download) instead of living in the repo.
The "AI coach" card on the site walks the user through it:

| State | What the card shows |
|---|---|
| Ollama not installed | **Install Ollama** button: downloads the official installer from ollama.com, checks its digital signature, installs it for the current user (no admin prompt) and starts it, with a progress bar. On Linux (needs root) it shows the one-line command instead |
| Installed, not running | "Start Ollama" button (runs `ollama serve`) |
| Running, no model | model picker + "Download AI coach (1.9 GB)" button |
| Downloading | progress bar with MB, Cancel (a later click resumes where it stopped) |
| Ready | "Rewrite tips"; tips stream in per mistake |

Offered models (`llm.py`, `MODELS`): `qwen2.5:3b` (default, 1.9 GB), `qwen2.5:1.5b` (1.0 GB), `qwen2.5:7b` (4.7 GB).
Without an LLM the site still shows its built-in tips, so nothing breaks.

Command line equivalent of the button: `ollama pull qwen2.5:3b`.
Use a different Ollama address with `OLLAMA_HOST=http://host:port`.

## API

| Route | Purpose |
|---|---|
| `GET /api/health` | server up, model trained, default stage and reference |
| `GET /api/laps` | laps in `dat/real/runs.csv` |
| `POST /api/analyze?lap=<file>` or body = lap CSV | model + rules + corner info -> `{segments: [...]}` (the format in `CONTEXT.md` 3b) |
| `POST /api/tips` `{segments, model}` | streams `{index, tip}` lines (NDJSON) as the LLM writes each tip |
| `GET /api/llm/status` | Ollama installed / running, models, download progress |
| `POST /api/llm/start` | start Ollama |
| `POST /api/llm/pull` `{model}` / `GET /api/llm/pull/status` / `POST /api/llm/cancel` | the download button |

## Files

- `app.py` HTTP routes and static files for `web/`
- `analysis.py` loads the model once, validates a lap CSV, runs `detect()`
- `llm.py` Ollama client, downloader, and the coaching prompt (edit `SYSTEM_PROMPT` to change the tone)

## Known limits

- Tested against a stand-in for Ollama's API and a stand-in installer (nothing real was installed), so expect to check the first real
  run: install, model download, then tips. The silent-install flags (/SILENT /NORESTART) are the standard ones for Ollama's Windows
  installer but were not run against the real OllamaSetup.exe. The macOS path is untested.
- Written for one stage: the reference lap and corner map are `dat/real/runs.csv`'s reference and `stage1_map.json`.
- A 3B model on CPU takes roughly 10-20 s per tip. Tips stream in, so the page stays usable.
