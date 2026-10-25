# Rally Coach website

A static site (plain HTML/CSS/JS, no build step, no dependencies) that shows a lap against the reference lap.

```
web/
  index.html      page layout: Lap analysis + How it works
  style.css       styles (light + dark mode)
  app.js          map, charts, coaching tips, uploads
  build_data.py   packs dat/real/ into data/bundle.js
  data/
    bundle.js     generated: every lap resampled to a 5 m grid + stage map + mistake segments
    analysis/     optional: <lap>.json from infer.py (+ LLM tips), picked up by build_data.py
```

## Update the data

```
python web/build_data.py            # after recording / converting new laps
python web/build_data.py --infer    # also run Jay's model on every lap (needs models/)
```

Mistake segments per lap come from, in order:
1. `web/data/analysis/<lap>.json` (model output, optionally with LLM tips)
2. `--infer` (runs `infer.py` and saves to 1)
3. `dat/real/annotations.json` (driver's notes; the site labels these as such)

## View it locally

**With the model and AI coach (recommended):** `python server/app.py --open` (see `server/README.md`). Every lap you open is
analysed live, and an on-device LLM can write the tips; the *AI coach* card downloads that LLM with one click.

**Static only:** open `web/index.html` in a browser. It works straight from disk because the data is a `<script>`, not a fetch.
`backend.js` notices there is no server and stays out of the way.

## Host it (GitHub Pages, free)

1. Commit `web/` including `web/data/bundle.js`.
2. GitHub repo → Settings → Pages → Source: *Deploy from a branch* → branch `main`, folder `/ (root)`.
3. The site is at `https://<user>.github.io/HackMercedXI_RallyCar/web/`.

(Netlify / Vercel also work: drag the `web/` folder in, no build command.)

## For the LLM (Teammate B): how tips get onto the page

Save `infer.py`'s output to `web/data/analysis/<lap>.json` and add a `"tip"` string to each segment:

```json
[
  {"start_m": 3380, "end_m": 3460, "suggested_type": "oversteer", "confidence": 0.91,
   "evidence": {"time_lost_s": 7.9}, "corner": {"number": 23, "direction": "left", "kind": "hairpin"},
   "tip": "The rear stepped out at corner 23. Brake a little earlier and be gentler on the throttle on exit."}
]
```

Then run `python web/build_data.py`. Any segment without `"tip"` gets an auto-written one from the telemetry.
Visitors can also load a lap CSV and an analysis JSON from the page itself (top right).

## What the site shows

- **Summary:** stage time, reference time, gap, number of mistakes, biggest single loss
- **Stage map:** drawn from x/z, colored by time gained/lost or by speed, corners numbered, mistakes highlighted
- **Coaching list:** one card per mistake: type, corner, distance range, time lost, tip, source/confidence; sort by order or by loss
- **Linked charts vs reference:** time gap, speed, throttle, brake, steering, slide angle; mistakes shaded;
  hover shows every value at that metre on all charts and the map; drag to zoom; click a tip to zoom to it
- **How it works:** pipeline, mistake glossary, what data is recorded, team
