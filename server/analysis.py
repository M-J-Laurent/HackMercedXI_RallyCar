"""The analysis layer: our anomaly model + rules + stage map, behind two functions the server calls.

Loads the trained model once (joblib, about 14 MB) and keeps the reference lap in memory.
"""
import csv
import io
import json
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from anomalyDetection.schema import DIST, TELEMETRY_COLUMNS  # noqa: E402
from anomalyDetection.stage_map import load_stage_map  # noqa: E402
from anomalyDetection.training.infer import detect  # noqa: E402

REAL = REPO / "dat" / "real"
MODEL_PATH = REPO / "models" / "anomaly_model.joblib"
MAP_PATH = REAL / "stage1_map.json"


class ModelMissing(RuntimeError):
    pass


class BadLap(ValueError):
    pass


class Analyzer:
    def __init__(self):
        self._lock = threading.Lock()
        self._bundle = None
        self._ref_cache: dict[str, pd.DataFrame] = {}
        self.stage_map = load_stage_map(MAP_PATH) if MAP_PATH.exists() else None

    # ---- what laps exist
    def laps(self) -> list[dict]:
        path = REAL / "runs.csv"
        if not path.exists():
            return []
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("file")]
        return [{"file": r["file"], "stage": r.get("stage", ""), "car": r.get("car", ""), "type": r.get("type", ""),
                 "notes": r.get("notes", ""), "is_reference": "REFERENCE" in r.get("type", "").upper()}
                for r in rows if (REAL / r["file"]).exists()]

    def default_reference(self) -> str | None:
        laps = self.laps()
        return next((l["file"] for l in laps if l["is_reference"]), laps[0]["file"] if laps else None)

    def stage_name(self) -> str:
        return (self.stage_map or {}).get("name", "")

    # ---- model
    @property
    def model_ready(self) -> bool:
        return MODEL_PATH.exists()

    def _model(self):
        with self._lock:
            if self._bundle is None:
                if not MODEL_PATH.exists():
                    raise ModelMissing(
                        "The anomaly model has not been trained yet. From the src folder run:\n"
                        "  python -m anomalyDetection.generation.build_real_dataset --runs 400\n"
                        "  python -m anomalyDetection.training.train --data ../dat/real_aug --split-by run_id")
                self._bundle = joblib.load(MODEL_PATH)
            return self._bundle

    def _reference(self, name: str) -> pd.DataFrame:
        if name not in self._ref_cache:
            path = (REAL / name).resolve()
            if REAL.resolve() not in path.parents or not path.exists():
                raise BadLap(f"unknown reference lap {name!r}")
            self._ref_cache[name] = pd.read_csv(path)
        return self._ref_cache[name]

    # ---- analysis
    @staticmethod
    def check(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in TELEMETRY_COLUMNS if c not in df.columns]
        if missing:
            raise BadLap(f"missing columns: {', '.join(missing)}")
        if len(df) < 100:
            raise BadLap("that lap has too few rows")
        if not (np.diff(df[DIST].to_numpy()) > 0).all():
            raise BadLap("lap_distance must be strictly increasing (is this a single, cleaned lap?)")
        return df

    def analyze_csv(self, text: str, ref: str | None = None) -> dict:
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:  # noqa: BLE001
            raise BadLap(f"could not read that as a CSV: {e}") from e
        return self.analyze(self.check(df), ref)

    def analyze_named(self, lap: str, ref: str | None = None) -> dict:
        path = (REAL / lap).resolve()
        if REAL.resolve() not in path.parents or not path.exists():
            raise BadLap(f"unknown lap {lap!r}")
        return self.analyze(self.check(pd.read_csv(path)), ref)

    def analyze(self, df: pd.DataFrame, ref: str | None) -> dict:
        ref = ref or self.default_reference()
        if not ref:
            raise BadLap("no reference lap found in dat/real/runs.csv")
        segments = detect(df, self._reference(ref), self._model(), stage_map=self.stage_map)
        return {"reference": ref, "stage": self.stage_name(), "segments": segments}


analyzer = Analyzer()
