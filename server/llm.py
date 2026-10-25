"""The coaching LLM: talks to a local Ollama, can download a model on demand, and writes one tip
per flagged segment.

Why Ollama and not the model file in the repo: a usable model is 2-5 GB. GitHub rejects files over
100 MB and Git LFS's free tier is about 1 GB, so the model is downloaded on the user's machine, by a
button on the site (/api/llm/pull), instead of living in git.

Standard library only.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL

# What the site offers. Sizes are the approximate download size.
MODELS = [
    {"name": "qwen2.5:3b", "label": "Qwen 2.5 · 3B", "size_gb": 1.9,
     "note": "Recommended. Good tips, runs on most laptops."},
    {"name": "qwen2.5:1.5b", "label": "Qwen 2.5 · 1.5B", "size_gb": 1.0,
     "note": "Smallest and fastest. Tips are simpler."},
    {"name": "qwen2.5:7b", "label": "Qwen 2.5 · 7B", "size_gb": 4.7,
     "note": "Best tips. Needs about 8 GB of free memory and is slow without a GPU."},
]
DEFAULT_MODEL = MODELS[0]["name"]
OLLAMA_DOWNLOAD_PAGE = "https://ollama.com/download"
# Official installers, fetched over HTTPS from ollama.com. Linux needs root, so there we only show the command.
INSTALLERS = {
    "win32": ("https://ollama.com/download/OllamaSetup.exe", "OllamaSetup.exe"),
    "darwin": ("https://ollama.com/download/Ollama-darwin.zip", "Ollama-darwin.zip"),
}
LINUX_INSTALL_COMMAND = "curl -fsSL https://ollama.com/install.sh | sh"
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,100}$")

SYSTEM_PROMPT = (
    "You are a rally driving coach reviewing one moment from a driver's stage. You get what a model "
    "flagged and the telemetry around it: the driver's trace next to the reference (ideal) lap.\n"
    "Rules:\n"
    "- Use only the numbers you are given. Never invent corners, speeds or causes you cannot see.\n"
    "- Say where it happened using the corner number or the metres.\n"
    "- Write exactly two short sentences: first what the driver did differently from the reference "
    "(brake point, throttle, steering, speed), then one concrete thing to do next time.\n"
    "- The type hint is a guess and may be wrong. If the traces disagree, trust the traces.\n"
    "- If a rule says the engine stalled or the car hit something, say that plainly.\n"
    "- Plain language for a gamer, no hardware or force-feedback advice, no bullet points, at most 45 words."
)


# ------------------------------------------------------------------ plumbing
def _open(path: str, payload: dict | None = None, timeout: float = 5.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(OLLAMA_URL + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    return urllib.request.urlopen(req, timeout=timeout)


def find_exe() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidates = []
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", "")
        candidates += [Path(base) / "Programs" / "Ollama" / "ollama.exe", Path("C:/Program Files/Ollama/ollama.exe")]
    elif sys.platform == "darwin":
        candidates += [Path("/Applications/Ollama.app/Contents/Resources/ollama"),
                       Path.home() / "Applications" / "Ollama.app" / "Contents" / "Resources" / "ollama",
                       Path("/usr/local/bin/ollama")]
    return next((str(c) for c in candidates if c.exists()), None)


def running() -> bool:
    try:
        with _open("/api/version", timeout=1.5):
            return True
    except Exception:
        return False


def installed_models() -> list[str]:
    try:
        with _open("/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def has_model(name: str) -> bool:
    have = installed_models()
    return name in have or (":" not in name and f"{name}:latest" in have)


def start_ollama() -> str:
    """Start `ollama serve` in the background. Returns a short message."""
    if running():
        return "Ollama is already running."
    exe = find_exe()
    if not exe:
        return "Ollama is not installed."
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    for _ in range(20):
        if running():
            return "Ollama started."
        time.sleep(0.5)
    return "Ollama was launched but is not answering yet. Try again in a few seconds."


def allowed_model(name: str) -> bool:
    """Only offered models (or ones already on this machine), so a web page cannot ask for anything huge."""
    if not isinstance(name, str) or not _MODEL_NAME.match(name):
        return False
    return name in {m["name"] for m in MODELS} or name in installed_models() or os.environ.get("RALLY_ALLOW_ANY_MODEL") == "1"


# ------------------------------------------------------------------ downloads
class Downloader:
    """Runs `ollama pull` in a thread and keeps progress the page can poll."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = self._blank()

    @staticmethod
    def _blank(model: str | None = None) -> dict:
        return {"model": model, "active": False, "percent": 0.0, "completed_mb": 0.0, "total_mb": 0.0,
                "status": "idle", "error": None, "done": False}

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def start(self, model: str) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return dict(self.state)
            self._cancel.clear()
            self.state = self._blank(model)
            self.state.update(active=True, status="starting")
            self._thread = threading.Thread(target=self._run, args=(model,), daemon=True)
            self._thread.start()
            return dict(self.state)

    def cancel(self):
        self._cancel.set()

    def _set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def _run(self, model: str):
        layers: dict[str, list[float]] = {}  # digest -> [total, completed]
        try:
            with _open("/api/pull", {"name": model, "stream": True}, timeout=60) as r:
                for raw in r:
                    if self._cancel.is_set():
                        self._set(active=False, status="cancelled")
                        return  # closing the connection stops the pull; Ollama keeps partial data and resumes later
                    if not raw.strip():
                        continue
                    msg = json.loads(raw)
                    if msg.get("error"):
                        raise RuntimeError(msg["error"])
                    if msg.get("total"):
                        layers[msg.get("digest", "?")] = [float(msg["total"]), float(msg.get("completed", 0))]
                    total = sum(t for t, _ in layers.values())
                    done = sum(c for _, c in layers.values())
                    self._set(status=msg.get("status", ""), total_mb=round(total / 1e6, 1),
                              completed_mb=round(done / 1e6, 1),
                              percent=round(100 * done / total, 1) if total else 0.0)
            self._set(active=False, done=True, percent=100.0, status="ready")
        except urllib.error.URLError:
            self._set(active=False, error="Could not reach Ollama. Is it running?", status="error")
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._set(active=False, error=str(e), status="error")


downloader = Downloader()


# ------------------------------------------------------------------ installing Ollama itself
def install_supported() -> bool:
    return sys.platform in INSTALLERS


def _installer_url() -> tuple[str, str]:
    return INSTALLERS[sys.platform]


def verify_signature(path: Path) -> str:
    """Windows: refuse an installer whose digital signature is not valid. Returns the signer."""
    quoted = str(path).replace("'", "''")  # PowerShell single-quote escaping
    script = (f"$s = Get-AuthenticodeSignature -LiteralPath '{quoted}'; "
              "Write-Output $s.Status; Write-Output $s.SignerCertificate.Subject")
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                         capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    if not out or out[0] != "Valid":
        raise RuntimeError(f"The downloaded installer's digital signature is not valid ({out[0] if out else 'unknown'}). "
                           "It was deleted and nothing was installed.")
    return out[1] if len(out) > 1 else ""


def _run_installer(path: Path) -> None:
    """Install for the current user only: no admin prompt, no windows."""
    if sys.platform == "win32":
        done = subprocess.run([str(path), "/SILENT", "/NORESTART", "/SUPPRESSMSGBOXES"], timeout=1200)
        if done.returncode != 0:
            raise RuntimeError(f"The Ollama installer stopped with code {done.returncode}.")
    elif sys.platform == "darwin":
        apps = Path.home() / "Applications"
        apps.mkdir(exist_ok=True)
        subprocess.run(["ditto", "-x", "-k", str(path), str(apps)], check=True, timeout=600)  # keeps the executable bits


class Installer:
    """Downloads the official Ollama installer and runs it, reporting progress the page can poll."""

    STAGES = {"downloading": "Downloading Ollama", "verifying": "Checking the download",
              "installing": "Installing Ollama", "starting": "Starting Ollama"}

    def __init__(self):
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = self._blank()

    @staticmethod
    def _blank() -> dict:
        return {"active": False, "stage": "idle", "label": "", "percent": 0.0, "completed_mb": 0.0,
                "total_mb": 0.0, "error": None, "done": False}

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def _set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def _stage(self, stage: str):
        self._set(stage=stage, label=self.STAGES.get(stage, stage))

    def start(self) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return dict(self.state)
            self._cancel.clear()
            self.state = self._blank()
            self.state.update(active=True, stage="downloading", label=self.STAGES["downloading"])
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return dict(self.state)

    def cancel(self):
        self._cancel.set()

    def _download(self, url: str, dest: Path):
        with urllib.request.urlopen(url, timeout=30) as r, open(dest, "wb") as out:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                if self._cancel.is_set():
                    raise InterruptedError
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                self._set(completed_mb=round(got / 1e6, 1), total_mb=round(total / 1e6, 1),
                          percent=round(100 * got / total, 1) if total else 0.0)

    def _run(self):
        work = Path(tempfile.gettempdir()) / "rally-coach-ollama"
        dest = None
        try:
            url, name = _installer_url()
            work.mkdir(exist_ok=True)
            dest = work / name
            self._download(url, dest)
            if sys.platform == "win32":
                self._stage("verifying")
                verify_signature(dest)
            self._stage("installing")
            _run_installer(dest)
            self._stage("starting")
            for tick in range(60):  # the Windows installer launches Ollama itself; otherwise start it
                if running():
                    break
                if tick == 10 and find_exe():
                    start_ollama()
                time.sleep(1)
            if not (find_exe() or running()):
                raise RuntimeError("Ollama was installed but could not be found. Restart this program and try again.")
            self._set(active=False, done=True, stage="done", label="Ollama is ready", percent=100.0)
        except InterruptedError:
            self._set(active=False, stage="cancelled", label="Cancelled")
        except urllib.error.URLError as e:
            self._set(active=False, stage="error", error=f"Could not download Ollama ({e.reason}). Check your internet connection.")
        except subprocess.TimeoutExpired:
            self._set(active=False, stage="error", error="The installer took too long and was stopped.")
        except Exception as e:  # noqa: BLE001 - shown to the user
            self._set(active=False, stage="error", error=str(e))
        finally:
            if dest is not None:
                dest.unlink(missing_ok=True)  # also removes a partial or rejected download


installer = Installer()


def status(model: str | None = None) -> dict:
    model = model or DEFAULT_MODEL
    up = running()
    exe = find_exe()
    have = installed_models() if up else []
    return {
        "ollama_installed": bool(exe) or up,
        "ollama_running": up,
        "install_supported": install_supported(),
        "install_command": None if install_supported() else LINUX_INSTALL_COMMAND,
        "install": installer.snapshot(),
        "download_page": OLLAMA_DOWNLOAD_PAGE,
        "models": MODELS,
        "default_model": DEFAULT_MODEL,
        "installed": have,
        "model": model,
        "model_ready": up and (model in have or f"{model}:latest" in have),
        "download": downloader.snapshot(),
    }


# ------------------------------------------------------------------ writing tips
def _fmt_table(ctx: dict, start_m: float, end_m: float) -> str:
    cols = ctx["columns"]
    ix = {c: i for i, c in enumerate(cols)}
    out = ["metre | in segment | speed km/h you/ref | throttle you/ref | brake you/ref | steer you/ref | gear you/ref"
           + (" | slip deg you/ref" if "slip_angle_deg" in ix else "")]
    for a, b in zip(ctx["driver"], ctx["reference"]):
        d = a[ix["lap_distance"]]
        row = (f"{d:.0f} | {'yes' if start_m <= d <= end_m else 'no'} | "
               f"{a[ix['speed']] * 3.6:.0f}/{b[ix['speed']] * 3.6:.0f} | "
               f"{a[ix['throttle']]:.2f}/{b[ix['throttle']]:.2f} | {a[ix['brake']]:.2f}/{b[ix['brake']]:.2f} | "
               f"{a[ix['steer']]:+.2f}/{b[ix['steer']]:+.2f} | {a[ix['gear']]:.0f}/{b[ix['gear']]:.0f}")
        if "slip_angle_deg" in ix:
            row += f" | {a[ix['slip_angle_deg']]:.0f}/{b[ix['slip_angle_deg']]:.0f}"
        out.append(row)
    return "\n".join(out)


def build_prompt(seg: dict, stage: str = "") -> str:
    ev, c = seg.get("evidence", {}), seg.get("corner") or {}
    if c.get("number"):
        where = (f"corner {c['number']} ({c['direction']} {c['kind']}, radius {c['radius_m']:.0f} m), "
                 f"from the {c['phase_at_start']} to the {c['phase_at_end']}, {c['offset_from_apex_m']:+d} m from the apex; "
                 f"the reference carries {c['ref_apex_speed_kmh']} km/h here, clean laps {c['clean_apex_speed_kmh'][0]}-{c['clean_apex_speed_kmh'][1]} km/h")
    elif c.get("on_straight"):
        where = f"on the straight after corner {c.get('after_corner')}, {c.get('to_next_corner_m')} m before corner {c.get('before_corner')}"
    else:
        where = "location not mapped"
    lines = [
        f"Stage: {stage}" if stage else "",
        f"Where: {seg['start_m']:.0f}-{seg['end_m']:.0f} m, {where}",
        f"Time lost against the reference in this segment: {ev.get('time_lost_s', 0):+.1f} s",
        f"Average speed difference (you minus reference): {ev.get('speed_delta_ms', 0) * 3.6:+.0f} km/h",
        f"Detected by: {seg.get('source', 'model')}. Type hint (may be wrong): {seg.get('suggested_type')}",
    ]
    if seg.get("rule"):
        r = seg["rule"]
        lines.append(f"Rule: the engine stalled for {r['stall_s']} s; speed went from {r['speed_before_kmh']} to "
                     f"{r['speed_min_kmh']} km/h; peak deceleration {r['peak_decel_g']} g with brake at most {r['max_brake']}")
    lines.append("Traces (driver/reference):\n" + _fmt_table(seg["context"], seg["start_m"], seg["end_m"]))
    return "\n".join(x for x in lines if x)


def write_tip(seg: dict, model: str, stage: str = "", timeout: float = 300.0) -> str:
    payload = {
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_prompt(seg, stage)}],
        "options": {"temperature": 0.2, "num_predict": 110, "num_thread": max((os.cpu_count() or 4) // 2, 2)},
    }
    with _open("/api/chat", payload, timeout=timeout) as r:
        text = json.load(r)["message"]["content"].strip()
    return re.sub(r"\s+", " ", text)[:500]


def stream_tips(segments: list[dict], model: str, stage: str = ""):
    """Yield (index, tip, error) per segment, as soon as each is written."""
    for i, seg in enumerate(segments):
        try:
            if "context" not in seg:
                raise ValueError("segment has no telemetry context")
            yield i, write_tip(seg, model, stage), None
        except Exception as e:  # noqa: BLE001 - reported per segment, the rest still run
            yield i, None, str(e)
