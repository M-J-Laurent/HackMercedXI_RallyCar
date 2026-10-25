"""Rally Coach local server: the website, the anomaly model and the LLM in one process.

    python server/app.py            # then open http://127.0.0.1:8000
    python server/app.py --open     # also opens the browser

    browser  <->  this server  <->  anomaly model + rules + stage map  (analysis.py)
                               <->  Ollama (local LLM, downloaded on demand)  (llm.py)

Only talks to 127.0.0.1. Standard library only, plus what the model needs (numpy, pandas, lightgbm).
"""
import argparse
import json
import sys
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm  # noqa: E402
from analysis import BadLap, ModelMissing, analyzer  # noqa: E402

WEB = Path(__file__).resolve().parents[1] / "web"
MAX_BODY = 200 * 1024 * 1024  # a lap CSV with every column is about 10 MB


class Handler(SimpleHTTPRequestHandler):
    server_version = "RallyCoach/1.0"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    # ---------------------------------------------------------------- helpers
    def log_message(self, fmt, *args):
        if self.path.startswith("/api/") and "pull/status" not in self.path and "llm/status" not in self.path:
            sys.stderr.write("%s %s\n" % (self.command, self.path))

    def _json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message, status=HTTPStatus.BAD_REQUEST):
        self._json({"error": message}, status)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise BadLap("that file is too large")
        return self.rfile.read(n) if n else b""

    def _same_origin(self) -> bool:
        """Stops other websites from making your browser POST to this local server."""
        origin = self.headers.get("Origin")
        return origin is None or origin == f"http://{self.headers.get('Host')}"

    # ---------------------------------------------------------------- routing
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        if url.path == "/api/health":
            return self._json({"ok": True, "model_ready": analyzer.model_ready, "stage": analyzer.stage_name(),
                               "reference": analyzer.default_reference()})
        if url.path == "/api/laps":
            return self._json({"laps": analyzer.laps(), "reference": analyzer.default_reference()})
        if url.path == "/api/llm/status":
            return self._json(llm.status((q.get("model") or [None])[0]))
        if url.path == "/api/llm/pull/status":
            return self._json(llm.downloader.snapshot())
        if url.path.startswith("/api/"):
            return self._error("not found", HTTPStatus.NOT_FOUND)
        return super().do_GET()

    def do_POST(self):
        if not self._same_origin():
            return self._error("cross-origin request refused", HTTPStatus.FORBIDDEN)
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        try:
            body = self._body()
            if url.path == "/api/analyze":
                ref = (q.get("ref") or [None])[0]
                lap = (q.get("lap") or [None])[0]
                result = analyzer.analyze_named(lap, ref) if lap else analyzer.analyze_csv(body.decode("utf-8-sig"), ref)
                return self._json(result)
            if url.path == "/api/tips":
                return self._tips(json.loads(body or b"{}"))
            if url.path == "/api/llm/start":
                return self._json({"message": llm.start_ollama(), **llm.status()})
            if url.path == "/api/llm/install":
                if llm.find_exe() or llm.running():
                    return self._error("Ollama is already installed.", HTTPStatus.CONFLICT)
                if not llm.install_supported():
                    return self._error(f"Automatic install isn't available here. Run: {llm.LINUX_INSTALL_COMMAND}",
                                       HTTPStatus.NOT_IMPLEMENTED)
                return self._json(llm.installer.start())
            if url.path == "/api/llm/install/cancel":
                llm.installer.cancel()
                return self._json(llm.installer.snapshot())
            if url.path == "/api/llm/pull":
                model = (json.loads(body or b"{}")).get("model") or llm.DEFAULT_MODEL
                if not llm.running():
                    return self._error("Ollama is not running. Start it first.", HTTPStatus.CONFLICT)
                if not llm.allowed_model(model):
                    return self._error(f"{model!r} is not an offered model", HTTPStatus.BAD_REQUEST)
                return self._json(llm.downloader.start(model))
            if url.path == "/api/llm/cancel":
                llm.downloader.cancel()
                return self._json(llm.downloader.snapshot())
        except ModelMissing as e:
            return self._error(str(e), HTTPStatus.SERVICE_UNAVAILABLE)
        except (BadLap, json.JSONDecodeError, UnicodeDecodeError) as e:
            return self._error(str(e))
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"error in {url.path}: {e!r}\n")
            return self._error(f"{type(e).__name__}: {e}", HTTPStatus.INTERNAL_SERVER_ERROR)
        return self._error("not found", HTTPStatus.NOT_FOUND)

    def _tips(self, req: dict):
        """Stream one JSON line per segment as its tip is written (newline-delimited JSON)."""
        segments = req.get("segments")
        model = req.get("model") or llm.DEFAULT_MODEL
        if not isinstance(segments, list):
            return self._error("expected {segments: [...]}")
        if not llm.running() or not llm.has_model(model):
            return self._error(f"The AI coach ({model}) is not ready.", HTTPStatus.CONFLICT)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for i, tip, err in llm.stream_tips(segments, model, req.get("stage") or analyzer.stage_name()):
                line = {"index": i, "tip": tip, "error": err, "model": model}
                self.wfile.write((json.dumps(line) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the page was closed or navigated away

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open", action="store_true", help="open the site in the browser")
    a = p.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print(f"Rally Coach running at {url}  (Ctrl+C to stop)")
    print(f"  anomaly model: {'ready' if analyzer.model_ready else 'NOT TRAINED - see README'}")
    print(f"  AI coach (Ollama): {'running' if llm.running() else 'not running'}")
    if a.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
