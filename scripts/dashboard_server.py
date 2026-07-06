#!/usr/bin/env python3
"""
Serve the benchmark reports as a modern web dashboard over the LAN.

The Markdown reports written by ollama_bench.py / foundry_bench.py are great for
version control but awkward to read and compare. This server parses them and
serves:

  - a single-page dashboard (dashboard/index.html + app.js + styles.css)
  - a small JSON API the page consumes:
      GET /api/runs             -> every report (metadata only), newest first
      GET /api/run?path=<rel>   -> one report fully parsed + its raw markdown
      GET /api/compare          -> latest report per machine (for cross-machine view)

Bind to 0.0.0.0 (the default) and open http://<this-machine-ip>:<port> from any
device on the LAN. Standard library only, so it runs anywhere Python 3.8+ does
with no install step.

Security: the process is reachable from the LAN, so path parameters are resolved
and confined to the reports directory; only *.md report files are ever read.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as bc  # noqa: E402
import compare_latest_reports as clr  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = bc.repo_root()
_DASHBOARD_DIR = os.path.join(_REPO_ROOT, "dashboard")

_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


# ---------------------------------------------------------------------------
# Report parsing (reuses the summary/metadata parser from the compare script)
# ---------------------------------------------------------------------------


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _parse_kv_table(lines: List[str], header: str) -> List[Dict[str, str]]:
    """Parse a two-column Markdown table under a `## <header>` section."""
    try:
        start = lines.index(header)
    except ValueError:
        return []
    rows: List[Dict[str, str]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if not stripped.startswith("|"):
            continue
        cells = clr._split_md_row(stripped)
        if len(cells) < 2:
            continue
        key = cells[0].strip()
        if not key or key.lower() in {"component", "field"}:
            continue
        if set(key) <= {"-", ":"}:
            continue
        rows.append({"key": key, "value": cells[1].strip()})
    return rows


def _parse_bullet(lines: List[str], label: str) -> Optional[str]:
    prefix = f"- {label}:"
    for line in lines:
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip().replace("`", "")
    return None


def parse_report_file(path: str) -> Dict[str, Any]:
    lines = _read_lines(path)
    models: Dict[str, Any] = {}
    try:
        summary = clr._parse_summary(lines)
        for name, ms in summary.items():
            row = dataclasses.asdict(ms)
            row["model"] = name
            models[name] = row
    except Exception:
        models = {}

    rel = os.path.relpath(path, _REPO_ROOT).replace(os.sep, "/")
    return {
        "path": rel,
        "machine_label": clr._metadata_value(lines, "Machine label"),
        "engine": clr._metadata_value(lines, "Engine"),
        "platform": clr._metadata_value(lines, "Platform"),
        "started": clr._metadata_value(lines, "Started"),
        "runs_per_model": clr._metadata_value(lines, "Runs per model"),
        "options": clr._metadata_value(lines, "Options"),
        "hardware": _parse_kv_table(lines, "## Hardware"),
        "peak_cpu": _parse_bullet(lines, "Peak observed CPU load"),
        "peak_ram": _parse_bullet(lines, "Peak observed RAM used"),
        "models": list(models.values()),
        "raw": "\n".join(lines),
    }


def _report_meta(path: str) -> Dict[str, Any]:
    full = parse_report_file(path)
    full.pop("raw", None)
    full.pop("hardware", None)
    full["models"] = [m.get("model") for m in full.get("models", [])]
    return full


def _discover_reports(reports_dir: str) -> List[str]:
    paths: List[str] = []
    if not os.path.isdir(reports_dir):
        return paths
    for name in sorted(os.listdir(reports_dir)):
        machine_dir = os.path.join(reports_dir, name)
        if not os.path.isdir(machine_dir) or name == "comparisons":
            continue
        for filename in os.listdir(machine_dir):
            if clr._REPORT_RE.match(filename):
                paths.append(os.path.join(machine_dir, filename))
    return paths


def _run_sort_key(path: str) -> str:
    match = clr._REPORT_RE.match(os.path.basename(path))
    return match.group(2) if match else ""


def _safe_report_path(reports_dir: str, rel_path: str) -> Optional[str]:
    """Resolve a client-supplied path and confine it to the reports dir."""
    candidate = os.path.normpath(os.path.join(_REPO_ROOT, rel_path))
    reports_real = os.path.realpath(reports_dir)
    candidate_real = os.path.realpath(candidate)
    if not candidate_real.startswith(reports_real + os.sep):
        return None
    if not candidate_real.endswith(".md") or not os.path.isfile(candidate_real):
        return None
    return candidate_real


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "BenchLabDashboard/1.0"
    reports_dir: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        bc.log(f"{self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)

        if route in _STATIC_FILES:
            return self._serve_static(*_STATIC_FILES[route])
        if route == "/api/runs":
            return self._api_runs()
        if route == "/api/run":
            return self._api_run(query)
        if route == "/api/compare":
            return self._api_compare()
        if route == "/api/status":
            return self._api_status()
        return self._send_error_json(404, f"Not found: {route}")

    def _serve_static(self, filename: str, content_type: str) -> None:
        file_path = os.path.join(_DASHBOARD_DIR, filename)
        if not os.path.isfile(file_path):
            return self._send_error_json(404, f"Missing asset: {filename}")
        with open(file_path, "rb") as f:
            self._send(200, f.read(), content_type)

    def _api_runs(self) -> None:
        paths = sorted(_discover_reports(self.reports_dir), key=_run_sort_key, reverse=True)
        runs = [_report_meta(p) for p in paths]
        # Flag the newest run per machine so the UI can badge it.
        seen: set = set()
        for run in runs:
            label = run.get("machine_label")
            run["is_latest"] = label not in seen
            seen.add(label)
        self._send_json({"runs": runs, "generated": bc.iso_now_local()})

    def _api_run(self, query: Dict[str, List[str]]) -> None:
        rel = (query.get("path") or [""])[0]
        if not rel:
            return self._send_error_json(400, "Missing 'path' parameter")
        safe = _safe_report_path(self.reports_dir, rel)
        if safe is None:
            return self._send_error_json(403, "Invalid report path")
        self._send_json(parse_report_file(safe))

    def _api_compare(self) -> None:
        by_machine: Dict[str, str] = {}
        for path in sorted(_discover_reports(self.reports_dir), key=_run_sort_key):
            meta = _report_meta(path)
            by_machine[meta["machine_label"]] = path  # later (newer) wins
        reports = [parse_report_file(p) for p in by_machine.values()]
        for r in reports:
            r.pop("raw", None)
        self._send_json({"reports": reports})

    def _api_status(self) -> None:
        """Report whether a benchmark is currently running (heartbeat file)."""
        path = os.path.join(self.reports_dir, bc.BENCH_STATUS_FILE)
        if not os.path.isfile(path):
            return self._send_json({"running": False})
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return self._send_json({"running": False})
        data["running"] = bool(data.get("running"))
        self._send_json(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _lan_ip() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    bc.bootstrap_venv_if_needed(__file__, argv)

    p = argparse.ArgumentParser(description="Serve the benchmark reports as a LAN dashboard.")
    p.add_argument("--no-venv", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0, i.e. all interfaces / LAN)")
    p.add_argument("--port", type=int, default=8680, help="Port (default: 8680)")
    p.add_argument(
        "--reports-dir",
        default=os.path.join(_REPO_ROOT, "reports"),
        help="Reports root directory (default: repo reports/)",
    )
    args = p.parse_args(argv)

    reports_dir = os.path.abspath(args.reports_dir)
    DashboardHandler.reports_dir = reports_dir

    httpd = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    count = len(_discover_reports(reports_dir))
    bc.log(f"Serving {count} report(s) from {reports_dir}")
    bc.log("Dashboard URLs:")
    bc.log(f"  local:  http://localhost:{args.port}")
    lan = _lan_ip()
    if lan and args.host in {"0.0.0.0", "::", ""}:
        bc.log(f"  LAN:    http://{lan}:{args.port}   <- open this on other devices")
    bc.log("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        bc.log("Shutting down.")
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
