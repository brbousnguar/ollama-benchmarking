#!/usr/bin/env python3
"""
Benchmark local Ollama models via the HTTP API and write a Markdown report.

Requires:
  - Python 3.8+
  - Ollama running locally (default: http://localhost:11434)

This script uses /api/tags to discover models (unless --models is provided)
and /api/generate (stream=false) to collect server-side timing counters.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


_VENV_ENV = "OLLAMA_BENCH_IN_VENV"
_NO_VENV_ENV = "OLLAMA_BENCH_NO_VENV"
_VENV_DIR_ENV = "OLLAMA_BENCH_VENV_DIR"


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_DEFAULT_CONFIG_PATH = os.path.join(_repo_root(), "ollama-bench.json")


def _running_in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _virtualenv_path() -> Optional[str]:
    return os.environ.get("VIRTUAL_ENV") or (sys.prefix if _running_in_virtualenv() else None)


def _venv_python(venv_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _bootstrap_venv_if_needed(argv: List[str]) -> None:
    if "--no-venv" in argv:
        return
    if os.environ.get(_NO_VENV_ENV) == "1":
        return
    if os.environ.get(_VENV_ENV) == "1" or _running_in_virtualenv():
        return

    venv_dir = os.environ.get(_VENV_DIR_ENV) or os.path.join(_repo_root(), ".venv")
    python_path = _venv_python(venv_dir)
    if not os.path.exists(python_path):
        sys.stderr.write(f"Creating virtual environment: {venv_dir}\n")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

    env = os.environ.copy()
    env[_VENV_ENV] = "1"
    env["VIRTUAL_ENV"] = venv_dir
    cmd = [python_path, os.path.abspath(__file__), *argv]
    try:
        raise SystemExit(subprocess.call(cmd, env=env))
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        raise SystemExit(130)


def _iso_now_local() -> str:
    # Include offset when available; keep it readable in reports.
    return _dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def _log(message: str) -> None:
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    sys.stdout.write(f"[{ts}] {message}\n")
    sys.stdout.flush()


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}.{ms:03d}s"
    if m:
        return f"{m}m {s}.{ms:03d}s"
    return f"{s}.{ms:03d}s"


def _slugify_path_part(value: Any, fallback: str = "unknown-machine") -> str:
    raw = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw)
    slug = slug.strip(".-_")
    return slug or fallback


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config file must contain a JSON object: {path}")
    return data


def _http_json(
    url: str,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    timeout_s: float = 600.0,
) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        raise RuntimeError(f"HTTP {e.code} calling {url}: {raw[:4000]!r}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to call {url}: {e}") from e

    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Non-JSON response from {url}: {raw[:4000]!r}") from e


def _ns_to_s(ns: Optional[int]) -> Optional[float]:
    if ns is None:
        return None
    return ns / 1_000_000_000.0


def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None or d == 0:
        return None
    return n / d


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _stdev_sample(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)


def _percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    if p <= 0:
        return min(xs)
    if p >= 100:
        return max(xs)
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs_sorted[int(k)]
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return d0 + d1


def _fmt_float(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _fmt_int(x: Optional[int]) -> str:
    if x is None:
        return "-"
    return str(int(x))


def _fmt_maybe(x: Any) -> str:
    if x is None or x == "":
        return "-"
    return str(x)


def _fmt_bytes(x: Optional[float]) -> str:
    if x is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(x)
    i = 0
    while value >= 1024 and i < len(units) - 1:
        value /= 1024
        i += 1
    return f"{value:.2f} {units[i]}"


def _md_escape(s: str) -> str:
    # Minimal; mainly protect table pipes.
    return s.replace("|", "\\|")


def _run_command(args: List[str], timeout_s: float = 5.0) -> Optional[str]:
    try:
        p = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip()


def _run_streaming_command(args: List[str]) -> int:
    p = subprocess.Popen(args)
    try:
        return p.wait()
    except KeyboardInterrupt:
        try:
            p.terminate()
        except OSError:
            pass
        raise


def _get_windows_wmic_value(args: List[str], field: str) -> Optional[str]:
    if platform.system().lower() != "windows":
        return None
    out = _run_command(["wmic", *args, "get", field, "/value"])
    if not out:
        return None
    prefix = f"{field}="
    for line in out.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            return value or None
    return None


def _get_windows_wmic_list(args: List[str], field: str) -> List[str]:
    if platform.system().lower() != "windows":
        return []
    out = _run_command(["wmic", *args, "get", field, "/value"])
    if not out:
        return []
    prefix = f"{field}="
    values: List[str] = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            if value:
                values.append(value)
    return sorted(set(values))


def _get_sysctl_value(name: str) -> Optional[str]:
    out = _run_command(["sysctl", "-n", name])
    return out.strip() if out else None


def _read_first_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = f.read().strip()
        except OSError:
            continue
        if value:
            return value
    return None


def _read_linux_meminfo() -> Dict[str, float]:
    values: Dict[str, float] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        values[key] = float(parts[1]) * 1024
                    except ValueError:
                        pass
    except OSError:
        pass
    return values


def _get_linux_cpu_name() -> Optional[str]:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    out = _run_command(["lscpu"])
    if out:
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    return None


def _get_linux_vendor_model() -> Tuple[Optional[str], Optional[str]]:
    manufacturer = _read_first_existing(
        [
            "/sys/devices/virtual/dmi/id/sys_vendor",
            "/sys/class/dmi/id/sys_vendor",
        ]
    )
    model = _read_first_existing(
        [
            "/sys/devices/virtual/dmi/id/product_name",
            "/sys/class/dmi/id/product_name",
        ]
    )
    return manufacturer, model


def _get_macos_model() -> Optional[str]:
    out = _run_command(["system_profiler", "SPHardwareDataType"], timeout_s=10.0)
    if not out:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Model Name:"):
            return line.split(":", 1)[1].strip()
    return None


def _get_windows_gpu_names() -> List[str]:
    return _get_windows_wmic_list(["path", "Win32_VideoController"], "Name")


def _get_macos_gpu_names() -> List[str]:
    out = _run_command(["system_profiler", "SPDisplaysDataType"], timeout_s=10.0)
    if not out:
        return []
    names = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Chipset Model:"):
            value = line.split(":", 1)[1].strip()
            if value:
                names.append(value)
    return sorted(set(names))


def _get_macos_ram_used_bytes(total_bytes: Optional[float]) -> Optional[float]:
    if total_bytes is None:
        return None
    out = _run_command(["vm_stat"])
    if not out:
        return None
    page_size = 4096.0
    values: Dict[str, float] = {}
    for line in out.splitlines():
        if "page size of" in line:
            parts = line.replace(")", "").split()
            for i, part in enumerate(parts):
                if part == "of" and i + 1 < len(parts):
                    try:
                        page_size = float(parts[i + 1])
                    except ValueError:
                        pass
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        raw = raw.strip().rstrip(".")
        try:
            values[key.strip()] = float(raw)
        except ValueError:
            pass

    available_pages = (
        values.get("Pages free", 0.0)
        + values.get("Pages inactive", 0.0)
        + values.get("Pages speculative", 0.0)
    )
    available_bytes = available_pages * page_size
    return max(0.0, total_bytes - available_bytes)


def _get_linux_gpu_names() -> List[str]:
    out = _run_command(["lspci"], timeout_s=5.0)
    if not out:
        return []
    names = []
    for line in out.splitlines():
        lower = line.lower()
        if " vga compatible controller:" in lower or " 3d controller:" in lower or " display controller:" in lower:
            names.append(line.split(":", 2)[-1].strip())
    return sorted(set(names))


def _get_gpu_names() -> List[str]:
    system = platform.system().lower()
    nvidia = [g.get("name") for g in _get_nvidia_gpu_stats() if g.get("name")]
    if system == "windows":
        return sorted(set([*nvidia, *_get_windows_gpu_names()]))
    if system == "darwin":
        return sorted(set([*nvidia, *_get_macos_gpu_names()]))
    if system == "linux":
        return sorted(set([*nvidia, *_get_linux_gpu_names()]))
    return sorted(set(nvidia))


def _get_pc_metadata() -> Dict[str, Any]:
    system = platform.system().lower()
    ram_bytes: Optional[float] = None
    cores: Optional[int] = None
    logical: Optional[int] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    os_caption = platform.platform()
    os_version = platform.version()
    cpu = platform.processor()

    if system == "windows":
        ram_raw = _get_windows_wmic_value(["computersystem"], "TotalPhysicalMemory")
        if ram_raw:
            try:
                ram_bytes = float(ram_raw)
            except ValueError:
                ram_bytes = None
        for key, attr in [("NumberOfCores", "cores"), ("NumberOfLogicalProcessors", "logical")]:
            raw = _get_windows_wmic_value(["cpu"], key)
            if raw:
                try:
                    if attr == "cores":
                        cores = int(raw)
                    else:
                        logical = int(raw)
                except ValueError:
                    pass
        manufacturer = _get_windows_wmic_value(["computersystem"], "Manufacturer")
        model = _get_windows_wmic_value(["computersystem"], "Model")
        os_caption = _get_windows_wmic_value(["os"], "Caption") or os_caption
        os_version = _get_windows_wmic_value(["os"], "Version") or os_version
        cpu = _get_windows_wmic_value(["cpu"], "Name") or cpu
    elif system == "darwin":
        ram_raw = _get_sysctl_value("hw.memsize")
        if ram_raw:
            try:
                ram_bytes = float(ram_raw)
            except ValueError:
                pass
        cores_raw = _get_sysctl_value("hw.physicalcpu")
        logical_raw = _get_sysctl_value("hw.logicalcpu")
        try:
            cores = int(cores_raw) if cores_raw else None
            logical = int(logical_raw) if logical_raw else None
        except ValueError:
            pass
        manufacturer = "Apple"
        model = _get_macos_model()
        cpu = _get_sysctl_value("machdep.cpu.brand_string") or cpu
        os_caption = f"macOS {platform.mac_ver()[0]}".strip()
    elif system == "linux":
        meminfo = _read_linux_meminfo()
        ram_bytes = meminfo.get("MemTotal")
        cores = os.cpu_count()
        logical = os.cpu_count()
        manufacturer, model = _get_linux_vendor_model()
        cpu = _get_linux_cpu_name() or cpu

    return {
        "computer_name": platform.node() or os.environ.get("COMPUTERNAME"),
        "user_name": os.environ.get("USERNAME") or os.environ.get("USER"),
        "manufacturer": manufacturer,
        "model": model,
        "os_caption": os_caption,
        "os_version": os_version,
        "cpu": cpu,
        "cpu_cores": cores,
        "cpu_logical_processors": logical or os.cpu_count(),
        "ram_bytes": ram_bytes,
        "gpus": _get_gpu_names(),
    }


def _get_nvidia_gpu_stats() -> List[Dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    rows: List[Dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "driver_version": parts[2],
                    "vram_total_mb": float(parts[3]),
                    "vram_used_mb": float(parts[4]),
                    "gpu_util_percent": float(parts[5]),
                    "temperature_c": float(parts[6]),
                    "power_w": float(parts[7]) if parts[7].replace(".", "", 1).isdigit() else None,
                }
            )
        except ValueError:
            continue
    return rows


def _get_resource_snapshot(stage: str) -> Dict[str, Any]:
    system = platform.system().lower()
    cpu_load: Optional[float] = None
    ram_total: Optional[float] = None
    ram_used: Optional[float] = None

    if system == "windows":
        cpu_raw = _get_windows_wmic_value(["cpu"], "LoadPercentage")
        if cpu_raw:
            try:
                cpu_load = float(cpu_raw)
            except ValueError:
                cpu_load = None

        total_kb = _get_windows_wmic_value(["os"], "TotalVisibleMemorySize")
        free_kb = _get_windows_wmic_value(["os"], "FreePhysicalMemory")
        try:
            if total_kb:
                ram_total = float(total_kb) * 1024
            if free_kb and ram_total is not None:
                ram_used = ram_total - (float(free_kb) * 1024)
        except ValueError:
            ram_total = None
            ram_used = None
    elif system == "linux":
        try:
            load1 = os.getloadavg()[0]
            cpu_count = os.cpu_count() or 1
            cpu_load = min(100.0, (load1 / cpu_count) * 100.0)
        except OSError:
            cpu_load = None
        meminfo = _read_linux_meminfo()
        ram_total = meminfo.get("MemTotal")
        available = meminfo.get("MemAvailable")
        if ram_total is not None and available is not None:
            ram_used = ram_total - available
    elif system == "darwin":
        try:
            load1 = os.getloadavg()[0]
            cpu_count = os.cpu_count() or 1
            cpu_load = min(100.0, (load1 / cpu_count) * 100.0)
        except OSError:
            cpu_load = None
        ram_raw = _get_sysctl_value("hw.memsize")
        if ram_raw:
            try:
                ram_total = float(ram_raw)
            except ValueError:
                ram_total = None
        ram_used = _get_macos_ram_used_bytes(ram_total)

    return {
        "timestamp": _iso_now_local(),
        "stage": stage,
        "cpu_load_percent": cpu_load,
        "ram_used_bytes": ram_used,
        "ram_total_bytes": ram_total,
        "nvidia_gpus": _get_nvidia_gpu_stats(),
    }


def _max_or_none(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return max(vals) if vals else None


def _aggregate_gpu_resource_samples(snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_gpu: Dict[str, List[Dict[str, Any]]] = {}
    for snap in snapshots:
        for gpu in snap.get("nvidia_gpus", []):
            key = f"{gpu.get('index')}|{gpu.get('name')}"
            by_gpu.setdefault(key, []).append(gpu)

    rows: List[Dict[str, Any]] = []
    for key in sorted(by_gpu):
        items = by_gpu[key]
        first = items[0]
        rows.append(
            {
                "index": first.get("index"),
                "name": first.get("name"),
                "driver_version": first.get("driver_version"),
                "vram_total_mb": first.get("vram_total_mb"),
                "peak_vram_used_mb": _max_or_none(i.get("vram_used_mb") for i in items),
                "peak_gpu_util_percent": _max_or_none(i.get("gpu_util_percent") for i in items),
                "peak_temperature_c": _max_or_none(i.get("temperature_c") for i in items),
                "peak_power_w": _max_or_none(i.get("power_w") for i in items),
            }
        )
    return rows


def _load_config_models(config_path: str) -> Optional[List[str]]:
    if not os.path.exists(config_path):
        return None
    config = _read_json_file(config_path)
    raw_models = config.get("models")
    if raw_models is None:
        raise RuntimeError(f"Config file does not define a 'models' array: {config_path}")
    if not isinstance(raw_models, list):
        raise RuntimeError(f"Config file 'models' must be an array: {config_path}")

    models: List[str] = []
    for item in raw_models:
        if not isinstance(item, str):
            raise RuntimeError(f"Config file 'models' entries must be strings: {config_path}")
        model = item.strip()
        if model:
            models.append(model)
    return sorted(set(models)) or []


@dataclass(frozen=True)
class RunResult:
    model: str
    ok: bool
    error: Optional[str]
    wall_s: Optional[float]
    load_s: Optional[float]
    total_s: Optional[float]
    prompt_eval_s: Optional[float]
    eval_s: Optional[float]
    prompt_tokens: Optional[int]
    gen_tokens: Optional[int]
    prompt_toks_per_s: Optional[float]
    gen_toks_per_s: Optional[float]


def _is_cloud_model(model: str) -> bool:
    return "cloud" in model.lower()


def _filter_cloud_models(models: List[str]) -> Tuple[List[str], List[str]]:
    local_models = [m for m in models if not _is_cloud_model(m)]
    cloud_models = [m for m in models if _is_cloud_model(m)]
    return local_models, cloud_models


def _list_ollama_models(host: str, timeout_s: float) -> List[str]:
    tags = _http_json(f"{host}/api/tags", timeout_s=timeout_s)
    models = []
    for m in tags.get("models", []) or []:
        name = m.get("name")
        if isinstance(name, str) and name.strip():
            models.append(name.strip())
    return sorted(set(models))


def _discover_models(host: str, timeout_s: float, include_cloud: bool) -> Tuple[List[str], List[str]]:
    models = _list_ollama_models(host, timeout_s)
    skipped_cloud: List[str] = []
    if not include_cloud:
        models, skipped_cloud = _filter_cloud_models(models)
    if not models:
        if skipped_cloud:
            raise RuntimeError("Only cloud models were found from /api/tags. Re-run with --include-cloud to benchmark them.")
        raise RuntimeError("No local models found from /api/tags. Is Ollama running and has models pulled?")
    return models, skipped_cloud


def _ensure_models_available(host: str, models: List[str], timeout_s: float) -> None:
    installed = set(_list_ollama_models(host, timeout_s))
    missing = [m for m in models if m not in installed]
    if not missing:
        _log("All selected models are already available locally")
        return
    if shutil.which("ollama") is None:
        raise RuntimeError("Some models are missing locally and the 'ollama' CLI was not found in PATH.")

    for model in missing:
        _log(f"Model {model} not found locally; pulling with `ollama pull {model}`")
        t0 = time.perf_counter()
        rc = _run_streaming_command(["ollama", "pull", model])
        if rc != 0:
            raise RuntimeError(f"`ollama pull {model}` failed with exit code {rc}")
        _log(f"Completed pull for {model} in {_fmt_duration(time.perf_counter() - t0)}")


def _generate_once(
    host: str,
    model: str,
    prompt: str,
    timeout_s: float,
    options: Dict[str, Any],
    keep_alive: Optional[str],
) -> RunResult:
    url = f"{host}/api/generate"
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        body["options"] = options
    if keep_alive is not None:
        body["keep_alive"] = keep_alive

    t0 = time.perf_counter()
    try:
        resp = _http_json(url, method="POST", body=body, timeout_s=timeout_s)
        t1 = time.perf_counter()
    except Exception as e:
        t1 = time.perf_counter()
        return RunResult(
            model=model,
            ok=False,
            error=str(e),
            wall_s=t1 - t0,
            load_s=None,
            total_s=None,
            prompt_eval_s=None,
            eval_s=None,
            prompt_tokens=None,
            gen_tokens=None,
            prompt_toks_per_s=None,
            gen_toks_per_s=None,
        )

    load_s = _ns_to_s(resp.get("load_duration"))
    total_s = _ns_to_s(resp.get("total_duration"))
    prompt_eval_s = _ns_to_s(resp.get("prompt_eval_duration"))
    eval_s = _ns_to_s(resp.get("eval_duration"))
    prompt_tokens = resp.get("prompt_eval_count")
    gen_tokens = resp.get("eval_count")

    prompt_tps = _safe_div(
        float(prompt_tokens) if isinstance(prompt_tokens, int) else None,
        prompt_eval_s,
    )
    gen_tps = _safe_div(
        float(gen_tokens) if isinstance(gen_tokens, int) else None,
        eval_s,
    )

    return RunResult(
        model=model,
        ok=True,
        error=None,
        wall_s=t1 - t0,
        load_s=load_s,
        total_s=total_s,
        prompt_eval_s=prompt_eval_s,
        eval_s=eval_s,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        gen_tokens=gen_tokens if isinstance(gen_tokens, int) else None,
        prompt_toks_per_s=prompt_tps,
        gen_toks_per_s=gen_tps,
    )


def _aggregate(results: List[RunResult]) -> Dict[str, Any]:
    ok = [r for r in results if r.ok]
    errs = [r for r in results if not r.ok]

    gen_tps = [r.gen_toks_per_s for r in ok if r.gen_toks_per_s is not None]
    prompt_tps = [r.prompt_toks_per_s for r in ok if r.prompt_toks_per_s is not None]
    wall_s = [r.wall_s for r in ok if r.wall_s is not None]
    total_s = [r.total_s for r in ok if r.total_s is not None]
    eval_s = [r.eval_s for r in ok if r.eval_s is not None]

    return {
        "runs": len(results),
        "ok_runs": len(ok),
        "err_runs": len(errs),
        "gen_tps_mean": _mean(gen_tps),
        "gen_tps_stdev": _stdev_sample(gen_tps),
        "gen_tps_p50": _percentile(gen_tps, 50),
        "gen_tps_p90": _percentile(gen_tps, 90),
        "prompt_tps_mean": _mean(prompt_tps),
        "wall_s_mean": _mean(wall_s),
        "total_s_mean": _mean(total_s),
        "eval_s_mean": _mean(eval_s),
        "errors": [e.error for e in errs if e.error],
    }


def _render_report(
    *,
    started_at: str,
    host: str,
    models: List[str],
    prompt_desc: str,
    runs: int,
    warmup: int,
    timeout_s: float,
    options: Dict[str, Any],
    keep_alive: Optional[str],
    all_results: Dict[str, List[RunResult]],
    pc_metadata: Dict[str, Any],
    resource_snapshots: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Ollama Benchmark Report")
    lines.append("")
    lines.append(f"- Started: `{started_at}`")
    lines.append(f"- Host: `{host}`")
    lines.append(f"- Python: `{platform.python_version()}`")
    lines.append(f"- Python executable: `{sys.executable}`")
    lines.append(f"- Virtual env: `{_virtualenv_path() or '-'}`")
    lines.append(f"- Platform: `{platform.platform()}`")
    lines.append(f"- Models: `{', '.join(models)}`")
    lines.append(f"- Runs per model: `{runs}` (warmup: `{warmup}`)")
    lines.append(f"- Timeout (s): `{timeout_s}`")
    if keep_alive is not None:
        lines.append(f"- keep_alive: `{keep_alive}`")
    if options:
        lines.append(f"- Options: `{json.dumps(options, sort_keys=True)}`")
    lines.append(f"- Prompt: `{prompt_desc}`")
    lines.append("")

    lines.append("## PC")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Computer | `{_md_escape(_fmt_maybe(pc_metadata.get('computer_name')))}` |")
    lines.append(f"| User | `{_md_escape(_fmt_maybe(pc_metadata.get('user_name')))}` |")
    lines.append(f"| Manufacturer | `{_md_escape(_fmt_maybe(pc_metadata.get('manufacturer')))}` |")
    lines.append(f"| Model | `{_md_escape(_fmt_maybe(pc_metadata.get('model')))}` |")
    os_value = f"{_fmt_maybe(pc_metadata.get('os_caption'))} {_fmt_maybe(pc_metadata.get('os_version'))}".strip()
    lines.append(f"| OS | `{_md_escape(os_value)}` |")
    lines.append(f"| CPU | `{_md_escape(_fmt_maybe(pc_metadata.get('cpu')))}` |")
    lines.append(
        f"| CPU cores / logical processors | `{_fmt_maybe(pc_metadata.get('cpu_cores'))} / {_fmt_maybe(pc_metadata.get('cpu_logical_processors'))}` |"
    )
    lines.append(f"| RAM | `{_fmt_bytes(pc_metadata.get('ram_bytes'))}` |")
    gpu_list = ", ".join(pc_metadata.get("gpus") or []) or "-"
    lines.append(f"| GPUs | `{_md_escape(gpu_list)}` |")
    lines.append("")

    lines.append("## Observed resources")
    lines.append("")
    lines.append(
        "Resource values are point-in-time samples captured around warmup and measured runs; GPU/VRAM rows require NVIDIA `nvidia-smi`."
    )
    lines.append("")
    peak_cpu = _max_or_none(s.get("cpu_load_percent") for s in resource_snapshots)
    peak_ram = _max_or_none(s.get("ram_used_bytes") for s in resource_snapshots)
    ram_total = _max_or_none(s.get("ram_total_bytes") for s in resource_snapshots)
    lines.append(f"- Peak observed CPU load: `{_fmt_float(peak_cpu, 2)}%`")
    lines.append(f"- Peak observed RAM used: `{_fmt_bytes(peak_ram)}` / `{_fmt_bytes(ram_total)}`")
    gpu_agg = _aggregate_gpu_resource_samples(resource_snapshots)
    if gpu_agg:
        lines.append("")
        lines.append("| GPU | Driver | VRAM used peak / total (MB) | GPU util peak (%) | Temp peak (C) | Power peak (W) |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for g in gpu_agg:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_escape(f"{g.get('index')}: {g.get('name')}"),
                        _md_escape(_fmt_maybe(g.get("driver_version"))),
                        f"{_fmt_float(g.get('peak_vram_used_mb'), 0)} / {_fmt_float(g.get('vram_total_mb'), 0)}",
                        _fmt_float(g.get("peak_gpu_util_percent"), 0),
                        _fmt_float(g.get("peak_temperature_c"), 0),
                        _fmt_float(g.get("peak_power_w"), 2),
                    ]
                )
                + " |"
            )
    else:
        lines.append("- NVIDIA GPU samples: `not available`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Model | OK/Total | Gen tok/s (mean) | Gen tok/s (p50) | Gen tok/s (p90) | Gen tok/s (stdev) | Prompt tok/s (mean) | Total s (mean) | Wall s (mean) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in models:
        agg = _aggregate(all_results.get(model, []))
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(model),
                    f"{agg['ok_runs']}/{agg['runs']}",
                    _fmt_float(agg["gen_tps_mean"], 2),
                    _fmt_float(agg["gen_tps_p50"], 2),
                    _fmt_float(agg["gen_tps_p90"], 2),
                    _fmt_float(agg["gen_tps_stdev"], 2),
                    _fmt_float(agg["prompt_tps_mean"], 2),
                    _fmt_float(agg["total_s_mean"], 2),
                    _fmt_float(agg["wall_s_mean"], 2),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Details")
    lines.append("")
    for model in models:
        results = all_results.get(model, [])
        lines.append(f"### {model}")
        lines.append("")
        lines.append(
            "| Run | OK | Gen tok/s | Prompt tok/s | Gen toks | Prompt toks | Eval s | Prompt eval s | Load s | Total s | Wall s | Error |"
        )
        lines.append("|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for i, r in enumerate(results, start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(i),
                        "Y" if r.ok else "N",
                        _fmt_float(r.gen_toks_per_s, 2),
                        _fmt_float(r.prompt_toks_per_s, 2),
                        _fmt_int(r.gen_tokens),
                        _fmt_int(r.prompt_tokens),
                        _fmt_float(r.eval_s, 2),
                        _fmt_float(r.prompt_eval_s, 2),
                        _fmt_float(r.load_s, 2),
                        _fmt_float(r.total_s, 2),
                        _fmt_float(r.wall_s, 2),
                        _md_escape(r.error or ""),
                    ]
                )
                + " |"
            )
        agg = _aggregate(results)
        if agg["errors"]:
            lines.append("")
            lines.append("Errors:")
            for e in agg["errors"]:
                lines.append(f"- `{e}`")
        lines.append("")

    return "\n".join(lines)


def _parse_models_arg(models_arg: Optional[str]) -> Optional[List[str]]:
    if models_arg is None:
        return None
    raw = [m.strip() for m in models_arg.split(",")]
    models = [m for m in raw if m]
    return models or None


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    _bootstrap_venv_if_needed(argv)
    session_t0 = time.perf_counter()

    p = argparse.ArgumentParser(description="Benchmark local Ollama models and write a Markdown report.")
    p.add_argument("--no-venv", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--host", default="http://localhost:11434", help="Ollama host (default: http://localhost:11434)")
    p.add_argument("--config", default=_DEFAULT_CONFIG_PATH, help=f"Config JSON path (default: {_DEFAULT_CONFIG_PATH})")
    p.add_argument("--models", default=None, help="Comma-separated models. If omitted, auto-discover via /api/tags")
    p.add_argument("--include-cloud", action="store_true", help="Include Ollama cloud models. By default models with 'cloud' in the name are skipped.")
    p.add_argument("--runs", type=int, default=3, help="Measured runs per model (default: 3)")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per model, not included in summary (default: 1)")
    p.add_argument("--timeout-s", type=float, default=600.0, help="Per-request timeout in seconds (default: 600)")
    p.add_argument(
        "--out",
        default=None,
        help="Output Markdown path (default: reports/<machine>/ollama-bench-<timestamp>.md)",
    )
    p.add_argument("--prompt", default=None, help="Prompt text (overrides --prompt-file)")
    p.add_argument("--prompt-file", default=None, help="Path to a text file containing the prompt")
    p.add_argument("--keep-alive", default="5m", help="Ollama keep_alive value (default: 5m). Use '0' to disable.")
    p.add_argument("--num-predict", type=int, default=256, help="Generation length, options.num_predict (default: 256)")
    p.add_argument("--temperature", type=float, default=0.0, help="options.temperature (default: 0.0)")
    p.add_argument("--top-p", type=float, default=None, help="options.top_p (default: unset)")
    p.add_argument("--seed", type=int, default=42, help="options.seed (default: 42)")
    p.add_argument("--num-ctx", type=int, default=None, help="options.num_ctx (default: unset)")
    p.add_argument("--stop", default=None, help="Comma-separated stop sequences, options.stop (default: unset)")
    args = p.parse_args(argv)

    host = args.host.rstrip("/")
    started_at = _iso_now_local()
    _log(f"Starting Ollama benchmark session at {started_at}")
    _log(f"Ollama host: {host}")
    _log(f"Python executable: {sys.executable}")
    if _virtualenv_path():
        _log(f"Virtual env: {_virtualenv_path()}")
    else:
        _log("Virtual env: not active")

    _log("Collecting PC metadata and initial resource snapshot")
    pc_metadata = _get_pc_metadata()
    resource_snapshots: List[Dict[str, Any]] = [_get_resource_snapshot("start")]

    config_models = _load_config_models(args.config)
    if config_models is not None:
        _log(f"Loaded config file: {args.config}")

    models = _parse_models_arg(args.models)
    model_source = "command line"
    if models is None and config_models is not None:
        models = config_models
        model_source = f"config file ({args.config})"

    if models is None:
        _log("Discovering models from Ollama /api/tags")
        models, skipped_cloud = _discover_models(host, timeout_s=args.timeout_s, include_cloud=args.include_cloud)
        if skipped_cloud:
            _log(f"Skipping {len(skipped_cloud)} cloud model(s): {', '.join(skipped_cloud)}")
        _log(f"Discovered {len(models)} model(s): {', '.join(models)}")
    else:
        skipped_cloud = []
        if not args.include_cloud:
            models, skipped_cloud = _filter_cloud_models(models)
            if skipped_cloud:
                _log(f"Skipping {len(skipped_cloud)} cloud model(s) from {model_source}: {', '.join(skipped_cloud)}")
            if not models:
                raise SystemExit("Only cloud models were selected. Re-run with --include-cloud to benchmark them.")
        _log(f"Using {len(models)} model(s) from {model_source}: {', '.join(models)}")

    if not models:
        raise SystemExit("No models selected. Add models to the config file, pass --models, or let the script auto-discover models.")

    _ensure_models_available(host=host, models=models, timeout_s=args.timeout_s)

    if args.prompt is not None and args.prompt_file is not None:
        raise SystemExit("Provide only one of --prompt or --prompt-file")
    if args.prompt_file is not None:
        prompt = _read_text_file(args.prompt_file)
        prompt_desc = f"file:{args.prompt_file}"
    elif args.prompt is not None:
        prompt = args.prompt
        prompt_desc = "inline"
    else:
        prompt = (
            "You are a benchmarking assistant.\n"
            "Task: produce a long, deterministic output.\n"
            "Output: write the integers from 1 to 2000 separated by a single space.\n"
            "Do not add any other words.\n"
        )
        prompt_desc = "default: integers 1..2000"

    options: Dict[str, Any] = {
        "num_predict": args.num_predict,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    if args.top_p is not None:
        options["top_p"] = args.top_p
    if args.num_ctx is not None:
        options["num_ctx"] = args.num_ctx
    if args.stop is not None:
        stops = [s for s in (x.strip() for x in args.stop.split(",")) if s]
        options["stop"] = stops

    keep_alive: Optional[str]
    if args.keep_alive is None:
        keep_alive = None
    else:
        keep_alive = None if args.keep_alive.strip() == "" else args.keep_alive.strip()
        if keep_alive == "0":
            keep_alive = "0s"

    out_path = args.out
    if out_path is None:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        machine_slug = _slugify_path_part(pc_metadata.get("computer_name"))
        out_path = os.path.join("reports", machine_slug, f"ollama-bench-{ts}.md")
    out_dir = os.path.dirname(out_path) or "."
    _ensure_dir(out_dir)
    _log(f"Report path: {out_path}")
    _log(f"Runs per model: {args.runs}; warmup runs per model: {args.warmup}")
    _log(f"Generation options: {json.dumps(options, sort_keys=True)}")

    # Run benchmarks.
    all_results: Dict[str, List[RunResult]] = {m: [] for m in models}

    # Warmup (not recorded) to amortize first-token/model-load effects if desired.
    if args.warmup > 0:
        _log("Starting warmup phase")
        for model in models:
            _log(f"Starting warmup for model {model}")
            model_warmup_t0 = time.perf_counter()
            for i in range(args.warmup):
                run_t0 = time.perf_counter()
                _log(f"Warmup {i + 1}/{args.warmup} for {model}: started")
                resource_snapshots.append(_get_resource_snapshot(f"before warmup {model}"))
                warmup_result = _generate_once(
                    host=host,
                    model=model,
                    prompt=prompt,
                    timeout_s=args.timeout_s,
                    options=options,
                    keep_alive=keep_alive,
                )
                resource_snapshots.append(_get_resource_snapshot(f"after warmup {model}"))
                status = "ok" if warmup_result.ok else f"failed: {warmup_result.error}"
                _log(f"Warmup {i + 1}/{args.warmup} for {model}: {status} in {_fmt_duration(time.perf_counter() - run_t0)}")
            _log(f"Completed warmup for model {model} in {_fmt_duration(time.perf_counter() - model_warmup_t0)}")

    _log("Starting measured benchmark phase")
    for model in models:
        _log(f"Starting model {model}")
        model_t0 = time.perf_counter()
        for i in range(args.runs):
            run_t0 = time.perf_counter()
            _log(f"Run {i + 1}/{args.runs} for {model}: started")
            resource_snapshots.append(_get_resource_snapshot(f"before run {model} #{i + 1}"))
            r = _generate_once(
                host=host,
                model=model,
                prompt=prompt,
                timeout_s=args.timeout_s,
                options=options,
                keep_alive=keep_alive,
            )
            all_results[model].append(r)
            resource_snapshots.append(_get_resource_snapshot(f"after run {model} #{i + 1}"))
            if r.ok:
                _log(
                    f"Run {i + 1}/{args.runs} for {model}: ok in {_fmt_duration(time.perf_counter() - run_t0)} "
                    f"(gen tok/s: {_fmt_float(r.gen_toks_per_s, 2)}, wall: {_fmt_float(r.wall_s, 2)}s)"
                )
            else:
                _log(f"Run {i + 1}/{args.runs} for {model}: failed in {_fmt_duration(time.perf_counter() - run_t0)}: {r.error}")
        model_elapsed = time.perf_counter() - model_t0
        agg = _aggregate(all_results[model])
        _log(
            f"Completed model {model} in {_fmt_duration(model_elapsed)} "
            f"({agg['ok_runs']}/{agg['runs']} ok, mean gen tok/s: {_fmt_float(agg['gen_tps_mean'], 2)})"
        )

    resource_snapshots.append(_get_resource_snapshot("end"))

    _log("Rendering markdown report")
    report = _render_report(
        started_at=started_at,
        host=host,
        models=models,
        prompt_desc=prompt_desc,
        runs=args.runs,
        warmup=args.warmup,
        timeout_s=args.timeout_s,
        options=options,
        keep_alive=keep_alive,
        all_results=all_results,
        pc_metadata=pc_metadata,
        resource_snapshots=resource_snapshots,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    session_elapsed = time.perf_counter() - session_t0
    _log(f"Report written: {out_path}")
    _log(f"Benchmark session completed in {_fmt_duration(session_elapsed)}")
    sys.stdout.write(f"{out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
