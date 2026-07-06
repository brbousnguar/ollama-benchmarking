#!/usr/bin/env python3
"""
Shared helpers for the local LLM benchmark scripts.

This module is engine-agnostic. It is used by:
  - scripts/ollama_bench.py   (Ollama HTTP API)
  - scripts/foundry_bench.py  (Microsoft Foundry Local, OpenAI-compatible API)

It provides:
  - venv bootstrap so the scripts run in a consistent isolated interpreter
  - PC metadata + richer hardware detail (memory speed/type/bandwidth, CPU
    clock/cache, GPU dedicated/shared memory, NPU presence)
  - point-in-time resource snapshots (CPU/RAM + NVIDIA GPU via nvidia-smi)
  - a generic per-run result type and the Markdown report renderer shared by
    both engines (so reports/comparisons stay consistent across backends)

No third-party Python dependencies; richer hardware data on Windows is read via
PowerShell (Get-CimInstance) and dxdiag, both guarded so they never block a run.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


_VENV_ENV = "OLLAMA_BENCH_IN_VENV"
_NO_VENV_ENV = "OLLAMA_BENCH_NO_VENV"
_VENV_DIR_ENV = "OLLAMA_BENCH_VENV_DIR"


# ---------------------------------------------------------------------------
# Paths / venv bootstrap
# ---------------------------------------------------------------------------


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def running_in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def virtualenv_path() -> Optional[str]:
    return os.environ.get("VIRTUAL_ENV") or (sys.prefix if running_in_virtualenv() else None)


def _venv_python(venv_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def bootstrap_venv_if_needed(script_path: str, argv: List[str]) -> None:
    """Re-exec the calling script inside a repo-local .venv unless opted out.

    ``script_path`` is the absolute path of the entry-point script so the
    re-exec runs the right file (not this shared module).
    """
    if "--no-venv" in argv:
        return
    if os.environ.get(_NO_VENV_ENV) == "1":
        return
    if os.environ.get(_VENV_ENV) == "1" or running_in_virtualenv():
        return

    venv_dir = os.environ.get(_VENV_DIR_ENV) or os.path.join(repo_root(), ".venv")
    python_path = _venv_python(venv_dir)
    if not os.path.exists(python_path):
        sys.stderr.write(f"Creating virtual environment: {venv_dir}\n")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

    env = os.environ.copy()
    env[_VENV_ENV] = "1"
    env["VIRTUAL_ENV"] = venv_dir
    cmd = [python_path, os.path.abspath(script_path), *argv]
    try:
        raise SystemExit(subprocess.call(cmd, env=env))
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        raise SystemExit(130)


# ---------------------------------------------------------------------------
# Time / logging / formatting
# ---------------------------------------------------------------------------


def iso_now_local() -> str:
    return _dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def log(message: str) -> None:
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    sys.stdout.write(f"[{ts}] {message}\n")
    sys.stdout.flush()


def fmt_duration(seconds: float) -> str:
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


def fmt_float(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def fmt_int(x: Optional[int]) -> str:
    if x is None:
        return "-"
    return str(int(x))


def fmt_maybe(x: Any) -> str:
    if x is None or x == "":
        return "-"
    return str(x)


def fmt_bytes(x: Optional[float]) -> str:
    if x is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(x)
    i = 0
    while value >= 1024 and i < len(units) - 1:
        value /= 1024
        i += 1
    return f"{value:.2f} {units[i]}"


def md_escape(s: str) -> str:
    # Minimal; mainly protect table pipes.
    return s.replace("|", "\\|")


def slugify_path_part(value: Any, fallback: str = "unknown-machine") -> str:
    raw = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", raw)
    slug = slug.strip(".-_")
    return slug or fallback


# ---------------------------------------------------------------------------
# Math / stats
# ---------------------------------------------------------------------------


def ns_to_s(ns: Optional[int]) -> Optional[float]:
    if ns is None:
        return None
    return ns / 1_000_000_000.0


def safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None or d == 0:
        return None
    return n / d


def mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def stdev_sample(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)


def percentile(xs: List[float], p: float) -> Optional[float]:
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


def max_or_none(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return max(vals) if vals else None


def effective_bandwidth_gbps(model_size_bytes: Optional[float], gen_tok_s: Optional[float]) -> Optional[float]:
    """Estimate achieved memory bandwidth (GB/s) during decode.

    LLM token generation is memory-bandwidth bound: the (quantized) weights are
    streamed from memory roughly once per generated token. So the effective
    bandwidth a run achieved is approximately model_size_bytes * tokens/sec.
    Compare against the theoretical bandwidth to gauge efficiency.
    """
    if model_size_bytes is None or gen_tok_s is None:
        return None
    return (model_size_bytes * gen_tok_s) / 1_000_000_000.0


def bandwidth_utilization_pct(
    eff_bw_gbps: Optional[float], theoretical_bw_gbps: Optional[float]
) -> Optional[float]:
    """Achieved decode bandwidth as a percentage of the theoretical ceiling.

    This is the headline efficiency KPI for local LLM inference: decode is
    memory-bandwidth bound, so how close a run gets to the machine's theoretical
    bandwidth says how well the engine/quant/model combination uses the silicon.
    """
    if eff_bw_gbps is None or not theoretical_bw_gbps:
        return None
    return (eff_bw_gbps / theoretical_bw_gbps) * 100.0


def tokens_per_s_per_gb(gen_tok_s: Optional[float], model_size_bytes: Optional[float]) -> Optional[float]:
    """Decode throughput normalized by model footprint (tokens/s per GiB).

    Lets small and large models be compared on efficiency-per-byte rather than
    raw speed, which raw tok/s hides (a 4B model is always faster in absolute
    terms; this shows which model does the most work per GB resident in memory).
    """
    if gen_tok_s is None or not model_size_bytes:
        return None
    size_gib = model_size_bytes / (1024 ** 3)
    if size_gib == 0:
        return None
    return gen_tok_s / size_gib


# ---------------------------------------------------------------------------
# Files / HTTP
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config file must contain a JSON object: {path}")
    return data


def http_json(
    url: str,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    timeout_s: float = 600.0,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
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


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def run_command(args: List[str], timeout_s: float = 5.0) -> Optional[str]:
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


def run_streaming_command(args: List[str]) -> Tuple[int, str]:
    p = subprocess.Popen(args)
    try:
        rc = p.wait()
        sys.stdout.write("\n")
        sys.stdout.flush()
        return rc, ""
    except KeyboardInterrupt:
        try:
            p.terminate()
        except OSError:
            pass
        raise


def _powershell_exe() -> Optional[str]:
    return shutil.which("pwsh") or shutil.which("powershell")


def powershell_json(script: str, timeout_s: float = 15.0) -> Any:
    """Run a PowerShell snippet that emits JSON and return the parsed value.

    Returns None on any failure. Single objects and arrays are both handled.
    """
    exe = _powershell_exe()
    if exe is None:
        return None
    out = run_command(
        [exe, "-NoProfile", "-NonInteractive", "-Command", script],
        timeout_s=timeout_s,
    )
    if not out:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Platform metadata collectors (wmic / sysctl / proc)
# ---------------------------------------------------------------------------


def _get_windows_wmic_value(args: List[str], field: str) -> Optional[str]:
    if platform.system().lower() != "windows":
        return None
    out = run_command(["wmic", *args, "get", field, "/value"])
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
    out = run_command(["wmic", *args, "get", field, "/value"])
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


def _cim_instances(class_name: str, properties: List[str], timeout_s: float = 15.0) -> List[Dict[str, Any]]:
    """Return Win32_* CIM instances as dicts (modern replacement for wmic).

    Windows 11 no longer ships wmic by default, so CIM via PowerShell is the
    primary path; the wmic helpers remain as a fallback for older systems.
    """
    if platform.system().lower() != "windows":
        return []
    props = ",".join(properties)
    data = powershell_json(
        f"Get-CimInstance {class_name} | Select-Object {props} | ConvertTo-Json -Compress",
        timeout_s=timeout_s,
    )
    return [d for d in _as_list(data) if isinstance(d, dict)]


def _cim_first_value(class_name: str, prop: str) -> Optional[str]:
    rows = _cim_instances(class_name, [prop])
    if not rows:
        return None
    value = rows[0].get(prop)
    if value is None or value == "":
        return None
    return str(value)


def _get_sysctl_value(name: str) -> Optional[str]:
    out = run_command(["sysctl", "-n", name])
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
    out = run_command(["lscpu"])
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
    out = run_command(["system_profiler", "SPHardwareDataType"], timeout_s=10.0)
    if not out:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Model Name:"):
            return line.split(":", 1)[1].strip()
    return None


def _get_windows_gpu_names() -> List[str]:
    rows = _cim_instances("Win32_VideoController", ["Name"])
    names = [str(r.get("Name")).strip() for r in rows if r.get("Name")]
    if names:
        return sorted(set(names))
    return _get_windows_wmic_list(["path", "Win32_VideoController"], "Name")


def _get_macos_gpu_names() -> List[str]:
    out = run_command(["system_profiler", "SPDisplaysDataType"], timeout_s=10.0)
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
    out = run_command(["vm_stat"])
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
    out = run_command(["lspci"], timeout_s=5.0)
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
    nvidia = [g.get("name") for g in get_nvidia_gpu_stats() if g.get("name")]
    if system == "windows":
        return sorted(set([*nvidia, *_get_windows_gpu_names()]))
    if system == "darwin":
        return sorted(set([*nvidia, *_get_macos_gpu_names()]))
    if system == "linux":
        return sorted(set([*nvidia, *_get_linux_gpu_names()]))
    return sorted(set(nvidia))


def get_pc_metadata() -> Dict[str, Any]:
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
        cs = _cim_instances("Win32_ComputerSystem", ["TotalPhysicalMemory", "Manufacturer", "Model"])
        cs = cs[0] if cs else {}
        proc = _cim_instances("Win32_Processor", ["Name", "NumberOfCores", "NumberOfLogicalProcessors"])
        proc = proc[0] if proc else {}
        osi = _cim_instances("Win32_OperatingSystem", ["Caption", "Version"])
        osi = osi[0] if osi else {}

        ram_raw = cs.get("TotalPhysicalMemory") or _get_windows_wmic_value(["computersystem"], "TotalPhysicalMemory")
        if ram_raw:
            try:
                ram_bytes = float(ram_raw)
            except (TypeError, ValueError):
                ram_bytes = None
        cores_raw = proc.get("NumberOfCores") or _get_windows_wmic_value(["cpu"], "NumberOfCores")
        logical_raw = proc.get("NumberOfLogicalProcessors") or _get_windows_wmic_value(["cpu"], "NumberOfLogicalProcessors")
        try:
            cores = int(cores_raw) if cores_raw else None
        except (TypeError, ValueError):
            cores = None
        try:
            logical = int(logical_raw) if logical_raw else None
        except (TypeError, ValueError):
            logical = None
        manufacturer = cs.get("Manufacturer") or _get_windows_wmic_value(["computersystem"], "Manufacturer")
        model = cs.get("Model") or _get_windows_wmic_value(["computersystem"], "Model")
        os_caption = osi.get("Caption") or _get_windows_wmic_value(["os"], "Caption") or os_caption
        os_version = osi.get("Version") or _get_windows_wmic_value(["os"], "Version") or os_version
        cpu = proc.get("Name") or _get_windows_wmic_value(["cpu"], "Name") or cpu
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


# ---------------------------------------------------------------------------
# Richer hardware detail (memory bandwidth, CPU clock/cache, GPU mem, NPU)
# ---------------------------------------------------------------------------


# SMBIOS memory type codes (subset commonly seen on laptops/desktops).
_SMBIOS_MEMORY_TYPES = {
    20: "DDR",
    21: "DDR2",
    24: "DDR3",
    26: "DDR4",
    27: "LPDDR",
    28: "LPDDR2",
    29: "LPDDR3",
    30: "LPDDR4",
    34: "LPDDR5",
    35: "DDR5",
}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get_windows_memory_info() -> Optional[Dict[str, Any]]:
    modules = powershell_json(
        "Get-CimInstance Win32_PhysicalMemory | "
        "Select-Object Capacity,Speed,ConfiguredClockSpeed,SMBIOSMemoryType,DataWidth,Manufacturer | "
        "ConvertTo-Json -Compress"
    )
    modules = _as_list(modules)
    if not modules:
        return None

    speeds: List[float] = []
    widths: List[float] = []
    total_capacity = 0.0
    type_label = "unknown"
    for m in modules:
        if not isinstance(m, dict):
            continue
        cap = m.get("Capacity")
        if cap:
            try:
                total_capacity += float(cap)
            except (TypeError, ValueError):
                pass
        speed = m.get("ConfiguredClockSpeed") or m.get("Speed")
        if speed:
            try:
                speeds.append(float(speed))
            except (TypeError, ValueError):
                pass
        width = m.get("DataWidth")
        if width:
            try:
                widths.append(float(width))
            except (TypeError, ValueError):
                pass
        code = m.get("SMBIOSMemoryType")
        if isinstance(code, (int, float)) and int(code) in _SMBIOS_MEMORY_TYPES:
            type_label = _SMBIOS_MEMORY_TYPES[int(code)]

    speed_mts = min(speeds) if speeds else None
    total_bus_bits = sum(widths) if widths else None
    theoretical_bw = None
    if speed_mts and total_bus_bits:
        # bytes-per-transfer * transfers-per-second -> GB/s
        theoretical_bw = (total_bus_bits / 8.0) * speed_mts / 1000.0

    return {
        "type": type_label,
        "speed_mts": speed_mts,
        "modules": len(modules),
        "channels": len([w for w in widths if w]) or len(modules),
        "total_capacity_bytes": total_capacity or None,
        "theoretical_bandwidth_gbps": theoretical_bw,
    }


def _get_windows_cpu_extra() -> Optional[Dict[str, Any]]:
    info = powershell_json(
        "Get-CimInstance Win32_Processor | "
        "Select-Object MaxClockSpeed,L2CacheSize,L3CacheSize | "
        "ConvertTo-Json -Compress"
    )
    info = _as_list(info)
    if not info or not isinstance(info[0], dict):
        return None
    first = info[0]
    return {
        "max_clock_mhz": first.get("MaxClockSpeed"),
        "l2_cache_kb": first.get("L2CacheSize"),
        "l3_cache_kb": first.get("L3CacheSize"),
    }


def _parse_dxdiag_memory(value: Any) -> Optional[float]:
    """Parse a dxdiag memory string like '2048 MB' / '16384 MB' to MB float."""
    if value is None:
        return None
    text = str(value).strip()
    m = re.match(r"([\d,.]+)\s*(MB|GB)?", text, re.IGNORECASE)
    if not m:
        return None
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = (m.group(2) or "MB").upper()
    return num * 1024.0 if unit == "GB" else num


def _get_windows_gpu_detail() -> List[Dict[str, Any]]:
    """True dedicated/shared GPU memory via dxdiag XML (Win32 AdapterRAM is capped)."""
    if platform.system().lower() != "windows":
        return []
    exe = _powershell_exe()
    tmp_path = os.path.join(tempfile.gettempdir(), f"dxdiag-bench-{os.getpid()}.xml")
    try:
        # dxdiag /x writes the XML and exits; can take a few seconds.
        subprocess.run(
            ["dxdiag", "/x", tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60.0,
            check=False,
        )
    except Exception:
        return []

    if not os.path.exists(tmp_path):
        return []
    try:
        tree = ET.parse(tmp_path)
    except Exception:
        return []
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    devices: List[Dict[str, Any]] = []
    seen: set = set()
    for dd in tree.getroot().iter("DisplayDevice"):
        name = dd.findtext("CardName")
        if not name:
            continue
        name = name.strip()
        if name in seen:
            continue
        seen.add(name)
        devices.append(
            {
                "name": name,
                "dedicated_mb": _parse_dxdiag_memory(dd.findtext("DedicatedMemory")),
                "shared_mb": _parse_dxdiag_memory(dd.findtext("SharedMemory")),
                "driver_version": (dd.findtext("DriverVersion") or "").strip() or None,
            }
        )
    return devices


def _get_windows_npu() -> Optional[Dict[str, Any]]:
    devices = powershell_json(
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.Name -match 'NPU|AI Boost|Neural|XDNA' } | "
        "Select-Object -ExpandProperty Name | ConvertTo-Json -Compress"
    )
    names = [n for n in _as_list(devices) if isinstance(n, str) and n.strip()]
    if not names:
        return {"present": False, "name": None}
    return {"present": True, "name": sorted(set(names))[0]}


# Nominal unified-memory bandwidth (GB/s) per Apple Silicon variant. Apple does
# not expose this via sysctl, but decode is bandwidth-bound so it is the key
# denominator for the bandwidth-utilization KPI. Values are the published
# figures per chip tier; match the most specific (Ultra/Max/Pro) label first.
_APPLE_SILICON_BANDWIDTH_GBPS = [
    ("m1 ultra", 800.0),
    ("m1 max", 400.0),
    ("m1 pro", 200.0),
    ("m1", 68.25),
    ("m2 ultra", 800.0),
    ("m2 max", 400.0),
    ("m2 pro", 200.0),
    ("m2", 100.0),
    ("m3 ultra", 800.0),
    ("m3 max", 400.0),
    ("m3 pro", 150.0),
    ("m3", 100.0),
    ("m4 max", 546.0),
    ("m4 pro", 273.0),
    ("m4", 120.0),
]


def _apple_silicon_bandwidth_gbps(brand_string: Optional[str]) -> Optional[float]:
    if not brand_string:
        return None
    text = brand_string.lower()
    # Longest / most specific labels first so "m4 pro" wins over "m4".
    for label, gbps in sorted(_APPLE_SILICON_BANDWIDTH_GBPS, key=lambda kv: -len(kv[0])):
        if label in text:
            return gbps
    return None


def _get_macos_memory_type() -> Optional[Dict[str, Any]]:
    out = run_command(["system_profiler", "SPMemoryDataType"], timeout_s=10.0)
    mem_type = None
    speed = None
    if out:
        for line in out.splitlines():
            stripped = line.strip()
            if stripped.startswith("Type:") and mem_type is None:
                mem_type = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Speed:") and speed is None:
                speed = stripped.split(":", 1)[1].strip()
    # Apple Silicon reports unified memory; type/speed often absent. Bandwidth is
    # looked up from the chip name so the utilization KPI works on Macs too.
    brand = _get_sysctl_value("machdep.cpu.brand_string")
    theoretical_bw = _apple_silicon_bandwidth_gbps(brand)
    return {
        "type": mem_type or "unified",
        "speed_label": speed,
        "unified": True,
        "theoretical_bandwidth_gbps": theoretical_bw,
    }


def get_hardware_detail(pc_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Collect richer, best-effort hardware detail for the report.

    Windows: deep (memory bandwidth, CPU clock/cache, GPU dedicated/shared, NPU).
    macOS/Linux: best-effort. Every probe is guarded so a failure yields None.
    """
    system = platform.system().lower()
    detail: Dict[str, Any] = {
        "memory": None,
        "cpu_extra": None,
        "gpus": [],
        "npu": None,
    }
    if system == "windows":
        try:
            detail["memory"] = _get_windows_memory_info()
        except Exception:
            pass
        try:
            detail["cpu_extra"] = _get_windows_cpu_extra()
        except Exception:
            pass
        try:
            detail["gpus"] = _get_windows_gpu_detail()
        except Exception:
            pass
        try:
            detail["npu"] = _get_windows_npu()
        except Exception:
            pass
    elif system == "darwin":
        try:
            detail["memory"] = _get_macos_memory_type()
        except Exception:
            pass
    return detail


# ---------------------------------------------------------------------------
# NVIDIA + resource snapshots
# ---------------------------------------------------------------------------


def get_nvidia_gpu_stats() -> List[Dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = run_command(
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


def get_resource_snapshot(stage: str) -> Dict[str, Any]:
    system = platform.system().lower()
    cpu_load: Optional[float] = None
    ram_total: Optional[float] = None
    ram_used: Optional[float] = None

    if system == "windows":
        cpu_raw = _cim_first_value("Win32_Processor", "LoadPercentage") or _get_windows_wmic_value(["cpu"], "LoadPercentage")
        if cpu_raw:
            try:
                cpu_load = float(cpu_raw)
            except (TypeError, ValueError):
                cpu_load = None

        osi = _cim_instances("Win32_OperatingSystem", ["TotalVisibleMemorySize", "FreePhysicalMemory"])
        osi = osi[0] if osi else {}
        total_kb = osi.get("TotalVisibleMemorySize") or _get_windows_wmic_value(["os"], "TotalVisibleMemorySize")
        free_kb = osi.get("FreePhysicalMemory") or _get_windows_wmic_value(["os"], "FreePhysicalMemory")
        try:
            if total_kb:
                ram_total = float(total_kb) * 1024
            if free_kb and ram_total is not None:
                ram_used = ram_total - (float(free_kb) * 1024)
        except (TypeError, ValueError):
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
        "timestamp": iso_now_local(),
        "stage": stage,
        "cpu_load_percent": cpu_load,
        "ram_used_bytes": ram_used,
        "ram_total_bytes": ram_total,
        "nvidia_gpus": get_nvidia_gpu_stats(),
    }


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
                "peak_vram_used_mb": max_or_none(i.get("vram_used_mb") for i in items),
                "peak_gpu_util_percent": max_or_none(i.get("gpu_util_percent") for i in items),
                "peak_temperature_c": max_or_none(i.get("temperature_c") for i in items),
                "peak_power_w": max_or_none(i.get("power_w") for i in items),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Anonymized labels
# ---------------------------------------------------------------------------


def rounded_ram_gb(ram_bytes: Optional[float]) -> Optional[int]:
    if ram_bytes is None or ram_bytes <= 0:
        return None
    return int(round(ram_bytes / (1024 ** 3)))


def primary_gpu_label(gpus: List[str]) -> str:
    joined = " ".join(gpus).lower()
    if "nvidia" in joined:
        return "nvidia"
    if "arc" in joined:
        return "arc"
    if "amd" in joined or "radeon" in joined:
        return "amd"
    if "intel" in joined or "iris" in joined:
        return "intel"
    if "apple" in joined:
        return "apple-gpu"
    if gpus:
        return slugify_path_part(gpus[0], "gpu")
    return "cpu"


def os_family_label(pc_metadata: Dict[str, Any]) -> str:
    os_caption = str(pc_metadata.get("os_caption") or "").lower()
    os_version = str(pc_metadata.get("os_version") or "").lower()
    if "windows" in os_caption:
        if "11" in os_caption or os_version.startswith("10.0.2"):
            return "windows11"
        return "windows"
    if "macos" in os_caption:
        return "macos"
    if "linux" in os_caption:
        return "linux"
    return slugify_path_part(os_caption or platform.system(), "unknown-os")


def cpu_family_label(pc_metadata: Dict[str, Any]) -> str:
    cpu = str(pc_metadata.get("cpu") or "").lower()
    if "apple m4" in cpu:
        return "apple-m4"
    if "apple m3" in cpu:
        return "apple-m3"
    if "apple m2" in cpu:
        return "apple-m2"
    if "apple m1" in cpu:
        return "apple-m1"
    if "ryzen" in cpu:
        return "ryzen"
    if "xeon" in cpu:
        return "xeon"
    if "intel" in cpu or "core(" in cpu or "core(tm)" in cpu:
        return "intel"
    return "cpu"


def anonymized_pc_profile(pc_metadata: Dict[str, Any], engine: Optional[str] = None) -> Dict[str, str]:
    ram_gb = rounded_ram_gb(pc_metadata.get("ram_bytes"))
    return {
        "machine_label": machine_label_parts(pc_metadata, engine)[0],
        "os_family": os_family_label(pc_metadata),
        "cpu_family": cpu_family_label(pc_metadata),
        "gpu_family": primary_gpu_label(pc_metadata.get("gpus") or []),
        "ram_class": f"{ram_gb}gb" if ram_gb is not None else "unknown-ram",
    }


def machine_label_parts(pc_metadata: Dict[str, Any], engine: Optional[str] = None) -> Tuple[str, str]:
    """Return (display label, path slug).

    When ``engine`` is given the label is prefixed with it (e.g. ``ollama-`` or
    ``foundry-``) so each engine+machine combination is its own reports folder
    and comparison column.
    """
    ram_gb = rounded_ram_gb(pc_metadata.get("ram_bytes"))
    ram_label = f"{ram_gb}gb" if ram_gb is not None else "unknown-ram"
    os_label = os_family_label(pc_metadata)
    cpu_family = cpu_family_label(pc_metadata)
    gpu_label = primary_gpu_label(pc_metadata.get("gpus") or [])

    if os_label == "macos" and cpu_family.startswith("apple-m"):
        base = f"{cpu_family}-{ram_label}"
    else:
        base = f"{os_label}-{gpu_label}-{ram_label}"

    if engine:
        display = f"{engine}-{base}"
    else:
        display = base
    return display, slugify_path_part(display, "unknown-machine")


# ---------------------------------------------------------------------------
# Generic per-run result + aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    model: str
    ok: bool
    error: Optional[str]
    wall_s: Optional[float]
    gen_tokens: Optional[int]
    prompt_tokens: Optional[int]
    prompt_toks_per_s: Optional[float]
    gen_toks_per_s: Optional[float]
    ttft_s: Optional[float] = None
    inter_token_ms: Optional[float] = None
    # Ollama exposes server-side timing; Foundry leaves these None.
    load_s: Optional[float] = None
    total_s: Optional[float] = None
    prompt_eval_s: Optional[float] = None
    eval_s: Optional[float] = None


def aggregate(results: List[RunResult]) -> Dict[str, Any]:
    ok = [r for r in results if r.ok]
    errs = [r for r in results if not r.ok]

    gen_tps = [r.gen_toks_per_s for r in ok if r.gen_toks_per_s is not None]
    prompt_tps = [r.prompt_toks_per_s for r in ok if r.prompt_toks_per_s is not None]
    wall_s = [r.wall_s for r in ok if r.wall_s is not None]
    total_s = [r.total_s for r in ok if r.total_s is not None]
    eval_s = [r.eval_s for r in ok if r.eval_s is not None]
    ttft_ms = [r.ttft_s * 1000.0 for r in ok if r.ttft_s is not None]

    return {
        "runs": len(results),
        "ok_runs": len(ok),
        "err_runs": len(errs),
        "gen_tps_mean": mean(gen_tps),
        "gen_tps_stdev": stdev_sample(gen_tps),
        "gen_tps_p50": percentile(gen_tps, 50),
        "gen_tps_p90": percentile(gen_tps, 90),
        "prompt_tps_mean": mean(prompt_tps),
        "ttft_ms_mean": mean(ttft_ms),
        "wall_s_mean": mean(wall_s),
        "total_s_mean": mean(total_s),
        "eval_s_mean": mean(eval_s),
        "errors": [e.error for e in errs if e.error],
    }


# ---------------------------------------------------------------------------
# Report rendering (shared by all engines)
# ---------------------------------------------------------------------------


def _render_hardware_section(lines: List[str], pc_metadata: Dict[str, Any], hardware: Dict[str, Any]) -> None:
    lines.append("## Hardware")
    lines.append("")
    lines.append("| Component | Detail |")
    lines.append("|---|---|")

    # CPU
    cpu_extra = hardware.get("cpu_extra") or {}
    cpu_bits = [str(pc_metadata.get("cpu") or "-")]
    if cpu_extra.get("max_clock_mhz"):
        cpu_bits.append(f"{fmt_maybe(cpu_extra.get('max_clock_mhz'))} MHz max")
    cores = pc_metadata.get("cpu_cores")
    logical = pc_metadata.get("cpu_logical_processors")
    cpu_bits.append(f"{fmt_maybe(cores)}C/{fmt_maybe(logical)}T")
    if cpu_extra.get("l2_cache_kb"):
        cpu_bits.append(f"L2 {fmt_maybe(cpu_extra.get('l2_cache_kb'))} KB")
    if cpu_extra.get("l3_cache_kb"):
        cpu_bits.append(f"L3 {fmt_maybe(cpu_extra.get('l3_cache_kb'))} KB")
    lines.append(f"| CPU | {md_escape(', '.join(cpu_bits))} |")

    # Memory
    mem = hardware.get("memory") or {}
    if mem:
        mem_bits = []
        if mem.get("type"):
            mem_bits.append(str(mem.get("type")))
        if mem.get("speed_mts"):
            mem_bits.append(f"{fmt_float(mem.get('speed_mts'), 0)} MT/s")
        if mem.get("channels"):
            mem_bits.append(f"{fmt_maybe(mem.get('channels'))} ch")
        if mem.get("theoretical_bandwidth_gbps"):
            mem_bits.append(f"~{fmt_float(mem.get('theoretical_bandwidth_gbps'), 1)} GB/s theoretical")
        if mem.get("unified"):
            mem_bits.append("unified")
        total_ram = pc_metadata.get("ram_bytes") or mem.get("total_capacity_bytes")
        if total_ram:
            mem_bits.append(f"{fmt_bytes(total_ram)} total")
        lines.append(f"| Memory | {md_escape(', '.join(mem_bits) or '-')} |")
    else:
        total_ram = pc_metadata.get("ram_bytes")
        lines.append(f"| Memory | {md_escape(fmt_bytes(total_ram) + ' total' if total_ram else '-')} |")

    # GPU(s)
    gpu_details = hardware.get("gpus") or []
    if gpu_details:
        for g in gpu_details:
            parts = [str(g.get("name") or "-")]
            if g.get("dedicated_mb"):
                parts.append(f"{fmt_float(g.get('dedicated_mb'), 0)} MB dedicated")
            if g.get("shared_mb"):
                parts.append(f"{fmt_float(g.get('shared_mb'), 0)} MB shared")
            if g.get("driver_version"):
                parts.append(f"driver {g.get('driver_version')}")
            lines.append(f"| GPU | {md_escape(', '.join(parts))} |")
    else:
        gpu_names = pc_metadata.get("gpus") or []
        lines.append(f"| GPU | {md_escape(', '.join(gpu_names) or '-')} |")

    # NPU
    npu = hardware.get("npu")
    if npu is not None:
        if npu.get("present"):
            lines.append(f"| NPU | {md_escape(fmt_maybe(npu.get('name')))} |")
        else:
            lines.append("| NPU | not detected |")
    lines.append("")


def render_report(
    *,
    engine: str,
    engine_title: str,
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
    hardware: Dict[str, Any],
    resource_snapshots: List[Dict[str, Any]],
    model_sizes: Optional[Dict[str, Optional[float]]] = None,
) -> str:
    model_sizes = model_sizes or {}
    lines: List[str] = []
    anon = anonymized_pc_profile(pc_metadata, engine)
    machine_label = anon["machine_label"]
    lines.append(f"# {engine_title} Benchmark Report")
    lines.append("")
    lines.append(f"- Started: `{started_at}`")
    lines.append(f"- Engine: `{engine}`")
    lines.append(f"- Host: `{host}`")
    lines.append(f"- Python: `{platform.python_version()}`")
    lines.append(f"- Python env: `{'venv' if virtualenv_path() else 'system'}`")
    lines.append(f"- Platform: `{anon['os_family']}`")
    lines.append(f"- Machine label: `{machine_label}`")
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
    lines.append(f"| Machine label | `{md_escape(machine_label)}` |")
    lines.append(f"| OS family | `{anon['os_family']}` |")
    lines.append(f"| CPU family | `{anon['cpu_family']}` |")
    lines.append(f"| GPU family | `{anon['gpu_family']}` |")
    lines.append(f"| RAM class | `{anon['ram_class']}` |")
    lines.append(
        f"| CPU cores / logical processors | `{fmt_maybe(pc_metadata.get('cpu_cores'))} / {fmt_maybe(pc_metadata.get('cpu_logical_processors'))}` |"
    )
    lines.append("")

    _render_hardware_section(lines, pc_metadata, hardware)

    lines.append("## Observed resources")
    lines.append("")
    lines.append(
        "Resource values are point-in-time samples captured around warmup and measured runs; GPU/VRAM rows require NVIDIA `nvidia-smi`."
    )
    lines.append("")
    peak_cpu = max_or_none(s.get("cpu_load_percent") for s in resource_snapshots)
    peak_ram = max_or_none(s.get("ram_used_bytes") for s in resource_snapshots)
    ram_total = max_or_none(s.get("ram_total_bytes") for s in resource_snapshots)
    lines.append(f"- Peak observed CPU load: `{fmt_float(peak_cpu, 2)}%`")
    lines.append(f"- Peak observed RAM used: `{fmt_bytes(peak_ram)}` / `{fmt_bytes(ram_total)}`")
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
                        md_escape(f"{g.get('index')}: {g.get('name')}"),
                        md_escape(fmt_maybe(g.get("driver_version"))),
                        f"{fmt_float(g.get('peak_vram_used_mb'), 0)} / {fmt_float(g.get('vram_total_mb'), 0)}",
                        fmt_float(g.get("peak_gpu_util_percent"), 0),
                        fmt_float(g.get("peak_temperature_c"), 0),
                        fmt_float(g.get("peak_power_w"), 2),
                    ]
                )
                + " |"
            )
    else:
        lines.append("- NVIDIA GPU samples: `not available`")
    lines.append("")

    theoretical_bw = None
    mem = hardware.get("memory") or {}
    if mem:
        theoretical_bw = mem.get("theoretical_bandwidth_gbps")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "KPIs: **Gen tok/s** (decode throughput), **Prompt tok/s** (prefill), **TTFT** "
        "(time to first token), **Eff BW** (achieved decode bandwidth ~= model_size x gen tok/s), "
        "**BW util %** (Eff BW as a share of the machine's theoretical memory bandwidth"
        + (f", ~{fmt_float(theoretical_bw, 1)} GB/s here" if theoretical_bw else "")
        + ") and **Tok/s/GB** (throughput per GiB of model resident in memory). BW util % and "
        "Tok/s/GB are the efficiency metrics; higher is better."
    )
    lines.append("")
    lines.append(
        "> Note: Eff BW and BW util assume *dense* weight streaming (all weights read once per "
        "token). Mixture-of-Experts / elastic models (e.g. `*-a3b`, `gpt-oss`, `nemotron-*-nano`, "
        "`gemma4:e4b`) stream only their active parameters, so their Eff BW and BW util are "
        "overestimated and can exceed 100% — a useful signal that the model is sparse rather than dense."
    )
    lines.append("")
    lines.append(
        "| Model | OK/Total | Gen tok/s (mean) | Gen tok/s (p50) | Gen tok/s (p90) | Gen tok/s (stdev) | Prompt tok/s (mean) | TTFT ms (mean) | Eff BW GB/s (mean) | BW util % (mean) | Tok/s/GB (mean) | Total s (mean) | Wall s (mean) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in models:
        agg = aggregate(all_results.get(model, []))
        eff_bw = effective_bandwidth_gbps(model_sizes.get(model), agg["gen_tps_mean"])
        bw_util = bandwidth_utilization_pct(eff_bw, theoretical_bw)
        toks_per_gb = tokens_per_s_per_gb(agg["gen_tps_mean"], model_sizes.get(model))
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(model),
                    f"{agg['ok_runs']}/{agg['runs']}",
                    fmt_float(agg["gen_tps_mean"], 2),
                    fmt_float(agg["gen_tps_p50"], 2),
                    fmt_float(agg["gen_tps_p90"], 2),
                    fmt_float(agg["gen_tps_stdev"], 2),
                    fmt_float(agg["prompt_tps_mean"], 2),
                    fmt_float(agg["ttft_ms_mean"], 1),
                    fmt_float(eff_bw, 1),
                    fmt_float(bw_util, 1),
                    fmt_float(toks_per_gb, 2),
                    fmt_float(agg["total_s_mean"], 2),
                    fmt_float(agg["wall_s_mean"], 2),
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
        size = model_sizes.get(model)
        if size:
            lines.append("")
            lines.append(f"Model size on disk: `{fmt_bytes(size)}`")
        lines.append("")
        lines.append(
            "| Run | OK | Gen tok/s | Prompt tok/s | TTFT ms | Inter-tok ms | Gen toks | Prompt toks | Eval s | Load s | Total s | Wall s | Error |"
        )
        lines.append("|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for i, r in enumerate(results, start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(i),
                        "Y" if r.ok else "N",
                        fmt_float(r.gen_toks_per_s, 2),
                        fmt_float(r.prompt_toks_per_s, 2),
                        fmt_float(r.ttft_s * 1000.0 if r.ttft_s is not None else None, 1),
                        fmt_float(r.inter_token_ms, 2),
                        fmt_int(r.gen_tokens),
                        fmt_int(r.prompt_tokens),
                        fmt_float(r.eval_s, 2),
                        fmt_float(r.load_s, 2),
                        fmt_float(r.total_s, 2),
                        fmt_float(r.wall_s, 2),
                        md_escape(r.error or ""),
                    ]
                )
                + " |"
            )
        agg = aggregate(results)
        if agg["errors"]:
            lines.append("")
            lines.append("Errors:")
            for e in agg["errors"]:
                lines.append(f"- `{e}`")
        lines.append("")

    return "\n".join(lines)
