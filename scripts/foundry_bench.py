#!/usr/bin/env python3
"""
Benchmark Microsoft Foundry Local models and write a Markdown report.

Foundry Local exposes an OpenAI-compatible HTTP API but, unlike Ollama, returns
no server-side timing counters. So throughput is measured client-side from a
streaming chat completion: time-to-first-token (TTFT), inter-token latency, and
generation tokens/sec over the decode window. Token counts come from the final
`usage` chunk (stream_options.include_usage).

Requires:
  - Python 3.8+
  - Foundry Local installed (`winget install Microsoft.FoundryLocal`) and its
    service running (this script will try to start it).

Reports share the format produced by bench_common so they compare directly with
Ollama reports via scripts/compare_latest_reports.py.

Note: Foundry Local is targeted here at the Windows machines. It runs on Apple
silicon too, but per the project's setup macOS stays on Ollama; pass
--allow-non-windows to override the guard.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as bc  # noqa: E402

_DEFAULT_CONFIG_PATH = os.path.join(bc.repo_root(), "foundry-bench.json")
_DEFAULT_ENDPOINT = "http://localhost:5273"


# ---------------------------------------------------------------------------
# Foundry service / endpoint discovery
# ---------------------------------------------------------------------------


def _foundry_cli() -> Optional[str]:
    return shutil.which("foundry")


def _parse_endpoint_from_status(text: str) -> Optional[str]:
    """Pull the base service URL out of `foundry service status` output."""
    m = re.search(r"https?://[\w.\-]+:\d+", text)
    if m:
        return m.group(0)
    return None


def _ensure_service_and_endpoint(explicit_endpoint: Optional[str]) -> str:
    """Return the base endpoint URL (e.g. http://localhost:5273).

    Order: explicit --endpoint > parse `foundry service status` > start the
    service then re-check > documented default.
    """
    if explicit_endpoint:
        return explicit_endpoint.rstrip("/")

    cli = _foundry_cli()
    if cli is None:
        raise RuntimeError(
            "Foundry Local CLI ('foundry') not found in PATH. Install it with "
            "`winget install Microsoft.FoundryLocal`, or pass --endpoint."
        )

    status = bc.run_command([cli, "service", "status"], timeout_s=20.0) or ""
    endpoint = _parse_endpoint_from_status(status)
    if endpoint is None:
        bc.log("Foundry service not reporting an endpoint; attempting `foundry service start`")
        bc.run_command([cli, "service", "start"], timeout_s=60.0)
        status = bc.run_command([cli, "service", "status"], timeout_s=20.0) or ""
        endpoint = _parse_endpoint_from_status(status)

    if endpoint is None:
        bc.log(f"Could not parse endpoint from status; falling back to {_DEFAULT_ENDPOINT}")
        endpoint = _DEFAULT_ENDPOINT
    return endpoint.rstrip("/")


def _api_base(endpoint: str) -> str:
    """Normalize to the OpenAI-compatible base ending in /v1 (no trailing slash)."""
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint
    return f"{endpoint}/v1"


# ---------------------------------------------------------------------------
# Model discovery / availability / size
# ---------------------------------------------------------------------------


def _list_loaded_models(api_base: str, timeout_s: float) -> List[str]:
    try:
        data = bc.http_json(f"{api_base}/models", timeout_s=timeout_s)
    except Exception:  # noqa: BLE001
        return []
    models = []
    for item in data.get("data", []) or []:
        mid = item.get("id")
        if isinstance(mid, str) and mid.strip():
            models.append(mid.strip())
    return sorted(set(models))


def _ensure_models_available(models: List[str], timeout_s: float) -> None:
    cli = _foundry_cli()
    if cli is None:
        bc.log("foundry CLI not found; skipping model download step (requests may fail if models are absent)")
        return
    for model in models:
        bc.log(f"Ensuring Foundry model is available: `foundry model download {model}`")
        rc, _ = bc.run_streaming_command([cli, "model", "download", model])
        if rc != 0:
            bc.log(f"Warning: `foundry model download {model}` exited {rc}; continuing (it may already be cached)")


def _foundry_model_sizes() -> Dict[str, Optional[float]]:
    """Best-effort {model: size_bytes} parsed from `foundry cache list`/`model list`."""
    cli = _foundry_cli()
    if cli is None:
        return {}
    out = bc.run_command([cli, "cache", "list"], timeout_s=30.0) or bc.run_command(
        [cli, "model", "list"], timeout_s=30.0
    )
    if not out:
        return {}
    sizes: Dict[str, Optional[float]] = {}
    for line in out.splitlines():
        # Lines vary by version; match a model token and a trailing size like "2.1 GB".
        size_m = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB|KB)\b", line, re.IGNORECASE)
        name_m = re.search(r"[\w][\w.\-:]+", line)
        if size_m and name_m:
            num = float(size_m.group(1))
            unit = size_m.group(2).upper()
            mult = {"KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}[unit]
            sizes[name_m.group(0)] = num * mult
    return sizes


# ---------------------------------------------------------------------------
# Streaming chat completion -> client-side timing
# ---------------------------------------------------------------------------


def _generate_once(
    api_base: str,
    model: str,
    prompt: str,
    timeout_s: float,
    options: Dict[str, Any],
) -> bc.RunResult:
    url = f"{api_base}/chat/completions"
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if "num_predict" in options:
        body["max_tokens"] = options["num_predict"]
    if "temperature" in options:
        body["temperature"] = options["temperature"]
    if options.get("seed") is not None:
        body["seed"] = options["seed"]
    if options.get("top_p") is not None:
        body["top_p"] = options["top_p"]

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    ttft_s: Optional[float] = None
    chunk_times: List[float] = []
    chunk_count = 0
    usage: Dict[str, Any] = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    msg = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                choices = msg.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if ttft_s is None:
                            ttft_s = now - t0
                        chunk_times.append(now)
                        chunk_count += 1
                if msg.get("usage"):
                    usage = msg["usage"]
        wall_s = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return bc.RunResult(
            model=model, ok=False, error=str(e), wall_s=time.perf_counter() - t0,
            gen_tokens=None, prompt_tokens=None, prompt_toks_per_s=None, gen_toks_per_s=None,
        )

    prompt_tokens = usage.get("prompt_tokens")
    gen_tokens = usage.get("completion_tokens")
    if not isinstance(gen_tokens, int):
        gen_tokens = chunk_count or None
    if not isinstance(prompt_tokens, int):
        prompt_tokens = None

    # Decode window = first content chunk to last content chunk.
    decode_window = (chunk_times[-1] - chunk_times[0]) if len(chunk_times) >= 2 else None
    gen_tps: Optional[float] = None
    if gen_tokens and decode_window and decode_window > 0:
        # tokens after the first arrive across the decode window
        gen_tps = (gen_tokens - 1) / decode_window if gen_tokens > 1 else None
    if gen_tps is None and gen_tokens and ttft_s is not None and wall_s > ttft_s:
        gen_tps = gen_tokens / (wall_s - ttft_s)

    prompt_tps = bc.safe_div(
        float(prompt_tokens) if isinstance(prompt_tokens, int) else None, ttft_s
    )
    inter_token_ms = (decode_window / (len(chunk_times) - 1) * 1000.0) if decode_window else None

    return bc.RunResult(
        model=model,
        ok=True,
        error=None,
        wall_s=wall_s,
        gen_tokens=gen_tokens if isinstance(gen_tokens, int) else None,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        prompt_toks_per_s=prompt_tps,
        gen_toks_per_s=gen_tps,
        ttft_s=ttft_s,
        inter_token_ms=inter_token_ms,
        # Foundry exposes no server-side timing.
        total_s=wall_s,
    )


# ---------------------------------------------------------------------------
# Config / args
# ---------------------------------------------------------------------------


def _load_config_models(config_path: str) -> Optional[List[str]]:
    if not os.path.exists(config_path):
        return None
    config = bc.read_json_file(config_path)
    raw_models = config.get("models")
    if raw_models is None:
        raise RuntimeError(f"Config file does not define a 'models' array: {config_path}")
    if not isinstance(raw_models, list):
        raise RuntimeError(f"Config file 'models' must be an array: {config_path}")
    models = [m.strip() for m in raw_models if isinstance(m, str) and m.strip()]
    return sorted(set(models)) or []


def _parse_models_arg(models_arg: Optional[str]) -> Optional[List[str]]:
    if models_arg is None:
        return None
    models = [m.strip() for m in models_arg.split(",") if m.strip()]
    return models or None


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    bc.bootstrap_venv_if_needed(__file__, argv)
    session_t0 = time.perf_counter()

    p = argparse.ArgumentParser(description="Benchmark Microsoft Foundry Local models and write a Markdown report.")
    p.add_argument("--no-venv", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--endpoint", default=None, help="Foundry base endpoint (default: auto-discover via `foundry service status`)")
    p.add_argument("--config", default=_DEFAULT_CONFIG_PATH, help=f"Config JSON path (default: {_DEFAULT_CONFIG_PATH})")
    p.add_argument("--models", default=None, help="Comma-separated models. If omitted, use config then /v1/models discovery")
    p.add_argument("--allow-non-windows", action="store_true", help="Run even though the host is not Windows")
    p.add_argument("--no-download", action="store_true", help="Skip `foundry model download` for missing models")
    p.add_argument("--runs", type=int, default=3, help="Measured runs per model (default: 3)")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per model, not included in summary (default: 1)")
    p.add_argument("--timeout-s", type=float, default=600.0, help="Per-request timeout in seconds (default: 600)")
    p.add_argument("--out", default=None, help="Output Markdown path (default: reports/foundry-<machine>/foundry-bench-<timestamp>.md)")
    p.add_argument("--prompt", default=None, help="Prompt text (overrides --prompt-file)")
    p.add_argument("--prompt-file", default=None, help="Path to a text file containing the prompt")
    p.add_argument("--num-predict", type=int, default=256, help="max_tokens for generation (default: 256)")
    p.add_argument("--temperature", type=float, default=0.0, help="temperature (default: 0.0)")
    p.add_argument("--top-p", type=float, default=None, help="top_p (default: unset)")
    p.add_argument("--seed", type=int, default=42, help="seed (default: 42)")
    args = p.parse_args(argv)

    if platform.system().lower() != "windows" and not args.allow_non_windows:
        sys.stderr.write(
            "Foundry Local benchmarking is configured for the Windows machines in this project.\n"
            "macOS stays on Ollama (scripts/ollama_bench.py). To run anyway, pass --allow-non-windows.\n"
        )
        return 2

    started_at = bc.iso_now_local()
    bc.log(f"Starting Foundry Local benchmark session at {started_at}")
    endpoint = _ensure_service_and_endpoint(args.endpoint)
    api_base = _api_base(endpoint)
    bc.log(f"Foundry endpoint: {endpoint} (API base: {api_base})")
    bc.log(f"Python executable: {sys.executable}")

    bc.log("Collecting PC metadata and initial resource snapshot")
    pc_metadata = bc.get_pc_metadata()
    machine_label, machine_slug = bc.machine_label_parts(pc_metadata, "foundry")
    resource_snapshots: List[Dict[str, Any]] = [bc.get_resource_snapshot("start")]
    bc.log(f"Anonymized machine label: {machine_label}")
    bc.log("Collecting richer hardware detail (memory/CPU/GPU/NPU)")
    hardware = bc.get_hardware_detail(pc_metadata)

    config_models = _load_config_models(args.config)
    if config_models is not None:
        bc.log(f"Loaded config file: {args.config}")

    models = _parse_models_arg(args.models)
    model_source = "command line"
    if models is None and config_models:
        models = config_models
        model_source = f"config file ({args.config})"
    if models is None:
        bc.log("Discovering models from Foundry /v1/models")
        models = _list_loaded_models(api_base, timeout_s=args.timeout_s)
        model_source = "Foundry /v1/models"
    if not models:
        raise SystemExit(
            "No models selected. Add models to foundry-bench.json, pass --models, or load a model in Foundry first."
        )
    bc.log(f"Using {len(models)} model(s) from {model_source}: {', '.join(models)}")

    if not args.no_download:
        _ensure_models_available(models, timeout_s=args.timeout_s)

    model_sizes = _foundry_model_sizes()

    if args.prompt is not None and args.prompt_file is not None:
        raise SystemExit("Provide only one of --prompt or --prompt-file")
    if args.prompt_file is not None:
        prompt = bc.read_text_file(args.prompt_file)
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

    out_path = args.out
    if out_path is None:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join("reports", machine_slug, f"foundry-bench-{ts}.md")
    bc.ensure_dir(os.path.dirname(out_path) or ".")
    bc.log(f"Report path: {out_path}")
    bc.log(f"Runs per model: {args.runs}; warmup runs per model: {args.warmup}")

    all_results: Dict[str, List[bc.RunResult]] = {m: [] for m in models}

    if args.warmup > 0:
        bc.log("Starting warmup phase")
        for model in models:
            for i in range(args.warmup):
                run_t0 = time.perf_counter()
                bc.log(f"Warmup {i + 1}/{args.warmup} for {model}: started")
                resource_snapshots.append(bc.get_resource_snapshot(f"before warmup {model}"))
                wr = _generate_once(api_base, model, prompt, args.timeout_s, options)
                resource_snapshots.append(bc.get_resource_snapshot(f"after warmup {model}"))
                status = "ok" if wr.ok else f"failed: {wr.error}"
                bc.log(f"Warmup {i + 1}/{args.warmup} for {model}: {status} in {bc.fmt_duration(time.perf_counter() - run_t0)}")

    bc.log("Starting measured benchmark phase")
    for model in models:
        bc.log(f"Starting model {model}")
        model_t0 = time.perf_counter()
        for i in range(args.runs):
            run_t0 = time.perf_counter()
            bc.log(f"Run {i + 1}/{args.runs} for {model}: started")
            resource_snapshots.append(bc.get_resource_snapshot(f"before run {model} #{i + 1}"))
            r = _generate_once(api_base, model, prompt, args.timeout_s, options)
            all_results[model].append(r)
            resource_snapshots.append(bc.get_resource_snapshot(f"after run {model} #{i + 1}"))
            if r.ok:
                bc.log(
                    f"Run {i + 1}/{args.runs} for {model}: ok in {bc.fmt_duration(time.perf_counter() - run_t0)} "
                    f"(gen tok/s: {bc.fmt_float(r.gen_toks_per_s, 2)}, ttft: {bc.fmt_float((r.ttft_s or 0) * 1000, 1)}ms, wall: {bc.fmt_float(r.wall_s, 2)}s)"
                )
            else:
                bc.log(f"Run {i + 1}/{args.runs} for {model}: failed in {bc.fmt_duration(time.perf_counter() - run_t0)}: {r.error}")
        agg = bc.aggregate(all_results[model])
        bc.log(
            f"Completed model {model} in {bc.fmt_duration(time.perf_counter() - model_t0)} "
            f"({agg['ok_runs']}/{agg['runs']} ok, mean gen tok/s: {bc.fmt_float(agg['gen_tps_mean'], 2)})"
        )

    resource_snapshots.append(bc.get_resource_snapshot("end"))

    bc.log("Rendering markdown report")
    report = bc.render_report(
        engine="foundry",
        engine_title="Foundry Local",
        started_at=started_at,
        host=endpoint,
        models=models,
        prompt_desc=prompt_desc,
        runs=args.runs,
        warmup=args.warmup,
        timeout_s=args.timeout_s,
        options=options,
        keep_alive=None,
        all_results=all_results,
        pc_metadata=pc_metadata,
        hardware=hardware,
        resource_snapshots=resource_snapshots,
        model_sizes=model_sizes,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    bc.log(f"Report written: {out_path}")
    bc.log(f"Benchmark session completed in {bc.fmt_duration(time.perf_counter() - session_t0)}")
    sys.stdout.write(f"{out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
