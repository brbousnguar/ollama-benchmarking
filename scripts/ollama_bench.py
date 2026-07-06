#!/usr/bin/env python3
"""
Benchmark local Ollama models via the HTTP API and write a Markdown report.

Requires:
  - Python 3.8+
  - Ollama running locally (default: http://localhost:11434)

This script uses /api/tags to discover models (unless --models is provided)
and /api/generate (stream=true) to collect server-side timing counters plus a
client-side time-to-first-token (TTFT) measurement.

Shared metadata/hardware collection and report rendering live in bench_common.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as bc  # noqa: E402

_DEFAULT_CONFIG_PATH = os.path.join(bc.repo_root(), "ollama-bench.json")


def _is_cloud_model(model: str) -> bool:
    return "cloud" in model.lower()


def _filter_cloud_models(models: List[str]) -> Tuple[List[str], List[str]]:
    local_models = [m for m in models if not _is_cloud_model(m)]
    cloud_models = [m for m in models if _is_cloud_model(m)]
    return local_models, cloud_models


def _list_ollama_model_sizes(host: str, timeout_s: float) -> Dict[str, Optional[float]]:
    """Return {model_name: size_bytes} from /api/tags (disk weight size)."""
    tags = bc.http_json(f"{host}/api/tags", timeout_s=timeout_s)
    sizes: Dict[str, Optional[float]] = {}
    for m in tags.get("models", []) or []:
        name = m.get("name")
        if isinstance(name, str) and name.strip():
            size = m.get("size")
            sizes[name.strip()] = float(size) if isinstance(size, (int, float)) else None
    return sizes


def _list_ollama_models(host: str, timeout_s: float) -> List[str]:
    return sorted(_list_ollama_model_sizes(host, timeout_s).keys())


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
        bc.log("All selected models are already available locally")
        return
    if shutil.which("ollama") is None:
        raise RuntimeError("Some models are missing locally and the 'ollama' CLI was not found in PATH.")

    for model in missing:
        bc.log(f"Model {model} not found locally; pulling with `ollama pull {model}`")
        t0 = time.perf_counter()
        rc, output = bc.run_streaming_command(["ollama", "pull", model])
        if rc != 0:
            detail = output.strip().splitlines()[-1] if output.strip() else "see Ollama output above"
            raise RuntimeError(f"`ollama pull {model}` failed with exit code {rc}: {detail}")
        bc.log(f"Completed pull for {model} in {bc.fmt_duration(time.perf_counter() - t0)}")


def _generate_once(
    host: str,
    model: str,
    prompt: str,
    timeout_s: float,
    options: Dict[str, Any],
    keep_alive: Optional[str],
) -> bc.RunResult:
    """Stream /api/generate so we can measure client-side TTFT, then read the
    final message's server-side timing counters for accurate tokens/sec."""
    url = f"{host}/api/generate"
    body: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
    if options:
        body["options"] = options
    if keep_alive is not None:
        body["keep_alive"] = keep_alive

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "application/x-ndjson", "Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    ttft_s: Optional[float] = None
    chunk_times: List[float] = []
    final: Dict[str, Any] = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                msg = json.loads(line)
                now = time.perf_counter()
                # Reasoning models (qwen3, deepseek-r1, ...) stream their tokens
                # under "thinking" with an empty "response" until the answer
                # begins, so count either as an emitted token for TTFT / latency.
                if msg.get("response") or msg.get("thinking"):
                    if ttft_s is None:
                        ttft_s = now - t0
                    chunk_times.append(now)
                if msg.get("done"):
                    final = msg
        wall_s = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - report the failure as a run error
        return bc.RunResult(
            model=model, ok=False, error=str(e), wall_s=time.perf_counter() - t0,
            gen_tokens=None, prompt_tokens=None, prompt_toks_per_s=None, gen_toks_per_s=None,
        )

    load_s = bc.ns_to_s(final.get("load_duration"))
    total_s = bc.ns_to_s(final.get("total_duration"))
    prompt_eval_s = bc.ns_to_s(final.get("prompt_eval_duration"))
    eval_s = bc.ns_to_s(final.get("eval_duration"))
    prompt_tokens = final.get("prompt_eval_count")
    gen_tokens = final.get("eval_count")

    prompt_tps = bc.safe_div(float(prompt_tokens) if isinstance(prompt_tokens, int) else None, prompt_eval_s)
    gen_tps = bc.safe_div(float(gen_tokens) if isinstance(gen_tokens, int) else None, eval_s)

    inter_token_ms: Optional[float] = None
    if len(chunk_times) >= 2:
        inter_token_ms = ((chunk_times[-1] - chunk_times[0]) / (len(chunk_times) - 1)) * 1000.0

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
        load_s=load_s,
        total_s=total_s,
        prompt_eval_s=prompt_eval_s,
        eval_s=eval_s,
    )


def _load_config_models(config_path: str) -> Optional[List[str]]:
    if not os.path.exists(config_path):
        return None
    config = bc.read_json_file(config_path)
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


def _parse_models_arg(models_arg: Optional[str]) -> Optional[List[str]]:
    if models_arg is None:
        return None
    raw = [m.strip() for m in models_arg.split(",")]
    models = [m for m in raw if m]
    return models or None


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    bc.bootstrap_venv_if_needed(__file__, argv)
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
        help="Output Markdown path (default: reports/ollama-<machine>/ollama-bench-<timestamp>.md)",
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
    started_at = bc.iso_now_local()
    bc.log(f"Starting Ollama benchmark session at {started_at}")
    bc.log(f"Ollama host: {host}")
    bc.log(f"Python executable: {sys.executable}")
    if bc.virtualenv_path():
        bc.log(f"Virtual env: {bc.virtualenv_path()}")
    else:
        bc.log("Virtual env: not active")

    bc.log("Collecting PC metadata and initial resource snapshot")
    pc_metadata = bc.get_pc_metadata()
    machine_label, machine_slug = bc.machine_label_parts(pc_metadata, "ollama")
    resource_snapshots: List[Dict[str, Any]] = [bc.get_resource_snapshot("start")]
    bc.log(f"Anonymized machine label: {machine_label}")
    bc.log("Collecting richer hardware detail (memory/CPU/GPU/NPU)")
    hardware = bc.get_hardware_detail(pc_metadata)

    config_models = _load_config_models(args.config)
    if config_models is not None:
        bc.log(f"Loaded config file: {args.config}")

    models = _parse_models_arg(args.models)
    model_source = "command line"
    if models is None and config_models is not None:
        models = config_models
        model_source = f"config file ({args.config})"

    if models is None:
        bc.log("Discovering models from Ollama /api/tags")
        models, skipped_cloud = _discover_models(host, timeout_s=args.timeout_s, include_cloud=args.include_cloud)
        if skipped_cloud:
            bc.log(f"Skipping {len(skipped_cloud)} cloud model(s): {', '.join(skipped_cloud)}")
        bc.log(f"Discovered {len(models)} model(s): {', '.join(models)}")
    else:
        skipped_cloud = []
        if not args.include_cloud:
            models, skipped_cloud = _filter_cloud_models(models)
            if skipped_cloud:
                bc.log(f"Skipping {len(skipped_cloud)} cloud model(s) from {model_source}: {', '.join(skipped_cloud)}")
            if not models:
                raise SystemExit("Only cloud models were selected. Re-run with --include-cloud to benchmark them.")
        bc.log(f"Using {len(models)} model(s) from {model_source}: {', '.join(models)}")

    if not models:
        raise SystemExit("No models selected. Add models to the config file, pass --models, or let the script auto-discover models.")

    _ensure_models_available(host=host, models=models, timeout_s=args.timeout_s)

    try:
        model_sizes = _list_ollama_model_sizes(host, timeout_s=args.timeout_s)
    except Exception:  # noqa: BLE001 - size is a nice-to-have for effective bandwidth
        model_sizes = {}

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
        out_path = os.path.join("reports", machine_slug, f"ollama-bench-{ts}.md")
    out_dir = os.path.dirname(out_path) or "."
    bc.ensure_dir(out_dir)
    bc.log(f"Report path: {out_path}")
    bc.log(f"Runs per model: {args.runs}; warmup runs per model: {args.warmup}")
    bc.log(f"Generation options: {json.dumps(options, sort_keys=True)}")

    all_results: Dict[str, List[bc.RunResult]] = {m: [] for m in models}

    # Heartbeat file so the dashboard can show an "analysis running" banner.
    status_dir = os.path.join(bc.repo_root(), "reports")
    run_status: Dict[str, Any] = {
        "running": True,
        "engine": "ollama",
        "machine_label": machine_label,
        "started": started_at,
        "pid": os.getpid(),
        "models": models,
        "total": len(models),
        "completed": 0,
        "current": models[0] if models else None,
        "phase": "warmup" if args.warmup > 0 else "measure",
    }

    def push_status(**changes: Any) -> None:
        run_status.update(changes)
        bc.write_bench_status(status_dir, run_status)

    push_status()

    if args.warmup > 0:
        bc.log("Starting warmup phase")
        for model in models:
            push_status(current=model, phase="warmup")
            bc.log(f"Starting warmup for model {model}")
            model_warmup_t0 = time.perf_counter()
            for i in range(args.warmup):
                run_t0 = time.perf_counter()
                bc.log(f"Warmup {i + 1}/{args.warmup} for {model}: started")
                resource_snapshots.append(bc.get_resource_snapshot(f"before warmup {model}"))
                warmup_result = _generate_once(
                    host=host, model=model, prompt=prompt, timeout_s=args.timeout_s,
                    options=options, keep_alive=keep_alive,
                )
                resource_snapshots.append(bc.get_resource_snapshot(f"after warmup {model}"))
                status = "ok" if warmup_result.ok else f"failed: {warmup_result.error}"
                bc.log(f"Warmup {i + 1}/{args.warmup} for {model}: {status} in {bc.fmt_duration(time.perf_counter() - run_t0)}")
            bc.log(f"Completed warmup for model {model} in {bc.fmt_duration(time.perf_counter() - model_warmup_t0)}")

    bc.log("Starting measured benchmark phase")
    for model_index, model in enumerate(models):
        push_status(current=model, phase="measure")
        bc.log(f"Starting model {model}")
        model_t0 = time.perf_counter()
        for i in range(args.runs):
            run_t0 = time.perf_counter()
            bc.log(f"Run {i + 1}/{args.runs} for {model}: started")
            resource_snapshots.append(bc.get_resource_snapshot(f"before run {model} #{i + 1}"))
            r = _generate_once(
                host=host, model=model, prompt=prompt, timeout_s=args.timeout_s,
                options=options, keep_alive=keep_alive,
            )
            all_results[model].append(r)
            resource_snapshots.append(bc.get_resource_snapshot(f"after run {model} #{i + 1}"))
            if r.ok:
                bc.log(
                    f"Run {i + 1}/{args.runs} for {model}: ok in {bc.fmt_duration(time.perf_counter() - run_t0)} "
                    f"(gen tok/s: {bc.fmt_float(r.gen_toks_per_s, 2)}, ttft: {bc.fmt_float((r.ttft_s or 0) * 1000, 1)}ms, wall: {bc.fmt_float(r.wall_s, 2)}s)"
                )
            else:
                bc.log(f"Run {i + 1}/{args.runs} for {model}: failed in {bc.fmt_duration(time.perf_counter() - run_t0)}: {r.error}")
        model_elapsed = time.perf_counter() - model_t0
        agg = bc.aggregate(all_results[model])
        bc.log(
            f"Completed model {model} in {bc.fmt_duration(model_elapsed)} "
            f"({agg['ok_runs']}/{agg['runs']} ok, mean gen tok/s: {bc.fmt_float(agg['gen_tps_mean'], 2)})"
        )
        push_status(completed=model_index + 1)

    resource_snapshots.append(bc.get_resource_snapshot("end"))

    # Benchmarks are done; drop the heartbeat so the dashboard clears its banner.
    bc.clear_bench_status(status_dir)

    bc.log("Rendering markdown report")
    report = bc.render_report(
        engine="ollama",
        engine_title="Ollama",
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
        hardware=hardware,
        resource_snapshots=resource_snapshots,
        model_sizes=model_sizes,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    session_elapsed = time.perf_counter() - session_t0
    bc.log(f"Report written: {out_path}")
    bc.log(f"Benchmark session completed in {bc.fmt_duration(session_elapsed)}")
    sys.stdout.write(f"{out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
