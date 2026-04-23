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
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iso_now_local() -> str:
    # Include offset when available; keep it readable in reports.
    return _dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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


def _md_escape(s: str) -> str:
    # Minimal; mainly protect table pipes.
    return s.replace("|", "\\|")


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


def _discover_models(host: str, timeout_s: float) -> List[str]:
    tags = _http_json(f"{host}/api/tags", timeout_s=timeout_s)
    models = []
    for m in tags.get("models", []) or []:
        name = m.get("name")
        if isinstance(name, str) and name.strip():
            models.append(name.strip())
    models = sorted(set(models))
    if not models:
        raise RuntimeError("No models found from /api/tags. Is Ollama running and has models pulled?")
    return models


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
) -> str:
    lines: List[str] = []
    lines.append(f"# Ollama Benchmark Report")
    lines.append("")
    lines.append(f"- Started: `{started_at}`")
    lines.append(f"- Host: `{host}`")
    lines.append(f"- Python: `{platform.python_version()}`")
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
    p = argparse.ArgumentParser(description="Benchmark local Ollama models and write a Markdown report.")
    p.add_argument("--host", default="http://localhost:11434", help="Ollama host (default: http://localhost:11434)")
    p.add_argument("--models", default=None, help="Comma-separated models. If omitted, auto-discover via /api/tags")
    p.add_argument("--runs", type=int, default=3, help="Measured runs per model (default: 3)")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per model, not included in summary (default: 1)")
    p.add_argument("--timeout-s", type=float, default=600.0, help="Per-request timeout in seconds (default: 600)")
    p.add_argument("--out", default=None, help="Output Markdown path (default: reports/ollama-bench-<timestamp>.md)")
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

    models = _parse_models_arg(args.models)
    if models is None:
        models = _discover_models(host, timeout_s=args.timeout_s)

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
        out_path = os.path.join("reports", f"ollama-bench-{ts}.md")
    out_dir = os.path.dirname(out_path) or "."
    _ensure_dir(out_dir)

    # Run benchmarks.
    all_results: Dict[str, List[RunResult]] = {m: [] for m in models}

    # Warmup (not recorded) to amortize first-token/model-load effects if desired.
    if args.warmup > 0:
        for model in models:
            for _ in range(args.warmup):
                _generate_once(
                    host=host,
                    model=model,
                    prompt=prompt,
                    timeout_s=args.timeout_s,
                    options=options,
                    keep_alive=keep_alive,
                )

    for model in models:
        for _ in range(args.runs):
            r = _generate_once(
                host=host,
                model=model,
                prompt=prompt,
                timeout_s=args.timeout_s,
                options=options,
                keep_alive=keep_alive,
            )
            all_results[model].append(r)

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
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    # Print a friendly one-liner for CLI users.
    sys.stdout.write(out_path + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

