#!/usr/bin/env python3
"""
Agentic tool-use eval for local Ollama models.

Where ollama_bench.py measures *speed*, this measures *capability*: can the model
use tools correctly? Each task in evals/agentic-tools.json gives the model real
tool schemas via Ollama's tools API (/api/chat) and grades the tool call it
returns — deterministically, locally, with no frontier-model judge:

  - tool_selection : pick the right tool among several
  - arg_extraction : fill the arguments correctly
  - abstention     : do NOT call a tool when none applies

Writes a Markdown report next to the speed reports:
  reports/<machine>/eval-agentic-<timestamp>.md

Standard library only; shares hardware/metadata/heartbeat helpers with
bench_common so a run shows up in the dashboard's "analysis running" banner.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as bc  # noqa: E402

_DEFAULT_SUITE = os.path.join(bc.repo_root(), "evals", "agentic-tools.json")


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _s(v: Any) -> str:
    return str(v).strip().lower()


def _match_value(value: Any, matcher: Dict[str, Any]) -> bool:
    if "equals" in matcher:
        return _s(value) == _s(matcher["equals"])
    if "contains" in matcher:
        return _s(matcher["contains"]) in _s(value)
    if "one_of" in matcher:
        opts = [_s(x) for x in matcher["one_of"]]
        return any(o in _s(value) or _s(value) in o for o in opts)
    if "regex" in matcher:
        return re.search(matcher["regex"], str(value), re.IGNORECASE) is not None
    if "numeric" in matcher:
        try:
            return abs(float(value) - float(matcher["numeric"])) <= float(matcher.get("tol", 0))
        except (TypeError, ValueError):
            return False
    return True  # only structural constraints (e.g. just "optional"): presence is enough


def _first_tool_call(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    fn = calls[0].get("function", {}) or {}
    args = fn.get("arguments")
    if isinstance(args, str):  # some runners return a JSON string
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return {"name": fn.get("name"), "args": args or {}}


def grade(task: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
    exp = task.get("expect", {})
    expected_call = exp.get("call", True)
    call = _first_tool_call(message)
    called = call is not None

    row: Dict[str, Any] = {
        "id": task.get("id"),
        "category": task.get("category", "?"),
        "expected_call": expected_call,
        "expected_tool": exp.get("tool"),
        "called": called,
        "got_tool": call["name"] if called else None,
        "got_args": call["args"] if called else None,
    }

    if not expected_call:
        # Abstention: correct behaviour is to answer without calling a tool.
        row.update(passed=not called, tool_ok=None, arg_score=None)
        return row

    if not called:
        row.update(passed=False, tool_ok=False, arg_score=0.0)
        return row

    tool_ok = call["name"] == exp.get("tool")
    matchers: Dict[str, Any] = exp.get("args", {})
    checks: List[bool] = []
    for arg_name, matcher in matchers.items():
        present = arg_name in call["args"] and call["args"][arg_name] not in (None, "")
        if not present:
            checks.append(bool(matcher.get("optional", False)))
            continue
        checks.append(_match_value(call["args"][arg_name], matcher))
    arg_score = (sum(1 for c in checks if c) / len(checks)) if checks else 1.0
    row.update(passed=bool(tool_ok and all(checks)), tool_ok=tool_ok, arg_score=arg_score)
    return row


# ---------------------------------------------------------------------------
# Ollama calls
# ---------------------------------------------------------------------------


def _list_local_models(host: str, timeout_s: float) -> List[str]:
    tags = bc.http_json(f"{host}/api/tags", timeout_s=timeout_s)
    names = []
    for m in tags.get("models", []) or []:
        name = m.get("name")
        if isinstance(name, str) and name.strip() and "cloud" not in name.lower():
            names.append(name.strip())
    return sorted(names)


def run_task(host: str, model: str, task: Dict[str, Any], tools_lib: Dict[str, Any],
             temperature: float, seed: int, timeout_s: float) -> Tuple[Dict[str, Any], float, Optional[str]]:
    tools = [tools_lib[name] for name in task.get("tools", []) if name in tools_lib]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "tools": tools,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed},
    }
    t0 = time.perf_counter()
    try:
        resp = bc.http_json(f"{host}/api/chat", method="POST", body=body, timeout_s=timeout_s)
    except Exception as e:  # noqa: BLE001
        return {}, (time.perf_counter() - t0) * 1000.0, str(e)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return resp.get("message", {}) or {}, latency_ms, None


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------


def aggregate(rows: List[Dict[str, Any]], latencies: List[float]) -> Dict[str, Any]:
    total = len(rows)
    passed = sum(1 for r in rows if r.get("passed"))
    call_rows = [r for r in rows if r["expected_call"]]
    abstain_rows = [r for r in rows if not r["expected_call"]]

    def _rate(items: List[bool]) -> Optional[float]:
        return (100.0 * sum(1 for x in items if x) / len(items)) if items else None

    tool_select = _rate([bool(r["tool_ok"]) for r in call_rows])
    arg_pct = (100.0 * sum(r["arg_score"] for r in call_rows) / len(call_rows)) if call_rows else None
    abstain_pct = _rate([bool(r["passed"]) for r in abstain_rows])
    return {
        "total": total,
        "passed": passed,
        "overall_pct": (100.0 * passed / total) if total else None,
        "tool_select_pct": tool_select,
        "arg_pct": arg_pct,
        "abstain_pct": abstain_pct,
        "mean_latency_ms": bc.mean(latencies),
    }


def _yn(v: Optional[bool]) -> str:
    if v is None:
        return "—"
    return "PASS" if v else "FAIL"


def render_report(*, started_at: str, host: str, models: List[str], suite_path: str,
                  num_tasks: int, temperature: float, seed: int,
                  results: Dict[str, Dict[str, Any]], pc_metadata: Dict[str, Any],
                  hardware: Dict[str, Any]) -> str:
    anon = bc.anonymized_pc_profile(pc_metadata, "ollama")
    lines: List[str] = []
    lines.append("# Agentic Tool-Use Eval Report")
    lines.append("")
    lines.append(f"- Started: `{started_at}`")
    lines.append(f"- Engine: `ollama`")
    lines.append(f"- Kind: `eval-agentic`")
    lines.append(f"- Host: `{host}`")
    lines.append(f"- Platform: `{anon['os_family']}`")
    lines.append(f"- Machine label: `{anon['machine_label']}`")
    lines.append(f"- Suite: `{os.path.relpath(suite_path, bc.repo_root())}` (`{num_tasks}` tasks)")
    lines.append(f"- Models: `{', '.join(models)}`")
    lines.append(f"- Options: `{json.dumps({'temperature': temperature, 'seed': seed}, sort_keys=True)}`")
    lines.append("")

    bc._render_hardware_section(lines, pc_metadata, hardware)

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "Deterministic local grading of Ollama tool calls. **Overall** = tasks fully correct. "
        "**Tool-select** = right tool chosen (call-expected tasks). **Arg** = mean argument correctness. "
        "**Abstention** = correctly did *not* call a tool when none applied. Higher is better."
    )
    lines.append("")
    lines.append("| Model | Passed | Overall % | Tool-select % | Arg % | Abstention % | Mean latency ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model in models:
        agg = results[model]["agg"]
        lines.append(
            "| " + " | ".join([
                bc.md_escape(model),
                f"{agg['passed']}/{agg['total']}",
                bc.fmt_float(agg["overall_pct"], 1),
                bc.fmt_float(agg["tool_select_pct"], 1),
                bc.fmt_float(agg["arg_pct"], 1),
                bc.fmt_float(agg["abstain_pct"], 1),
                bc.fmt_float(agg["mean_latency_ms"], 1),
            ]) + " |"
        )
    lines.append("")

    lines.append("## Details")
    lines.append("")
    for model in models:
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Task | Category | Expected | Got | Result |")
        lines.append("|---|---|---|---|:--:|")
        for r in results[model]["rows"]:
            if r["expected_call"]:
                expected = f"{r['expected_tool']}(...)"
            else:
                expected = "(no tool call)"
            if r.get("error"):
                got = f"ERROR: {r['error']}"
            elif r["called"]:
                args = json.dumps(r["got_args"], ensure_ascii=False) if r["got_args"] else "{}"
                got = f"{r['got_tool']}({args})"
            else:
                got = "(no tool call)"
            lines.append(
                "| " + " | ".join([
                    bc.md_escape(str(r["id"])),
                    bc.md_escape(r["category"]),
                    bc.md_escape(expected),
                    bc.md_escape(got[:160]),
                    _yn(r["passed"]),
                ]) + " |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    bc.bootstrap_venv_if_needed(__file__, argv)
    session_t0 = time.perf_counter()

    p = argparse.ArgumentParser(description="Agentic tool-use eval for local Ollama models.")
    p.add_argument("--no-venv", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--suite", default=_DEFAULT_SUITE, help=f"Task suite JSON (default: {_DEFAULT_SUITE})")
    p.add_argument("--models", default=None, help="Comma-separated models. If omitted, auto-discover local models.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    host = args.host.rstrip("/")
    started_at = bc.iso_now_local()
    bc.log(f"Starting agentic tool-use eval at {started_at}")

    suite = bc.read_json_file(args.suite)
    tools_lib: Dict[str, Any] = suite.get("tools", {})
    tasks: List[Dict[str, Any]] = suite.get("tasks", [])
    if not tasks:
        raise SystemExit(f"Suite has no tasks: {args.suite}")
    bc.log(f"Loaded {len(tasks)} tasks from {args.suite}")

    installed = set(_list_local_models(host, args.timeout_s))
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        models = [m for m in requested if m in installed]
        missing = [m for m in requested if m not in installed]
        if missing:
            bc.log(f"Skipping {len(missing)} model(s) not installed locally: {', '.join(missing)}")
    else:
        bc.log("Discovering local models from /api/tags")
        models = sorted(installed)
    if not models:
        raise SystemExit("No installed models selected. Pull them first or pass installed --models.")
    bc.log(f"Evaluating {len(models)} model(s): {', '.join(models)}")

    pc_metadata = bc.get_pc_metadata()
    machine_label, machine_slug = bc.machine_label_parts(pc_metadata, "ollama")
    hardware = bc.get_hardware_detail(pc_metadata)

    status_dir = os.path.join(bc.repo_root(), "reports")
    run_status: Dict[str, Any] = {
        "running": True, "engine": "ollama-eval", "machine_label": machine_label,
        "started": started_at, "pid": os.getpid(), "models": models,
        "total": len(models), "completed": 0, "current": models[0], "phase": "eval",
    }
    bc.write_bench_status(status_dir, run_status)

    results: Dict[str, Dict[str, Any]] = {}
    try:
        for mi, model in enumerate(models):
            run_status.update(current=model, completed=mi)
            bc.write_bench_status(status_dir, run_status)
            bc.log(f"Evaluating {model}")
            rows: List[Dict[str, Any]] = []
            latencies: List[float] = []
            for task in tasks:
                message, latency_ms, err = run_task(
                    host, model, task, tools_lib, args.temperature, args.seed, args.timeout_s)
                latencies.append(latency_ms)
                if err is not None:
                    rows.append({
                        "id": task.get("id"), "category": task.get("category", "?"),
                        "expected_call": task.get("expect", {}).get("call", True),
                        "expected_tool": task.get("expect", {}).get("tool"),
                        "called": False, "got_tool": None, "got_args": None,
                        "passed": False, "tool_ok": False, "arg_score": 0.0, "error": err,
                    })
                    continue
                row = grade(task, message)
                rows.append(row)
            agg = aggregate(rows, latencies)
            results[model] = {"rows": rows, "agg": agg}
            bc.log(f"  {model}: {agg['passed']}/{agg['total']} passed "
                   f"(overall {bc.fmt_float(agg['overall_pct'], 1)}%, "
                   f"tool-select {bc.fmt_float(agg['tool_select_pct'], 1)}%, "
                   f"args {bc.fmt_float(agg['arg_pct'], 1)}%)")
    finally:
        bc.clear_bench_status(status_dir)

    out_path = args.out
    if out_path is None:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join("reports", machine_slug, f"eval-agentic-{ts}.md")
    bc.ensure_dir(os.path.dirname(out_path) or ".")
    report = render_report(
        started_at=started_at, host=host, models=models, suite_path=args.suite,
        num_tasks=len(tasks), temperature=args.temperature, seed=args.seed,
        results=results, pc_metadata=pc_metadata, hardware=hardware,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")

    bc.log(f"Report written: {out_path}")
    bc.log(f"Eval session completed in {bc.fmt_duration(time.perf_counter() - session_t0)}")
    sys.stdout.write(f"{out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
