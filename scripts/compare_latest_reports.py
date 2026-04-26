#!/usr/bin/env python3
"""
Compare the latest Ollama benchmark report from each machine folder.

Reads Markdown reports written by scripts/ollama_bench.py and writes a
Markdown comparison focused on common model throughput and wall time.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


_REPORT_RE = re.compile(r"^ollama-bench-(\d{8}-\d{6})\.md$")


@dataclass(frozen=True)
class ModelSummary:
    ok_total: str
    gen_tps_mean: Optional[float]
    gen_tps_p50: Optional[float]
    gen_tps_p90: Optional[float]
    gen_tps_stdev: Optional[float]
    prompt_tps_mean: Optional[float]
    total_s_mean: Optional[float]
    wall_s_mean: Optional[float]


@dataclass(frozen=True)
class BenchmarkReport:
    path: str
    started: str
    machine_label: str
    platform: str
    models: Dict[str, ModelSummary]


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _strip_cell(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def _split_md_row(line: str) -> List[str]:
    return [_strip_cell(cell) for cell in line.strip().strip("|").split("|")]


def _parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _metadata_value(lines: Iterable[str], key: str) -> str:
    prefix = f"- {key}: `"
    for line in lines:
        if line.startswith(prefix) and line.rstrip().endswith("`"):
            return line[len(prefix) :].rstrip()[0:-1]
    return "-"


def _parse_summary(lines: List[str]) -> Dict[str, ModelSummary]:
    try:
        start = lines.index("## Summary")
    except ValueError as e:
        raise RuntimeError("Missing '## Summary' section") from e

    rows: Dict[str, ModelSummary] = {}
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("|---") or stripped.startswith("| Model "):
            continue

        cells = _split_md_row(stripped)
        if len(cells) < 9:
            continue
        model = cells[0]
        rows[model] = ModelSummary(
            ok_total=cells[1],
            gen_tps_mean=_parse_float(cells[2]),
            gen_tps_p50=_parse_float(cells[3]),
            gen_tps_p90=_parse_float(cells[4]),
            gen_tps_stdev=_parse_float(cells[5]),
            prompt_tps_mean=_parse_float(cells[6]),
            total_s_mean=_parse_float(cells[7]),
            wall_s_mean=_parse_float(cells[8]),
        )
    if not rows:
        raise RuntimeError("Summary table did not contain any model rows")
    return rows


def _parse_report(path: str) -> BenchmarkReport:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    machine_label = _metadata_value(lines, "Machine label")
    return BenchmarkReport(
        path=path,
        started=_metadata_value(lines, "Started"),
        machine_label=machine_label if machine_label != "-" else os.path.basename(os.path.dirname(path)),
        platform=_metadata_value(lines, "Platform"),
        models=_parse_summary(lines),
    )


def _latest_report_paths(reports_dir: str) -> List[str]:
    if not os.path.isdir(reports_dir):
        raise RuntimeError(f"Reports directory does not exist: {reports_dir}")

    latest: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(reports_dir)):
        machine_dir = os.path.join(reports_dir, name)
        if not os.path.isdir(machine_dir) or name == "comparisons":
            continue

        candidates: List[Tuple[str, str]] = []
        for filename in os.listdir(machine_dir):
            match = _REPORT_RE.match(filename)
            if match:
                candidates.append((match.group(1), os.path.join(machine_dir, filename)))
        if candidates:
            latest.append(max(candidates, key=lambda item: item[0]))

    return [path for _, path in sorted(latest, key=lambda item: item[1])]


def _fmt_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _pct_diff(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None or baseline == 0:
        return None
    return ((value - baseline) / baseline) * 100.0


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|")


def _relative_path(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _render_report(reports: List[BenchmarkReport]) -> str:
    now = _dt.datetime.now().astimezone().replace(microsecond=0).isoformat()
    machines = [r.machine_label for r in reports]
    common_models = sorted(set.intersection(*(set(r.models) for r in reports)))
    all_models = sorted(set.union(*(set(r.models) for r in reports)))

    lines: List[str] = []
    lines.append("# Ollama Benchmark Comparison")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Compared machines: `{', '.join(machines)}`")
    lines.append(f"- Report selection: `latest ollama-bench-*.md from each reports/<machine>/ folder`")
    lines.append(f"- Common models: `{len(common_models)}`")
    lines.append("")

    lines.append("## Source Reports")
    lines.append("")
    lines.append("| Machine | Platform | Started | Report |")
    lines.append("|---|---|---|---|")
    for report in reports:
        rel = _relative_path(report.path, _repo_root())
        lines.append(
            f"| {_md_escape(report.machine_label)} | {_md_escape(report.platform)} | `{_md_escape(report.started)}` | `{_md_escape(rel)}` |"
        )
    lines.append("")

    if len(reports) < 2:
        lines.append("At least two machine reports are needed for a comparison.")
        return "\n".join(lines)

    if not common_models:
        lines.append("No common models were found across the latest machine reports.")
        return "\n".join(lines)

    lines.append("## Fastest By Model")
    lines.append("")
    lines.append("| Model | Fastest gen tok/s | Best machine | Slowest gen tok/s | Speed ratio |")
    lines.append("|---|---:|---|---:|---:|")
    for model in common_models:
        values = [
            (report.machine_label, report.models[model].gen_tps_mean)
            for report in reports
            if report.models[model].gen_tps_mean is not None
        ]
        if not values:
            continue
        fastest = max(values, key=lambda item: item[1] or 0.0)
        slowest = min(values, key=lambda item: item[1] or 0.0)
        ratio = (fastest[1] / slowest[1]) if fastest[1] is not None and slowest[1] else None
        lines.append(
            f"| {_md_escape(model)} | {_fmt_float(fastest[1])} | {_md_escape(fastest[0])} | {_fmt_float(slowest[1])} | {_fmt_float(ratio)}x |"
        )
    lines.append("")

    baseline = reports[0]
    lines.append(f"## Generation Throughput vs {baseline.machine_label}")
    lines.append("")
    header = ["Model", *[r.machine_label for r in reports]]
    lines.append("| " + " | ".join(_md_escape(h) for h in header) + " |")
    lines.append("|---" + "|---:" * len(reports) + "|")
    for model in common_models:
        row = [_md_escape(model)]
        baseline_value = baseline.models[model].gen_tps_mean
        for report in reports:
            value = report.models[model].gen_tps_mean
            diff = _pct_diff(value, baseline_value)
            if report is baseline:
                row.append(f"{_fmt_float(value)}")
            else:
                row.append(f"{_fmt_float(value)} ({_fmt_pct(diff)})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Wall Time Mean")
    lines.append("")
    lines.append("| " + " | ".join(_md_escape(h) for h in header) + " |")
    lines.append("|---" + "|---:" * len(reports) + "|")
    for model in common_models:
        row = [_md_escape(model)]
        for report in reports:
            row.append(_fmt_float(report.models[model].wall_s_mean))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    missing = sorted(set(all_models) - set(common_models))
    if missing:
        lines.append("## Models Not Compared")
        lines.append("")
        lines.append("These models were not present in every selected report:")
        lines.append("")
        for model in missing:
            present = [report.machine_label for report in reports if model in report.models]
            lines.append(f"- `{model}`: present in `{', '.join(present)}`")
        lines.append("")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the latest Ollama benchmark report from each machine.")
    parser.add_argument("--reports-dir", default=os.path.join(_repo_root(), "reports"), help="Reports root directory")
    parser.add_argument("--out", default=None, help="Output Markdown path")
    args = parser.parse_args(argv)

    reports_dir = os.path.abspath(args.reports_dir)
    paths = _latest_report_paths(reports_dir)
    if not paths:
        raise SystemExit(f"No benchmark reports found under {reports_dir}")

    reports = [_parse_report(path) for path in paths]
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or os.path.join(reports_dir, "comparisons", f"latest-comparison-{timestamp}.md")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    comparison = _render_report(reports)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(comparison)
        f.write("\n")

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
