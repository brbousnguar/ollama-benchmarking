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
from typing import Any, Dict, Iterable, List, Optional, Tuple


_REPORT_RE = re.compile(r"^(ollama|foundry)-bench-(\d{8}-\d{6})\.md$")


@dataclass(frozen=True)
class ModelSummary:
    ok_total: str
    gen_tps_mean: Optional[float]
    gen_tps_p50: Optional[float]
    gen_tps_p90: Optional[float]
    gen_tps_stdev: Optional[float]
    prompt_tps_mean: Optional[float]
    ttft_ms_mean: Optional[float]
    eff_bw_mean: Optional[float]
    total_s_mean: Optional[float]
    wall_s_mean: Optional[float]


@dataclass(frozen=True)
class BenchmarkReport:
    path: str
    engine: str
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


# Map normalized summary header cells to ModelSummary fields. This makes the
# parser tolerant of new columns and backward-compatible with older reports
# that lack the TTFT / effective-bandwidth columns.
_COLUMN_FIELDS = {
    "ok/total": "ok_total",
    "gen tok/s (mean)": "gen_tps_mean",
    "gen tok/s (p50)": "gen_tps_p50",
    "gen tok/s (p90)": "gen_tps_p90",
    "gen tok/s (stdev)": "gen_tps_stdev",
    "prompt tok/s (mean)": "prompt_tps_mean",
    "ttft ms (mean)": "ttft_ms_mean",
    "eff bw gb/s (mean)": "eff_bw_mean",
    "total s (mean)": "total_s_mean",
    "wall s (mean)": "wall_s_mean",
}
_FLOAT_FIELDS = {
    "gen_tps_mean", "gen_tps_p50", "gen_tps_p90", "gen_tps_stdev",
    "prompt_tps_mean", "ttft_ms_mean", "eff_bw_mean", "total_s_mean", "wall_s_mean",
}


def _parse_summary(lines: List[str]) -> Dict[str, ModelSummary]:
    try:
        start = lines.index("## Summary")
    except ValueError as e:
        raise RuntimeError("Missing '## Summary' section") from e

    header_fields: Optional[List[str]] = None
    rows: Dict[str, ModelSummary] = {}
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if not stripped.startswith("|"):
            continue
        cells = _split_md_row(stripped)
        if not cells:
            continue
        first = cells[0].strip().lower()
        if first == "model":
            # Header row: map each column (after Model) to a field name.
            header_fields = [_COLUMN_FIELDS.get(c.strip().lower()) for c in cells[1:]]
            continue
        if set(cells[0]) <= {"-", ":"}:  # separator row like |---|
            continue
        if header_fields is None:
            continue

        values: Dict[str, Any] = {field: None for field in _COLUMN_FIELDS.values()}
        for field, cell in zip(header_fields, cells[1:]):
            if field is None:
                continue
            values[field] = _parse_float(cell) if field in _FLOAT_FIELDS else cell
        rows[cells[0]] = ModelSummary(
            ok_total=values.get("ok_total") or "-",
            gen_tps_mean=values.get("gen_tps_mean"),
            gen_tps_p50=values.get("gen_tps_p50"),
            gen_tps_p90=values.get("gen_tps_p90"),
            gen_tps_stdev=values.get("gen_tps_stdev"),
            prompt_tps_mean=values.get("prompt_tps_mean"),
            ttft_ms_mean=values.get("ttft_ms_mean"),
            eff_bw_mean=values.get("eff_bw_mean"),
            total_s_mean=values.get("total_s_mean"),
            wall_s_mean=values.get("wall_s_mean"),
        )
    if not rows:
        raise RuntimeError("Summary table did not contain any model rows")
    return rows


def _parse_report(path: str) -> BenchmarkReport:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    machine_label = _metadata_value(lines, "Machine label")
    engine = _metadata_value(lines, "Engine")
    if engine == "-":
        # Older reports have no Engine field; infer from the filename prefix.
        match = _REPORT_RE.match(os.path.basename(path))
        engine = match.group(1) if match else "ollama"
    return BenchmarkReport(
        path=path,
        engine=engine,
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
                candidates.append((match.group(2), os.path.join(machine_dir, filename)))
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

    engines = sorted({r.engine for r in reports})

    lines: List[str] = []
    lines.append("# Local LLM Benchmark Comparison")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Engines: `{', '.join(engines)}`")
    lines.append(f"- Compared machines: `{', '.join(machines)}`")
    lines.append(f"- Report selection: `latest (ollama|foundry)-bench-*.md from each reports/<machine>/ folder`")
    lines.append(f"- Common models: `{len(common_models)}`")
    lines.append("")

    lines.append("## Source Reports")
    lines.append("")
    lines.append("| Machine | Engine | Platform | Started | Report |")
    lines.append("|---|---|---|---|---|")
    for report in reports:
        rel = _relative_path(report.path, _repo_root())
        lines.append(
            f"| {_md_escape(report.machine_label)} | {_md_escape(report.engine)} | {_md_escape(report.platform)} | `{_md_escape(report.started)}` | `{_md_escape(rel)}` |"
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

    if any(report.models[m].ttft_ms_mean is not None for report in reports for m in common_models):
        lines.append("## Time To First Token (ms, mean)")
        lines.append("")
        lines.append("Lower is better. `-` means the report predates TTFT measurement.")
        lines.append("")
        lines.append("| " + " | ".join(_md_escape(h) for h in header) + " |")
        lines.append("|---" + "|---:" * len(reports) + "|")
        for model in common_models:
            row = [_md_escape(model)]
            for report in reports:
                row.append(_fmt_float(report.models[model].ttft_ms_mean, 1))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if any(report.models[m].eff_bw_mean is not None for report in reports for m in common_models):
        lines.append("## Effective Memory Bandwidth (GB/s, mean)")
        lines.append("")
        lines.append("Estimated achieved decode bandwidth (model_size x gen tok/s); higher is better.")
        lines.append("")
        lines.append("| " + " | ".join(_md_escape(h) for h in header) + " |")
        lines.append("|---" + "|---:" * len(reports) + "|")
        for model in common_models:
            row = [_md_escape(model)]
            for report in reports:
                row.append(_fmt_float(report.models[model].eff_bw_mean, 1))
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
