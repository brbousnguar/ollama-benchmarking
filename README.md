# benchmarking-local-ai

Local benchmarking helpers for locally hosted LLMs. Two inference engines are
supported:

- **Ollama** (`scripts/ollama_bench.py`) — all platforms (Windows, macOS, Linux).
- **Foundry Local** (`scripts/foundry_bench.py`) — the Windows machines. (Foundry
  Local also runs on Apple silicon, but in this project macOS stays on Ollama.)

Both write the same Markdown report format, so `scripts/compare_latest_reports.py`
compares engines and machines side by side.

## Shared module

`scripts/bench_common.py` holds the engine-agnostic logic: venv bootstrap, PC +
hardware metadata collection, resource sampling, the per-run result type, and the
Markdown report renderer. Both engine scripts import it. It has no third-party
dependencies; richer Windows hardware data is read via PowerShell `Get-CimInstance`
and `dxdiag` (with the legacy `wmic` kept only as a fallback, since Windows 11 no
longer ships `wmic`).

## Hardware detail and KPIs

Every report includes a **Hardware** section describing *which chips do the work*:

- **CPU**: model, max clock, core/thread count, L2/L3 cache.
- **Memory**: type (e.g. LPDDR5/DDR5), speed (MT/s), channel count, and the
  **theoretical memory bandwidth** (GB/s) — the dominant predictor of LLM decode
  speed. On Apple silicon this is reported as unified memory.
- **GPU**: true name plus **dedicated and shared memory** (e.g. an Intel Arc iGPU's
  shared-memory budget), read from `dxdiag` because the `Win32_VideoController`
  `AdapterRAM` field is capped at ~4 GB. NVIDIA GPUs also get live VRAM/util/temp/
  power samples via `nvidia-smi`.
- **NPU**: presence and name (Intel AI Boost / AMD XDNA).

Beyond raw tokens/sec, the **Summary** adds:

- **TTFT (ms)** — time to first token, measured client-side from the stream.
- **Inter-token latency (ms)** — per-token decode latency (in the Details tables).
- **Effective memory bandwidth (GB/s)** — `model_size × gen tok/s`, an estimate of
  the bandwidth actually achieved during decode. Compare it against the theoretical
  ceiling to see how efficiently a machine is being used.
- **Memory-bandwidth utilization (%)** — effective ÷ theoretical bandwidth. Since
  decode is bandwidth-bound, this is the headline efficiency KPI: how close a run
  gets to the silicon's ceiling. The theoretical ceiling is read from SMBIOS on
  Windows and from a per-chip lookup on Apple Silicon (e.g. M4 Pro ≈ 273 GB/s), so
  the KPI is populated on Macs too. It assumes *dense* weight streaming, so
  Mixture-of-Experts / elastic models (which stream only active params per token)
  can read above 100% — a useful signal that a model is sparse rather than dense.
- **Throughput per GB (tok/s per GiB)** — decode throughput normalized by the
  model's resident footprint, so small and large models can be compared on
  efficiency-per-byte rather than raw speed.

## Model configs

Two curated model sets ship with the repo:

- **`ollama-bench.json`** (default) — a cross-platform, current-generation set of
  GGUF models spanning 4B–30B (`qwen3`, `granite4.1`, `deepseek-r1`, `gpt-oss`,
  `qwen3-coder`). These tags run on Windows, macOS and Linux.
- **`ollama-bench-mac-mlx.json`** — Apple Silicon **MLX** builds (`gemma4:*-mlx`,
  `qwen3.5:9b-mlx-bf16`, `qwen3.6:35b-mlx`). MLX tags require Ollama 0.31+ and run
  only on Apple Silicon, where they are materially faster than the GGUF/Metal path
  (e.g. Gemma 4 gains multi-token prediction). On a Mac, benchmark the MLX set with:

  ```bash
  python3 scripts/ollama_bench.py --config ollama-bench-mac-mlx.json
  ```

  Comparing the two configs on the same Mac shows the MLX vs GGUF speedup directly.

Larger local models (e.g. `nemotron-3-nano:30b`, `llama3.3:70b`) are left out of the
defaults so smaller machines don't auto-pull them; benchmark them ad hoc with
`--models nemotron-3-nano:30b`.

## Ollama benchmark

Script: `scripts/ollama_bench.py`

Default config: `ollama-bench.json`

Benchmarks one or more local Ollama models via the HTTP API (streaming so TTFT can
be measured) and writes a Markdown report including:

- Generation tokens/sec (from Ollama `eval_count` / `eval_duration`)
- Prompt tokens/sec (from `prompt_eval_count` / `prompt_eval_duration`)
- TTFT and inter-token latency (client-side, from the streamed response)
- Load/total/eval durations (server-side) and wall time (client-side)
- The Hardware section and KPIs described above
- Observed resource samples around benchmark runs: peak CPU/RAM and NVIDIA GPU/VRAM metrics when `nvidia-smi` is available
- Verbose console progress logs for discovery, warmup, each measured run, report writing, and total session time
- Local-only model selection by default; Ollama cloud models are skipped unless `--include-cloud` is provided
- Model selection order: `--models`, then `ollama-bench.json`, then Ollama auto-discovery
- Missing selected models are pulled automatically with `ollama pull <model>`
- Reports use an anonymized machine label instead of personal identifiers such as username, computer name, exact local paths, or exact hardware model names

By default, reports are grouped by an engine-prefixed anonymized machine label under
`reports/<engine>-<machine-label>/`, for example
`reports/ollama-windows11-arc-64gb/ollama-bench-20260423-151324.md` or
`reports/ollama-apple-m4-24gb/ollama-bench-20260423-151324.md`. (Older reports
created before the engine prefix remain under their original `reports/<machine>/`
folders and are still picked up by the comparison.)

## Foundry Local benchmark

Script: `scripts/foundry_bench.py`

Default config: `foundry-bench.json`

[Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/) is
Microsoft's on-device inference runtime. Install it on Windows with:

```powershell
winget install Microsoft.FoundryLocal
```

Foundry exposes an OpenAI-compatible HTTP API but returns no server-side timing
counters, so throughput, TTFT, and inter-token latency are all measured client-side
from a streaming chat completion (`stream_options.include_usage` provides the token
counts). Then:

```powershell
py -3 scripts\foundry_bench.py
```

The script starts the Foundry service if needed and auto-discovers its endpoint via
`foundry service status` (override with `--endpoint http://127.0.0.1:PORT`). Model
selection order is `--models`, then `foundry-bench.json`, then `/v1/models`
discovery; missing models are fetched with `foundry model download <model>`
(`--no-download` to skip). Foundry model names come from the Foundry catalog
(`foundry model list`), e.g. `phi-3.5-mini`, `qwen2.5-7b` — not the Ollama names.

Reports are written to `reports/foundry-<machine-label>/foundry-bench-<timestamp>.md`.
Because macOS stays on Ollama in this project, the script exits on non-Windows hosts
unless `--allow-non-windows` is passed.

## Compare machines

Script: `scripts/compare_latest_reports.py`

After collecting benchmark reports from more than one PC, generate a comparison from the newest report in each `reports/<machine-label>/` folder:

```bash
python3 scripts/compare_latest_reports.py
```

The comparison is written to `reports/comparisons/latest-comparison-<timestamp>.md` and includes:

- The source report chosen for each machine
- The fastest machine per common model
- Generation throughput per model, with percentage difference from the first machine in the report
- Mean wall time per model

To choose a specific output path:

```bash
python3 scripts/compare_latest_reports.py --out reports/comparisons/latest.md
```

### Usage

The script automatically creates and uses a local `.venv` on first run. It has no third-party Python dependencies, so the venv is only used to keep execution isolated and consistent across machines.

If `ollama-bench.json` exists, the script uses the `models` array from that file by default. Example:

```json
{
  "models": [
    "llama3.2:1b",
    "qwen2.5-coder:1.5b"
  ]
}
```

On macOS or Linux, use `python3` because stock macOS does not provide a `python` command by default:

```bash
python3 scripts/ollama_bench.py
```

The script is also executable on macOS/Linux, so this is equivalent:

```bash
./scripts/ollama_bench.py
```

On Windows, use Python Launcher or your existing Python 3 command:

```powershell
py -3 scripts\ollama_bench.py
```

Auto-discover models from the local Ollama instance and run 3 measured runs (1 warmup):

```bash
python3 scripts/ollama_bench.py
```

Benchmark specific models:

```bash
python3 scripts/ollama_bench.py --models llama3,phi3
```

Use a different config file:

```bash
python3 scripts/ollama_bench.py --config my-models.json
```

Cloud models are skipped by default, including names such as `kimi-k2.6:cloud` or `gpt-oss:20b-cloud`. To include them:

```bash
python3 scripts/ollama_bench.py --include-cloud
```

Control generation length / context:

```bash
python3 scripts/ollama_bench.py --models llama3 --num-predict 512 --num-ctx 8192
```

Custom prompt from a file, custom report path:

```bash
python3 scripts/ollama_bench.py --prompt-file prompts/throughput.txt --out reports/my-run.md
```

`--out` overrides the machine-grouped default path.

You do not need to activate `.venv` manually. If it is already activated, the script detects that and runs directly inside it:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\ollama_bench.py
```

To skip the automatic venv for a one-off run:

```bash
python3 scripts/ollama_bench.py --no-venv
```
