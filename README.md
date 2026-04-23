# benchmarking-local-ai

Local benchmarking helpers for locally hosted LLMs.

## Ollama benchmark

Script: `scripts/ollama_bench.py`

Benchmarks one or more local Ollama models via the HTTP API and writes a Markdown report including:

- Generation tokens/sec (from Ollama `eval_count` / `eval_duration`)
- Prompt tokens/sec (from `prompt_eval_count` / `prompt_eval_duration`)
- Load/total/eval durations (server-side) and wall time (client-side)
- PC metadata for comparing runs across machines: computer/model, OS, CPU, RAM, and GPU names
- Observed resource samples around benchmark runs: peak CPU/RAM and NVIDIA GPU/VRAM metrics when `nvidia-smi` is available
- Verbose console progress logs for discovery, warmup, each measured run, report writing, and total session time
- Local-only model selection by default; Ollama cloud models are skipped unless `--include-cloud` is provided

By default, reports are grouped by machine under `reports/<machine>/`, for example `reports/my-laptop/ollama-bench-20260423-151324.md`.

### Usage

The script automatically creates and uses a local `.venv` on first run. It has no third-party Python dependencies, so the venv is only used to keep execution isolated and consistent across machines.

Auto-discover models from the local Ollama instance and run 3 measured runs (1 warmup):

```bash
python scripts/ollama_bench.py
```

Benchmark specific models:

```bash
python scripts/ollama_bench.py --models llama3,phi3
```

Cloud models are skipped by default, including names such as `kimi-k2.6:cloud` or `gpt-oss:20b-cloud`. To include them:

```bash
python scripts/ollama_bench.py --include-cloud
```

Control generation length / context:

```bash
python scripts/ollama_bench.py --models llama3 --num-predict 512 --num-ctx 8192
```

Custom prompt from a file, custom report path:

```bash
python scripts/ollama_bench.py --prompt-file prompts/throughput.txt --out reports/my-run.md
```

`--out` overrides the machine-grouped default path.

On macOS or Linux, use `python3` if `python` is not mapped to Python 3. On Windows, `py -3 scripts\ollama_bench.py` is also fine.

You do not need to activate `.venv` manually. If it is already activated, the script detects that and runs directly inside it:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\ollama_bench.py
```

To skip the automatic venv for a one-off run:

```bash
python scripts/ollama_bench.py --no-venv
```
