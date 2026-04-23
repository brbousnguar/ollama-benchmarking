# benchmarking-local-ai

Local benchmarking helpers for locally hosted LLMs.

## Ollama benchmark

Default (Windows PowerShell): `scripts/ollama-bench.ps1`

Optional (Python): `scripts/ollama_bench.py`

Benchmarks one or more local Ollama models via the HTTP API and writes a Markdown report including:

- Generation tokens/sec (from Ollama `eval_count` / `eval_duration`)
- Prompt tokens/sec (from `prompt_eval_count` / `prompt_eval_duration`)
- Load/total/eval durations (server-side) and wall time (client-side)

### Usage

Auto-discover models from the local Ollama instance and run 3 measured runs (1 warmup):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ollama-bench.ps1
```

Benchmark specific models:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ollama-bench.ps1 -Models llama3,phi3
```

Control generation length / context:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ollama-bench.ps1 -Models llama3 -NumPredict 512 -NumCtx 8192
```

Custom prompt from a file, custom report path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\ollama-bench.ps1 -PromptFile prompts\\throughput.txt -Out reports\\my-run.md
```
