# Local LLM Benchmark Comparison

- Generated: `2026-06-07T07:43:22+02:00`
- Engines: `ollama`
- Compared machines: `apple-m4-24gb, apple-m4-64gb, ollama-windows11-arc-63gb, windows11-cpu-unknown-ram, windows11-nvidia-32gb`
- Report selection: `latest (ollama|foundry)-bench-*.md from each reports/<machine>/ folder`
- Common models: `2`

## Source Reports

| Machine | Engine | Platform | Started | Report |
|---|---|---|---|---|
| apple-m4-24gb | ollama | macos | `2026-04-26T08:18:17+02:00` | `reports/apple-m4-24gb/ollama-bench-20260426-081818.md` |
| apple-m4-64gb | ollama | macos | `2026-05-03T08:55:14+02:00` | `reports/apple-m4-64gb/ollama-bench-20260503-085833.md` |
| ollama-windows11-arc-63gb | ollama | windows11 | `2026-06-07T07:13:19+02:00` | `reports/ollama-windows11-arc-63gb/ollama-bench-20260607-071346.md` |
| windows11-cpu-unknown-ram | ollama | windows11 | `2026-05-28T13:57:57+02:00` | `reports/windows11-cpu-unknown-ram/ollama-bench-20260528-135800.md` |
| windows11-nvidia-32gb | ollama | windows11 | `2026-04-25T10:06:54+02:00` | `reports/windows11-nvidia-32gb/ollama-bench-20260425-100703.md` |

## Fastest By Model

| Model | Fastest gen tok/s | Best machine | Slowest gen tok/s | Speed ratio |
|---|---:|---|---:|---:|
| gemma3:1b | 142.42 | apple-m4-64gb | 39.83 | 3.58x |
| mistral:7b | 49.30 | apple-m4-64gb | 9.20 | 5.36x |

## Generation Throughput vs apple-m4-24gb

| Model | apple-m4-24gb | apple-m4-64gb | ollama-windows11-arc-63gb | windows11-cpu-unknown-ram | windows11-nvidia-32gb |
|---|---:|---:|---:|---:|---:|
| gemma3:1b | 91.40 | 142.42 (+55.8%) | 39.83 (-56.4%) | 48.13 (-47.3%) | 67.03 (-26.7%) |
| mistral:7b | 19.59 | 49.30 (+151.7%) | 9.20 (-53.0%) | 9.85 (-49.7%) | 14.59 (-25.5%) |

## Wall Time Mean

| Model | apple-m4-24gb | apple-m4-64gb | ollama-windows11-arc-63gb | windows11-cpu-unknown-ram | windows11-nvidia-32gb |
|---|---:|---:|---:|---:|---:|
| gemma3:1b | 3.12 | 2.29 | 4.79 | 7.93 | 5.19 |
| mistral:7b | 13.20 | 5.94 | 32.71 | 28.44 | 20.57 |

## Time To First Token (ms, mean)

Lower is better. `-` means the report predates TTFT measurement.

| Model | apple-m4-24gb | apple-m4-64gb | ollama-windows11-arc-63gb | windows11-cpu-unknown-ram | windows11-nvidia-32gb |
|---|---:|---:|---:|---:|---:|
| gemma3:1b | - | - | 3806.5 | - | - |
| mistral:7b | - | - | 4866.3 | - | - |

## Effective Memory Bandwidth (GB/s, mean)

Estimated achieved decode bandwidth (model_size x gen tok/s); higher is better.

| Model | apple-m4-24gb | apple-m4-64gb | ollama-windows11-arc-63gb | windows11-cpu-unknown-ram | windows11-nvidia-32gb |
|---|---:|---:|---:|---:|---:|
| gemma3:1b | - | - | 32.5 | - | - |
| mistral:7b | - | - | 40.2 | - | - |

## Models Not Compared

These models were not present in every selected report:

- `deepseek-r1:8b`: present in `ollama-windows11-arc-63gb`
- `gemma3:270m`: present in `apple-m4-24gb, apple-m4-64gb, windows11-cpu-unknown-ram, windows11-nvidia-32gb`
- `gpt-oss:20b`: present in `ollama-windows11-arc-63gb`
- `llama3:8b`: present in `apple-m4-24gb, apple-m4-64gb, windows11-cpu-unknown-ram, windows11-nvidia-32gb`
- `phi3:mini`: present in `apple-m4-24gb, apple-m4-64gb, windows11-cpu-unknown-ram, windows11-nvidia-32gb`
- `phi4-mini`: present in `ollama-windows11-arc-63gb`
- `qwen2.5:7b`: present in `apple-m4-24gb, apple-m4-64gb, windows11-cpu-unknown-ram, windows11-nvidia-32gb`
- `qwen3:30b`: present in `ollama-windows11-arc-63gb`
- `qwen3:4b`: present in `ollama-windows11-arc-63gb`
- `qwen3:8b`: present in `ollama-windows11-arc-63gb`

