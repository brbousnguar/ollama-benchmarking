# Ollama Benchmark Comparison

- Generated: `2026-05-28T13:57:41+02:00`
- Compared machines: `apple-m4-24gb, apple-m4-64gb, windows11-cpu-unknown-ram, windows11-nvidia-32gb`
- Report selection: `latest ollama-bench-*.md from each reports/<machine>/ folder`
- Common models: `6`

## Source Reports

| Machine | Platform | Started | Report |
|---|---|---|---|
| apple-m4-24gb | macos | `2026-04-26T08:18:17+02:00` | `reports/apple-m4-24gb/ollama-bench-20260426-081818.md` |
| apple-m4-64gb | macos | `2026-05-03T08:55:14+02:00` | `reports/apple-m4-64gb/ollama-bench-20260503-085833.md` |
| windows11-cpu-unknown-ram | windows11 | `2026-05-11T14:56:17+02:00` | `reports/windows11-cpu-unknown-ram/ollama-bench-20260511-145620.md` |
| windows11-nvidia-32gb | windows11 | `2026-04-25T10:06:54+02:00` | `reports/windows11-nvidia-32gb/ollama-bench-20260425-100703.md` |

## Fastest By Model

| Model | Fastest gen tok/s | Best machine | Slowest gen tok/s | Speed ratio |
|---|---:|---|---:|---:|
| gemma3:1b | 142.42 | apple-m4-64gb | 40.63 | 3.51x |
| gemma3:270m | 257.37 | apple-m4-64gb | 90.53 | 2.84x |
| llama3:8b | 51.53 | apple-m4-64gb | 10.17 | 5.07x |
| mistral:7b | 49.30 | apple-m4-64gb | 9.94 | 4.96x |
| phi3:mini | 79.26 | apple-m4-64gb | 28.80 | 2.75x |
| qwen2.5:7b | 48.13 | apple-m4-64gb | 13.01 | 3.70x |

## Generation Throughput vs apple-m4-24gb

| Model | apple-m4-24gb | apple-m4-64gb | windows11-cpu-unknown-ram | windows11-nvidia-32gb |
|---|---:|---:|---:|---:|
| gemma3:1b | 91.40 | 142.42 (+55.8%) | 40.63 (-55.5%) | 67.03 (-26.7%) |
| gemma3:270m | 217.28 | 257.37 (+18.5%) | 90.53 (-58.3%) | 127.38 (-41.4%) |
| llama3:8b | 20.45 | 51.53 (+152.0%) | 10.17 (-50.3%) | 12.48 (-39.0%) |
| mistral:7b | 19.59 | 49.30 (+151.7%) | 9.94 (-49.3%) | 14.59 (-25.5%) |
| phi3:mini | 37.32 | 79.26 (+112.4%) | 29.07 (-22.1%) | 28.80 (-22.8%) |
| qwen2.5:7b | 19.85 | 48.13 (+142.5%) | 13.01 (-34.5%) | 13.86 (-30.2%) |

## Wall Time Mean

| Model | apple-m4-24gb | apple-m4-64gb | windows11-cpu-unknown-ram | windows11-nvidia-32gb |
|---|---:|---:|---:|---:|
| gemma3:1b | 3.12 | 2.29 | 8.87 | 5.19 |
| gemma3:270m | 0.59 | 0.59 | 3.87 | 3.98 |
| llama3:8b | 14.16 | 5.87 | 29.78 | 25.58 |
| mistral:7b | 13.20 | 5.94 | 29.08 | 20.57 |
| phi3:mini | 7.85 | 4.29 | 15.51 | 10.70 |
| qwen2.5:7b | 14.63 | 6.33 | 26.22 | 21.94 |

