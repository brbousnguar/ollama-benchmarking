# Ollama Benchmark Comparison

- Generated: `2026-04-26T18:16:56+02:00`
- Compared machines: `apple-m4-24gb, windows11-cpu-unknown-ram, windows11-nvidia-32gb`
- Report selection: `latest ollama-bench-*.md from each reports/<machine>/ folder`
- Common models: `6`

## Source Reports

| Machine | Platform | Started | Report |
|---|---|---|---|
| apple-m4-24gb | macos | `2026-04-26T08:18:17+02:00` | `reports/apple-m4-24gb/ollama-bench-20260426-081818.md` |
| windows11-cpu-unknown-ram | windows11 | `2026-04-26T17:53:44+02:00` | `reports/windows11-cpu-unknown-ram/ollama-bench-20260426-175818.md` |
| windows11-nvidia-32gb | windows11 | `2026-04-25T10:06:54+02:00` | `reports/windows11-nvidia-32gb/ollama-bench-20260425-100703.md` |

## Fastest By Model

| Model | Fastest gen tok/s | Best machine | Slowest gen tok/s | Speed ratio |
|---|---:|---|---:|---:|
| gemma3:1b | 91.40 | apple-m4-24gb | 34.59 | 2.64x |
| gemma3:270m | 217.28 | apple-m4-24gb | 70.85 | 3.07x |
| llama3:8b | 20.45 | apple-m4-24gb | 9.02 | 2.27x |
| mistral:7b | 19.59 | apple-m4-24gb | 7.11 | 2.76x |
| phi3:mini | 37.32 | apple-m4-24gb | 18.57 | 2.01x |
| qwen2.5:7b | 19.85 | apple-m4-24gb | 8.61 | 2.31x |

## Generation Throughput vs apple-m4-24gb

| Model | apple-m4-24gb | windows11-cpu-unknown-ram | windows11-nvidia-32gb |
|---|---:|---:|---:|
| gemma3:1b | 91.40 | 34.59 (-62.2%) | 67.03 (-26.7%) |
| gemma3:270m | 217.28 | 70.85 (-67.4%) | 127.38 (-41.4%) |
| llama3:8b | 20.45 | 9.02 (-55.9%) | 12.48 (-39.0%) |
| mistral:7b | 19.59 | 7.11 (-63.7%) | 14.59 (-25.5%) |
| phi3:mini | 37.32 | 18.57 (-50.2%) | 28.80 (-22.8%) |
| qwen2.5:7b | 19.85 | 8.61 (-56.6%) | 13.86 (-30.2%) |

## Wall Time Mean

| Model | apple-m4-24gb | windows11-cpu-unknown-ram | windows11-nvidia-32gb |
|---|---:|---:|---:|
| gemma3:1b | 3.12 | 10.51 | 5.19 |
| gemma3:270m | 0.59 | 4.51 | 3.98 |
| llama3:8b | 14.16 | 34.11 | 25.58 |
| mistral:7b | 13.20 | 38.75 | 20.57 |
| phi3:mini | 7.85 | 20.63 | 10.70 |
| qwen2.5:7b | 14.63 | 35.91 | 21.94 |

