# Ollama Benchmark Comparison

- Generated: `2026-04-26T08:46:57+02:00`
- Compared machines: `apple-m4-24gb, windows11-nvidia-32gb`
- Report selection: `latest ollama-bench-*.md from each reports/<machine>/ folder`
- Common models: `6`

## Source Reports

| Machine | Platform | Started | Report |
|---|---|---|---|
| apple-m4-24gb | macos | `2026-04-26T08:18:17+02:00` | `reports/apple-m4-24gb/ollama-bench-20260426-081818.md` |
| windows11-nvidia-32gb | windows11 | `2026-04-25T10:06:54+02:00` | `reports/windows11-nvidia-32gb/ollama-bench-20260425-100703.md` |

## Fastest By Model

| Model | Fastest gen tok/s | Best machine | Slowest gen tok/s | Speed ratio |
|---|---:|---|---:|---:|
| gemma3:1b | 91.40 | apple-m4-24gb | 67.03 | 1.36x |
| gemma3:270m | 217.28 | apple-m4-24gb | 127.38 | 1.71x |
| llama3:8b | 20.45 | apple-m4-24gb | 12.48 | 1.64x |
| mistral:7b | 19.59 | apple-m4-24gb | 14.59 | 1.34x |
| phi3:mini | 37.32 | apple-m4-24gb | 28.80 | 1.30x |
| qwen2.5:7b | 19.85 | apple-m4-24gb | 13.86 | 1.43x |

## Generation Throughput vs apple-m4-24gb

| Model | apple-m4-24gb | windows11-nvidia-32gb |
|---|---:|---:|
| gemma3:1b | 91.40 | 67.03 (-26.7%) |
| gemma3:270m | 217.28 | 127.38 (-41.4%) |
| llama3:8b | 20.45 | 12.48 (-39.0%) |
| mistral:7b | 19.59 | 14.59 (-25.5%) |
| phi3:mini | 37.32 | 28.80 (-22.8%) |
| qwen2.5:7b | 19.85 | 13.86 (-30.2%) |

## Wall Time Mean

| Model | apple-m4-24gb | windows11-nvidia-32gb |
|---|---:|---:|
| gemma3:1b | 3.12 | 5.19 |
| gemma3:270m | 0.59 | 3.98 |
| llama3:8b | 14.16 | 25.58 |
| mistral:7b | 13.20 | 20.57 |
| phi3:mini | 7.85 | 10.70 |
| qwen2.5:7b | 14.63 | 21.94 |

