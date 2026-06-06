# Ollama Benchmark Report

- Started: `2026-06-06T08:31:41+02:00`
- Engine: `ollama`
- Host: `http://localhost:11434`
- Python: `3.14.4`
- Python env: `venv`
- Platform: `windows11`
- Machine label: `ollama-windows11-arc-63gb`
- Models: `qwen3:4b, deepseek-r1:8b`
- Runs per model: `1` (warmup: `0`)
- Timeout (s): `600.0`
- keep_alive: `5m`
- Options: `{"num_predict": 256, "seed": 42, "temperature": 0.0}`
- Prompt: `default: integers 1..2000`

## PC

| Field | Value |
|---|---|
| Machine label | `ollama-windows11-arc-63gb` |
| OS family | `windows11` |
| CPU family | `intel` |
| GPU family | `arc` |
| RAM class | `63gb` |
| CPU cores / logical processors | `16 / 16` |

## Hardware

| Component | Detail |
|---|---|
| CPU | Intel(R) Core(TM) Ultra 7 255H, 2000 MHz max, 16C/16T, L2 28672 KB, L3 24576 KB |
| Memory | LPDDR5, 5600 MT/s, 2 ch, ~89.6 GB/s theoretical, 63.43 GB total |
| GPU | Intel(R) Arc(TM) 140T GPU (32GB), 128 MB dedicated, 37025 MB shared, driver 32.0.101.8508 |
| NPU | Intel(R) AI Boost |

## Observed resources

Resource values are point-in-time samples captured around warmup and measured runs; GPU/VRAM rows require NVIDIA `nvidia-smi`.

- Peak observed CPU load: `23.00%`
- Peak observed RAM used: `53.28 GB` / `63.43 GB`
- NVIDIA GPU samples: `not available`

## Summary

Eff. mem BW (GB/s) ~= model_size x gen tok/s, an estimate of achieved memory bandwidth during decode (theoretical ~89.6 GB/s).

| Model | OK/Total | Gen tok/s (mean) | Gen tok/s (p50) | Gen tok/s (p90) | Gen tok/s (stdev) | Prompt tok/s (mean) | TTFT ms (mean) | Eff BW GB/s (mean) | Total s (mean) | Wall s (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3:4b | 1/1 | 16.35 | 16.35 | 16.35 | - | 168.10 | 2538.4 | 40.8 | 16.24 | 18.29 |
| deepseek-r1:8b | 1/1 | 9.65 | 9.65 | 9.65 | - | 95.94 | 11386.3 | 50.4 | 35.76 | 37.83 |

## Details

### qwen3:4b

Model size on disk: `2.33 GB`

| Run | OK | Gen tok/s | Prompt tok/s | TTFT ms | Inter-tok ms | Gen toks | Prompt toks | Eval s | Load s | Total s | Wall s | Error |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Y | 16.35 | 168.10 | 2538.4 | 61.76 | 256 | 53 | 15.66 | 0.17 | 16.24 | 18.29 |  |

### deepseek-r1:8b

Model size on disk: `4.87 GB`

| Run | OK | Gen tok/s | Prompt tok/s | TTFT ms | Inter-tok ms | Gen toks | Prompt toks | Eval s | Load s | Total s | Wall s | Error |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Y | 9.65 | 95.94 | 11386.3 | 104.50 | 256 | 45 | 26.54 | 8.58 | 35.76 | 37.83 |  |

