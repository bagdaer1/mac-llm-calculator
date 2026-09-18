# mac-llm-calculator

Will this LLM run on this Mac, and how fast. One formula, fitted to measured
benchmarks, applied to every Apple Silicon chip from M1 to M6 and to the
NVIDIA and AMD hardware people compare them against.

No dependencies, no network calls, no telemetry. The data files are the whole
substance of this repo and they are readable on their own.

```bash
pip install mac-llm-calculator

maccalc llama-3-3-70b m4-pro --ram 64
# Llama 3.3 70B on M4 Pro with 64 GB: fits
#   weights        42.8 GB
#   context        1 GB
#   needs          53.2 GB including a 3 GB system reserve
#   generation     5.7 tokens per second
```

```python
from maccalc import fit_model

fit = fit_model(params_b=70.6, bits_per_weight=4.85, memory_gb=64,
                bandwidth_gbs=273, ctx_overhead_gb=3, ctx_tokens=8192)
fit.fits, fit.tokens_per_second   # True, 5.7
```

## The two questions

**Does it fit.** Weights are `params * bits_per_weight / 8`. The runtime
allocates about 15% on top, the KV cache adds a per model allowance that
scales with context length, and macOS wants roughly 3 GB. A discrete GPU with
its own VRAM gets a 1 GB reserve instead.

**How fast.** Generation reads every active weight once per token, so speed is
set by memory bandwidth, not by core count:

```
seconds_per_token = read_gb / (bandwidth_gbs * 0.915) + 0.003885
```

Both constants were fitted by least squares to the six runs in
`data/benchmarks.json`, measured on a base M4 Mac mini through the Ollama API,
median of three runs each.

| Model | Weights | Predicted | Measured | Error |
| --- | --- | --- | --- | --- |
| Llama 3.2 3B | 1.94 GB | 46.4 | 46.7 | 0.7% |
| Mistral 7B | 4.37 GB | 22.9 | 22.8 | 0.5% |
| Qwen 2.5 7B | 4.61 GB | 21.8 | 22.3 | 2.2% |
| Llama 3.1 8B | 4.85 GB | 20.8 | 21.2 | 1.8% |
| DeepSeek R1 8B | 4.85 GB | 20.8 | 20.0 | 4.0% |
| Qwen 2.5 14B | 8.97 GB | 11.7 | 11.7 | 0.2% |

The second term is the point of the model. Without it the same arithmetic
claims an RTX 5090 generates over 300 tokens per second on an 8B model, which
nobody observes, and it overestimates small models on wide buses by a factor
of two. With it, the same formula gives 146 for that case, which matches what
people report.

Two external checks landed after the fit, both from the same public thread on
first M5 Ultra benchmarks: 50 tokens per second reported on a 27B at Q4
against 53 predicted, and 20 to 25 reported on the same model at Q8 on an M3
Ultra against 24 predicted. Sources are in `data/benchmarks.json`.

## What is wrong with it

Stated before someone else does.

- The overhead constant was measured on Apple hardware running Ollama.
  Applying it to CUDA and ROCm assumes a comparable per token cost, which is
  not verified. It is not a rounding error: it is what keeps the GPU rows
  believable, and it is an assumption.
- Prompt processing is compute bound, not bandwidth bound. This model says
  nothing useful about it. The measured prompt numbers are in the data file
  and they vary by a factor of three across models with near identical
  generation speed.
- Everything is single stream. Batched serving behaves differently and the
  gap favours discrete GPUs.
- Mixture of experts models take memory from the total parameter count and
  speed from the active one, which is implemented, but the routing overhead
  is ignored.
- Quantisation sizes are nominal bits per weight. Real GGUF files differ by a
  few percent.

## Data

- `data/benchmarks.json` measured runs and the exact method, CC BY 4.0.
- `data/chips.json` Apple Silicon M1 to M6 with rated bandwidth and memory
  options, verified against Apple published specs on 2026-09-11, plus RTX
  3090, 4090, 5090, RTX PRO 6000, DGX Spark and Strix Halo with source links,
  verified 2026-09-13.
- `data/models.json` parameter counts, KV cache allowances and quantisation
  levels.

Corrections are welcome, especially measured numbers on hardware we do not
have. Open an issue with the run details and we will check the model against
them.

## Who made this

[Macyou](https://macyou.co) rents dedicated Apple Silicon Macs for AI, so we
have an obvious interest in one of the answers this calculator gives. That is
exactly why the formula, the constants and the raw data are all in this repo:
check them. The hosted version is at
[macyou.co/mac-llm-calculator](https://macyou.co/mac-llm-calculator), and the
cross vendor comparison, including the part about where a Mac is the wrong
purchase, is at
[macyou.co/compare/local-llm-hardware](https://macyou.co/compare/local-llm-hardware).

Code is MIT. Benchmark data is CC BY 4.0.
