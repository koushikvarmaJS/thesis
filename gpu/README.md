# GPU Pipeline (Google Colab / RTX 5090)

This is where the real thesis experiments run. Energy measured with NVML via `pynvml`.

## Colab setup checklist

- Runtime → Change runtime type → **GPU** (T4 free, A100/L4 paid)
- `!nvidia-smi` to confirm driver + GPU
- `!pip install nvidia-ml-py torch transformers accelerate`
- HuggingFace token for gated models (Llama): `from huggingface_hub import login; login()`
- Mount Drive if you want to persist results: `from google.colab import drive; drive.mount('/content/drive')`

## Sanity notebook

`nvml_sanity.ipynb` — checks `pynvml` works and reads GPU power. Run this first.

---

## Methods to benchmark (from thesis §4.2)

| Category | Method | Paper (arXiv) | Reference code |
|---|---|---|---|
| Eviction | H2O | 2306.14048 | `FMInference/H2O` on GitHub |
| Eviction | StreamingLLM | 2309.17453 | `mit-han-lab/streaming-llm` |
| Eviction | SnapKV | 2404.14469 | `FasterDecoding/SnapKV` |
| Quantization | KIVI (2-bit / 4-bit) | 2402.02750 | `jy-yuan/KIVI` |
| Compression | PyramidInfer | 2405.12532 | `mutonix/pyramidinfer` |
| Low-rank | PALU | 2407.21118 | `shadowpa0327/PALU` |
| Hybrid | H2O + KIVI 4-bit | — | stack the two above |

**Unified implementations (prefer these over per-paper repos):**

- **KVCache-Factory** — thesis recommends this as the unified library: <https://github.com/Zefan-Cai/KVCache-Factory>
- **kvpress** (NVIDIA) — `SnapKVPress`, `StreamingLLMPress`, etc., integrated with HF transformers: search GitHub for `NVIDIA/kvpress`
- **Awesome-KV-Cache-Compression** — paper index: <https://github.com/October2001/Awesome-KV-Cache-Compression>

Pick one unified lib instead of cobbling per-paper forks — they diverge on transformers versions.

---

## Models (thesis §5.2)

- `meta-llama/Llama-3.1-8B-Instruct` (gated — need HF access)
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

Free Colab T4 (16 GB) fits 7–8B models only in 4-bit (bitsandbytes) — pure FP16 baselines need A100/L4 or paid tier. On free T4, start with a smaller model (e.g. Llama-3.2-3B) to validate the pipeline, then rerun on the lab RTX 5090.

---

## Datasets (thesis §5.4)

- NarrativeQA — `deepmind/narrativeqa` on HuggingFace (long-context QA)
- CNN/DailyMail — `abisee/cnn_dailymail` or `cnn_dailymail` on HuggingFace
- MT-Bench — `lmsys/mt_bench_human_judgments`
- WikiText-2 — `Salesforce/wikitext` config `wikitext-2-raw-v1` (perplexity)

---

## Measurement protocol (thesis §4.3)

1. `pynvml.nvmlDeviceGetPowerUsage(handle) / 1000` → watts, sampled every 100 ms in a background thread
2. Energy = ∫ P(t) dt ≈ Σ P·0.1 over the inference window
3. Per run: 10 warm-up (discard) + 50 measured
4. Idle baseline: sample GPU for N seconds with no load, subtract `idle_watts × duration` from each run
5. Report mean ± std of joules/token, tokens/joule, throughput

Known pitfall in old code: NVML sampling at 100 ms on short (<1 s) generations gives ≤10 samples — noise dominates. Either generate enough tokens to last ≥5 s, or use a tighter sampling loop (NVML driver-side resolution is still ~50–100 ms though).

---

## Experimental sweep (thesis §5.3)

- Context length: 2K, 8K, 16K, 32K tokens
- Batch size: 1, 4, 8, 16
- Output length: 50, 100, 200 tokens
- Methods: 7 (baseline + 6)

That's 4·4·3·7 = 336 cells × (10 warmup + 50 measured) × ~3 models. Free Colab won't finish this; plan for the lab GPU.

---

## Related energy-benchmark papers (thesis §2)

- TokenPowerBench (AAAI 2026) — search arXiv; tests inference energy but not KV compression
- "From Prompts to Power" — arXiv:2511.05597
- "A Systematic Characterization of LLM Inference" — arXiv:2512.01644
- Kelle (MICRO 2025) — custom eDRAM hardware, not comparable

These are the "what's already measured" baselines — cite them for the gap, don't reimplement.
