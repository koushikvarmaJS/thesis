# Mac Pipeline (Apple Silicon, local dev)

**Not where the thesis energy numbers come from.** macOS has no NVML and no per-process GPU energy API. Use this folder for:

- Code-reading and prototyping pipeline logic before pushing to Colab/GPU
- Plotting and analyzing result JSONs pulled from the GPU run
- Sanity-checking model outputs (generation correctness, tokenizer behavior) against a small MLX model
- Perplexity / accuracy evaluation that doesn't need GPU energy (runs on CPU/MPS, just slow)

## What not to do here

- Don't measure "energy" via `psutil.cpu_times()` with an arbitrary multiplier — that's what the old `mac_pipelines/` did and the numbers were meaningless.
- `powermetrics` (sudo) can give SoC-level package energy but it's not per-process and mixes everything on the SoC. Not publishable as KV-cache-method energy.

## If you want approximate local energy anyway

- `sudo powermetrics --samplers cpu_power,gpu_power -i 100 -n 1` → prints package power in watts; same integration approach as NVML. Only useful for rough ordering, never for the thesis plots.

## Sources for the methods (same as `gpu/README.md`)

See `../gpu/README.md` — the method papers and reference repos are identical; only the runtime differs.

For local testing of compression *logic* (not energy), MLX ports:

- `mlx-lm` — base framework: search GitHub for `ml-explore/mlx-examples` (the `llms/` subfolder)
- MLX doesn't yet have first-class KV compression hooks like `kvpress`. Compression would have to be implemented against MLX's cache API (`mlx_lm.models.cache`). Scope decision: probably not worth implementing — just prototype on HF + CUDA in Colab.

## Model suggestions for local prototyping (fast, small)

- `mlx-community/Llama-3.2-1B-Instruct-4bit`
- `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `mlx-community/Qwen2.5-3B-Instruct-4bit`

These are for correctness testing only — final numbers come from the GPU runs.
