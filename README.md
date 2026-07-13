# vLLM DSpark + NVFP4 (4-bit) KV cache — DeepSeek-V4 sparse-MLA on SM120

True 4-bit (E2M1) KV cache for **DeepSeek-V4-Flash** sparse-MLA on **RTX PRO 6000 Blackwell /
SM120**. This image is [`fraserpricee/vllm:dspark-cu132-20260627`](https://hub.docker.com/r/fraserpricee/vllm)
(tonyd2wild's DSpark vLLM fork, flashinfer 0.6.13+cu132) plus a ~1,000-line patch series adding a
new KV cache dtype: **`nvfp4_ds_mla`**. The patched sparse-MLA kernel is pre-compiled into the
image — no JIT at startup.

## Why

DeepSeek-V4's sparse-MLA decode is bandwidth-bound on workstation Blackwell. Halving KV bytes
(fp8 → fp4 NoPE) buys both capacity and speed:

| | fp8_ds_mla | **nvfp4_ds_mla** |
|---|---|---|
| KV footprint | 15.3 KB/token | **10.4 KB/token (1.47x more tokens/GiB)** |
| KV pool @262K max-len, util 0.96, TP=2 | ~592K tokens | **~1.54M tokens** (2.9M max observed) |
| Decode (DSpark spec-decode, 2x PRO 6000) | ~195 tok/s | **243–262 tok/s (+25–34%)** |
| HumanEval-164 / MBPP-100 | 85.0% / — | **90.2% / 94%** |
| Needle recall | — | **all-green through 1,030,039 real tokens** |

fp8 regression on the same build: all-green (the fp8 path is byte-identical; nvfp4 is a strict
addition).

## Usage

Identical to the base DSpark image, plus one flag: `--kv-cache-dtype nvfp4_ds_mla`.

The model is a `MODEL` env variable — the image's entrypoint runs
`vllm serve "$MODEL" "$@"`, so `-e MODEL=…` selects the checkpoint (default: the official
DeepSeek-V4-Flash-DSpark) and any extra args are forwarded to `vllm serve`.

```bash
docker run --gpus '"device=0,1"' --ipc=host --shm-size 64g -p 8000:8000 \
  -v /path/to/hf-cache:/hf -e HF_HOME=/hf \
  -e MODEL=deepseek-ai/DeepSeek-V4-Flash-DSpark \
  danielwoz/vllm:dspark-nvfp4-cu132 \
    --tensor-parallel-size 2 \
    --kv-cache-dtype nvfp4_ds_mla --block-size 256 \
    --max-model-len 1048576 --gpu-memory-utilization 0.95 \
    --max-cudagraph-capture-size 16 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":4}' \
    --kernel-config.moe_backend marlin \
    --enable-flashinfer-autotune \
    --trust-remote-code --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 --enable-auto-tool-choice --tool-call-parser deepseek_v4
```

Model variants (change `-e MODEL=…`):
- **Official (default):** `deepseek-ai/DeepSeek-V4-Flash-DSpark`
- **Abliterated:** `fraserprice/DeepSeek-V4-Flash-Abliterated-DSpark`
- **NVIDIA NVFP4 (no DSpark draft):** `nvidia/DeepSeek-V4-Flash-NVFP4` — also drop `--speculative-config`

Notes:
- **SM120 only** (RTX PRO 6000 Blackwell Workstation/Max-Q; anything that JITs `sparse_mla_sm120`).
- `--enable-flashinfer-autotune` is on by default here (kernel autotune at startup; adds boot time,
  neutral-to-positive on throughput).
- Tuned config: **spec-4 / cudagraph-capture-16 / marlin MoE** (+~11% decode vs capture-8 in our sweep).
- `--kv-cache-dtype fp8_ds_mla` and all other base-image behavior is unchanged.

## What's in the patch series (shipped in the image at `/opt/nvfp4/patches/`)

- `01-flashinfer-cuda/` — `ModelType::DSV4_NVFP4` + KV-cache traits (360 B/token page: 448x E2M1
  nibbles + 64x bf16 RoPE + per-64 UE8M0 footer), TMA-packed IO, in-place E2M1→FP8 nibble expand
  in the decode/prefill consumers. The fp8 tensor-core math path is reused unchanged — E2M1
  magnitudes are exactly representable in e4m3, so the existing block-scaled MMA reproduces
  values losslessly after expansion.
- `02-flashinfer-python/` — dispatch gates, `kv_scale_format="nvfp4"`, page-size assert relaxation.
- `03-vllm-nvfp4/` — `nvfp4_ds_mla` dtype registration and plumbing, two new Triton kernels
  (SWA store + compressed-cache store variant) and a standalone q-transform kernel (verified
  bit-exact vs the fused fp8 C++ op), plus two small bugfixes that also benefit fp8.

Validation harnesses are shipped at `/opt/nvfp4/validation/` (needle/garble KV test, HumanEval/
MBPP quality bench, 1M-token depth ladder).

## Provenance & credits

- **DSpark vLLM fork & base image**: tonyd2wild / fraserpricee — this work builds directly on the
  DSpark spec-decode stack. The DSpark repo documented an earlier true-4-bit attempt failing at
  ~411 prompt tokens; root cause was GLM-geometry (512-wide) page borrowing — this series
  implements the DeepSeek-V4-native 448-wide layout instead.
- **vLLM** (Apache-2.0) and **FlashInfer** (Apache-2.0): all modifications retain their licenses.
  Full diffs vs the pristine base image are in `/opt/nvfp4/patches/`.

Built and validated on 2x RTX PRO 6000 Blackwell Workstation Edition, driver 580.159, CUDA 13.2
runtime, TP=2.
