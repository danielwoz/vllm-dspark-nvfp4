# vLLM DSpark fork + true 4-bit (NVFP4/E2M1) KV cache for DeepSeek-V4 sparse-MLA on SM120.
#
# Base: tonyd2wild's DSpark vLLM fork image (flashinfer 0.6.13+cu132), unmodified.
# This layer applies the nvfp4_ds_mla patch series (flashinfer CUDA kernels,
# flashinfer Python dispatch, vLLM plumbing + Triton store/qpad kernels), then
# recompiles the sparse_mla_sm120 module from the patched source and bakes the
# resulting .so into the AOT cache so users never JIT at runtime.
FROM fraserpricee/vllm:dspark-cu132-20260627

LABEL org.opencontainers.image.title="vLLM DSpark + NVFP4 KV cache (DeepSeek-V4 sparse-MLA, SM120)" \
      org.opencontainers.image.description="True 4-bit E2M1 KV cache for DeepSeek-V4-Flash on RTX PRO 6000 / SM120: 1.47x KV tokens/GiB vs fp8_ds_mla, +25-34% decode, needle-verified to 1M+ tokens. Enable with --kv-cache-dtype nvfp4_ds_mla." \
      org.opencontainers.image.base.name="docker.io/fraserpricee/vllm:dspark-cu132-20260627" \
      org.opencontainers.image.licenses="Apache-2.0"

ARG SP=/opt/venv/lib/python3.12/site-packages

# Patch series + validation harnesses kept in the image for transparency:
# users can inspect exactly what changed vs the base image.
COPY patches /opt/nvfp4/patches
COPY validation /opt/nvfp4/validation
COPY README.md /opt/nvfp4/README.md

# 1) Apply the series to the installed packages (paths in the patches are
#    a/flashinfer/..., a/vllm/... relative to site-packages).
RUN set -eu; \
    for p in /opt/nvfp4/patches/0*/*.patch; do \
        echo ">> $p"; patch -p1 -d "$SP" --forward < "$p"; \
    done; \
    cp /opt/nvfp4/patches/03-vllm-nvfp4/NEW_nvfp4_store.py "$SP/vllm/models/deepseek_v4/nvfp4_store.py"; \
    cp /opt/nvfp4/patches/03-vllm-nvfp4/NEW_nvfp4_qpad.py  "$SP/vllm/models/deepseek_v4/nvfp4_qpad.py"; \
    grep -q nvfp4_ds_mla "$SP/vllm/config/cache.py"; \
    grep -q DSV4_NVFP4 "$SP/flashinfer/data/include/flashinfer/attention/sparse_mla_sm120/model/model_type.h"

# 2) Recompile sparse_mla_sm120 from the patched source (no GPU needed at
#    build time; arch pinned to SM120f) and replace the stale AOT .so so the
#    runtime loads the nvfp4-capable kernel without any JIT step.
RUN set -eu; \
    export FLASHINFER_CUDA_ARCH_LIST="12.0f"; \
    /opt/venv/bin/python3 -c "from flashinfer.jit.mla import gen_sparse_mla_sm120_module as g; g().build(verbose=False); print('KERNEL_BUILD_OK')"; \
    JSO="$(find /cache -name sparse_mla_sm120.so -path '*cached_ops*' 2>/dev/null | head -1)"; \
    test -n "$JSO"; \
    cp "$JSO" "$SP/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so"; \
    rm -rf "$(dirname "$JSO")"

# 3) Import smoke test (no GPU): the patched python modules must at least parse
#    and resolve their imports.
RUN /opt/venv/bin/python3 - <<'PY'
import importlib
for m in ("vllm.models.deepseek_v4.nvfp4_store",
          "vllm.models.deepseek_v4.nvfp4_qpad",
          "vllm.config.cache",
          "flashinfer.mla._core"):
    importlib.import_module(m)
print("IMPORT_SMOKE_OK")
PY
