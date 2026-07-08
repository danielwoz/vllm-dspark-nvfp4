#!/usr/bin/env python3
"""Diff-test nvfp4_qpad.qnorm_rope_pad against the fused fp8 C++ op's returned q."""
import sys, torch
import torch as _t; _t.ops.load_library("/opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so")
sys.path.insert(0, "/work/vllm-patches")
from nvfp4_qpad import qnorm_rope_pad

torch.manual_seed(0)
dev = "cuda:0"
T, H, PH, HD = 33, 16, 32, 512
PBS = 64
EPS = 1e-6

q = torch.randn(T, H, HD, dtype=torch.bfloat16, device=dev)
kv = torch.randn(T, HD, dtype=torch.bfloat16, device=dev)
positions = torch.randint(0, 60000, (T,), dtype=torch.int64, device=dev)
# cos_sin cache like rotary_emb: [max_pos, 64] = cos(32)|sin(32), fp32
maxpos = 65536
inv = 1.0 / (10000 ** (torch.arange(0, 32, device=dev, dtype=torch.float32) / 32))
tpos = torch.arange(maxpos, device=dev, dtype=torch.float32)
ang = tpos[:, None] * inv[None, :]
cos_sin = torch.cat([ang.cos(), ang.sin()], dim=1).contiguous()

scratch = torch.zeros((1, PBS * 584), dtype=torch.uint8, device=dev)
slots = torch.zeros(T, dtype=torch.int64, device=dev)

q_ref = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q.clone(), kv, scratch, slots, positions, cos_sin, PH, EPS, PBS)
q_new = qnorm_rope_pad(q, positions, cos_sin, PH, EPS)

print("ref shape:", tuple(q_ref.shape), "new shape:", tuple(q_new.shape))
d = (q_ref.float() - q_new.float()).abs()
print(f"max abs diff: {d.max().item():.6f}   mean: {d.mean().item():.8f}")
print(f"pad rows zero (ref): {q_ref[:, H:].abs().max().item():.6f}  (new): {q_new[:, H:].abs().max().item():.6f}")
for tag, sl in (("nope", slice(0, 448)), ("rope", slice(448, 512))):
    dd = (q_ref[:, :H, sl].float() - q_new[:, :H, sl].float()).abs()
    print(f"  {tag}: max {dd.max().item():.6f} mean {dd.mean().item():.8f}")
ok = d.max().item() < 2e-2
print("PASS" if ok else "FAIL")
