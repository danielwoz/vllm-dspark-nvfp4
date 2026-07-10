# DeepSeek-V4-Flash-Abliterated-DSpark — Benchmark Results

**Date:** 2026-07-08 → 2026-07-10
**Hardware:** 2× RTX PRO 6000 Blackwell (SM120, 96 GB), TP=2, 600 W caps, chassis-fan control active
**Serving stack:** `danielwoz/vllm:dspark-nvfp4-cu132-20260708` (dspark-vllm fork + flashinfer SM120
patches, true 4-bit E2M1 KV cache `nvfp4_ds_mla`), 1,048,576-token window, ~1.63M-token KV pool
(1.55× concurrent 1M sessions), CUDA graphs FULL_AND_PIECEWISE capture 8, gpu-util 0.95,
DSpark speculative decoding (k=4, probabilistic), Marlin MoE, `FLASHINFER_MLA_SPARSE_DSV4`.

## Thinking modes

The vLLM `deepseek_v4` chat template has exactly three states, selected per request via
`chat_template_kwargs` (or server-wide via `--default-chat-template-kwargs`):

| Column | kwargs | Meaning |
|---|---|---|
| **false** | `{"thinking": false}` / effort `"none"` | chat mode, no thinking channel |
| **high** | `{"thinking": true}` | thinking channel, model-calibrated depth ("high" is a no-op label — any effort string other than none/max renders nothing) |
| **max** | `{"thinking": true, "reasoning_effort": "max"}` | injects a maximal-deliberation directive at conversation start |

## Quality results

| Benchmark | false | high | max |
|---|---|---|---|
| SWE-bench Verified (50-inst subset, mini-SWE-agent) | **64%** | 62% | 56% |
| HumanEval-164 pass@1 | **86.0%** | 82.3% | 83.5% |
| MBPP-100 pass@1 | **94%** | **94%** | 84% |
| GSM8K-200 (exact match) | **98.5%** | 97.0% | 94.5% |
| MATH-500 (150, strict boxed match) | **80.7%** | 75.3% | — |
| MMLU-STEM-210 | 85.7% | **89.5%** | — |
| WMT14 en→de (150, BLEU / chrF2) | **32.3 / 61.2** | 28.7 / 54.7 | — |
| WMT14 en→fr (150, BLEU / chrF2) | **42.3 / 66.2** | 37.3 / 60.1 | — |
| GPQA-Diamond (198) | **68.7%** | 61.6% | 59.1% |
| SimpleQA-Verified (1000, self-graded) | 24.0% | — | — |
| Deep needles (8K/32K/64K start-position) | 3/3 | 3/3 | 3/3 |
| Long-gen loop/repetition check | clean | clean | clean |

"—" = not measured (max abandoned after being refuted; SimpleQA high/HLE/Codeforces runs stopped early).

### Wall-clock cost of thinking (same work, same hardware)

| Benchmark | false | high | max |
|---|---|---|---|
| HumanEval-164 | 6 min | 28 min | 100 min |
| MBPP-100 | 54 s | 17 min | 81 min |
| GSM8K-200 | 4.7 min | 17 min | 39 min |
| GPQA-Diamond | 74 min | 5.7 h | 8.2 h |

### SWE-bench Verified detail (same shuffled 50-instance subset, 4 workers)

| | false | high | max |
|---|---|---|---|
| Resolved | 32/50 | 31/50 | 28/50 |
| Patches evaluated | 44 | 44 | 41 |
| Lost to step-limits / timeouts / no patch | 5 | 6 | 9 |
| Accuracy of evaluated patches | 73% | 70% | 68% |

Scaffold: mini-SWE-agent v2.4.5 (`--subset verified --shuffle --slice 0:50`), scored with the
official swebench harness. Max-effort thinking nearly doubles budget-exhaustion casualties —
the model deliberates exhaustively over trivial tool steps.

## Serving performance (vllm bench serve, 2048-in / 256-out random tokens)

| Metric (median) | eager @ util 0.95 | cudagraph capture-8 @ util 0.95 |
|---|---|---|
| Single-stream inter-token latency | 137.6 ms | **14.5 ms (9.5×)** |
| Single-stream effective decode | ~17 tok/s | **~160 tok/s** |
| TTFT | 762 ms | **481 ms** |
| Total throughput @ concurrency 8 | 476 tok/s | **735 tok/s** |
| KV pool @ 1M window | 2.41M tokens | 1.63M tokens |

- DSpark acceptance is text-dependent: 2.3/4 on random-token benches, **3.7/4 on real
  benchmark text** (87% first-position) → real-workload single-stream ≈ 220-250 tok/s.
- Capture 64 is not bootable at the 1M window (graph reservation ~17 GiB/GPU starves KV
  below the one-request minimum); capture 8 keeps one spec-decode session on the FULL-graph
  fast path. Capture 16 (~3 sessions) would cost ≈ 350-400K pool tokens (extrapolated).
- GPUs are latency-bound, not compute-bound, at small batch (~200 W of 600 during benches);
  aggregate throughput scales with client concurrency (72 → 476 → 735 tok/s at 1 → 8 streams).

## Conclusions

1. **Chat mode (thinking: false) wins or ties everything except MMLU-STEM** — the model's
   prompted step-by-step in chat mode already extracts its reasoning ability. This matches
   DeepSeek hybrid tuning guidance: agentic/tool work belongs in chat mode.
2. **Max effort is strictly harmful**: 0-for-6 vs chat (including −9.6 pts on GPQA, its
   supposed home turf) at 4-17× the compute. The directive induces overthinking: budget
   exhaustion in agents, self-argued wrong answers in QA.
3. **Standard thinking (high) costs 2-5 points on most benches** and one order of magnitude
   of latency; its one win is MMLU-STEM (+3.8). Kept as the server default for interactive
   use; benchmarks and agents should prefer chat mode per request.
4. **The 4-bit NVFP4-KV serving stack shows no quality regression** vs its fp8 references
   (HumanEval 86 vs 85 fp8; MBPP 94 vs 94 at 262K; needle recall intact to 1.03M tokens —
   see STATUS.md for the KV-cache A/B history).

## Caveats & artifacts (read before quoting numbers)

- 50-instance SWE subset ⇒ ±7% sampling error; single runs at temp 1.0 (HumanEval/MBPP)
  carry ±2-3%. MATH-500 uses strict boxed-match with a simple normalizer (scores are floors).
- SimpleQA/HLE grading uses the local model as its own judge (official harnesses use a
  frontier judge); SimpleQA 24.0% had zero abstentions — the model always guesses.
- **Runaway-thinking artifact**: at thinking-high/max with greedy decoding under batched
  load, ~1-2% of generations ramble in the thinking channel to the token cap. This
  corrupts corpus-level metrics (one 6,000-word "hypothesis" collapses BLEU for the whole
  set) and crashes naive harnesses via HTTP timeouts. Harnesses here use content-only
  extraction for translation and per-item exception guards.
- Prompt-suffix folklore test: appending "Your final answer MUST include the implementation."
  changed HumanEval 82.3→81.1 and MBPP 94→91 (both within noise) — no benefit on this model.
- Thinking selection for clients that can't send template kwargs (Claude Code) is done by
  model alias via the :8012 proxy: `deepseekv4-flash[1m]` (thinking) /
  `deepseekv4-nothink[1m]` (chat) / `deepseekv4-think-max[1m]` (max). An alias overriding
  effort must also override `thinking` explicitly, or vLLM's parser and template disagree
  and all output lands in the thinking block.
