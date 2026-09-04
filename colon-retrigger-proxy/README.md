# colon-retrigger-proxy

A tiny reverse proxy for OpenAI-compatible vLLM serves that recovers turns some
reasoning models end too early.

## The problem

DeepSeek-V4-Flash (and similar reasoning models) sometimes emit their
end-of-turn token right after a lead-in that ends in a colon — *"Let me check
the git version:"* — instead of proceeding to the tool call or code block they
were leading into. The turn ends with `finish_reason: "stop"`, no tool call, and
agent clients such as opencode simply stop. It reads as "a colon halts the
model." At temperature 1.0 it happens a meaningful fraction of the time; lower
temperature reduces but does not remove it.

## What it does

The proxy sits between the client and the vLLM serve. On a **streaming**
`/v1/chat/completions` turn that ends with `finish_reason: "stop"`, **no tool
call**, and **content ending in a colon**, it continues the assistant turn
(`continue_final_message`) and splices the recovered tokens — text and/or the
tool call — into the same client stream. Everything else is forwarded unchanged.

Design points that keep it safe:

- **Scoped to tool-bearing turns**, where a colon-stop is a dropped tool call
  rather than an intended ending, so it never extends a normal chat answer.
- **The continuation runs with thinking off** so `continue_final_message`
  appends instead of re-generating (a thinking template breaks the resume).
- **The continuation runs non-streaming internally**, then is re-emitted as
  stream chunks — the one-shot tool-call parser is reliable, whereas streaming
  continuations can leak raw tool-call markup.
- **No-progress guard + retry cap** stop a stubborn model from piling up colons.

It also **breaks degenerate repetition loops**: when a turn's output collapses
into the same phrase, line, or character repeated over and over, the proxy cuts
the turn cleanly (and drops the upstream request) instead of letting it run to
tens of KB of garbage. Controlled by `LOOP_DETECT` / `LOOP_MAX_REPEAT`.

## Run

In front of an existing serve:

```bash
docker run -d --network host \
  -e UPSTREAM=http://127.0.0.1:8008 \
  -e LISTEN_PORT=8012 \
  danielwoz/colon-retrigger-proxy
```

Point your client's base URL at `http://<host>:8012/v1`. See
`docker-compose.yml` for a full serve + proxy example.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `UPSTREAM` | `http://127.0.0.1:8000` | Upstream vLLM base URL |
| `LISTEN_PORT` | `8012` | Port to listen on |
| `COLON_RETRIGGER` | `1` | Enable the fix (`0` = pure passthrough) |
| `MAX_RETRIES` | `2` | Continuation attempts per turn |
| `CONT_CHAT_TEMPLATE_KWARGS` | `{"thinking": false}` | Kwargs for the continuation request; set `{}` for models without a thinking template |
| `MODEL_ALIASES` | `{}` | Optional JSON mapping an exposed model name to `{"model": ..., "chat_template_kwargs": ...}`, so clients that can't send `chat_template_kwargs` can select variants by model name |
| `LOOP_DETECT` | `1` | Cut a turn when its output degenerates into repetition (`0` = off) |
| `LOOP_MAX_REPEAT` | `6` | Repeats of a phrase/line/character that count as a loop |
| `HIDE_REASONING` | `0` | Drop reasoning deltas on `/v1/chat/completions` so the client shows no thinking; the model still reasons upstream (answer unchanged). `/v1/messages` is unaffected |

Firings are logged: `[colon-retrigger] fired …`, `[loop-break] cut at … chars …`.

## Endpoints

Covers the streaming path of both the OpenAI `/v1/chat/completions` and the
Anthropic `/v1/messages` APIs (so it works for opencode and for Claude Code
pointed at a local serve). Non-streaming requests are forwarded unchanged.
