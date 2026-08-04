#!/usr/bin/env python3
"""Colon-retrigger proxy for OpenAI-compatible vLLM serves.

Some reasoning models (e.g. DeepSeek-V4-Flash) occasionally emit their
end-of-turn token right after a lead-in that ends in a colon ("Let me check
the version:") instead of proceeding to the tool call or code block they were
about to make. The turn ends early with finish_reason "stop", no tool call,
and the client (opencode and other agents) just stops.

This proxy sits between the client and the vLLM serve. On a streaming
/v1/chat/completions turn that ends that way, it continues the assistant turn
(continue_final_message) and splices the recovered tokens into the same client
stream. Everything else is forwarded unchanged.

Configuration (environment):
  UPSTREAM                 upstream vLLM base URL   (default http://127.0.0.1:8000)
  LISTEN_PORT              port to listen on        (default 8012)
  COLON_RETRIGGER          enable the fix, "1"/"0"  (default 1)
  MAX_RETRIES              continuation attempts    (default 2)
  CONT_CHAT_TEMPLATE_KWARGS  JSON kwargs for the continuation request
                           (default {"thinking": false} — reasoning already
                           happened in the initial turn; thinking off lets
                           continue_final_message append instead of regenerate;
                           set {} for models without a thinking template)
  MODEL_ALIASES            optional JSON mapping an exposed model name to a
                           rewrite, e.g.
                           {"myalias": {"model": "real-model",
                                        "chat_template_kwargs": {"thinking": false}}}
                           lets thinking variants be selected by model name for
                           clients that cannot send chat_template_kwargs.
"""
import json
import os
from collections import Counter

import aiohttp
from aiohttp import web

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8012"))
COLON_RETRIGGER = os.environ.get("COLON_RETRIGGER", "1") == "1"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))
CONT_KWARGS = json.loads(os.environ.get("CONT_CHAT_TEMPLATE_KWARGS", '{"thinking": false}'))
ALIASES = json.loads(os.environ.get("MODEL_ALIASES", "{}"))
LOOP_DETECT = os.environ.get("LOOP_DETECT", "1") == "1"
LOOP_MAX_REPEAT = int(os.environ.get("LOOP_MAX_REPEAT", "6"))
HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}


# Rule/box/underline characters: long runs of these are legitimate formatting,
# not a decode loop, so short-period detection skips pure runs of them.
_FMT = set("=-_*#~|+. \t\r\n─━═")


def _looping(text, k=LOOP_MAX_REPEAT):
    """True when the tail is a short unit repeated many times (a degenerate
    decode loop): a repeated phrase/line, or a repeated character. Long units
    (words, phrases, lines) trigger at k repeats; sub-4-char units need a much
    higher bar and skip pure formatting runs, to avoid flagging real output."""
    tail = text[-2000:]
    if len(tail) < 24:
        return False
    maxp = min(128, len(tail) // k)
    for p in range(1, maxp + 1):
        unit = tail[-p:]
        if not unit.strip():
            continue
        if p >= 4:
            need = k
        else:
            if all(ch in _FMT for ch in unit):
                continue
            need = max(k, 30)
        if len(tail) >= p * need and tail.endswith(unit * need):
            return True
    # Long-unit (period > 128) line loops.
    lines = [l.strip() for l in tail.splitlines() if len(l.strip()) >= 8]
    if len(lines) >= k and Counter(lines).most_common(1)[0][1] >= k:
        return True
    return False


def rewrite(body: bytes) -> bytes:
    """Apply MODEL_ALIASES: swap the model and/or inject chat_template_kwargs."""
    if not ALIASES:
        return body
    try:
        d = json.loads(body)
    except Exception:
        return body
    spec = ALIASES.get(d.get("model"))
    if spec:
        if spec.get("model"):
            d["model"] = spec["model"]
        if spec.get("chat_template_kwargs") is not None:
            d.setdefault("chat_template_kwargs", spec["chat_template_kwargs"])
        return json.dumps(d).encode()
    return body


def _delta(obj):
    ch = obj.get("choices") or []
    if not ch:
        return "", False
    d = ch[0].get("delta") or {}
    return (d.get("content") or ""), bool(d.get("tool_calls"))


def _chunk(meta, delta, finish=None):
    return {"id": meta[0], "object": "chat.completion.chunk",
            "created": meta[1], "model": meta[2],
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _sse(obj):
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


async def _pump(sess, body, headers, resp):
    """Stream one upstream call to the client, suppressing a finish_reason that
    rides on a content chunk so the stream stays open, and holding terminal-only
    chunks. Returns (content, tool_seen, finish, terminals, meta)."""
    content = ""
    tool = False
    finish = None
    terminals = []
    meta = None
    last_check = 0
    rtext = ""
    last_check_r = 0
    async with sess.request("POST", UPSTREAM + "/v1/chat/completions",
                            data=body, headers=headers) as up:
        async for raw in up.content:
            raw = raw.strip()
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if meta is None and obj.get("id"):
                meta = (obj.get("id"), obj.get("created"), obj.get("model"))
            c, t = _delta(obj)
            content += c
            tool = tool or t
            ch = obj.get("choices") or []
            delta = ch[0].get("delta") if ch else None
            fr = ch[0].get("finish_reason") if ch else None
            if fr is not None:
                finish = fr
            rc = (delta.get("reasoning") or delta.get("reasoning_content") or "") if delta else ""
            # Forward any chunk carrying a delta payload (content, reasoning, tool,
            # or the opening role); a finish_reason riding on it is suppressed so
            # the stream stays open and the turn end is emitted once at the close.
            if delta and (c or t or rc or delta.get("role")):
                if fr is not None:
                    obj["choices"][0]["finish_reason"] = None
                await resp.write(_sse(obj))
                if LOOP_DETECT and c and len(content) - last_check >= 48:
                    last_check = len(content)
                    if _looping(content):
                        print(f"[loop-break] cut at {len(content)} chars "
                              f"tail={content[-40:]!r}", flush=True)
                        finish = "stop"
                        break
                if LOOP_DETECT and rc:
                    rtext += rc
                    if len(rtext) - last_check_r >= 48:
                        last_check_r = len(rtext)
                        if _looping(rtext):
                            print(f"[loop-break] cut reasoning at {len(rtext)} chars "
                                  f"tail={rtext[-40:]!r}", flush=True)
                            finish = "stop"
                            break
            else:
                terminals.append(obj)
    return content, tool, finish, terminals, meta


async def _continue_once(sess, d, content, headers):
    """Continue the assistant turn from `content`. Non-streaming: the upstream
    tool-call parser is reliable one-shot (streaming continuations can leak raw
    tool-call markup). Returns (new_content, tool_calls, finish)."""
    cont = dict(d)
    cont["messages"] = d["messages"] + [{"role": "assistant", "content": content}]
    cont["continue_final_message"] = True
    cont["add_generation_prompt"] = False
    if CONT_KWARGS:
        cont["chat_template_kwargs"] = CONT_KWARGS
    cont["stream"] = False
    cont.pop("stream_options", None)
    async with sess.request("POST", UPSTREAM + "/v1/chat/completions",
                            data=json.dumps(cont).encode(), headers=headers) as up:
        raw = await up.read()
    o = json.loads(raw)
    ch = o["choices"][0]
    m = ch.get("message", {})
    return (m.get("content") or ""), (m.get("tool_calls") or []), ch.get("finish_reason")


async def _chat_stream(request, body, headers):
    d = json.loads(body)
    out_headers = {"Content-Type": "text/event-stream; charset=utf-8",
                   "Cache-Control": "no-cache"}
    resp = web.StreamResponse(status=200, headers=out_headers)
    await resp.prepare(request)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as sess:
        content, tool, finish, terminals, meta = await _pump(sess, body, headers, resp)
        # Only continue agentic (tool-bearing) turns, where a colon-stop is a
        # dropped tool call rather than an intended ending.
        has_tools = bool(d.get("tools"))
        retries = 0
        from_cont = False
        while (has_tools and finish == "stop" and not tool
               and content.rstrip().endswith(":") and retries < MAX_RETRIES):
            retries += 1
            from_cont = True
            c2, tc2, f2 = await _continue_once(sess, d, content, headers)
            if c2:
                await resp.write(_sse(_chunk(meta, {"content": c2})))
                content += c2
            if tc2:
                for i, tc in enumerate(tc2):
                    delta = {"tool_calls": [{"index": i, "id": tc.get("id"),
                             "type": "function", "function": tc.get("function", {})}]}
                    await resp.write(_sse(_chunk(meta, delta)))
                tool = True
                finish = "tool_calls"
                break
            finish = f2 or finish
            # Stop if the continuation made no real progress (empty, or only
            # more colons/whitespace) so a stubborn stop can't pile up colons.
            if not c2.strip(" :\n\t"):
                break

    if from_cont:
        print(f"[colon-retrigger] fired retries={retries} finish={finish} "
              f"tool={tool} tail={content[-40:]!r}", flush=True)
        await resp.write(_sse(_chunk(meta, {}, "tool_calls" if tool else (finish or "stop"))))
    elif terminals:
        for tobj in terminals:
            await resp.write(_sse(tobj))
    elif meta:
        await resp.write(_sse(_chunk(meta, {}, finish or "stop")))
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def _aevent(etype, data):
    return b"event: " + etype.encode() + b"\ndata: " + json.dumps(data).encode() + b"\n\n"


async def _continue_msgs(sess, d, text, headers):
    """Anthropic /v1/messages continuation of the assistant turn from `text`.
    Returns (new_text, tool_use_blocks, stop_reason)."""
    cont = dict(d)
    cont["messages"] = d["messages"] + [{"role": "assistant", "content": text}]
    cont["continue_final_message"] = True
    cont["add_generation_prompt"] = False
    if CONT_KWARGS:
        cont["chat_template_kwargs"] = CONT_KWARGS
    cont["stream"] = False
    async with sess.request("POST", UPSTREAM + "/v1/messages",
                            data=json.dumps(cont).encode(), headers=headers) as up:
        raw = await up.read()
    o = json.loads(raw)
    blocks = o.get("content") or []
    nt = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    tus = [b for b in blocks if b.get("type") == "tool_use"]
    return nt, tus, o.get("stop_reason")


async def _messages_stream(request, body, headers):
    """Anthropic /v1/messages equivalent of _chat_stream. The message_delta /
    message_stop terminal events are held so recovered content blocks can be
    injected before the turn is closed to the client."""
    d = json.loads(body)
    out_headers = {"Content-Type": "text/event-stream; charset=utf-8",
                   "Cache-Control": "no-cache"}
    resp = web.StreamResponse(status=200, headers=out_headers)
    await resp.prepare(request)
    text = ""
    tool_use_seen = False
    max_index = 0
    stop_reason = None
    held_md = None
    last_check = 0
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as sess:
        cur = None
        async with sess.request("POST", UPSTREAM + "/v1/messages",
                                data=body, headers=headers) as up:
            async for raw in up.content:
                s = raw.strip()
                if s.startswith(b"event:"):
                    cur = s[6:].strip().decode()
                    continue
                if not s.startswith(b"data:"):
                    continue
                try:
                    obj = json.loads(s[5:].strip())
                except Exception:
                    continue
                et = obj.get("type") or cur
                if et == "content_block_start":
                    max_index = max(max_index, obj.get("index", 0))
                    if (obj.get("content_block") or {}).get("type") == "tool_use":
                        tool_use_seen = True
                    await resp.write(_aevent(et, obj))
                elif et == "content_block_delta":
                    max_index = max(max_index, obj.get("index", 0))
                    delta = obj.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text += delta.get("text", "")
                    await resp.write(_aevent(et, obj))
                    if (LOOP_DETECT and delta.get("type") == "text_delta"
                            and len(text) - last_check >= 48):
                        last_check = len(text)
                        if _looping(text):
                            print(f"[loop-break:messages] cut at {len(text)} chars "
                                  f"tail={text[-40:]!r}", flush=True)
                            await resp.write(_aevent("content_block_stop",
                                {"type": "content_block_stop", "index": max_index}))
                            stop_reason = "end_turn"
                            break
                elif et == "message_delta":
                    held_md = obj
                    stop_reason = (obj.get("delta") or {}).get("stop_reason")
                elif et == "message_stop":
                    pass
                else:
                    await resp.write(_aevent(et, obj))

        has_tools = bool(d.get("tools"))
        retries = 0
        from_cont = False
        while (has_tools and stop_reason == "end_turn" and not tool_use_seen
               and text.rstrip().endswith(":") and retries < MAX_RETRIES):
            retries += 1
            from_cont = True
            nt, tus, sr = await _continue_msgs(sess, d, text, headers)
            if nt:
                max_index += 1
                await resp.write(_aevent("content_block_start", {"type": "content_block_start",
                    "index": max_index, "content_block": {"type": "text", "text": ""}}))
                await resp.write(_aevent("content_block_delta", {"type": "content_block_delta",
                    "index": max_index, "delta": {"type": "text_delta", "text": nt}}))
                await resp.write(_aevent("content_block_stop",
                    {"type": "content_block_stop", "index": max_index}))
                text += nt
            if tus:
                for tu in tus:
                    max_index += 1
                    await resp.write(_aevent("content_block_start", {"type": "content_block_start",
                        "index": max_index, "content_block": {"type": "tool_use",
                        "id": tu.get("id"), "name": tu.get("name"), "input": {}}}))
                    await resp.write(_aevent("content_block_delta", {"type": "content_block_delta",
                        "index": max_index, "delta": {"type": "input_json_delta",
                        "partial_json": json.dumps(tu.get("input", {}))}}))
                    await resp.write(_aevent("content_block_stop",
                        {"type": "content_block_stop", "index": max_index}))
                tool_use_seen = True
                stop_reason = "tool_use"
                break
            stop_reason = sr or stop_reason
            if not nt.strip(" :\n\t"):
                break

        if from_cont:
            print(f"[colon-retrigger:messages] fired retries={retries} "
                  f"stop={stop_reason} tail={text[-40:]!r}", flush=True)
        if held_md is None:
            held_md = {"type": "message_delta", "usage": {"output_tokens": 0},
                       "delta": {"stop_sequence": None}}
        held_md.setdefault("delta", {})["stop_reason"] = stop_reason or "end_turn"
        await resp.write(_aevent("message_delta", held_md))
        await resp.write(_aevent("message_stop", {"type": "message_stop"}))
    await resp.write_eof()
    return resp


async def handle(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    if request.method == "POST" and body[:1] == b"{":
        body = rewrite(body)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}

    if (COLON_RETRIGGER and request.method == "POST" and body[:1] == b"{"):
        try:
            path = request.rel_url.path
            if json.loads(body).get("stream"):
                if path.endswith("/chat/completions"):
                    return await _chat_stream(request, body, headers)
                if path.endswith("/messages"):
                    return await _messages_stream(request, body, headers)
        except Exception:
            pass

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as sess:
        async with sess.request(request.method, UPSTREAM + str(request.rel_url),
                                data=body if body else None, headers=headers) as up:
            out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP_HEADERS}
            resp = web.StreamResponse(status=up.status, headers=out_headers)
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp


app = web.Application(client_max_size=512 * 1024 * 1024)
app.router.add_route("*", "/{tail:.*}", handle)

if __name__ == "__main__":
    print(f"colon-retrigger proxy: :{LISTEN_PORT} -> {UPSTREAM} "
          f"(retrigger={'on' if COLON_RETRIGGER else 'off'}, max_retries={MAX_RETRIES}, "
          f"aliases={len(ALIASES)})", flush=True)
    web.run_app(app, host="0.0.0.0", port=LISTEN_PORT, access_log=None)
