#!/usr/bin/env python3
"""Extended-context benchmark: needle recall + prefill rate + decode tok/s at depth.
Start-position needle (hardest direction), streaming to separate TTFT from decode."""
import json, time, urllib.request, sys

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "DeepSeek-V4-Flash-Abliterated-DSpark"
FILLER = ("The quick brown fox jumps over the lazy dog while counting widgets "
          "in the warehouse near the harbor at dawn. ")  # ~13 tokens

def bench_depth(target_tokens, code):
    prompt = (f"IMPORTANT: The secret passcode is {code}. Remember it.\n\n"
              + FILLER * (target_tokens // 21)
              + "\n\nWhat was the secret passcode stated at the very beginning? "
                "Reply with the passcode, then explain in about 150 words how you found it.")
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 320, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; ptoks = 0; ctoks = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                ptoks = d["usage"]["prompt_tokens"]; ctoks = d["usage"]["completion_tokens"]
            for ch in d.get("choices", []):
                delta = (ch.get("delta") or {}).get("content") or ""
                if delta and ttft is None:
                    ttft = time.time() - t0
                text += delta
    total = time.time() - t0
    dec_t = total - (ttft or total)
    dec_rate = (ctoks / dec_t) if dec_t > 0.05 and ctoks else float("nan")
    pre_rate = ptoks / ttft if ttft else float("nan")
    ok = code in text
    print(f"{target_tokens:>9,} tgt | {ptoks:>9,} real | TTFT {(ttft if ttft is not None else -1):7.1f}s "
          f"({pre_rate:7.0f} tok/s prefill) | decode {dec_rate:6.1f} tok/s | "
          f"needle {'OK ' if ok else 'MISS'} {text.strip()[:24]!r}", flush=True)
    return ok

if __name__ == "__main__":
    import hashlib
    depths = [int(a) for a in sys.argv[1:]] or [600_000]
    codes = [f"Q{int(hashlib.md5(str(d).encode()).hexdigest()[:4],16)%9000+1000}Z" for d in depths]
    print("== extended-context bench (needle at START, thinking off) ==", flush=True)
    hits = 0
    for d, c in zip(depths, codes):
        try:
            hits += bench_depth(d, c)
        except Exception as e:
            print(f"{d:>9,} tgt | ERROR: {str(e)[:90]}", flush=True)
    print(f"== DONE: {hits}/{len(depths)} needles ==", flush=True)
