#!/usr/bin/env python3
"""NVFP4-KV correctness probe for DeepSeek-V4-Flash sparse-MLA.

Purpose: pin the "~411 real prompt token" corruption cliff and validate a fix.
Sends prompts of increasing REAL length and checks whether the model still
produces coherent, on-task output (needle-in-context recall + garble detection).

Usage:  python3 kvtest.py [--port 8012] [--model NAME]
A healthy fp8 baseline should pass ALL lengths. A broken nvfp4 layout is expected
to pass short prompts and start failing around the page/block boundary.
"""
import argparse, json, re, sys, urllib.request

# Real-token filler (~1 token per short word). We embed a unique needle near the
# END so the model must attend across the whole (quantized) KV to answer.
FILLER_UNIT = ("The quick brown fox jumps over the lazy dog while counting "
               "widgets in the warehouse near the harbor at dawn. ")

def build_prompt(approx_tokens, needle):
    # ~13 tokens per FILLER_UNIT; pad to target then append the needle question.
    reps = max(1, approx_tokens // 13)
    body = (FILLER_UNIT * reps)
    return (f"{body}\n\nIMPORTANT: The secret passcode is {needle}. "
            f"Reply with ONLY the secret passcode, nothing else.")

def gen(port, model, prompt, max_tokens=32):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    ch = d["choices"][0]
    txt = ch["message"].get("content") or ""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)
    return txt.strip(), d["usage"]["prompt_tokens"], ch.get("finish_reason")

def garbled(txt):
    if not txt: return True
    # repetition loop or CJK leakage in an English task = corruption signature
    if re.search(r"([\w ]{3,20})\1{4,}", txt): return True
    if len(re.findall(r"[一-鿿]", txt)) > 2: return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--model", default="DeepSeek-V4-Flash-Abliterated-DSpark")
    args = ap.parse_args()
    lengths = [100, 300, 400, 411, 450, 512, 800, 1200, 2000, 4000, 8000]
    print(f"{'target':>7} {'real_toks':>9} {'recall':>7} {'garble':>7}  reply")
    first_fail = None
    for i, n in enumerate(lengths):
        needle = f"ZQ{7000+i*137}X"
        try:
            txt, real, fin = gen(args.port, args.model, build_prompt(n, needle))
        except Exception as e:
            print(f"{n:>7} {'ERR':>9}  {str(e)[:60]}"); continue
        ok = needle in txt
        gb = garbled(txt)
        flag = "OK" if (ok and not gb) else "FAIL"
        if flag == "FAIL" and first_fail is None and real > 200:
            first_fail = real
        print(f"{n:>7} {real:>9} {str(ok):>7} {str(gb):>7}  [{flag}] {txt[:50]!r}")
    print("\nfirst corruption at real prompt tokens:",
          first_fail if first_fail else "none (all coherent)")

if __name__ == "__main__":
    main()
