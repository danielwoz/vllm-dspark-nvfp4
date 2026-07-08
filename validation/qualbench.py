#!/usr/bin/env python3
"""Quality suite for the nvfp4 server: HumanEval-164, MBPP-100, deep start-position
needles, long-gen repetition. Methodology matches the July-1 fp8 benchmark
(temp=1.0, top_p=0.95, last ```python block, subprocess test)."""
import os, re, json, time, subprocess, sys, tempfile, urllib.request, concurrent.futures as cf
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_HOME", "/hf")

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "DeepSeek-V4-Flash-Abliterated-DSpark"
W = 4  # concurrency

def gen(prompt, max_tokens=16384, temp=1.0, top_p=0.95):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
         "temperature": temp, "top_p": top_p, "max_tokens": max_tokens}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, json.dumps(b).encode(), {"content-type": "application/json"}), timeout=900))
    ch = r["choices"][0]
    return (ch["message"].get("content") or ""), ch.get("finish_reason")

def extract(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    t = re.sub(r"^.*?</think>", "", t, flags=re.DOTALL)
    b = re.findall(r"```(?:python|py)?\s*\n(.*?)```", t, flags=re.DOTALL)
    return (b[-1].strip() if b else t.strip())

def run_prog(prog, timeout=15):
    try:
        return subprocess.run([sys.executable, "-c", prog], capture_output=True,
                              timeout=timeout, cwd=tempfile.gettempdir()).returncode == 0
    except Exception:
        return False

def humaneval():
    from datasets import load_dataset
    probs = list(load_dataset("openai_humaneval", split="test"))
    def one(p):
        c, _ = gen("Complete this Python function. Reply with ONLY the complete "
                   "function in a single ```python code block.\n\n```python\n"
                   + p["prompt"] + "\n```")
        code = extract(c); ep = p["entry_point"]
        body = code if f"def {ep}" in code else p["prompt"] + "\n" + code
        return run_prog(body + "\n\n" + p["test"] + f"\n\ncheck({ep})\n")
    t0 = time.time(); passed = 0
    with cf.ThreadPoolExecutor(max_workers=W) as ex:
        for i, ok in enumerate(ex.map(one, probs)):
            passed += ok
            if (i + 1) % 40 == 0: print(f"  HE {i+1}/164: {passed} passed", flush=True)
    print(f"HUMANEVAL-164 pass@1 = {passed}/164 ({100*passed/164:.1f}%)  [{time.time()-t0:.0f}s]", flush=True)

def mbpp():
    from datasets import load_dataset
    probs = list(load_dataset("mbpp", "sanitized", split="test"))[:100]
    def one(p):
        tests = "\n".join(p["test_list"])
        c, _ = gen("Write a Python function for this task. It must pass these tests. "
                   "Reply with ONLY the code in a single ```python code block.\n\nTask: "
                   + p["prompt"] + "\n\nTests:\n```python\n" + tests + "\n```")
        return run_prog(extract(c) + "\n\n" + tests + "\n")
    t0 = time.time(); passed = 0
    with cf.ThreadPoolExecutor(max_workers=W) as ex:
        for i, ok in enumerate(ex.map(one, probs)):
            passed += ok
            if (i + 1) % 25 == 0: print(f"  MBPP {i+1}/100: {passed} passed", flush=True)
    print(f"MBPP-100 pass@1 = {passed}/100 ({passed}%)  [{time.time()-t0:.0f}s]", flush=True)

def deep_needles():
    filler = ("The quick brown fox jumps over the lazy dog while counting widgets "
              "in the warehouse near the harbor at dawn. ")
    ok = 0; total = 0
    for tgt, code in [(8000, "KX3141Z"), (32000, "PB2718Q"), (64000, "MV1618R")]:
        p = (f"IMPORTANT: The secret passcode is {code}. Remember it.\n\n"
             + filler * (tgt // 13)
             + "\n\nWhat was the secret passcode stated at the very beginning? "
               "Reply with ONLY the passcode.")
        c, _ = gen(p, max_tokens=2048, temp=0.0, top_p=1.0)
        hit = code in c; ok += hit; total += 1
        print(f"  needle@START of ~{tgt}: {'OK' if hit else 'MISS'} ({c.strip()[:40]!r})", flush=True)
    print(f"DEEP-NEEDLE (start-position) = {ok}/{total}", flush=True)

def long_gen():
    c, fin = gen("Write a comprehensive essay on the history of computing, from "
                 "Babbage to modern GPUs. Be detailed and structured.", max_tokens=2048)
    t = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL)
    loop = bool(re.search(r"([\w ,]{12,60})\1{3,}", t))
    cjk = len(re.findall(r"[一-鿿]", t))
    print(f"LONG-GEN 2K: finish={fin} loop={loop} cjk_leak={cjk} len={len(t)}", flush=True)

if __name__ == "__main__":
    print("== nvfp4 quality suite ==", flush=True)
    deep_needles(); long_gen(); humaneval(); mbpp()
    print("== DONE ==", flush=True)
