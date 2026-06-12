#!/usr/bin/env python3
"""
— performance + light-quality benchmark for one served quant.

Run once per quant against a live llama-server (single-stream, CONCURRENCY=1):
  - generation throughput (tok/s) and prefill throughput (tok/s) from llama-server's
    own `timings` block (authoritative, no client-side token guessing);
  - TTFT (time to first token) via SSE streaming;
  - end-to-end latency;
  - VRAM used (GPU0+GPU1) measured via nvidia-smi right after load and after a long request;
  - a light quality proxy: deterministic Q&A (temp=0), substring-checked accuracy.

Usage: perf_bench.py <label> <out.json>
Stdlib only.
"""
import json
import re
import subprocess
import sys
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8000"
MODEL = "gemma-4-31b-it"
REPS = 5 # repetitions for throughput averaging
GEN_TOKENS = 256
LONG_PROMPT_WORDS = 1200 # to exercise prefill at depth

SHORT_PROMPT = "Explain in detail how a hash map works, including collision handling."
LONG_PROMPT = ("Read the following text and then summarize it in three sentences.\n\n"
               + ("The quick brown fox jumps over the lazy dog. " * LONG_PROMPT_WORDS)
               + "\n\nSummary:")

QUALITY = [
    ("What is 17 times 23? Reply with only the number.", r"\b391\b"),
    ("What is 144 divided by 12? Reply with only the number.", r"\b12\b"),
    ("What is 2 to the power of 10? Reply with only the number.", r"\b1024\b"),
    ("What is the capital of France? One word.", r"(?i)paris"),
    ("Who wrote the play 'Romeo and Juliet'? Surname only.", r"(?i)shakespeare"),
    ("What is the chemical symbol for gold? Reply with only the symbol.", r"\bAu\b"),
    ("How many continents are there on Earth? Reply with only the number.", r"\b7\b"),
    ("What is the square root of 81? Reply with only the number.", r"\b9\b"),
    ("In what year did World War II end? Reply with only the year.", r"\b1945\b"),
    ("What is the boiling point of water in Celsius at sea level? Number only.", r"\b100\b"),
]


def vram_used_mib():
    """Sum of memory.used on GPU 0 and 1 (CUDA_VISIBLE_DEVICES GPUs)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        per = {}
        for line in out.strip().splitlines():
            idx, used = [x.strip() for x in line.split(",")]
            per[int(idx)] = int(used)
        return {"gpu0": per.get(0), "gpu1": per.get(1),
                "sum_0_1": (per.get(0, 0) + per.get(1, 0))}
    except Exception as e: # noqa
        return {"error": str(e)}


def chat(prompt, max_tokens, temperature=0.0):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode()
    req = urllib.request.Request(BASE_URL + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    wall = time.time() - t0
    return data, wall


def ttft(prompt, max_tokens=64):
    """Stream and measure seconds to the first token chunk."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(BASE_URL + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                first = time.time() - t0
                break
    return first


def avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def run(label, out_path):
    result = {"label": label, "model": MODEL, "base_url": BASE_URL,
              "reps": REPS, "gen_tokens": GEN_TOKENS}

    # VRAM right after load (idle)
    result["vram_idle_mib"] = vram_used_mib()

    # generation throughput + latency (short prompt, GEN_TOKENS gen)
    gen_tps, prefill_tps_short, walls = [], [], []
    for _ in range(REPS):
        data, wall = chat(SHORT_PROMPT, GEN_TOKENS)
        tm = data.get("timings", {})
        gen_tps.append(tm.get("predicted_per_second"))
        prefill_tps_short.append(tm.get("prompt_per_second"))
        walls.append(wall)
    result["gen_tokps"] = avg(gen_tps)
    result["latency_s_256tok"] = avg(walls)
    result["prefill_tokps_shortprompt"] = avg(prefill_tps_short)

    # prefill throughput at depth (long prompt)
    prefill_long, prompt_n = [], []
    for _ in range(3):
        data, _ = chat(LONG_PROMPT, 16)
        tm = data.get("timings", {})
        prefill_long.append(tm.get("prompt_per_second"))
        prompt_n.append(tm.get("prompt_n"))
    result["prefill_tokps_longprompt"] = avg(prefill_long)
    result["longprompt_tokens"] = prompt_n[0] if prompt_n else None

    # TTFT
    result["ttft_s_shortprompt"] = avg([ttft(SHORT_PROMPT) for _ in range(3)])
    result["ttft_s_longprompt"] = avg([ttft(LONG_PROMPT) for _ in range(2)])

    # VRAM after a long request (KV populated)
    result["vram_after_longctx_mib"] = vram_used_mib()

    # quality probe
    q_pass, q_detail = 0, []
    for q, pat in QUALITY:
        data, _ = chat(q, 256)
        msg = data["choices"][0]["message"]
        text = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
        ok = bool(re.search(pat, text))
        q_pass += 1 if ok else 0
        q_detail.append({"q": q, "ok": ok, "answer": (msg.get("content") or "").strip()[:80]})
    result["quality_accuracy"] = round(q_pass / len(QUALITY), 4)
    result["quality_pass"] = f"{q_pass}/{len(QUALITY)}"
    result["quality_detail"] = q_detail

    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"\n==== {label} ====")
    print(f" gen throughput: {result['gen_tokps']} tok/s")
    print(f" prefill (short): {result['prefill_tokps_shortprompt']} tok/s")
    print(f" prefill (long {result['longprompt_tokens']}t): {result['prefill_tokps_longprompt']} tok/s")
    print(f" TTFT short / long: {result['ttft_s_shortprompt']}s / {result['ttft_s_longprompt']}s")
    print(f" latency (256 gen): {result['latency_s_256tok']}s")
    print(f" VRAM idle (G0+G1): {result['vram_idle_mib'].get('sum_0_1')} MiB "
          f"(g0={result['vram_idle_mib'].get('gpu0')}, g1={result['vram_idle_mib'].get('gpu1')})")
    print(f" VRAM after longctx: {result['vram_after_longctx_mib'].get('sum_0_1')} MiB")
    print(f" quality: {result['quality_pass']} ({result['quality_accuracy']*100:.0f}%)")
    print(f" wrote {out_path}")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    out = sys.argv[2] if len(sys.argv) > 2 else f"tests/bench/perf-{label}.json"
    run(label, out)
