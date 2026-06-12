#!/usr/bin/env python3
"""
— tool-calling reliability benchmark for the local Gemma 4 (llama-server).

Engine-agnostic: hits any OpenAI-compatible /v1/chat/completions. Stdlib only
(urllib + concurrent.futures) so it needs no pip install in the conda env.

Categories (each = a distinct tool-calling skill):
  1. single_arg - extract one argument, obvious tool -> must call, correct arg
  2. multi_arg - extract several arguments from prose -> must call, all required args
  3. tool_select - pick the right tool from the full toolset -> must call the EXPECTED tool
  4. no_tool - general question; tools present but irrelevant-> must NOT call (answer in content)
  5. missing_info - tool fits but a required arg is absent -> must NOT fabricate a call (ask instead)

Scoring per trial:
  - call_expected categories (1,2,3): pass = correct tool name + valid-JSON args
        + all required args present + (when given) expected arg values match.
  - no_tool / missing_info: pass = finish_reason != tool_calls AND no tool_calls emitted.

Run at several (temperature, trials) settings to get a statistical rate.
Outputs JSON (per-trial + aggregates) and prints a summary table.
"""
import json
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://127.0.0.1:8000"
MODEL = "gemma-4-31b-it"
CONCURRENCY = 4 # llama-server n_parallel
SETTINGS = [(0.0, 1), (0.7, 3), (1.0, 2)] # (temperature, trials) -> 6 trials/case
MAX_TOKENS = 384

# ---------------------------------------------------------------- toolset
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "city name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. 17*23"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web for a query",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "send_email", "description": "Send an email",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "recipient email address"},
            "subject": {"type": "string"},
            "body": {"type": "string"}},
            "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {
        "name": "create_calendar_event", "description": "Create a calendar event",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "time": {"type": "string", "description": "HH:MM 24h"}},
            "required": ["title", "date", "time"]}}},
    {"type": "function", "function": {
        "name": "get_stock_price", "description": "Get the latest stock price by ticker symbol",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "ticker, e.g. AAPL"}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "translate", "description": "Translate text into a target language",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"},
            "target_language": {"type": "string"}},
            "required": ["text", "target_language"]}}},
    {"type": "function", "function": {
        "name": "set_timer", "description": "Set a countdown timer",
        "parameters": {"type": "object", "properties": {
            "duration_minutes": {"type": "integer"}},
            "required": ["duration_minutes"]}}},
]
REQUIRED = {f["function"]["name"]: f["function"]["parameters"].get("required", [])
            for f in TOOLS}

# ---------------------------------------------------------------- cases
# each: id, category, prompt, expect_tool (None = must NOT call),
# check = optional dict of {arg: (mode, value)} where mode in
# {"eq","ieq","contains","icontains","num"} for value verification.
def C(cid, cat, prompt, expect_tool=None, check=None):
    return {"id": cid, "category": cat, "prompt": prompt,
            "expect_tool": expect_tool, "check": check or {}}

CASES = [
    # 1. single_arg ---------------------------------------------------------
    C("s01", "single_arg", "What's the weather in Tokyo?", "get_weather", {"city": ("ieq", "Tokyo")}),
    C("s02", "single_arg", "Tell me the current weather in Seoul.", "get_weather", {"city": ("ieq", "Seoul")}),
    C("s03", "single_arg", "How's the weather in New York right now?", "get_weather", {"city": ("icontains", "New York")}),
    C("s04", "single_arg", "Compute 17 * 23 using the calculator.", "calculator", {"expression": ("contains", "17")}),
    C("s05", "single_arg", "What is the stock price of AAPL?", "get_stock_price", {"symbol": ("ieq", "AAPL")}),
    C("s06", "single_arg", "Get me Tesla's stock price (ticker TSLA).", "get_stock_price", {"symbol": ("ieq", "TSLA")}),
    C("s07", "single_arg", "Set a timer for 10 minutes.", "set_timer", {"duration_minutes": ("num", 10)}),
    C("s08", "single_arg", "Search the web for 'best ramen in Osaka'.", "web_search", {"query": ("icontains", "ramen")}),
    C("s09", "single_arg", "What's the weather in Paris?", "get_weather", {"city": ("ieq", "Paris")}),
    C("s10", "single_arg", "Start a 25 minute timer.", "set_timer", {"duration_minutes": ("num", 25)}),

    # 2. multi_arg ----------------------------------------------------------
    C("m01", "multi_arg", "Send an email to john@example.com with subject 'Lunch' and body 'Are you free at noon?'",
      "send_email", {"to": ("icontains", "john@example.com"), "subject": ("icontains", "Lunch")}),
    C("m02", "multi_arg", "Email alice@corp.io, subject 'Report', body 'The Q3 report is attached.'",
      "send_email", {"to": ("icontains", "alice@corp.io"), "subject": ("icontains", "Report")}),
    C("m03", "multi_arg", "Create a calendar event titled 'Dentist' on 2026-06-10 at 14:30.",
      "create_calendar_event", {"title": ("icontains", "Dentist"), "date": ("contains", "2026-06-10"), "time": ("contains", "14:30")}),
    C("m04", "multi_arg", "Schedule an event called 'Team Sync' for 2026-07-01 at 09:00.",
      "create_calendar_event", {"title": ("icontains", "Team Sync"), "date": ("contains", "2026-07-01")}),
    C("m05", "multi_arg", "Translate 'Good morning' into Korean.",
      "translate", {"text": ("icontains", "Good morning"), "target_language": ("icontains", "korean")}),
    C("m06", "multi_arg", "Translate the phrase 'thank you very much' to Japanese.",
      "translate", {"target_language": ("icontains", "japanese")}),
    C("m07", "multi_arg", "Get the weather in Berlin in fahrenheit.",
      "get_weather", {"city": ("ieq", "Berlin"), "units": ("ieq", "fahrenheit")}),
    C("m08", "multi_arg", "Send an email to bob@x.com titled 'Hi' that says 'See you tomorrow.'",
      "send_email", {"to": ("icontains", "bob@x.com")}),
    C("m09", "multi_arg", "Search the web for 'python asyncio tutorial' and give me 5 results.",
      "web_search", {"query": ("icontains", "asyncio"), "max_results": ("num", 5)}),
    C("m10", "multi_arg", "Add a calendar event 'Flight to NRT' on 2026-08-15 at 18:45.",
      "create_calendar_event", {"date": ("contains", "2026-08-15"), "time": ("contains", "18:45")}),

    # 3. tool_select (full toolset; must pick the right one) -----------------
    C("t01", "tool_select", "I need to know if it'll be cold in Helsinki today.", "get_weather", {"city": ("icontains", "Helsinki")}),
    C("t02", "tool_select", "Remind me in 5 minutes.", "set_timer", {"duration_minutes": ("num", 5)}),
    C("t03", "tool_select", "How much is 144 divided by 12?", "calculator"),
    C("t04", "tool_select", "Look up the latest news on the Mars mission.", "web_search", {"query": ("icontains", "Mars")}),
    C("t05", "tool_select", "What's Microsoft trading at? (MSFT)", "get_stock_price", {"symbol": ("ieq", "MSFT")}),
    C("t06", "tool_select", "Put 'Yoga class' on my calendar for 2026-06-20 at 07:00.", "create_calendar_event", {"date": ("contains", "2026-06-20")}),
    C("t07", "tool_select", "How do you say 'where is the station' in French?", "translate", {"target_language": ("icontains", "french")}),
    C("t08", "tool_select", "Drop an email to hr@firm.com about my leave, subject 'Leave request', body 'I'll be out next week.'", "send_email", {"to": ("icontains", "hr@firm.com")}),
    C("t09", "tool_select", "Is it raining in Mumbai?", "get_weather", {"city": ("icontains", "Mumbai")}),
    C("t10", "tool_select", "Calculate 2 to the power of 10.", "calculator"),

    # 4. no_tool (must NOT call) -------------------------------------------
    C("n01", "no_tool", "What is the capital of France?", None),
    C("n02", "no_tool", "Explain what a black hole is in two sentences.", None),
    C("n03", "no_tool", "Who wrote 'Pride and Prejudice'?", None),
    C("n04", "no_tool", "Give me a fun fact about octopuses.", None),
    C("n05", "no_tool", "What does the acronym 'HTTP' stand for?", None),
    C("n06", "no_tool", "Tell me a short joke.", None),
    C("n07", "no_tool", "Summarize the plot of Romeo and Juliet briefly.", None),
    C("n08", "no_tool", "What's the difference between TCP and UDP?", None),
    C("n09", "no_tool", "How are you doing today?", None),
    C("n10", "no_tool", "Define the word 'serendipity'.", None),

    # 5. missing_info (required arg absent; must NOT fabricate a call) -------
    C("x01", "missing_info", "Send an email.", None),
    C("x02", "missing_info", "Can you email someone for me?", None),
    C("x03", "missing_info", "Add an event to my calendar.", None),
    C("x04", "missing_info", "Translate this for me.", None),
    C("x05", "missing_info", "What's the stock price?", None),
    C("x06", "missing_info", "Set a timer.", None),
    C("x07", "missing_info", "Look something up for me.", None),
    C("x08", "missing_info", "Schedule a meeting please.", None),
    C("x09", "missing_info", "Tell me the weather.", None),
    C("x10", "missing_info", "Send a message to my boss.", None),
]

# ---------------------------------------------------------------- request
def call(prompt, temperature):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(BASE_URL + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    ch = data["choices"][0]
    msg = ch["message"]
    return {"finish_reason": ch.get("finish_reason"),
            "tool_calls": msg.get("tool_calls"),
            "content": msg.get("content") or "",
            "latency_s": round(dt, 2),
            "usage": data.get("usage", {})}

# ---------------------------------------------------------------- scoring
def check_value(actual, mode, expected):
    a = actual
    if mode == "num":
        try:
            return float(a) == float(expected)
        except (TypeError, ValueError):
            return False
    s = str(a)
    if mode == "eq":
        return s == str(expected)
    if mode == "ieq":
        return s.strip().lower() == str(expected).strip().lower()
    if mode == "contains":
        return str(expected) in s
    if mode == "icontains":
        return str(expected).lower() in s.lower()
    return False

def score(case, resp):
    tcs = resp["tool_calls"] or []
    called = len(tcs) > 0
    expect = case["expect_tool"]
    detail = ""

    if expect is None:
        # must NOT call
        if called:
            return False, f"unexpected call to {tcs[0]['function']['name']}"
        return True, "no call (correct)"

    # must call `expect`
    if not called:
        return False, "no tool_call emitted (expected one)"
    fc = tcs[0]["function"]
    name = fc["name"]
    if name != expect:
        return False, f"wrong tool: {name} (expected {expect})"
    # args valid JSON?
    try:
        args = json.loads(fc["arguments"]) if isinstance(fc["arguments"], str) else fc["arguments"]
    except (json.JSONDecodeError, TypeError):
        return False, "arguments not valid JSON"
    # required present?
    missing = [r for r in REQUIRED.get(name, []) if r not in args or args[r] in (None, "")]
    if missing:
        return False, f"missing required args: {missing}"
    # value checks
    for arg, (mode, exp) in case["check"].items():
        if arg not in args:
            return False, f"check arg '{arg}' absent"
        if not check_value(args[arg], mode, exp):
            return False, f"arg '{arg}'={args[arg]!r} failed {mode} {exp!r}"
    return True, "ok"

# ---------------------------------------------------------------- run
def run():
    jobs = []
    for case in CASES:
        for temp, trials in SETTINGS:
            for k in range(trials):
                jobs.append((case, temp, k))
    print(f"Running {len(jobs)} trials ({len(CASES)} cases x {sum(t for _,t in SETTINGS)} trials) "
          f"at concurrency {CONCURRENCY} ...", flush=True)

    results = []
    def work(job):
        case, temp, k = job
        try:
            resp = call(case["prompt"], temp)
            ok, detail = score(case, resp)
        except Exception as e: # noqa
            return {"id": case["id"], "category": case["category"], "temp": temp,
                    "trial": k, "ok": False, "detail": f"ERROR: {e}",
                    "tool_calls": None, "content": "", "latency_s": None}
        return {"id": case["id"], "category": case["category"], "temp": temp,
                "trial": k, "ok": ok, "detail": detail,
                "tool": (resp["tool_calls"][0]["function"]["name"] if resp["tool_calls"] else None),
                "finish_reason": resp["finish_reason"],
                "latency_s": resp["latency_s"]}

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 20 == 0:
                print(f" {done}/{len(jobs)} done", flush=True)

    # aggregate
    cats = {}
    for r in results:
        cats.setdefault(r["category"], {"pass": 0, "n": 0})
        cats[r["category"]]["n"] += 1
        cats[r["category"]]["pass"] += 1 if r["ok"] else 0
    by_temp = {}
    for r in results:
        by_temp.setdefault(r["temp"], {"pass": 0, "n": 0})
        by_temp[r["temp"]]["n"] += 1
        by_temp[r["temp"]]["pass"] += 1 if r["ok"] else 0
    total_pass = sum(1 for r in results if r["ok"])
    lats = [r["latency_s"] for r in results if r["latency_s"]]

    summary = {
        "model": MODEL, "n_cases": len(CASES), "n_trials": len(results),
        "settings": SETTINGS, "overall_rate": round(total_pass / len(results), 4),
        "by_category": {c: {"rate": round(v["pass"]/v["n"], 4), **v} for c, v in cats.items()},
        "by_temperature": {str(t): {"rate": round(v["pass"]/v["n"], 4), **v} for t, v in by_temp.items()},
        "latency_s": {"mean": round(sum(lats)/len(lats), 2), "max": max(lats)} if lats else {},
        "failures": [r for r in results if not r["ok"]],
    }

    out = sys.argv[1] if len(sys.argv) > 1 else "tests/bench/results-q6.json"
    with open(out, "w") as fh:
        json.dump({"summary": summary, "results": results}, fh, indent=2, ensure_ascii=False)

    # print
    print("\n================ TOOL-CALLING RELIABILITY ================")
    print(f"model={MODEL} cases={len(CASES)} trials={len(results)} overall={summary['overall_rate']*100:.1f}%")
    print("\nby category:")
    for c, v in summary["by_category"].items():
        print(f" {c:13s} {v['rate']*100:6.1f}% ({v['pass']}/{v['n']})")
    print("\nby temperature:")
    for t, v in summary["by_temperature"].items():
        print(f" temp={t:4s} {v['rate']*100:6.1f}% ({v['pass']}/{v['n']})")
    if summary["latency_s"]:
        print(f"\nlatency: mean={summary['latency_s']['mean']}s max={summary['latency_s']['max']}s")
    print(f"\nfailures: {len(summary['failures'])}")
    for fr in summary["failures"][:25]:
        print(f" [{fr['category']}/{fr['id']} t={fr['temp']}] -> {fr.get('tool')}: {fr['detail']}")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    run()
