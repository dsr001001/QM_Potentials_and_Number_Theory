"""Thin wrapper around the Claude API for the book pipeline.

Every call (sync or batch) is logged to ledger/costs.csv with token counts and
cost; no call is made once the ledger total reaches CAP_USD.
The API key is read from $BOOK_API_KEY_FILE (default ~/book/.apikey); it is
never stored in this repository.
"""
import csv, json, os, time, datetime
import anthropic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "ledger", "costs.csv")
CAP_USD = 79.5
KEYFILE = os.environ.get("BOOK_API_KEY_FILE", "/home/claude/book/.apikey")

# USD per million tokens: input, output, cache read, cache write (5 min)
PRICES = {
    "claude-fable-5-2": (8.0, 40.0, 0.20, 10.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=open(KEYFILE).read().strip(), max_retries=5)
    return _client


def cost(model, usage, batch=False):
    pi, po, pcr, pcw = PRICES[model]
    u = usage if isinstance(usage, dict) else usage.model_dump()
    c = (u.get("input_tokens", 0) * pi + u.get("output_tokens", 0) * po
         + (u.get("cache_read_input_tokens") or 0) * pcr
         + (u.get("cache_creation_input_tokens") or 0) * pcw) / 1e6
    return c * (0.5 if batch else 1.0)


def spent():
    if not os.path.exists(LEDGER):
        return 0.0
    with open(LEDGER) as f:
        return sum(float(r["usd"]) for r in csv.DictReader(f))


def log(tag, model, usage, batch=False):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    u = usage if isinstance(usage, dict) else usage.model_dump()
    new = not os.path.exists(LEDGER)
    c = cost(model, u, batch)
    with open(LEDGER, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "tag", "model", "batch", "input", "cache_read", "cache_write", "output", "usd"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), tag, model, int(batch),
                    u.get("input_tokens", 0), u.get("cache_read_input_tokens") or 0,
                    u.get("cache_creation_input_tokens") or 0, u.get("output_tokens", 0), f"{c:.4f}"])
    return c


def guard(estimate=0.0):
    s = spent()
    if s + estimate > CAP_USD:
        raise RuntimeError(f"budget cap: spent {s:.2f} + estimate {estimate:.2f} > {CAP_USD}")
    return s


def system_blocks(*texts):
    """System prompt as blocks; the last block carries the cache breakpoint."""
    blocks = [{"type": "text", "text": t} for t in texts]
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def params(model, system, user, max_tokens=128000, effort="medium"):
    msgs = user if isinstance(user, list) else [{"role": "user", "content": user}]
    return dict(model=model, max_tokens=max_tokens, system=system,
                thinking={"type": "adaptive"}, output_config={"effort": effort},
                messages=msgs)


def call(tag, model, system, user, max_tokens=128000, effort="medium"):
    """Streaming synchronous call; returns final text."""
    guard()
    with client().messages.stream(**params(model, system, user, max_tokens, effort)) as s:
        msg = s.get_final_message()
    c = log(tag, model, msg.usage)
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text, msg.usage, c, msg.stop_reason


def submit_batch(requests, label):
    """requests: list of (custom_id, model, system, user, max_tokens, effort)."""
    guard()
    reqs = [{"custom_id": cid, "params": params(m, sy, u, mt, ef)} for cid, m, sy, u, mt, ef in requests]
    b = client().messages.batches.create(requests=reqs)
    os.makedirs(os.path.join(ROOT, "ledger"), exist_ok=True)
    with open(os.path.join(ROOT, "ledger", "batches.jsonl"), "a") as f:
        f.write(json.dumps({"id": b.id, "label": label, "n": len(reqs),
                            "models": {cid: m for cid, m, *_ in requests},
                            "time": time.time()}) + "\n")
    return b.id


def batch_status(bid):
    return client().messages.batches.retrieve(bid)


def collect_batch(bid, models):
    """Returns {custom_id: (text, stop_reason)}; logs cost of each result once."""
    out = {}
    for r in client().messages.batches.results(bid):
        if r.result.type != "succeeded":
            out[r.custom_id] = (None, r.result.type)
            continue
        msg = r.result.message
        log(f"batch:{bid}:{r.custom_id}", models[r.custom_id], msg.usage, batch=True)
        out[r.custom_id] = ("".join(b.text for b in msg.content if b.type == "text"), msg.stop_reason)
    return out
