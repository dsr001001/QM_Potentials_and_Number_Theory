"""Targeted critique pass: directional, conditional and self-consistency claims.

Critique only. Nothing is patched automatically: each item is verified by hand
(or by computation) before any edit.  Usage:
  python scripts/direction_check.py submit            # one batch, all chapters
  python scripts/direction_check.py collect <batch>   # writes work/<slug>.direction.json
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import llm, book

INSTR = """You are checking one chapter of the textbook for a narrow class of errors that numerical assertion scripts cannot catch. Ignore style, clarity, layout, pedagogy and bible-format rules entirely. The `didyouknow` environment is defined in the preamble; do not comment on those boxes.

Go through the chapter sentence by sentence and report ONLY statements that are actually wrong, in these classes:

DIRECTION: a sign, inequality, ordering or monotonic trend stated the wrong way round ("more" for "fewer", "increases" for "decreases", "above" for "below", "positive" for "negative", "upper bound" for "lower bound", "before" for "after", wrong index order), or a causal or heuristic argument that runs backwards.
CONDITION: a theorem, lemma or general statement asserted without a hypothesis it needs, or asserted more strongly than is true (a bound stated as an equality; "all" where only "some"; a special case stated as general), such that a counterexample exists. Give the counterexample.
SELFCONTRA: the prose contradicts the chapter's own tables, equations, sample-problem results, figure captions or an earlier or later sentence.
PAPER: a statement about what the paper says, does or reports that disagrees with the paper source given above (wrong number, wrong section, wrong object, a claim the paper does not make).
MATH: a derivation step or a formula that is wrong (not merely terse).

For each candidate, work the claim out yourself before reporting it: test the inequality with numbers, compute the counterexample, look up the table row, find the passage of the paper. Do not report something you have not checked, and do not report matters of taste. If the chapter is clean, return an empty list. Precision matters more than recall padding: every false alarm costs a human check.

Output a single JSON array and nothing else. Each item:
{"class": "DIRECTION|CONDITION|SELFCONTRA|PAPER|MATH", "confidence": "certain|likely", "quote": "<exact substring of the chapter, one or two sentences, copied verbatim>", "problem": "<what is wrong, with the check you did: numbers, counterexample or paper passage>", "fix": "<the corrected wording>"}"""


def prompt(slug):
    n, title = book.BY_SLUG[slug]
    ch = open(book.chfile(slug)).read()
    ch = re.sub(r"\n% DYK-BEGIN.*?% DYK-END\n", "\n", ch, flags=re.S)
    return f"{INSTR}\n\nChapter {n} `{slug}` \"{title}\":\n\n<chapter>\n{ch}\n</chapter>"


if sys.argv[1] == "submit":
    sl = [s for _, s, _ in book.CHAPTERS] if len(sys.argv) < 3 else sys.argv[2].split(",")
    reqs = [(f"direction__{s}", book.CRITIC_MODEL, book.SYSTEM, prompt(s), 64000, "high") for s in sl]
    print(llm.submit_batch(reqs, "direction-check"))
elif sys.argv[1] == "collect":
    bid = sys.argv[2]
    models = {}
    for l in open(os.path.join(llm.ROOT, "ledger", "batches.jsonl")):
        d = json.loads(l)
        if d["id"] == bid:
            models = d["models"]
    for cid, (text, stop) in llm.collect_batch(bid, models).items():
        slug = cid.split("__")[1]
        if text is None:
            print(cid, "FAILED", stop); continue
        open(os.path.join(book.WORK, f"{slug}.direction.raw.txt"), "w").write(text)
        m = re.search(r"\[.*\]", text, re.S)
        items = json.loads(m.group(0)) if m else []
        json.dump(items, open(os.path.join(book.WORK, f"{slug}.direction.json"), "w"), indent=1, ensure_ascii=False)
        print(cid, stop, len(items))
    print(f"spent so far {llm.spent():.2f}")
