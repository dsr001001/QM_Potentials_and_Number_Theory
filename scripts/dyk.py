"""Draft 'Did you know?' facts for the book's slots (one API call)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import llm
slots = json.load(open("/home/claude/book/dyk_slots.json"))
listing = "\n".join(f"{i}: Chapter {s['chapter']} \"{s['ctitle']}\", before section {s['sec']} \"{s['stitle']}\"" for i, s in enumerate(slots))
prompt = f"""You are writing the "Did you know?" boxes for a textbook that takes a first-year physics student to the point of understanding a research paper on the Riemann zeta zeros, quantum potentials (the Wu--Sprung inverse-spectral construction), the explicit formula, random matrices, numerical methods and the statistics of prediction (memorization vs generalization).

There are {len(slots)} boxes, one per slot below. For each slot write one fun, surprising, true fact related to that chapter's subject (preferably to the section it sits before). Mix: history and people, surprising numbers, real-world applications, connections between physics and number theory, famous computations, curiosities of notation. Do not repeat a fact; spread topics.

Hard rules:
- Accuracy above all. Only well-documented facts. No apocryphal anecdotes, no quotations unless the wording is well attested (and then keep them short), no invented numbers. If unsure of a date or figure, leave it out or phrase it safely ("in the 1970s").
- 40 to 90 words each, friendly HRW tone, understandable by a first-year physics student at that point of the book.
- LaTeX-safe text: math in $...$, use \\emph{{}} for emphasis, escape % and &. Never use an em dash (no --- and no Unicode em dash); use commas, colons, semicolons.
- A short hook title of 2 to 6 words.

Output only a JSON array of objects: {{"slot": <int>, "title": "...", "text": "...", "source": "<where this is documented, for the editor; not printed>"}}.

Slots:
{listing}"""
text, usage, c, stop = llm.call("dyk-facts", "claude-opus-5-5", [{"type": "text", "text": "You are a careful science writer and fact-checker."}], prompt, max_tokens=64000, effort="medium")
open("/home/claude/book/dyk_raw.txt", "w").write(text)
print("stop", stop, "usd %.3f" % c, "spent %.2f" % llm.spent())
