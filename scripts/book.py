"""Chapter pipeline: draft, critique, revise.

Usage:
  python book.py draft <slug> [--sync]          # one chapter, synchronous (pilot)
  python book.py draft-batch <slug,...|all>     # submit a batch of drafts
  python book.py critique-batch <slug,...|all>
  python book.py revise-batch <slug,...|all>
  python book.py collect <batch_id>             # write results of a finished batch
  python book.py status
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import llm

ROOT = llm.ROOT
BIBLE = open(os.path.join(ROOT, "bible", "BIBLE.md")).read()
PAPER = open("/home/claude/book/paper_R1.tex").read()   # not in the public repo
TEX = os.path.join(ROOT, "tex")
CH_DIR = os.path.join(TEX, "chapters")
FIG_DIR = os.path.join(TEX, "figs", "scripts")
CHK_DIR = os.path.join(TEX, "checks")
WORK = os.path.join(ROOT, "work")          # raw responses, critiques, patches
for d in (CH_DIR, FIG_DIR, CHK_DIR, WORK):
    os.makedirs(d, exist_ok=True)

DRAFT_MODEL = "claude-opus-5-5"
CRITIC_MODEL = "claude-fable-5-2"
REVISE_MODEL = "claude-opus-5-5"

CHAPTERS = [(int(m.group(1)), m.group(2), m.group(3).strip())
            for m in re.finditer(r"^\*\*(\d+)\. `([a-z\-]+)` (.+?)\*\*", BIBLE, re.M)]
BY_SLUG = {s: (n, t) for n, s, t in CHAPTERS}

ROLE = """You are a member of the writing team of a textbook. You write with the clarity and warmth of Halliday, Resnick and Walker and with the completeness of Riley, Hobson and Bence: no logical jumps. The book bible below is your contract; follow it exactly. The full LaTeX source of the research paper that the book prepares readers for follows the bible."""

SYSTEM = llm.system_blocks(ROLE, "# BOOK BIBLE\n\n" + BIBLE, "# THE PAPER (LaTeX source, revision R1)\n\n" + PAPER)


def chfile(slug):
    n, _ = BY_SLUG[slug]
    return os.path.join(CH_DIR, f"{n:02d}-{slug}.tex")


def draft_prompt(slug):
    n, title = BY_SLUG[slug]
    return f"""Write Chapter {n}, slug `{slug}`, title "{title}".

Follow the contract for this chapter in Section 7 of the bible, the anatomy of Section 3, the output format of Section 4, notation of Section 6, citation keys of Section 9, running examples of Section 10 and the honesty rules of Section 11.

Remember:
- The reader has only first-year physics plus the chapters this one may assume. Define everything else from scratch. No logical jumps: every derivation line by line.
- Write the complete chapter, not an outline. Be generous with worked Sample Problems (at least 6), Checkpoints (at least 5), figures (3 to 8) and Problems (15 to 40).
- The "Paper connection" section must cite specific sections, equations, tables and figures of the paper by their R1 numbers.
- Every number you state must be verified in the CHECKS script or quoted from the paper with its location.
- Never use an em dash.

Plan briefly (a short outline is enough); spend your effort on the written chapter itself, which is long.

Respond with exactly the three parts: =====CHAPTER=====, =====FIGURES=====, =====CHECKS=====."""

CONTINUE = "Your previous message was cut off by the output limit. Continue exactly where it stopped, with no repetition and no preamble: the next character you write follows the last character of your previous message. Keep the same three-part format (the remaining marker lines still to come)."


def draft_with_continuation(slug, max_rounds=3):
    msgs = [{"role": "user", "content": draft_prompt(slug)}]
    text, total = "", 0.0
    for r in range(max_rounds):
        t, usage, c, stop = llm.call(f"draft:{slug}:r{r}", DRAFT_MODEL, SYSTEM, msgs)
        text += t; total += c
        print(f"  round {r}: stop {stop}, out {usage.output_tokens}, thinking {usage.output_tokens_details.thinking_tokens}, usd {c:.3f}", flush=True)
        if stop != "max_tokens":
            break
        msgs = [msgs[0], {"role": "assistant", "content": text.rstrip()}, {"role": "user", "content": CONTINUE}]
    return text, total


CRITIQUE_INSTR = """You are the critical reviewer for one chapter of the textbook. Read it as (1) a meticulous mathematical physicist checking every equation and number, (2) a first-year physics student who knows only what the bible says this chapter may assume, and (3) the book editor enforcing the bible.

Report every problem you find, most serious first. Categories:
- ERROR: a mathematical, physical or numerical mistake; a wrong statement about the paper; a misquoted number.
- JUMP: a step a first-year student could not follow; a concept or symbol used before it is defined in this chapter or in an assumed chapter; "it can be shown" style gaps; a missing intermediate line.
- OVERCLAIM: stating as proved what is conjectural or heuristic; overstating the paper.
- RULE: a violation of the bible (anatomy, label prefixes, citation keys, environments, em dashes, missing figure scripts, missing checks for a stated number, label references to other chapters' internal labels).
- CLARITY: confusing exposition, missing example or figure that a student needs.
- LATEX: code that will not compile.

Output a single JSON array and nothing else. Each element:
{"category": "...", "severity": "high|medium|low", "quote": "<exact verbatim text from the chapter, 5 to 40 words, that locates the problem; must be a substring of the chapter>", "problem": "<what is wrong>", "fix": "<precise instruction or replacement text>"}
Aim for completeness: a typical good chapter still has 15 to 40 issues. Do not report matters of taste."""


def critique_prompt(slug):
    n, title = BY_SLUG[slug]
    ch = open(chfile(slug)).read()
    chk = open(os.path.join(CHK_DIR, f"{slug}.py")).read() if os.path.exists(os.path.join(CHK_DIR, f"{slug}.py")) else ""
    rep = os.path.join(WORK, f"{slug}.checkreport.txt")
    rep = open(rep).read() if os.path.exists(rep) else "(not yet run)"
    return f"""{CRITIQUE_INSTR}

Chapter {n} `{slug}` "{title}":

<chapter>
{ch}
</chapter>

<checks_script>
{chk}
</checks_script>

<checks_run_report>
{rep}
</checks_run_report>"""


REVISE_INSTR = """You are revising one chapter of the textbook in response to a reviewer's issue list and the result of running the chapter's verification script. Fix every issue that is valid; for an issue you judge invalid, skip it and say why in the notes.

Output a single JSON object and nothing else:
{"patches": [{"old": "<exact substring of the current chapter, long enough to be unique>", "new": "<replacement text>"}], "new_checks": "<complete replacement CHECKS python script, or empty string if unchanged>", "new_figures": {"<slug_name>": "<complete python script>"}, "notes": "<which issues were skipped and why>"}

Rules: each "old" must match the chapter text exactly, including whitespace and backslashes, and must occur exactly once. Patches are applied in order to the evolving text. To add new material, patch an anchor (for example the line before) and include the anchor in "new". Keep every rule of the bible (no em dashes, label prefixes, citation keys, no logical jumps). Every number you add must be covered by new_checks."""


def revise_prompt(slug):
    n, title = BY_SLUG[slug]
    ch = open(chfile(slug)).read()
    issues = open(os.path.join(WORK, f"{slug}.critique.json")).read()
    chk = open(os.path.join(CHK_DIR, f"{slug}.py")).read()
    rep = os.path.join(WORK, f"{slug}.checkreport.txt")
    rep = open(rep).read() if os.path.exists(rep) else "(not run)"
    return f"""{REVISE_INSTR}

Chapter {n} `{slug}` "{title}":

<chapter>
{ch}
</chapter>

<checks_script>
{chk}
</checks_script>

<checks_run_report>
{rep}
</checks_run_report>

<issues>
{issues}
</issues>"""


# ------------------------------------------------------------ parsing
def normalize(ch):
    """Strip a redundant itemize directly inside keyideas/prerequisites; replace em dashes."""
    for env in ("keyideas", "prerequisites"):
        ch = re.sub(r"(\\begin\{%s\}\s*)\\begin\{itemize\}(.*?)\\end\{itemize\}(\s*\\end\{%s\})" % (env, env),
                    r"\1\2\3", ch, flags=re.S)
    ch = ch.replace("\u2014", ", ").replace("---", ", ")
    # floats are not allowed inside boxes: pin them with [H]
    boxes = "sample|checkpoint|keyideas|pitfall|beyond|chapterintro|prerequisites"
    def pin(m):
        return re.sub(r"\\begin\{(table|figure)\}(\[[^\]]*\])?", r"\\begin{\1}[H]", m.group(0))
    ch = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % boxes, pin, ch, flags=re.S)
    return ch


def save_draft(slug, text):
    open(os.path.join(WORK, f"{slug}.draft.raw.txt"), "w").write(text)
    parts = re.split(r"^=====(CHAPTER|FIGURES|CHECKS)=====\s*$", text, flags=re.M)
    d = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    ch = d.get("CHAPTER", "").strip()
    ch = re.sub(r"^```(?:latex|tex)?\s*\n|\n```\s*$", "", ch)
    open(chfile(slug), "w").write(normalize(ch) + "\n")
    figs = re.findall(r"^### FIG (\S+)\s*\n```python\s*\n(.*?)\n```", d.get("FIGURES", ""), re.M | re.S)
    for name, code in figs:
        open(os.path.join(FIG_DIR, f"{name}.py"), "w").write(code + "\n")
    chk = d.get("CHECKS", "").strip()
    chk = re.sub(r"^```(?:python)?\s*\n|\n```\s*$", "", chk)
    open(os.path.join(CHK_DIR, f"{slug}.py"), "w").write(chk + "\n")
    return len(ch.split()), len(figs), len(chk.splitlines())


def save_critique(slug, text):
    m = re.search(r"\[.*\]", text, re.S)
    issues = json.loads(m.group(0)) if m else []
    json.dump(issues, open(os.path.join(WORK, f"{slug}.critique.json"), "w"), indent=1, ensure_ascii=False)
    return len(issues)


def save_revision(slug, text):
    m = re.search(r"\{.*\}", text, re.S)
    obj = json.loads(m.group(0))
    json.dump(obj, open(os.path.join(WORK, f"{slug}.revision.json"), "w"), indent=1, ensure_ascii=False)
    ch = open(chfile(slug)).read()
    ok, bad = 0, []
    for p in obj.get("patches", []):
        if ch.count(p["old"]) == 1:
            ch = ch.replace(p["old"], p["new"]); ok += 1
        else:
            bad.append(p)
    open(chfile(slug), "w").write(normalize(ch))
    if obj.get("new_checks"):
        open(os.path.join(CHK_DIR, f"{slug}.py"), "w").write(obj["new_checks"].strip() + "\n")
    for name, code in (obj.get("new_figures") or {}).items():
        open(os.path.join(FIG_DIR, f"{name}.py"), "w").write(code.strip() + "\n")
    json.dump(bad, open(os.path.join(WORK, f"{slug}.patchfail.json"), "w"), indent=1, ensure_ascii=False)
    return ok, len(bad)


# ------------------------------------------------------------ commands
def slugs(arg):
    return [s for _, s, _ in CHAPTERS] if arg == "all" else arg.split(",")


def main():
    cmd = sys.argv[1]
    if cmd == "draft" and "--sync" in sys.argv:
        slug = sys.argv[2]
        text, c = draft_with_continuation(slug)
        print(slug, "usd %.3f" % c, "words/figs/checklines", save_draft(slug, text))
    elif cmd == "critique" and "--sync" in sys.argv:
        slug = sys.argv[2]
        text, usage, c, stop = llm.call(f"critique:{slug}", CRITIC_MODEL, SYSTEM, critique_prompt(slug), max_tokens=64000, effort="medium")
        open(os.path.join(WORK, f"{slug}.critique.raw.txt"), "w").write(text)
        print(slug, "stop", stop, "usd %.3f" % c, "issues", save_critique(slug, text))
    elif cmd == "revise" and "--sync" in sys.argv:
        slug = sys.argv[2]
        text, usage, c, stop = llm.call(f"revise:{slug}", REVISE_MODEL, SYSTEM, revise_prompt(slug))
        open(os.path.join(WORK, f"{slug}.revise.raw.txt"), "w").write(text)
        print(slug, "stop", stop, "usd %.3f" % c, "patches ok/failed", save_revision(slug, text))
    elif cmd.endswith("-batch"):
        kind = cmd.split("-")[0]
        reqs = []
        for s in slugs(sys.argv[2]):
            if kind == "draft":
                reqs.append((f"draft__{s}", DRAFT_MODEL, SYSTEM, draft_prompt(s), 100000, "high"))
            elif kind == "critique":
                reqs.append((f"critique__{s}", CRITIC_MODEL, SYSTEM, critique_prompt(s), 64000, "medium"))
            elif kind == "cont":
                raw = open(os.path.join(WORK, f"{s}.draft.raw.txt")).read().rstrip()
                msgs = [{"role": "user", "content": draft_prompt(s)}, {"role": "assistant", "content": raw},
                        {"role": "user", "content": CONTINUE}]
                reqs.append((f"cont__{s}", DRAFT_MODEL, SYSTEM, msgs, 128000, "low"))
            elif kind == "revise":
                reqs.append((f"revise__{s}", REVISE_MODEL, SYSTEM, revise_prompt(s), 128000, "medium"))
        print("submitted", llm.submit_batch(reqs, f"{kind}:{sys.argv[2]}"), len(reqs))
    elif cmd == "collect":
        bid = sys.argv[2]
        meta = [json.loads(l) for l in open(os.path.join(ROOT, "ledger", "batches.jsonl")) if json.loads(l)["id"] == bid][0]
        res = llm.collect_batch(bid, meta["models"])
        for cid, (text, stop) in res.items():
            kind, s = cid.split("__")
            if text is None:
                print(cid, "FAILED", stop); continue
            if kind == "cont":
                prev = open(os.path.join(WORK, f"{s}.draft.raw.txt")).read().rstrip()
                text = prev + text
                kind = "draft"
            open(os.path.join(WORK, f"{s}.{kind}.raw.txt"), "w").write(text)
            try:
                out = {"draft": save_draft, "critique": save_critique, "revise": save_revision}[kind](s, text)
            except Exception as e:
                out = f"PARSE ERROR {e}"
            print(cid, stop, out)
        print("spent so far %.2f" % llm.spent())
    elif cmd == "status":
        for l in open(os.path.join(ROOT, "ledger", "batches.jsonl")):
            m = json.loads(l); b = llm.batch_status(m["id"])
            print(m["id"], m["label"], b.processing_status, b.request_counts)
        print("spent %.2f" % llm.spent())


if __name__ == "__main__":
    main()
