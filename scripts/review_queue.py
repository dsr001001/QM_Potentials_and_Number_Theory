"""Review chapters one at a time: Fable critique -> Opus revision -> checks -> figures -> log -> commit.

Stops before a chapter if the ledger total plus the per-chapter estimate would pass the cap.
Log: REVIEW_LOG.md (one row per chapter)."""
import json, os, subprocess, sys, time, datetime
sys.path.insert(0, os.path.dirname(__file__))
import llm
ROOT = llm.ROOT
QUEUE = sys.argv[1].split(",")
EST = 1.5
LOG = os.path.join(ROOT, "REVIEW_LOG.md")
env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "code"), MPLBACKEND="Agg")


def sh(cmd, cwd=ROOT, timeout=3600):
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    return r.stdout + r.stderr


def run_batch(kind, slug):
    out = sh(f"python3 scripts/book.py {kind}-batch {slug}")
    bid = [l for l in out.split() if l.startswith("msgbatch_")][0]
    while llm.batch_status(bid).processing_status != "ended":
        time.sleep(30)
    return sh(f"python3 scripts/book.py collect {bid}")


if not os.path.exists(LOG):
    open(LOG, "w").write("# Chapter review log\n\nSecond review by Claude Fable 5.2 (critique) and Claude Opus 5.5 (revision), one chapter at a time. Earlier rounds are recorded in `work/round1`, `work/round2`.\n\n| Chapter | Issues found (high/med/low) | Patches applied / failed | Checks after revision | Cost (USD) | Finished |\n|---|---|---|---|---|---|\n")

for slug in QUEUE:
    s0 = llm.spent()
    if s0 + EST > llm.CAP_USD:
        print(f"STOP before {slug}: spent {s0:.2f}, cap {llm.CAP_USD}", flush=True)
        open(LOG, "a").write(f"| {slug} | not reviewed: credit exhausted (spent {s0:.2f}) | | | | |\n")
        break
    print(f"== {slug}: critique", flush=True)
    mt = os.path.getmtime(os.path.join(ROOT, "work", f"{slug}.critique.json")) if os.path.exists(os.path.join(ROOT, "work", f"{slug}.critique.json")) else 0
    try:
        out = run_batch("critique", slug)
    except Exception as e:
        out = f"API ERROR {e}"
    print(out.strip()[-400:], flush=True)
    cj = os.path.join(ROOT, "work", f"{slug}.critique.json")
    if "FAILED" in out or "API ERROR" in out or not os.path.exists(cj) or os.path.getmtime(cj) <= mt:
        print(f"API STOPPED at {slug}", flush=True)
        open(LOG, "a").write(f"| {slug} | API refused (credit exhausted at ledger {llm.spent():.2f}); handed to in-session review | | | | |\n")
        sh('git add -A && git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit -q -m "Review log: API credit exhausted\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01SuMwdyJD2Vh7EGuCvcG9hm" && git push -q origin HEAD:main')
        break
    iss = json.load(open(cj))
    sev = [sum(1 for i in iss if i.get("severity") == k) for k in ("high", "medium", "low")]
    print(f"== {slug}: revise ({len(iss)} issues)", flush=True)
    try:
        out = run_batch("revise", slug)
    except Exception as e:
        out = f"API ERROR {e}"
    if f"revise__{slug} end_turn" not in out and f"revise__{slug} max_tokens" not in out:
        print(f"API STOPPED during revision of {slug}: {out[-300:]}", flush=True)
        open(LOG, "a").write(f"| {slug} | {len(iss)} issues found by Fable; revision refused by API (credit exhausted); handed to in-session revision | | | | |\n")
        sh('git add -A && git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit -q -m "Review log: API credit exhausted during revision\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01SuMwdyJD2Vh7EGuCvcG9hm" && git push -q origin HEAD:main')
        break
    line = [l for l in out.splitlines() if l.startswith(f"revise__{slug}")]
    pat = line[0].split()[-2:] if line else ["?", "?"]
    chk = sh(f"python3 scripts/run_checks.py {slug}", timeout=1800).strip().splitlines()[-1]
    figs = sh(f"for f in tex/figs/scripts/{slug}_*.py; do (cd tex/figs && timeout 600 python3 scripts/$(basename $f) >/dev/null 2>&1) || echo FAIL $f; done")
    nfail = figs.count("FAIL")
    cost = llm.spent() - s0
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    chk_short = chk.split(" ", 1)[1] if " " in chk else chk
    open(LOG, "a").write(f"| {slug} | {len(iss)} ({sev[0]}/{sev[1]}/{sev[2]}) | {pat[0].strip('(,')} / {pat[1].strip(')')} | {chk_short}{'; figure failures: ' + str(nfail) if nfail else ''} | {cost:.2f} | {when} |\n")
    sh(f'git add -A && git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit -q -m "Fable review + revision: {slug}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01SuMwdyJD2Vh7EGuCvcG9hm" && git push -q origin HEAD:main')
    print(f"== {slug}: done, cost {cost:.2f}, total {llm.spent():.2f}; checks: {chk_short}; fig failures {nfail}", flush=True)
print("QUEUE FINISHED", flush=True)
