"""Log an in-session (no API) review: compile the chapter alone, run its checks, append a row, commit, push.
Usage: python scripts/log_session_review.py <slug> "<issues summary>" <n_fixes>"""
import os, subprocess, sys, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slug, summary, nfix = sys.argv[1], sys.argv[2], sys.argv[3]
sh = lambda c, **k: subprocess.run(c, shell=True, cwd=ROOT, capture_output=True, text=True, **k)
comp = sh(f"python3 scripts/chapter_errors.py {slug}").stdout.strip().splitlines()
chk = sh(f"python3 scripts/run_checks.py {slug}", timeout=1800).stdout.strip().splitlines()[-1].split(" ", 1)[1]
row = (f"| {slug} | in-session line-by-line read (Claude Opus 5.5, no API): {summary} | {nfix} edits by hand | "
       f"{chk}; standalone compile: {comp[0].split(': ',1)[1] if comp else '?'} | 0 (plan) | {datetime.datetime.now():%Y-%m-%d %H:%M} |\n")
open(os.path.join(ROOT, "REVIEW_LOG.md"), "a").write(row)
print(row)
sh(f'git add tex/chapters REVIEW_LOG.md work scripts && git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit -q -m "In-session review: {slug}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01SuMwdyJD2Vh7EGuCvcG9hm" && git push -q origin HEAD:main')
