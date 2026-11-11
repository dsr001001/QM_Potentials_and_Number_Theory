"""Run a chapter's CHECKS script; write the report to work/<slug>.checkreport.txt."""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "code"), MPLBACKEND="Agg")
for slug in sys.argv[1:]:
    f = os.path.join(ROOT, "tex", "checks", slug + ".py")
    try:
        r = subprocess.run([sys.executable, f], cwd=os.path.join(ROOT, "tex", "checks"), env=env,
                           capture_output=True, text=True, timeout=900)
        rep = f"exit code {r.returncode}\nSTDOUT:\n{r.stdout[-6000:]}\nSTDERR:\n{r.stderr[-6000:]}"
    except subprocess.TimeoutExpired:
        rep = "TIMEOUT after 900 s"
    open(os.path.join(ROOT, "work", slug + ".checkreport.txt"), "w").write(rep)
    print(slug, rep.splitlines()[0])
