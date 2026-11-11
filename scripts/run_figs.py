"""Run figure scripts (all, or those whose name starts with a slug) from tex/figs."""
import glob, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "tex", "figs")
pref = sys.argv[1] if len(sys.argv) > 1 else ""
env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "code"), MPLBACKEND="Agg")
for f in sorted(glob.glob(os.path.join(FIGS, "scripts", pref + "*.py"))):
    r = subprocess.run([sys.executable, f], cwd=FIGS, env=env, capture_output=True, text=True, timeout=600)
    print(("OK  " if r.returncode == 0 else "FAIL"), os.path.basename(f), r.stderr.strip().splitlines()[-1] if r.returncode else "")
