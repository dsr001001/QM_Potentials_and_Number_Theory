"""Compile each chapter standalone; print LaTeX errors with chapter line numbers."""
import glob, os, re, subprocess, sys
TEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tex")
os.makedirs("/tmp/chk", exist_ok=True)
files = sorted(glob.glob(os.path.join(TEX, "chapters", "*.tex")))
if len(sys.argv) > 1:
    files = [f for f in files if any(s in f for s in sys.argv[1:])]
labels = ""
for f in files:
    name = os.path.basename(f)[:-4]
    doc = (r"\documentclass[11pt,openany]{book}\input{%s/preamble}\graphicspath{{%s/figs/}}\begin{document}"
           r"\setcounter{chapter}{%d}\input{%s}\end{document}" % (TEX, TEX, int(name[:2]) - 1, f[:-4]))
    open("/tmp/chk/%s.tex" % name, "w").write(doc)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error" if False else "-interaction=nonstopmode", name + ".tex"],
                       cwd="/tmp/chk", capture_output=True, timeout=300)
    log = open("/tmp/chk/%s.log" % name, errors="replace").read()
    errs = re.findall(r"^! (.*?)\n.*?^l\.(\d+) (.*?)$", log, re.M | re.S)
    print(f"{name}: {len(errs)} errors")
    seen = set()
    for msg, line, ctx in errs:
        key = (msg[:60], line)
        if key in seen: continue
        seen.add(key)
        print(f"   line {line}: {msg.strip()[:90]} | {ctx.strip()[:70]}")
