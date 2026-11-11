"""Insert 'Did you know?' boxes at the start of chosen sections (idempotent).

Slots (chapter, section number) come from page positions in the compiled book
so that boxes fall roughly every ten pages; facts are in bible/did_you_know.json."""
import glob, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
facts = json.load(open(os.path.join(ROOT, "bible", "did_you_know.json")))
by_ch = {}
for f in facts:
    by_ch.setdefault(int(f["chapter"]), []).append(f)
for path in sorted(glob.glob(os.path.join(ROOT, "tex", "chapters", "*.tex"))):
    n = int(os.path.basename(path)[:2])
    s = open(path).read()
    s = re.sub(r"\n% DYK-BEGIN.*?% DYK-END\n", "\n", s, flags=re.S)   # remove old boxes
    lines = s.split("\n")
    secs = [i for i, l in enumerate(lines) if re.match(r"\s*\\section\{", l)]
    inserts = {}
    for f in by_ch.get(n, []):
        k = int(f["section"].split(".")[1])
        i = secs[k - 1]
        j = i + 1 if (i + 1 < len(lines) and lines[i + 1].strip().startswith("\\label")) else i
        if "\\label" in lines[i] or j == i + 1:
            pass
        box = ["% DYK-BEGIN", "\\begin{didyouknow}{%s}" % f["title"], f["text"], "\\end{didyouknow}", "% DYK-END"]
        inserts.setdefault(j, []).extend(box)
    out = []
    for i, l in enumerate(lines):
        out.append(l)
        if i in inserts:
            out.extend(inserts[i])
    open(path, "w").write("\n".join(out))
    if n in by_ch:
        print(os.path.basename(path), len(by_ch[n]))
