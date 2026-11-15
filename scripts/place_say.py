#!/usr/bin/env python3
"""Place pronunciation footnotes (\\say) at the first use of new-ish symbols and names.

Usage (from the repo root):
    python3 scripts/place_say.py            # dry run: report where each footnote would go
    python3 scripts/place_say.py --apply    # insert them

The catalogue is bible/say.json: a list of entries
    {"key": "...", "kind": "sym"|"name", "pat": "<python regex>", "show": "<LaTeX>",
     "say": "<how to say it>", "note": "<optional note>"}

Rules
- A footnote goes at the first occurrence in running text, in chapter order. Boxes, displayed
  equations, figures, tables, code, headings, captions and the problem and answer sections are
  skipped, so a symbol first met inside a box gets its note at the next mention in the text.
- "sym" entries are matched inside inline math and the note is put after the closing dollar;
  "name" entries are matched in text and the note is put after the word (and a possessive 's).
  Either way the mark goes after any punctuation that follows. A mark straight after mathematics
  would read like a power, so for "sym" entries it is carried to the end of the clause, or
  failing that hung on the word just before the mathematics.
- Optional fields: "from": "NN" starts the search at chapter NN (to skip a passing mention in an
  earlier chapter); "also": ["NN", ...] repeats the note at the first use in those chapters (for a
  symbol that returns as the main subject after a long gap); "avoid": regex of lines to skip.
- The script refuses to run on files that already contain \\say unless --force is given.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REACH = 140
SKIP_ENVS = {"chapterintro", "prerequisites", "keyideas", "sample", "checkpoint", "pitfall", "beyond",
             "paperbox", "didyouknow", "figure", "table", "tabular", "equation", "equation*", "align",
             "align*", "gather", "gather*", "multline", "pycode", "lstlisting", "problemlist", "verbatim",
             "tikzpicture", "center", "cases", "array"}
MASK_CMDS = ["chapter", "section", "subsection", "subsubsection", "paragraph", "caption", "checkanswer",
             "texttt", "label", "cite", "cref", "Cref", "ref", "eqref", "url", "href", "footnote", "say",
             "saytext", "footnotetext", "lstinline", "input", "includegraphics", "item\\["]
STOP_SECTIONS = re.compile(r"\\section\{(Problems|Answers to selected problems|Summary)\}")


def mask_groups(line):
    """Blank out the brace argument of commands whose argument must not receive a footnote."""
    out = list(line)
    for cmd in MASK_CMDS:
        for m in re.finditer(r"\\" + cmd + (r"" if cmd.endswith("[") else r"(?![A-Za-z])"), line):
            i = m.end()
            if cmd.endswith("["):
                j = line.find("]", i)
                j = len(line) if j < 0 else j
                for k in range(m.start(), j + 1):
                    out[k] = " "
                continue
            # optional [..] then {..}
            while i < len(line) and line[i] in " \t":
                i += 1
            if i < len(line) and line[i] == "[":
                j = line.find("]", i)
                i = j + 1 if j >= 0 else i
            if i < len(line) and line[i] == "{":
                depth, k = 0, i
                while k < len(line):
                    if line[k] == "{" and line[k - 1] != "\\":
                        depth += 1
                    elif line[k] == "}" and line[k - 1] != "\\":
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                for q in range(m.start(), min(k + 1, len(line))):
                    out[q] = " "
    return "".join(out)


def inline_math_spans(line):
    """(start, end) of each $...$ on the line; end is the index just after the closing dollar."""
    spans, start, i = [], None, 0
    while i < len(line):
        c = line[i]
        if c == "\\":
            i += 2
            continue
        if c == "$":
            if start is None:
                start = i
            else:
                spans.append((start, i + 1))
                start = None
        i += 1
    return spans


def safe_lines(text):
    """Yield (line_no, line, masked_line) for lines of running text."""
    stack, in_disp, stopped = [], False, False
    for n, line in enumerate(text.split("\n")):
        if STOP_SECTIONS.search(line):
            stopped = True
        elif line.startswith("\\section{"):
            stopped = False
        begins = re.findall(r"\\begin\{([^}]*)\}", line)
        ends = re.findall(r"\\end\{([^}]*)\}", line)
        was_skipping = bool(stack) or in_disp
        for b in begins:
            if b in SKIP_ENVS or stack:
                stack.append(b)
        for e in ends:
            if stack and stack[-1] == e:
                stack.pop()
        if "\\[" in line:
            in_disp = True
            was_skipping = True
        if "\\]" in line:
            in_disp = False
            continue
        if was_skipping or stack or in_disp or stopped or begins or ends:
            continue
        if line.lstrip().startswith("%") or not line.strip():
            continue
        yield n, line, mask_groups(line)


def find_spot(entry, line, masked):
    """Return the insertion index on this line, or None."""
    pat = re.compile(entry["pat"])
    spans = inline_math_spans(masked)
    if entry["kind"] == "sym":
        for a, b in spans:
            if pat.search(masked[a:b]):
                pos = b
                break
        else:
            return None
    else:
        pos = None
        for m in pat.finditer(masked):
            if any(a <= m.start() < b for a, b in spans):
                continue
            pos = m.end()
            break
        if pos is None:
            return None
        while pos < len(line) and line[pos].isalpha():   # finish the word (eigenvalue -> eigenvalues)
            pos += 1
        if line[pos:pos + 2] == "'s":
            pos += 2
        elif line[pos:pos + 1] == "'" and line[pos:pos + 2] != "''":
            pos += 1
    # never inside an unclosed brace group that we masked; never before a closing brace of \emph etc.: allowed
    if line[pos:pos + 2] == "''":
        pos += 2
    while pos < len(line) and line[pos] == "}":
        pos += 1
    if entry["kind"] == "sym" and not (pos < len(line) and line[pos] in ".,;:!?)"):
        # A footnote mark straight after mathematics reads like a power, so carry it to the end of
        # the clause (the next punctuation mark in the text), if that is near.
        moved = False
        for q in range(pos, min(len(line), pos + REACH)):
            if masked[q] in ".,;:?)" and not any(a < q < b for a, b in spans) and masked[q - 1] != "\\":
                if masked[q] == "." and (masked[q + 1:q + 2].isdigit() or masked[q + 1:q + 2] == "~"
                                         or re.search(r"(Sec|Secs|Eq|Eqs|Fig|Figs|App|Ref|cf|vs|e\.g|i\.e)$", masked[:q])):
                    continue
                pos, moved = q, True
                break
        if not moved:
            # no clause end nearby: hang the mark on the word before the mathematics instead
            start = max(a for a, b in spans if b <= pos)
            k = start
            while k > 0 and line[k - 1] == " ":
                k -= 1
            if k > 0 and (masked[k - 1].isalnum()):
                pos = k
    while pos < len(line) and line[pos] in ".,;:!?)}":
        pos += 1
    if line[pos:pos + 2] == "''":
        pos += 2
    return pos


def main():
    apply = "--apply" in sys.argv
    force = "--force" in sys.argv
    cat = json.load(open(os.path.join(ROOT, "bible", "say.json")))
    files = sorted(glob.glob(os.path.join(ROOT, "tex", "chapters", "*.tex")))
    texts = {f: open(f).read() for f in files}
    if not force and any("\\say{" in t for t in texts.values()):
        sys.exit("chapters already contain \\say; use --force to add more")
    inserts = {f: [] for f in files}   # (line_no, pos, latex)
    placed = 0
    for e in cat:
        found = False
        targets = [e.get("from", "00")] + list(e.get("also", []))
        avoid = re.compile(e["avoid"]) if e.get("avoid") else None
        for ti, start in enumerate(targets):
            for f in files:
                ch = os.path.basename(f)[:2]
                if (ti == 0 and ch < start) or (ti > 0 and ch != start):
                    continue
                hit = False
                for n, line, masked in safe_lines(texts[f]):
                    if avoid and avoid.search(line):
                        continue
                    pos = find_spot(e, line, masked)
                    if pos is None:
                        continue
                    note = "[" + e["note"] + "]" if e.get("note") else ""
                    tex = "\\say{%s}{%s}%s" % (e["show"], e["say"], note)
                    inserts[f].append((n, pos, tex))
                    ctx = line[max(0, pos - 70):pos] + " <<" + e["key"] + ">> " + line[pos:pos + 25]
                    print("%-16s %-14s L%-5d %s" % (os.path.basename(f)[:16], e["key"], n + 1, ctx))
                    placed += 1
                    hit = found = True
                    break
                if hit:
                    break
        if not found:
            print("%-16s %-14s NOT FOUND" % ("", e["key"]))
    print("footnotes:", placed)
    if apply:
        for f in files:
            lines = texts[f].split("\n")
            for n, pos, tex in sorted(inserts[f], key=lambda x: (x[0], -x[1])):
                lines[n] = lines[n][:pos] + tex + lines[n][pos:]
            open(f, "w").write("\n".join(lines))
        print("applied")


if __name__ == "__main__":
    main()
