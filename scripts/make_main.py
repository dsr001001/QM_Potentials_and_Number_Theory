"""Write tex/book.tex including every chapter file that exists, in order."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from book import CHAPTERS, TEX, chfile
PARTS = {1: "Mathematical Foundations", 10: "Quantum Mechanics of a Particle on a Line",
         15: "Primes and the Zeta Function", 19: "Spectra, Randomness and Operators",
         23: "Computation, Fitting and Generalization", 28: "The Paper, Section by Section"}
CAP = json.load(open(os.path.join(TEX, "figs", "art", "captions.json")))
art = lambda slug: [r"\renewcommand{\chapterart}{%s}" % slug, r"\renewcommand{\chapterartcaption}{%s}" % CAP.get(slug, "")]
lines = [r"\documentclass[11pt,openany]{book}", r"\input{preamble}", r"\begin{document}",
         r"\input{frontmatter}", r"\cleardoublepage\pdfbookmark[0]{Contents}{contents}\hypertarget{toc}{}", r"\tableofcontents", r"\mainmatter"]
for n, slug, title in CHAPTERS:
    if n in PARTS:
        lines.append(r"\part{%s}" % PARTS[n])
    lines += art(slug)
    if os.path.exists(chfile(slug)):
        lines.append(r"\input{chapters/%s}" % os.path.basename(chfile(slug))[:-4])
    else:
        lines.append(r"\chapter{%s}\label{ch:%s}\textit{Chapter in preparation.}" % (title, slug))
lines += [r"\appendix", r"\IfFileExists{appendices.tex}{\input{appendices}}{}",
          r"\backmatter", r"\bibliographystyle{unsrt}", r"\bibliography{refs}", r"\end{document}"]
open(os.path.join(TEX, "book.tex"), "w").write("\n".join(lines) + "\n")
