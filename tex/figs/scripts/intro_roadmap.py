import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
boxes = {
    "I":   (5.0, 9.0, "I  Mathematical foundations\n(Ch. 2-9)"),
    "II":  (2.5, 6.6, "II  Quantum mechanics\n(Ch. 10-14)"),
    "III": (7.5, 6.6, "III  Primes and zeta\n(Ch. 15-18)"),
    "IV":  (5.0, 4.4, "IV  Spectra, randomness,\noperators (Ch. 19-22)"),
    "V":   (5.0, 2.4, "V  Computation, fitting,\ngeneralization (Ch. 23-27)"),
    "VI":  (5.0, 0.6, "VI  The paper (Ch. 28-32)"),
}
for k, (x, y, txt) in boxes.items():
    ax.text(x, y, txt, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="k"))
def arrow(a, b):
    xa, ya, _ = boxes[a]; xb, yb, _ = boxes[b]
    ax.annotate("", xy=(xb, yb + 0.55), xytext=(xa, ya - 0.55),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
for a, b in [("I", "II"), ("I", "III"), ("II", "IV"), ("III", "IV"),
             ("IV", "V"), ("II", "V"), ("III", "V"), ("V", "VI")]:
    arrow(a, b)
plt.savefig("intro_roadmap.pdf", bbox_inches="tight")
