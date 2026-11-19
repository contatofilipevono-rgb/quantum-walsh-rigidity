"""Regenerate fig_magic_scaling.pdf comparing MM disjoint-CCZ vs RS degree-4 magic density."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
s2 = json.loads((ROOT / "results" / "full_study_summary.json").read_text(encoding="utf-8"))[
    "study_2_magic_scaling"
]
rs = json.loads((ROOT / "results" / "rs_quartic_bridge.json").read_text(encoding="utf-8"))

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
    }
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5))

# Panel (a): CCZ gate sweep on n = 8, 10, 12
colors = {"n8": "#2980b9", "n10": "#27ae60", "n12": "#8e44ad"}
markers = {"n8": "o-", "n10": "s-", "n12": "^-"}
for key, label in [("n8", "$n=8$"), ("n10", "$n=10$"), ("n12", "$n=12$")]:
    xs = [r["gates"] for r in s2[key]["ccz"]]
    ys = [r["m2"] for r in s2[key]["ccz"]]
    ax1.plot(xs, ys, markers[key], color=colors[key], lw=2, label=label)

ax1.set_xlabel("Number of $CCZ$ gates in $g(y)$")
ax1.set_ylabel("2-Rényi Stabilizer Entropy $M_2$ (bits)")
ax1.set_title("(a) Overlapping $CCZ$ Sweep (MM Flat States)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel (b): Magic density scaling n = 8, 16, 24, 32
ns = [8, 16, 24, 32]
m2_clifford = [0.0 for _ in ns]
m2_mm_disjoint = [(n // 6) * math.log2(32.0 / 11.0) for n in ns]
m2_rs_deg4 = [(n // 8) * rs["n8_m2_bits"] for n in ns]

ax2.plot(ns, m2_rs_deg4, "D-", color="#c0392b", lw=2.2, label="RS deg-4 interleaved ($0.638\\,n$ bits)")
ax2.plot(ns, m2_mm_disjoint, "s--", color="#1f4e79", lw=2.0, label="MM depth-2 disjoint $CCZ$ ($0.257\\,n$)")
ax2.plot(ns, m2_clifford, "o:", color="#7f8c8d", lw=1.6, label="Clifford graph states ($M_2 = 0$)")
ax2.set_xticks(ns)
ax2.set_xlabel("Number of qubits $n = 8q$")
ax2.set_ylabel("Exact $M_2(|\\psi_f\\rangle)$ (bits)")
ax2.set_title("(b) Linear Magic Scaling ($2.5\\times$ Higher in RS deg-4)")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

fig.tight_layout()
out_path = ROOT / "paper" / "figures" / "fig_magic_scaling.pdf"
fig.savefig(out_path)
plt.close(fig)
print("Updated:", out_path)
