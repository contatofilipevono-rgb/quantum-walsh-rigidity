"""Generate PNG versions of the key figures for GitHub README rendering."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

s1 = json.loads((ROOT / "results" / "full_study_summary.json").read_text(encoding="utf-8"))["study_1_modular_scaling"]
s2 = json.loads((ROOT / "results" / "full_study_summary.json").read_text(encoding="utf-8"))["study_2_magic_scaling"]
rs = json.loads((ROOT / "results" / "rs_quartic_bridge.json").read_text(encoding="utf-8"))
p3 = json.loads((ROOT / "results" / "pillars_1_2_3_summary.json").read_text(encoding="utf-8"))["pillar_3_hardware_topologies"]
gcp = json.loads((ROOT / "results" / "gcp_32q_openmp.json").read_text(encoding="utf-8"))["benchmarks"]

plt.rcParams.update({"font.size": 10, "figure.dpi": 160})

# 1. Cloud 24-36q Memory Comparison (complex128 vs int32 vs Certified Modular)
ns = [r["n_qubits"] for r in gcp]
c128 = [r["complex128_gb"] for r in gcp]
i32 = [r["exact_gb"] for r in gcp]
mod = [r["mod_gb"] for r in gcp]
x = np.arange(len(ns))
w = 0.25

fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.bar(x - w, c128, w, color="#95a5a6", label="Standard complex128")
ax.bar(x, i32, w, color="#1f4e79", label="Exact int32 FWHT")
ax.bar(x + w, mod, w, color="#c0392b", label="Certified 8/15 Modular (uint8/uint16)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"{n}q\n(p={r['p_8_15']})" for n, r in zip(ns, gcp)])
ax.set_ylabel("Working Memory (GB, log scale)")
ax.set_title("Memory Footprint up to 36 Qubits (68.7B Amplitudes: 128 GB vs 256 GB OOM / 1 TB)")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "memory_36q.png")
plt.close(fig)

# 2. Magic Density Scaling (RS deg-4 vs MM disjoint-CCZ vs Clifford)
ns_m = [8, 16, 24, 32]
m2_clifford = [0.0 for _ in ns_m]
m2_mm = [(n // 6) * math.log2(32.0 / 11.0) for n in ns_m]
m2_rs = [(n // 8) * rs["n8_m2_bits"] for n in ns_m]

fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.plot(ns_m, m2_rs, "D-", color="#c0392b", lw=2.2, label="RS deg-4 interleaved (0.638n bits)")
ax.plot(ns_m, m2_mm, "s--", color="#1f4e79", lw=2.0, label="MM depth-2 disjoint CCZ (0.257n bits)")
ax.plot(ns_m, m2_clifford, "o:", color="#7f8c8d", lw=1.6, label="Clifford graph states (M2 = 0)")
ax.set_xticks(ns_m)
ax.set_xlabel("Number of qubits n = 8q")
ax.set_ylabel("Exact 2-Renyi Stabilizer Entropy M2 (bits)")
ax.set_title("Linear Non-Clifford Magic Scaling in Flat Hypergraph States")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "magic_density.png")
plt.close(fig)

# 3. Hardware Topology SWAP Routing
ns3 = [r["n_qubits"] for r in p3]
fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.plot(ns3, [r["ring_1d"]["swaps"] for r in p3], "o-", color="#c0392b", lw=2, label="1D Ring")
ax.plot(ns3, [r["heavy_hex"]["swaps"] for r in p3], "s--", color="#d35400", lw=2, label="IBM Heavy-Hex")
ax.plot(ns3, [r["grid_2d"]["swaps"] for r in p3], "^-", color="#1f4e79", lw=2, label="2D Grid")
ax.set_xticks(ns3)
ax.set_xlabel("Number of qubits n = 2H")
ax.set_ylabel("Routed SWAP count for antipodal CZ orbit")
ax.set_title("Mandatory Antipodal Orbit Routing Across Hardware Topologies")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "topology_swaps.png")
plt.close(fig)

print("Generated 3 PNG figures in:", OUT)
