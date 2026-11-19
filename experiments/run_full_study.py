"""
Full numerical study & publication figure generator for:
"Modular Walsh Rigidity and Antipodal Obstructions for Flat Non-Clifford Hypergraph States"

Runs:
1. Scaled modular certification benchmark for n = 16, 20, 24, 28 qubits (up to 268M amplitudes)
   comparing exact int32 FWHT vs overflow-free uint8 modular FWHT at p = p_{8/15}.
2. Quantum magic (2-Renyi Stabilizer Entropy M_2) scaling for n = 8, 10, 12 qubits
   as a function of CCZ / CCCZ gate count in flat hypergraph states.
3. 1D ring SWAP & CNOT routing overhead imposed by Theorem 1 (antipodal necessity).
4. Generates 3 publication-quality PDF figures in paper/figures/.
"""

from __future__ import annotations

import gc
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypergraph_walsh import (
    HypergraphCircuit,
    build_mm_quartic_bent_circuit,
    minimal_rigidity_modulus,
    stabilizer_renyi_entropy_m2,
    verify_flat_exact,
    verify_flat_modular,
)


def fast_truth_table(n_qubits: int, gates: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Memory-friendly truth table evaluation in uint8 (handles up to n=28 in 256 MB)."""
    size = 1 << n_qubits
    f = np.zeros(size, dtype=np.uint8)
    # Process in chunks of 2^22 (4M) to avoid allocating a huge uint32 index array at n=28
    chunk = min(size, 1 << 22)
    base_idx = np.arange(chunk, dtype=np.uint32)
    masks = []
    for g in gates:
        m = 0
        for q in g:
            m |= 1 << q
        masks.append(np.uint32(m))

    for offset in range(0, size, chunk):
        idx = base_idx + np.uint32(offset)
        buf = np.zeros(chunk, dtype=np.uint8)
        for m in masks:
            buf ^= ((idx & m) == m)
        f[offset : offset + chunk] = buf
    return f


def study_1_modular_scaling() -> list[dict]:
    print("=" * 74)
    print("STUDY 1: Modular Certification Scaling (n = 16, 20, 24, 28 qubits)")
    print("=" * 74)
    rows = []

    for n in (16, 20, 24, 28):
        k = n // 2
        p_815 = minimal_rigidity_modulus(n, 8.0 / 15.0)
        p_half = minimal_rigidity_modulus(n, 0.5)
        size = 1 << n

        extra_y = [(k, k + 1, k + 2), (k, k + 1, k + 2, k + 3)]
        circ = build_mm_quartic_bent_circuit(n, extra_y)
        f = fast_truth_table(n, circ.gates)

        # Warm/measure modular uint8 first (lower memory)
        gc.collect()
        t0 = time.perf_counter()
        ok_mod, _ = verify_flat_modular(f, n, p_815)
        t_mod_ms = (time.perf_counter() - t0) * 1000.0
        gc.collect()

        # Measure exact int32
        t0 = time.perf_counter()
        ok_exact, _ = verify_flat_exact(f, n)
        t_exact_ms = (time.perf_counter() - t0) * 1000.0
        del f
        gc.collect()

        exact_mb = (size * 4) / (1024 * 1024)
        mod_mb = (size * (1 if 2 * p_815 <= 255 else 2)) / (1024 * 1024)
        speedup = t_exact_ms / t_mod_ms

        print(
            f"  n={n:2d} | 2^n={size:11,d} | p_8/15={p_815:2d} (p_1/2={p_half:2d}) | "
            f"exact={t_exact_ms:8.1f} ms ({exact_mb:6.2f} MB) | "
            f"mod={t_mod_ms:8.1f} ms ({mod_mb:6.2f} MB) | speedup={speedup:.2f}x | ok={ok_exact and ok_mod}"
        )

        rows.append(
            {
                "n_qubits": n,
                "amplitudes": size,
                "p_8_15": p_815,
                "p_1_2": p_half,
                "exact_ms": round(t_exact_ms, 2),
                "mod_ms": round(t_mod_ms, 2),
                "speedup": round(speedup, 2),
                "exact_mb": round(exact_mb, 3),
                "mod_mb": round(mod_mb, 3),
                "verified_ok": bool(ok_exact and ok_mod),
            }
        )
    return rows


def study_2_magic_scaling() -> dict:
    print("\n" + "=" * 74)
    print("STUDY 2: Quantum Magic (M_2) Scaling in Flat Hypergraph States (n=8, 10, 12)")
    print("=" * 74)
    out = {}

    for n in (8, 10, 12):
        k = n // 2
        y_qubits = list(range(k, n))
        all_ccz = list(itertools.combinations(y_qubits, 3))
        all_cccz = list(itertools.combinations(y_qubits, 4))

        # Sweep number of CCZ gates (0 .. min(6, len))
        ccz_curve = []
        max_ccz = min(6, len(all_ccz))
        for count in range(0, max_ccz + 1):
            circ = build_mm_quartic_bent_circuit(n, all_ccz[:count])
            f = circ.truth_table()
            is_flat, _ = verify_flat_exact(f, n)
            m2 = stabilizer_renyi_entropy_m2(f, n)
            ccz_curve.append({"gates": count, "m2": round(m2, 6), "is_flat": is_flat})

        # Sweep number of CCCZ gates (0 .. min(6, len))
        cccz_curve = []
        max_cccz = min(6, len(all_cccz))
        for count in range(0, max_cccz + 1):
            circ = build_mm_quartic_bent_circuit(n, all_cccz[:count])
            f = circ.truth_table()
            is_flat, _ = verify_flat_exact(f, n)
            m2 = stabilizer_renyi_entropy_m2(f, n)
            cccz_curve.append({"gates": count, "m2": round(m2, 6), "is_flat": is_flat})

        print(f"  n={n:2d} (k={k}):")
        print("    CCZ  sweep:", [(r["gates"], r["m2"]) for r in ccz_curve])
        print("    CCCZ sweep:", [(r["gates"], r["m2"]) for r in cccz_curve])
        out[f"n{n}"] = {"ccz": ccz_curve, "cccz": cccz_curve}

    return out


def study_3_ring_swap_overhead() -> list[dict]:
    print("\n" + "=" * 74)
    print("STUDY 3: 1D Ring Routing Overhead Imposed by Antipodal Necessity (Theorem 1)")
    print("=" * 74)
    rows = []
    for n in (6, 8, 10, 12, 16, 20, 24):
        H = n // 2
        # On a 1D ring of n=2H qubits, each antipodal pair (i, i+H) is at distance H.
        # Bringing i and i+H adjacent requires at least (H - 1) SWAPs (each SWAP = 3 CNOTs).
        # Across all H antipodal pairs in the mandatory orbit O_ant:
        #   - Minimum sequential/per-pair SWAPs: H - 1
        #   - Total pairwise SWAP lower bound (for all H pairs, uncomputing/re-routing):
        #     At least ceil(H*(H-1)/2) SWAPs (half-shift permutation on the ring moves each
        #     antipodal partner across distance H-1; each SWAP reduces sum of distances by <= 2).
        min_swaps_single = H - 1
        min_swaps_layer = (H * (H - 1) + 1) // 2
        min_cnots_layer = 3 * min_swaps_layer
        print(
            f"  n={n:2d} (H={H:2d}): ring_dist={H:2d} | min SWAPs/pair={min_swaps_single:2d} | "
            f"layer SWAP lower bound={min_swaps_layer:3d} ({min_cnots_layer:3d} CNOTs)"
        )
        rows.append(
            {
                "n_qubits": n,
                "H": H,
                "ring_distance": H,
                "min_swaps_per_pair": min_swaps_single,
                "min_swaps_orbit_lb": min_swaps_layer,
                "min_cnots_orbit_lb": min_cnots_layer,
            }
        )
    return rows


def generate_figures(s1: list[dict], s2: dict, s3: list[dict]) -> None:
    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )

    # --- Figure 1: Modular Scaling (Runtime + Memory) ---
    ns = [r["n_qubits"] for r in s1]
    t_exact = [r["exact_ms"] for r in s1]
    t_mod = [r["mod_ms"] for r in s1]
    m_exact = [r["exact_mb"] for r in s1]
    m_mod = [r["mod_mb"] for r in s1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax1.semilogy(ns, t_exact, "o-", color="#1f4e79", lw=2, label="Exact FWHT (int32)")
    ax1.semilogy(ns, t_mod, "s--", color="#c0392b", lw=2, label="Certified 8/15 Mod FWHT (uint8)")
    ax1.set_xticks(ns)
    ax1.set_xlabel("Number of qubits $n$")
    ax1.set_ylabel("Verification time (ms, log scale)")
    ax1.set_title("(a) Runtime Scaling ($n = 16$ to $28$)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    x = np.arange(len(ns))
    w = 0.35
    ax2.bar(x - w / 2, m_exact, w, color="#1f4e79", label="Exact (int32)")
    ax2.bar(x + w / 2, m_mod, w, color="#c0392b", label="Certified 8/15 (uint8)")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n}q\n(p={r['p_8_15']})" for n, r in zip(ns, s1)])
    ax2.set_xlabel("Qubits & certified modulus $p_{8/15}$")
    ax2.set_ylabel("Working memory (MB, log scale)")
    ax2.set_title("(b) $4\\times$ Memory Footprint Reduction")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_modular_scaling.pdf")
    plt.close(fig)

    # --- Figure 2: Quantum Magic Scaling (n=8, 10, 12) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    colors = {"n8": "#2980b9", "n10": "#27ae60", "n12": "#8e44ad"}
    markers = {"n8": "o-", "n10": "s-", "n12": "^-"}

    for key, label in [("n8", "$n=8$"), ("n10", "$n=10$"), ("n12", "$n=12$")]:
        xs = [r["gates"] for r in s2[key]["ccz"]]
        ys = [r["m2"] for r in s2[key]["ccz"]]
        ax1.plot(xs, ys, markers[key], color=colors[key], lw=2, label=label)

        xs4 = [r["gates"] for r in s2[key]["cccz"]]
        ys4 = [r["m2"] for r in s2[key]["cccz"]]
        ax2.plot(xs4, ys4, markers[key], color=colors[key], lw=2, label=label)

    ax1.set_xlabel("Number of $CCZ$ gates in $g(y)$")
    ax1.set_ylabel("2-Rényi Stabilizer Entropy $M_2$ (bits)")
    ax1.set_title("(a) Cubic Flat Hypergraph States ($CZ + CCZ$)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Number of $CCCZ$ gates in $g(y)$")
    ax2.set_title("(b) Quartic Flat Hypergraph States ($CZ + CCCZ$)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_magic_scaling.pdf")
    plt.close(fig)

    # --- Figure 3: 1D Ring SWAP / CNOT Overhead ---
    ns3 = [r["n_qubits"] for r in s3]
    swaps = [r["min_swaps_orbit_lb"] for r in s3]
    cnots = [r["min_cnots_orbit_lb"] for r in s3]

    fig, ax1 = plt.subplots(figsize=(5.6, 3.3))
    ax1.plot(ns3, swaps, "o-", color="#d35400", lw=2, label="Min. SWAPs for antipodal orbit $\\mathcal{O}_{\\mathrm{ant}}$")
    ax1.plot(ns3, cnots, "s--", color="#2c3e50", lw=2, label="Equivalent CNOT overhead ($3\\times$ SWAPs)")
    ax1.set_xticks(ns3)
    ax1.set_xlabel("Ring size $n = 2H$ qubits")
    ax1.set_ylabel("Gate count lower bound")
    ax1.set_title("1D Ring Routing Overhead Imposed by Antipodal Necessity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_ring_swap_overhead.pdf")
    plt.close(fig)
    print("\nGenerated 3 PDF figures in:", fig_dir)


def main() -> None:
    s1 = study_1_modular_scaling()
    s2 = study_2_magic_scaling()
    s3 = study_3_ring_swap_overhead()
    generate_figures(s1, s2, s3)

    summary = {
        "study_1_modular_scaling": s1,
        "study_2_magic_scaling": s2,
        "study_3_ring_swap_overhead": s3,
    }
    out_path = ROOT / "results" / "full_study_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved full study summary to:", out_path)


if __name__ == "__main__":
    main()
