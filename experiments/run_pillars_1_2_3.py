"""
Runs Pillars 1, 2, and 3 end-to-end and generates publication figures & JSON results:
  - Pillar 1: Exact M_2 dimensional reduction & linear magic scaling M_2 = floor(n/6)*log2(32/11)
  - Pillar 2: OpenQASM 3.0 export + NISQ noisy X-basis flatness testing (clean vs faulty vs fooling)
  - Pillar 3: Real SWAP routing of the mandatory antipodal CZ orbit on 1D Ring, 2D Grid, and Heavy-Hex
"""

from __future__ import annotations

import json
import math
import sys
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
    stabilizer_renyi_entropy_m2,
    verify_flat_exact,
)
from nisq_and_qasm import save_qasm_examples, simulate_nisq_flatness_test
from topology_compiler import (
    build_1d_ring,
    build_2d_grid,
    build_heavy_hex,
    route_cz_layer,
)


def pillar_1_exact_magic_theorem() -> dict:
    print("=" * 74)
    print("PILLAR 1: Exact Magic Reduction Theorem M_2(|psi_f>) = M_2(|psi_g>)")
    print("=" * 74)
    m2_ccz_exact = math.log2(32.0 / 11.0)
    m2_cccz_exact = math.log2(512.0 / 197.0)
    rows = []

    # Verify disjoint CCZ family on n = 6, 8, 10, 12 (up to 2 disjoint CCZs on n=12!)
    for n in (6, 8, 10, 12):
        k = n // 2
        num_triples = k // 3
        ccz_gates = [(k + 3 * j, k + 3 * j + 1, k + 3 * j + 2) for j in range(num_triples)]
        circ = build_mm_quartic_bent_circuit(n, ccz_gates)
        f = circ.truth_table()
        is_flat, _ = verify_flat_exact(f, n)
        m2_num = stabilizer_renyi_entropy_m2(f, n)
        m2_theory = num_triples * m2_ccz_exact
        err = abs(m2_num - m2_theory)
        print(
            f"  n={n:2d} | disjoint CCZs={num_triples} | flat={is_flat} | "
            f"M_2 numeric={m2_num:.6f} | theory={m2_theory:.6f} | diff={err:.2e}"
        )
        rows.append(
            {
                "n_qubits": n,
                "disjoint_ccz": num_triples,
                "is_flat": is_flat,
                "m2_numeric": round(m2_num, 6),
                "m2_theory": round(m2_theory, 6),
                "abs_error": float(f"{err:.2e}"),
            }
        )

    # Also verify single CCCZ exact formula log2(512/197)
    circ4 = build_mm_quartic_bent_circuit(8, [(4, 5, 6, 7)])
    m2_4 = stabilizer_renyi_entropy_m2(circ4.truth_table(), 8)
    print(f"  n= 8 | single CCCZ   | M_2 numeric={m2_4:.6f} | log2(512/197)={m2_cccz_exact:.6f}")

    return {
        "log2_32_over_11": round(m2_ccz_exact, 8),
        "log2_512_over_197": round(m2_cccz_exact, 8),
        "disjoint_ccz_verification": rows,
    }


def pillar_2_nisq_and_qasm() -> dict:
    print("\n" + "=" * 74)
    print("PILLAR 2: OpenQASM 3.0 Export & NISQ Noisy X-Basis Flatness Testing")
    print("=" * 74)
    rng = np.random.default_rng(20261119)

    # Circuits on n=8 qubits (256 outcomes) for clean NISQ testing
    clean_8 = build_mm_quartic_bent_circuit(8, [(4, 5, 6), (4, 5, 6, 7)])
    faulty_8 = HypergraphCircuit(8, clean_8.gates[1:])  # drop 1 CZ gate
    fool_8 = HypergraphCircuit(8, ((0, 1), (2, 3), (4, 5, 6, 7)))  # universal p=3 fooling

    # Export ready-to-run OpenQASM 3.0 files
    qasm_files = save_qasm_examples(
        ROOT / "qasm",
        {
            "flat_cubic_n6": build_mm_quartic_bent_circuit(6, [(3, 4, 5)]),
            "flat_quartic_n8": clean_8,
            "fooling_p3_n8": fool_8,
        },
    )
    print("  Exported OpenQASM 3.0 files:", [Path(p).name for p in qasm_files])

    noise_levels = [0.0, 1e-4, 1e-3, 5e-3, 1e-2]
    shots = 20000
    rows = []
    for p_err in noise_levels:
        r_clean = simulate_nisq_flatness_test(clean_8, p_err, shots, rng)
        r_fault = simulate_nisq_flatness_test(faulty_8, p_err, shots, rng)
        r_fool = simulate_nisq_flatness_test(fool_8, p_err, shots, rng)
        print(
            f"  p_2q={p_err:6.4f} (S={shots}): "
            f"Clean Z={r_clean['z_score']:6.2f} (rej={str(r_clean['rejects_flatness_5sigma']):5s}) | "
            f"Faulty Z={r_fault['z_score']:7.1f} | "
            f"Fooling(p=3) Z={r_fool['z_score']:7.1f}"
        )
        rows.append(
            {
                "p_2q_error": p_err,
                "shots": shots,
                "clean": r_clean,
                "faulty_drop_cz": r_fault,
                "fooling_p3": r_fool,
            }
        )

    return {"qasm_files": qasm_files, "nisq_n8_sweep": rows}


def pillar_3_hardware_topologies() -> list[dict]:
    print("\n" + "=" * 74)
    print("PILLAR 3: Hardware SWAP Routing (1D Ring vs 2D Grid vs IBM Heavy-Hex)")
    print("=" * 74)
    rows = []
    for n in (8, 12, 16, 20, 24):
        H = n // 2
        antipodal = [(i, i + H) for i in range(H)]
        r_ring = route_cz_layer(build_1d_ring(n), antipodal)
        r_grid = route_cz_layer(build_2d_grid(n), antipodal)
        r_hhex = route_cz_layer(build_heavy_hex(n), antipodal)
        print(
            f"  n={n:2d} | 1D Ring: {r_ring['swaps']:2d} SWAPs (F={r_ring['est_fidelity_1e3']:.3f}) | "
            f"2D Grid: {r_grid['swaps']:2d} SWAPs (F={r_grid['est_fidelity_1e3']:.3f}) | "
            f"Heavy-Hex: {r_hhex['swaps']:2d} SWAPs (F={r_hhex['est_fidelity_1e3']:.3f})"
        )
        rows.append(
            {
                "n_qubits": n,
                "ring_1d": r_ring,
                "grid_2d": r_grid,
                "heavy_hex": r_hhex,
            }
        )
    return rows


def generate_new_figures(p2: dict, p3: list[dict]) -> None:
    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 4: Hardware Topologies (SWAPs + Estimated Fidelity)
    ns = [r["n_qubits"] for r in p3]
    sw_ring = [r["ring_1d"]["swaps"] for r in p3]
    sw_grid = [r["grid_2d"]["swaps"] for r in p3]
    sw_hhex = [r["heavy_hex"]["swaps"] for r in p3]

    f_ring = [r["ring_1d"]["est_fidelity_1e3"] for r in p3]
    f_grid = [r["grid_2d"]["est_fidelity_1e3"] for r in p3]
    f_hhex = [r["heavy_hex"]["est_fidelity_1e3"] for r in p3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax1.plot(ns, sw_ring, "o-", color="#c0392b", lw=2, label="1D Ring")
    ax1.plot(ns, sw_hhex, "s--", color="#d35400", lw=2, label="IBM Heavy-Hex")
    ax1.plot(ns, sw_grid, "^-", color="#1f4e79", lw=2, label="2D Grid")
    ax1.set_xticks(ns)
    ax1.set_xlabel("Number of qubits $n = 2H$")
    ax1.set_ylabel("Routed SWAP count for $\\mathcal{O}_{\\mathrm{ant}}$")
    ax1.set_title("(a) Mandatory Antipodal Orbit Routing")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ns, f_ring, "o-", color="#c0392b", lw=2, label="1D Ring")
    ax2.plot(ns, f_hhex, "s--", color="#d35400", lw=2, label="IBM Heavy-Hex")
    ax2.plot(ns, f_grid, "^-", color="#1f4e79", lw=2, label="2D Grid")
    ax2.set_xticks(ns)
    ax2.set_ylim(0.5, 1.02)
    ax2.set_xlabel("Number of qubits $n = 2H$")
    ax2.set_ylabel("Estimated layer fidelity ($p_{2q}=10^{-3}$)")
    ax2.set_title("(b) Hardware Fidelity Impact")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_topology_comparison.pdf")
    plt.close(fig)

    # Figure 5: NISQ Flatness Testing (Z-score vs Noise)
    rows = p2["nisq_n8_sweep"]
    ps = [r["p_2q_error"] for r in rows]
    z_clean = [max(0.1, abs(r["clean"]["z_score"])) for r in rows]
    z_fault = [r["faulty_drop_cz"]["z_score"] for r in rows]
    z_fool = [r["fooling_p3"]["z_score"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.semilogy(ps, z_clean, "o-", color="#27ae60", lw=2, label="Clean flat deg-4 hypergraph")
    ax.semilogy(ps, z_fault, "s--", color="#c0392b", lw=2, label="Faulty circuit (1 dropped CZ)")
    ax.semilogy(ps, z_fool, "^-.", color="#8e44ad", lw=2, label="Universal $p=3$ fooling circuit")
    ax.axhline(5.0, color="gray", ls=":", lw=1.5, label="$5\\sigma$ rejection threshold")
    ax.set_xlabel("Two-qubit gate error rate $p_{2q}$")
    ax.set_ylabel("$\\chi^2$ flatness Z-score (log scale, $S=20{,}000$)")
    ax.set_title("NISQ $X$-Basis Flatness Discrimination ($n=8$ qubits)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_nisq_flatness.pdf")
    plt.close(fig)
    print("\nGenerated 2 new PDF figures in:", fig_dir)


def main() -> None:
    p1 = pillar_1_exact_magic_theorem()
    p2 = pillar_2_nisq_and_qasm()
    p3 = pillar_3_hardware_topologies()
    generate_new_figures(p2, p3)

    out = {
        "pillar_1_exact_magic": p1,
        "pillar_2_nisq_and_qasm": p2,
        "pillar_3_hardware_topologies": p3,
    }
    out_path = ROOT / "results" / "pillars_1_2_3_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved Pillars 1-3 summary to:", out_path)


if __name__ == "__main__":
    main()
