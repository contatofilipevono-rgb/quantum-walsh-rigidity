"""
End-to-end IQM Garnet (Crystal-20, QB1..QB20) 20-Qubit Non-Clifford Walsh Rigidity,
Steiner-Star Synthesis & OpenAI-#274 Antipodal Felinity Suite (v4).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iqm20_core import (
    HypergraphCircuit,
    build_1d_ring,
    build_grid_4x5,
    build_heavy_hex_20,
    build_iqm_garnet_20,
    compile_physically_routed_iqm_qasm,
    exact_openai274_felinity_and_l2,
    iqm_garnet_20_mm_matching,
    realistic_nisq_triple_protocol_sim,
    route_cz_pairs,
    stabilizer_entropy_m2_small,
    verify_flat_exact,
    verify_flat_modular,
    verify_halfspace_collapse_h20_4,
    verify_routed_qasm_exact_20q,
)


def main() -> None:
    print("=" * 84)
    print("IQM GARNET 20-QUBIT (QB1..QB20) SUITE v4: OPENAI #274 FELINITY + SELF-CANCELLING RO")
    print("=" * 84)

    g_iqm, coords_iqm, qb_labels = build_iqm_garnet_20()
    g_ring = build_1d_ring(20)
    g_grid = build_grid_4x5()
    g_hhex = build_heavy_hex_20()
    pairs_mm, x_reg, y_reg = iqm_garnet_20_mm_matching()

    y_path_3a  = (y_reg[3], y_reg[7], y_reg[9])           # QB11 - QB16(hub) - QB17 (0 SWAPs!)
    y_star_4   = (y_reg[3], y_reg[6], y_reg[7], y_reg[9]) # Star at QB16 (0 SWAPs!)
    y_path_3b  = (y_reg[0], y_reg[1], y_reg[2])           # Top y-triple: QB2, QB7, QB5
    y_path_3c  = (y_reg[4], y_reg[5], y_reg[8])           # Outer y-triple: QB13, QB9, QB20

    c_flat_ccz = HypergraphCircuit(20, tuple(pairs_mm) + (y_path_3a,), "iqm20_flat_ccz")
    c_flat_cccz = HypergraphCircuit(20, tuple(pairs_mm) + (y_star_4,), "iqm20_flat_cccz")
    c_flat_3ccz = HypergraphCircuit(
        20,
        tuple(pairs_mm) + (y_path_3a, y_path_3b, y_path_3c),
        "iqm20_flat_3ccz_highmagic",
    )
    c_faulty = HypergraphCircuit(20, tuple(pairs_mm[:-1]) + (y_star_4,), "iqm20_faulty_dropped_cz")
    fooling_4 = (x_reg[0], y_reg[0], x_reg[1], y_reg[1])
    c_fooling = HypergraphCircuit(20, tuple(pairs_mm[2:]) + (fooling_4,), "iqm20_fooling_p3")

    circuits = [c_flat_ccz, c_flat_cccz, c_flat_3ccz, c_faulty, c_fooling]

    m2_1ccz = stabilizer_entropy_m2_small(HypergraphCircuit(3, ((0, 1, 2),)).truth_table(), 3)
    m2_1cccz = stabilizer_entropy_m2_small(HypergraphCircuit(4, ((0, 1, 2, 3),)).truth_table(), 4)
    magic_map = {
        "iqm20_flat_ccz": round(m2_1ccz, 4),
        "iqm20_flat_cccz": round(m2_1cccz, 4),
        "iqm20_flat_3ccz_highmagic": round(3.0 * m2_1ccz, 4),
    }

    qasm_dir = ROOT / "qasm" / "iqm20_routed"
    qasm_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- [1] OpenAI #274 Exact Felinity F_20(rho), QAC^0 Disjoint Depth & QASM Audit ---")
    results_circuits = {}
    w_cache: dict[str, np.ndarray] = {}

    for circ in circuits:
        f = circ.truth_table()
        is_flat, viol_ex, w_ex = verify_flat_exact(f, 20)
        w_cache[circ.name] = w_ex
        pass_17, viol_17 = verify_flat_modular(f, 20, p=17)
        pass_3, _ = verify_flat_modular(f, 20, p=3)

        qasm_fwd, stats_fwd = compile_physically_routed_iqm_qasm(circ, g_iqm, qb_labels, mirror=False)
        qasm_mir, stats_mir = compile_physically_routed_iqm_qasm(circ, g_iqm, qb_labels, mirror=True)
        assert verify_routed_qasm_exact_20q(qasm_fwd, circ, g_iqm)

        (qasm_dir / f"{circ.name}_routed_iqm20.qasm").write_text(qasm_fwd, encoding="utf-8")
        (qasm_dir / f"{circ.name}_mirror_iqm20.qasm").write_text(qasm_mir, encoding="utf-8")

        oai274 = exact_openai274_felinity_and_l2(w_ex, 20)
        results_circuits[circ.name] = {
            "degree": circ.degree,
            "gate_counts": circ.gate_counts(),
            "exact_is_flat": is_flat,
            "mod17_passes": pass_17,
            "mod3_passes": pass_3,
            "exact_magic_M2_bits": magic_map.get(circ.name),
            "openai274_exact_metrics": oai274,
            "physically_routed_forward": stats_fwd,
            "physically_routed_mirror": stats_mir,
        }
        print(
            f"  {circ.name:25s} | I_self*2^20={oai274['i_self_times_2n']:5.2f} "
            f"| OAI#274 Felinity*2^20={oai274['openai274_felinity_times_2n']:5.2f} "
            f"| d_QAC(log/phys)={stats_fwd['qac0_logical_reflection_depth']}/{stats_fwd['qac0_physical_cz_layer_depth']:2d} "
            f"| CZs={stats_fwd['physical_native_cz']:2d}"
        )

    print("\n--- [2] Self-Cancelling Readout-Bias Triple Protocol (Strong Asym. RO: p01=3.0%, p10=0.5%) ---")
    rng = np.random.default_rng(20261121)
    shots = 16384
    noise_grid = [0.001, 0.003, 0.005, 0.008, 0.012]
    triple_sweep: dict[str, list[dict]] = {}
    for cname in [c.name for c in circuits]:
        cz_fwd = results_circuits[cname]["physically_routed_forward"]["physical_native_cz"]
        triple_sweep[cname] = []
        for p2q in noise_grid:
            sim = realistic_nisq_triple_protocol_sim(
                w_exact=w_cache[cname],
                n_qubits=20,
                physical_cz_forward=cz_fwd,
                p_2q=p2q,
                shots=shots,
                n_trials=25,
                rng=rng,
                p_ro_0given1=0.030,
                p_ro_1given0=0.005,
            )
            triple_sweep[cname].append(sim)
        r = triple_sweep[cname][2]  # p_2q = 0.5%
        print(
            f"  {cname:25s} | C_self={r['mean_c_self']:6.1f} | C_ant(OAI#274)={r['mean_c_ant_felinity']:6.1f} "
            f"| C_comb={r['mean_c_combined_self_cancelling']:6.1f} (ideal=128.0, Z_no_cal={r['mean_z_uncalibrated_combined']:+6.2f}, "
            f"det={r['detect_5sigma_combined_no_cal']*100:5.1f}%)"
        )

    # Also sweep readout asymmetry delta_ro at fixed p_2q = 0.5% on iqm20_flat_cccz
    # to visualize the exact +4*n*delta^2 vs -4*n*delta^2 cancellation!
    ro_p01_grid = [0.005, 0.015, 0.025, 0.035, 0.050]
    ro_cancel_curve = []
    for p01 in ro_p01_grid:
        sim_ro = realistic_nisq_triple_protocol_sim(
            w_exact=w_cache["iqm20_flat_cccz"],
            n_qubits=20,
            physical_cz_forward=26,
            p_2q=0.005,
            shots=shots,
            n_trials=30,
            rng=rng,
            p_ro_0given1=p01,
            p_ro_1given0=0.005,
        )
        ro_cancel_curve.append(sim_ro)

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_fig5_openai274_felinity(
        fig_dir / "fig5_openai274_felinity_self_cancelling_readout.png",
        ro_cancel_curve,
        ro_p01_grid,
        triple_sweep,
        noise_grid,
        shots,
    )

    report = {
        "version": "v4 (OpenAI #274 Corollary 8 Antipodal Felinity + Self-Cancelling Readout Estimator + QAC^0 Disjoint Layer Depth)",
        "circuits": results_circuits,
        "triple_protocol_sweep_strong_ro": triple_sweep,
        "readout_cancellation_curve_flat_cccz": ro_cancel_curve,
    }
    out_json = ROOT / "results" / "iqm20_benchmark_v4_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[OK] Generated Figure 5: {fig_dir / 'fig5_openai274_felinity_self_cancelling_readout.png'}")
    print(f"[OK] Saved v4 JSON report: {out_json}")


def _plot_fig5_openai274_felinity(
    out_path: Path,
    ro_cancel_curve: list[dict],
    ro_p01_grid: list[float],
    triple_sweep: dict[str, list[dict]],
    noise_grid: list[float],
    shots: int,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.1), dpi=180)

    # Panel (a): Self-Collisions vs OpenAI #274 Antipodal Collisions vs Combined Self-Cancelling
    x_ro = [p * 100.0 for p in ro_p01_grid]
    c_self = [r["mean_c_self"] for r in ro_cancel_curve]
    c_ant = [r["mean_c_ant_felinity"] for r in ro_cancel_curve]
    c_comb = [r["mean_c_combined_self_cancelling"] for r in ro_cancel_curve]

    ax1.plot(x_ro, c_self, "^--", color="#f97316", lw=2.2, markersize=7, label="Self-Collisions C_self (+4n·δ² drift)")
    ax1.plot(x_ro, c_ant, "v--", color="#3b82f6", lw=2.2, markersize=7, label="OpenAI #274 Antipodal C_ant (-4n·δ² drift)")
    ax1.plot(x_ro, c_comb, "o-", color="#10b981", lw=2.8, markersize=8, label="Combined (C_self + C_ant)/2 (Self-Cancelling!)")
    ax1.axhline(128.0, color="#1e293b", ls=":", lw=1.8, label="Exact Flat Baseline (128.0)")
    ax1.set_xlabel("Asymmetric Readout Error P(0|1) (%) [with P(1|0) = 0.5%]", fontsize=10)
    ax1.set_ylabel("Mean Collision Count (16,384 shots)", fontsize=10)
    ax1.set_title("(a) Self-Cancelling Readout Bias via OpenAI #274 Felinity\n(Flat CCCZ, 26 CZs — No Extra Calibration Circuit Needed!)", fontsize=10.3, fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend(fontsize=8.3, loc="upper left")

    # Panel (b): Uncalibrated Combined Z-Score across p_2q
    x_p2q = [p * 100.0 for p in noise_grid]
    styles = {
        "iqm20_flat_ccz": ("#10b981", "o-", "Flat CCZ (17 CZ, M2=1.54b)"),
        "iqm20_flat_cccz": ("#0ea5e9", "s-", "Flat CCCZ (26 CZ, M2=1.38b)"),
        "iqm20_flat_3ccz_highmagic": ("#8b5cf6", "P-", "Flat 3xCCZ (55 CZ, M2=4.62b)"),
        "iqm20_faulty_dropped_cz": ("#f97316", "^--", "Faulty (1 dropped CZ)"),
        "iqm20_fooling_p3": ("#dc2626", "D--", "Universal p=3 Fooling"),
    }
    for cname, (col, fmt, lab) in styles.items():
        zs = [r["mean_z_uncalibrated_combined"] for r in triple_sweep[cname]]
        ax2.plot(x_p2q, zs, fmt, color=col, lw=2.2, markersize=6.8, label=lab)
    ax2.axhline(5.0, color="#334155", ls=":", lw=1.7, label="5-sigma threshold (Uncalibrated!)")
    ax2.axvline(0.5, color="#94a3b8", ls="-.", lw=1.2)
    ax2.set_xlabel("2-Qubit Native CZ Error Rate p_2q (%)", fontsize=10)
    ax2.set_ylabel("Uncalibrated Combined Z-Score (C_self + C_ant)/2", fontsize=10)
    ax2.set_title("(b) Calibration-Free Fault Detection (Strong Asym. Readout 3.0% vs 0.5%)", fontsize=10.3, fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(fontsize=8.3, loc="upper right")

    plt.suptitle("IQM Garnet (QB1..QB20) v4: OpenAI #274 Antipodal Felinity Cancels Asymmetric Readout Bias In-Situ", fontsize=11.8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
