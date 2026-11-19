"""
Pillar 2: OpenQASM 3.0 exporter & NISQ noisy X-basis flatness testing simulator.

Exports HypergraphCircuit objects ({Z, CZ, CCZ, CCCZ} + H^{otimes n} measurement)
to OpenQASM 3.0 and simulates finite-shot X-basis measurements under depolarizing
and phase-flip gate noise to distinguish flat hypergraph states from faulty circuits
(including the universal p=3 fooling circuit).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from hypergraph_walsh import HypergraphCircuit, fwht_exact


def export_openqasm3(circuit: HypergraphCircuit, measure_x: bool = True) -> str:
    """Export a HypergraphCircuit to clean OpenQASM 3.0."""
    n = circuit.n_qubits
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{n}] q;",
        f"bit[{n}] c;",
        "",
        "// Prepare |+>^{\\otimes n}",
    ]
    for i in range(n):
        lines.append(f"h q[{i}];")
    lines.append("")
    lines.append("// Diagonal hypergraph phase oracle U_f")
    for g in circuit.gates:
        if len(g) == 1:
            lines.append(f"z q[{g[0]}];")
        elif len(g) == 2:
            lines.append(f"cz q[{g[0]}], q[{g[1]}];")
        elif len(g) == 3:
            lines.append(f"ctrl(2) @ z q[{g[0]}], q[{g[1]}], q[{g[2]}];")
        elif len(g) == 4:
            lines.append(f"ctrl(3) @ z q[{g[0]}], q[{g[1]}], q[{g[2]}], q[{g[3]}];")
        else:
            raise ValueError(f"Unsupported gate arity: {len(g)}")
    if measure_x:
        lines.append("")
        lines.append("// Final Hadamard layer H^{\\otimes n} and X-basis measurement")
        for i in range(n):
            lines.append(f"h q[{i}];")
        for i in range(n):
            lines.append(f"c[{i}] = measure q[{i}];")
    return "\n".join(lines) + "\n"


def ideal_x_probabilities(circuit: HypergraphCircuit) -> np.ndarray:
    """Exact X-basis measurement distribution P(b) = (W_f(b) / 2^n)^2."""
    f = circuit.truth_table()
    w = fwht_exact(f).astype(np.float64)
    probs = (w / float(1 << circuit.n_qubits)) ** 2
    probs /= probs.sum()
    return probs


def noisy_x_probabilities(circuit: HypergraphCircuit, p_2q_error: float) -> np.ndarray:
    """
    Effective global depolarizing + gate-weighted noise model for X-basis outcomes:
      - Each CZ counts as 1 two-qubit gate
      - Each CCZ decomposes into 6 CNOTs (2q-equivalent = 6)
      - Each CCCZ decomposes into 14 CNOTs (2q-equivalent = 14)
    Survival probability: F = (1 - p_2q_error) ** N_2q_equiv
    Noisy distribution: P_noisy(b) = F * P_ideal(b) + (1 - F) * (1 / 2^n).
    """
    n_2q = 0
    for g in circuit.gates:
        if len(g) == 2:
            n_2q += 1
        elif len(g) == 3:
            n_2q += 6
        elif len(g) == 4:
            n_2q += 14
    survival = (1.0 - p_2q_error) ** n_2q
    p_ideal = ideal_x_probabilities(circuit)
    uniform = np.full_like(p_ideal, 1.0 / p_ideal.size)
    return survival * p_ideal + (1.0 - survival) * uniform


def simulate_nisq_flatness_test(
    circuit: HypergraphCircuit,
    p_2q_error: float,
    shots: int,
    rng: np.random.Generator,
) -> dict:
    """
    Sample `shots` X-basis outcomes under `p_2q_error` and compute:
      - Total Variation Distance (TVD) to uniform 2^{-n}
      - Normalized Pearson chi-square statistic: chi2 / df (expected ~ 1.0 for flat)
      - Z-score of chi2: (chi2 - df) / sqrt(2 * df) (> 5.0 cleanly rejects flatness)
    """
    probs = noisy_x_probabilities(circuit, p_2q_error)
    dim = probs.size
    counts = rng.multinomial(shots, probs)
    emp = counts / float(shots)
    tvd = 0.5 * float(np.sum(np.abs(emp - (1.0 / dim))))
    expected = shots / float(dim)
    chi2 = float(np.sum(((counts - expected) ** 2) / expected))
    df = dim - 1
    z_score = (chi2 - df) / math.sqrt(2.0 * df)
    return {
        "tvd": round(tvd, 5),
        "chi2_over_df": round(chi2 / df, 4),
        "z_score": round(z_score, 2),
        "rejects_flatness_5sigma": bool(z_score > 5.0),
    }


def save_qasm_examples(out_dir: Path, circuits: dict[str, HypergraphCircuit]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, circ in circuits.items():
        path = out_dir / f"{name}.qasm"
        path.write_text(export_openqasm3(circ, measure_x=True), encoding="utf-8")
        written.append(str(path))
    return written
