"""
Construct and verify explicit Rotation-Symmetric (RS) degree-4 flat hypergraph states
on n = 8 and n = 16 qubits (via even/odd interleaving), bridging:
  - Cyclic ring symmetry (Theorem 1 context),
  - Degree-4 non-Clifford gates (CCCZ) and exact quantum magic M_2,
  - Certified modular flatness verification at p = 9 (Theorem 2, n = 16).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypergraph_walsh import (
    HypergraphCircuit,
    build_rotation_symmetric_circuit,
    stabilizer_renyi_entropy_m2,
    verify_flat_exact,
    verify_flat_modular,
)
from nisq_and_qasm import save_qasm_examples


def find_rs_quartic_bent_n8() -> tuple[list[tuple[int, ...]], HypergraphCircuit]:
    """Find a compact rotation-symmetric degree-4 bent hypergraph circuit on n=8 qubits."""
    # Known compact RS quartic bent on n=8: antipodal CZ (0,4) + quartic orbit(s)
    # Search over antipodal (0,4) + optional CZ (0,1),(0,2),(0,3) + 1 or 2 quartic orbits
    quad_seeds = [(0, 4), (0, 1), (0, 2), (0, 3)]
    quartic_candidates = [
        (0, 1, 2, 3),
        (0, 1, 2, 4),
        (0, 1, 2, 5),
        (0, 1, 3, 4),
        (0, 1, 3, 5),
        (0, 1, 4, 5),
        (0, 2, 4, 6),
    ]
    # Try single quartic orbit + subsets of quad seeds (must include (0,4) or any subset)
    for q4 in quartic_candidates:
        for mask in range(1, 1 << len(quad_seeds)):
            seeds = [quad_seeds[i] for i in range(len(quad_seeds)) if (mask >> i) & 1]
            seeds.append(q4)
            circ = build_rotation_symmetric_circuit(8, seeds)
            if circ.degree == 4:
                ok, _ = verify_flat_exact(circ.truth_table(), 8)
                if ok:
                    return seeds, circ

    # Try pairs of quartic orbits + subsets of quad seeds
    for i in range(len(quartic_candidates)):
        for j in range(i + 1, len(quartic_candidates)):
            for mask in range(1, 1 << len(quad_seeds)):
                seeds = [quad_seeds[k] for k in range(len(quad_seeds)) if (mask >> k) & 1]
                seeds.extend([quartic_candidates[i], quartic_candidates[j]])
                circ = build_rotation_symmetric_circuit(8, seeds)
                if circ.degree == 4:
                    ok, _ = verify_flat_exact(circ.truth_table(), 8)
                    if ok:
                        return seeds, circ
    raise RuntimeError("No RS quartic bent found in candidate set.")


def interleave_rs_n8_to_n16(seeds_n8: list[tuple[int, ...]]) -> HypergraphCircuit:
    """
    Interleave two copies of an n=8 RS circuit onto even and odd qubits of n=16:
        f(x_0..x_15) = h(x_0,x_2,...,x_14) xor h(x_1,x_3,...,x_15).
    Preserves cyclic rotation symmetry (sigma shifts even <-> odd), degree 4, and flatness.
    """
    seeds_n16 = [tuple(2 * q for q in s) for s in seeds_n8]
    return build_rotation_symmetric_circuit(16, seeds_n16)


def main() -> None:
    seeds_8, circ_8 = find_rs_quartic_bent_n8()
    f8 = circ_8.truth_table()
    ok8, _ = verify_flat_exact(f8, 8)
    m2_8 = stabilizer_renyi_entropy_m2(f8, 8)

    circ_16 = interleave_rs_n8_to_n16(seeds_8)
    f16 = circ_16.truth_table()
    ok16_exact, _ = verify_flat_exact(f16, 16)
    ok16_mod9, viol16_mod9 = verify_flat_modular(f16, 16, 9)
    # By tensor-product additivity of M_2 on disjoint even/odd registers:
    m2_16_exact = 2.0 * m2_8

    qasm_paths = save_qasm_examples(
        ROOT / "qasm",
        {
            "rs_quartic_flat_n8": circ_8,
            "rs_quartic_flat_n16": circ_16,
        },
    )

    print("Found RS degree-4 flat hypergraph state on n=8:")
    print(f"  Orbit seeds (n=8): {seeds_8}")
    print(f"  Total gates (n=8): {len(circ_8.gates)} | degree={circ_8.degree} | flat={ok8} | M_2={m2_8:.6f} bits")
    print("Interleaved RS degree-4 flat hypergraph state on n=16:")
    print(f"  Total gates (n=16): {len(circ_16.gates)} | degree={circ_16.degree}")
    print(f"  Exact flat={ok16_exact} | Certified mod p=9 flat={ok16_mod9} (violations={viol16_mod9})")
    print(f"  Exact M_2 (n=16, additivity) = 2 * M_2(n=8) = {m2_16_exact:.6f} bits")
    print(f"  Exported QASM: {[Path(p).name for p in qasm_paths]}")

    out = {
        "n8_orbit_seeds": [list(s) for s in seeds_8],
        "n8_num_gates": len(circ_8.gates),
        "n8_flat": ok8,
        "n8_m2_bits": round(m2_8, 6),
        "n16_num_gates": len(circ_16.gates),
        "n16_flat_exact": ok16_exact,
        "n16_flat_mod9": ok16_mod9,
        "n16_m2_bits_exact": round(m2_16_exact, 6),
        "qasm_files": qasm_paths,
    }
    (ROOT / "results" / "rs_quartic_bridge.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
