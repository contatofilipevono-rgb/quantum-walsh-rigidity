"""
Reproducible experiments connecting:
1. Modular Walsh rigidity (8/15 uniform, 1/2 asymptotic) -> compact modular flatness
   certification & fault detection for degree <= 4 quantum phase circuits {Z, CZ, CCZ, CCCZ}.
2. Antipodal necessity (Lean-certified HRSBF theorem) -> ring-topology obstruction &
   SWAP overhead for rotation-symmetric hypergraph states.
3. Quantum magic (2-Renyi Stabilizer Entropy M_2) of bent hypergraph states.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypergraph_walsh import (
    HypergraphCircuit,
    build_mm_quartic_bent_circuit,
    build_rotation_symmetric_circuit,
    minimal_rigidity_modulus,
    ring_distance,
    stabilizer_renyi_entropy_m2,
    verify_flat_exact,
    verify_flat_modular,
)


def inject_fault(circuit: HypergraphCircuit, rng: random.Random, fault_type: str) -> HypergraphCircuit:
    """Inject a single-gate compilation fault (drop or insert a Z/CZ/CCZ/CCCZ gate)."""
    gates = list(circuit.gates)
    n = circuit.n_qubits
    if fault_type == "drop_cz":
        cz_idx = [i for i, g in enumerate(gates) if len(g) == 2]
        gates.pop(rng.choice(cz_idx))
    elif fault_type == "insert_ccz_cross":
        # Cross-register CCZ breaks Maiorana-McFarland flatness
        k = n // 2
        g = tuple(sorted((rng.randrange(0, k), rng.randrange(0, k), rng.randrange(k, n))))
        while len(set(g)) < 3:
            g = tuple(sorted((rng.randrange(0, k), rng.randrange(0, k), rng.randrange(k, n))))
        gates.append(g)
    elif fault_type == "insert_cccz_cross":
        k = n // 2
        qs = rng.sample(range(0, k), 2) + rng.sample(range(k, n), 2)
        gates.append(tuple(sorted(qs)))
    elif fault_type == "swap_cz_target":
        # Misroute one CZ gate
        cz_idx = [i for i, g in enumerate(gates) if len(g) == 2]
        idx = rng.choice(cz_idx)
        a, b = gates[idx]
        b_new = (b + 1) % n
        if b_new == a:
            b_new = (b + 2) % n
        gates[idx] = tuple(sorted((a, b_new)))
    else:
        raise ValueError(f"Unknown fault_type: {fault_type}")
    return HypergraphCircuit(n_qubits=n, gates=tuple(gates))


def experiment_1_modular_certification() -> dict:
    print("=" * 72)
    print("EXPERIMENT 1: Modular Flatness Certification (n=16, n=20 qubits)")
    print("=" * 72)
    rng = random.Random(20261119)
    results = {}

    for n in (16, 20):
        k = n // 2
        p_815 = minimal_rigidity_modulus(n, 8.0 / 15.0)
        p_half = minimal_rigidity_modulus(n, 0.5)

        # Build a genuine non-Clifford degree-4 bent hypergraph circuit (CZ + CCZ + CCCZ)
        extra_y = [
            (k, k + 1, k + 2),
            (k + 1, k + 2, k + 3),
            (k, k + 1, k + 2, k + 3),
            (k + 2, k + 3, k + 4, k + 5),
        ]
        clean_circ = build_mm_quartic_bent_circuit(n, extra_y)
        f_clean = clean_circ.truth_table()

        t0 = time.perf_counter()
        ok_exact, viol_exact = verify_flat_exact(f_clean, n)
        t_exact = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ok_mod, viol_mod = verify_flat_modular(f_clean, n, p_815)
        t_mod = (time.perf_counter() - t0) * 1000.0

        # Memory footprint comparison: int8 (1 byte) vs int32 (4 bytes)
        exact_bytes = (1 << n) * 4
        mod_bytes = (1 << n) * (1 if 2 * p_815 <= 127 else 2)

        print(f"\n[n = {n} qubits | degree = {clean_circ.degree} | gates = {len(clean_circ.gates)}]")
        print(f"  8/15 modulus p = {p_815} | 1/2 modulus p = {p_half}")
        print(f"  Clean circuit: exact={ok_exact} ({t_exact:.1f} ms, {exact_bytes/1024:.0f} KB) "
              f"| mod {p_815}={ok_mod} ({t_mod:.1f} ms, {mod_bytes/1024:.0f} KB)")

        # Fault injection test across moduli (including sub-threshold p=3, 5, 7)
        fault_types = ["drop_cz", "insert_ccz_cross", "insert_cccz_cross", "swap_cz_target"]
        trials_per_type = 5 if n == 20 else 10
        moduli_to_test = [3, 5, 7, p_815] if p_815 > 7 else [3, 5, p_815]
        detection_counts = {p: 0 for p in moduli_to_test}
        non_flat_total = 0

        for ft in fault_types:
            for _ in range(trials_per_type):
                faulty = inject_fault(clean_circ, rng, ft)
                f_bad = faulty.truth_table()
                is_flat, _ = verify_flat_exact(f_bad, n)
                if not is_flat:
                    non_flat_total += 1
                    for p in moduli_to_test:
                        passes_p, _ = verify_flat_modular(f_bad, n, p)
                        if not passes_p:
                            detection_counts[p] += 1

        print(f"  Fault injection ({non_flat_total} non-flat faulty circuits):")
        for p in moduli_to_test:
            tag = " <-- CERTIFIED (8/15 theorem)" if p == p_815 else " (below threshold)"
            rate = 100.0 * detection_counts[p] / max(1, non_flat_total)
            print(f"    mod p={p:2d}: detected {detection_counts[p]:2d}/{non_flat_total} ({rate:5.1f}%){tag}")

        results[f"n{n}"] = {
            "n_qubits": n,
            "p_8_15": p_815,
            "p_1_2": p_half,
            "clean_exact_ok": ok_exact,
            "clean_mod_ok": ok_mod,
            "exact_ms": round(t_exact, 2),
            "mod_ms": round(t_mod, 2),
            "memory_ratio_exact_over_mod": exact_bytes // mod_bytes,
            "faults_tested": non_flat_total,
            "detection_by_modulus": detection_counts,
        }

    return results


def experiment_2_antipodal_ring_obstruction() -> dict:
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Antipodal Obstruction for Ring-Symmetric {Z, CZ, CCZ}")
    print("=" * 72)
    rng = random.Random(20261119)
    results = {}

    for n in (6, 8, 10, 12):
        H = n // 2
        # Quadratic orbit seeds: (0, d) for d = 1 .. H
        non_antipodal_cz = [(0, d) for d in range(1, H)]
        antipodal_cz = (0, H)
        # Sample cubic orbit seeds (0, a, b) with 0 < a < b < n
        cubic_orbits = []
        seen_canon = set()
        for a in range(1, n):
            for b in range(a + 1, n):
                orb = tuple(
                    sorted(tuple(sorted(((0 + s) % n, (a + s) % n, (b + s) % n))) for s in range(n))
                )
                if orb not in seen_canon:
                    seen_canon.add(orb)
                    cubic_orbits.append((0, a, b))

        n_samples = 200 if n <= 10 else 60
        flat_without_antipodal = 0
        flat_with_antipodal = 0

        for _ in range(n_samples):
            chosen_cz = [g for g in non_antipodal_cz if rng.random() < 0.5]
            chosen_ccz = rng.sample(cubic_orbits, k=min(len(cubic_orbits), rng.randint(1, 3)))

            # Without antipodal CZ orbit
            c_no = build_rotation_symmetric_circuit(n, chosen_cz + chosen_ccz)
            ok_no, _ = verify_flat_exact(c_no.truth_table(), n)
            if ok_no:
                flat_without_antipodal += 1

            # Purely quadratic + antipodal check (and with cubic)
            c_yes = build_rotation_symmetric_circuit(n, [antipodal_cz] + chosen_cz)
            ok_yes, _ = verify_flat_exact(c_yes.truth_table(), n)
            if ok_yes:
                flat_with_antipodal += 1

        # Minimum SWAP distance on a 1D ring for antipodal CZ_{i, i+H}
        min_ring_dist = ring_distance(n, 0, H)
        min_swaps_per_antipodal_cz = min_ring_dist - 1

        print(f"  n = {n:2d} qubits (H={H}): "
              f"flat WITHOUT antipodal CZ = {flat_without_antipodal}/{n_samples} (0% by theorem) | "
              f"flat WITH antipodal CZ (quadratic sample) = {flat_with_antipodal}/{n_samples} | "
              f"ring dist = {min_ring_dist} (min {min_swaps_per_antipodal_cz} SWAPs/pair)")

        results[f"n{n}"] = {
            "n_qubits": n,
            "H": H,
            "samples": n_samples,
            "flat_without_antipodal": flat_without_antipodal,
            "flat_with_antipodal_quad_sample": flat_with_antipodal,
            "antipodal_ring_distance": min_ring_dist,
            "min_swaps_per_antipodal_cz": min_swaps_per_antipodal_cz,
        }

    return results


def experiment_3_quantum_magic() -> dict:
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Quantum Magic (Stabilizer Renyi Entropy M_2) of Bent States")
    print("=" * 72)
    results = {}

    cases = [
        ("n=6 Clifford graph state (deg 2, bent)", 6, []),
        ("n=6 Cubic bent hypergraph (deg 3, CCZ)", 6, [(3, 4, 5)]),
        ("n=8 Clifford graph state (deg 2, bent)", 8, []),
        ("n=8 Cubic bent hypergraph (deg 3, CCZ)", 8, [(4, 5, 6)]),
        ("n=8 Quartic bent hypergraph (deg 4, CCCZ)", 8, [(4, 5, 6, 7)]),
        ("n=8 Mixed deg 3+4 bent hypergraph", 8, [(4, 5, 6), (4, 5, 6, 7)]),
    ]

    for label, n, extra_y in cases:
        circ = build_mm_quartic_bent_circuit(n, extra_y)
        f = circ.truth_table()
        is_flat, _ = verify_flat_exact(f, n)
        m2 = stabilizer_renyi_entropy_m2(f, n)
        print(f"  {label:44s} | flat={str(is_flat):5s} | M_2 = {m2:.6f} bits")
        results[label] = {
            "n_qubits": n,
            "degree": circ.degree,
            "is_flat": is_flat,
            "stabilizer_renyi_m2": round(m2, 6),
        }

    return results


def main() -> None:
    summary = {
        "experiment_1_modular_certification": experiment_1_modular_certification(),
        "experiment_2_antipodal_ring_obstruction": experiment_2_antipodal_ring_obstruction(),
        "experiment_3_quantum_magic": experiment_3_quantum_magic(),
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved results to:", out_path)


if __name__ == "__main__":
    main()
