"""
Scaled benchmark (n=16, 20, 24 qubits) + universal sub-threshold counterexample (p=3).

Demonstrates:
1. 24-qubit (16,777,216 amplitudes) modular certification at p=35 (8/15 theorem)
   using int8 (16 MB) vs exact int32 (64 MB) — 4x memory reduction and faster runtime.
2. Explicit degree-4 non-bent hypergraph family:
       f(x) = bent_{n-4}(x_0..x_{n-5}) xor (x_{n-4} x_{n-3} x_{n-2} x_{n-1})
   whose Walsh spectrum is { +/- (7/2) M^2, +/- (1/2) M^2 }:
   - 0% of frequencies have the bent magnitude M^2 (100% wrong!),
   - YET 100% of all 2^n frequencies pass modular verification at p=3 (false positive!),
   - While the certified 8/15 modulus (p=9, 19, 35) rejects 100% of frequencies.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hypergraph_walsh import (
    HypergraphCircuit,
    build_mm_quartic_bent_circuit,
    minimal_rigidity_modulus,
    verify_flat_exact,
    verify_flat_modular,
)


def build_p3_fooling_circuit(n_qubits: int) -> HypergraphCircuit:
    """
    Non-bent degree-4 hypergraph circuit on n = 4m qubits:
        bent quadratic graph state on first n-4 qubits + single CCCZ on last 4 qubits.
    Every Walsh coefficient is +/- (7/2) M^2 or +/- (1/2) M^2, so:
      - 0% of frequencies satisfy |W_f(b)| = M^2,
      - 100% of frequencies satisfy W_f(b) == +/- M^2 (mod 3).
    """
    gates = [(2 * i, 2 * i + 1) for i in range((n_qubits - 4) // 2)]
    gates.append((n_qubits - 4, n_qubits - 3, n_qubits - 2, n_qubits - 1))
    return HypergraphCircuit(n_qubits=n_qubits, gates=tuple(gates))


def main() -> None:
    print("=" * 74)
    print("SCALED BENCHMARK (n = 16, 20, 24 qubits) & p=3 FALSE-POSITIVE COUNTEREXAMPLE")
    print("=" * 74)
    results = {}

    for n in (16, 20, 24):
        k = n // 2
        p_815 = minimal_rigidity_modulus(n, 8.0 / 15.0)
        p_half = minimal_rigidity_modulus(n, 0.5)
        size = 1 << n

        # 1. Clean degree-4 bent hypergraph circuit
        extra_y = [
            (k, k + 1, k + 2),
            (k, k + 1, k + 2, k + 3),
        ]
        clean = build_mm_quartic_bent_circuit(n, extra_y)
        f_clean = clean.truth_table()

        t0 = time.perf_counter()
        ok_exact, _ = verify_flat_exact(f_clean, n)
        t_exact_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ok_mod, _ = verify_flat_modular(f_clean, n, p_815)
        t_mod_ms = (time.perf_counter() - t0) * 1000.0

        exact_mb = (size * 4) / (1024 * 1024)
        mod_mb = (size * (1 if 2 * p_815 <= 127 else 2)) / (1024 * 1024)

        # 2. Universal p=3 fooling counterexample (degree 4, 100% non-bent)
        fool = build_p3_fooling_circuit(n)
        f_fool = fool.truth_table()

        fool_exact_ok, fool_exact_viol = verify_flat_exact(f_fool, n)
        fool_p3_ok, fool_p3_viol = verify_flat_modular(f_fool, n, 3)
        fool_p815_ok, fool_p815_viol = verify_flat_modular(f_fool, n, p_815)

        print(f"\n[n = {n} qubits | 2^n = {size:,} amplitudes | p_8/15 = {p_815} | p_1/2 = {p_half}]")
        print(f"  Clean deg-4 bent : exact={ok_exact} ({t_exact_ms:7.1f} ms, {exact_mb:5.2f} MB)"
              f" | mod {p_815}={ok_mod} ({t_mod_ms:7.1f} ms, {mod_mb:5.2f} MB) | speedup={t_exact_ms/t_mod_ms:.2f}x")
        print(f"  Fooling circuit  : exact_ok={fool_exact_ok} (violations={fool_exact_viol:,}/{size:,})")
        print(f"    -> mod p=3     : PASS={fool_p3_ok} (violations={fool_p3_viol:,}/{size:,})  <-- FALSE POSITIVE!")
        print(f"    -> mod p={p_815:<2d}    : PASS={fool_p815_ok} (violations={fool_p815_viol:,}/{size:,})  <-- CERTIFIED REJECTION")

        results[f"n{n}"] = {
            "n_qubits": n,
            "amplitudes": size,
            "p_8_15": p_815,
            "p_1_2": p_half,
            "clean_exact_ok": ok_exact,
            "clean_mod_ok": ok_mod,
            "exact_ms": round(t_exact_ms, 2),
            "mod_ms": round(t_mod_ms, 2),
            "speedup": round(t_exact_ms / t_mod_ms, 2),
            "exact_mb": round(exact_mb, 3),
            "mod_mb": round(mod_mb, 3),
            "fool_exact_violations": fool_exact_viol,
            "fool_p3_pass": fool_p3_ok,
            "fool_p3_violations": fool_p3_viol,
            "fool_p815_pass": fool_p815_ok,
            "fool_p815_violations": fool_p815_viol,
        }

    out_path = ROOT / "results" / "benchmark_24q_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved results to:", out_path)


if __name__ == "__main__":
    main()
