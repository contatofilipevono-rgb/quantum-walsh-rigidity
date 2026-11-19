"""
Quantum Walsh Rigidity: Hypergraph State Certification & Ring Obstructions.

Implements:
- Degree <= 4 diagonal phase circuits {Z, CZ, CCZ, CCCZ} (hypergraph states)
- Exact and compact modular Fast Walsh-Hadamard Transform (FWHT mod p)
- Flatness (bentness) verification using the 8/15 uniform and 1/2 asymptotic bounds
- Rotation-symmetric ring hypergraph generators (antipodal necessity check)
- 2-Renyi Stabilizer Entropy M_2 (quantum magic) for phase states
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


Gate = tuple[int, ...]


@dataclass(frozen=True)
class HypergraphCircuit:
    """Diagonal quantum phase circuit with gates in {Z, CZ, CCZ, CCCZ}."""

    n_qubits: int
    gates: tuple[Gate, ...]

    @property
    def degree(self) -> int:
        return max((len(g) for g in self.gates), default=0)

    def truth_table(self) -> np.ndarray:
        """Evaluate f(x) in {0, 1} over all x in F_2^n (vectorized)."""
        size = 1 << self.n_qubits
        idx = np.arange(size, dtype=np.uint32)
        f = np.zeros(size, dtype=np.uint8)
        for gate in self.gates:
            mask = 0
            for q in gate:
                mask |= 1 << q
            # Monomial prod_{q in gate} x_q is 1 iff (x & mask) == mask
            f ^= ((idx & mask) == mask).astype(np.uint8)
        return f


def fwht_exact(f_bits: np.ndarray) -> np.ndarray:
    """Compute exact unnormalized Walsh-Hadamard transform W_f(b)."""
    # (-1)^f(x) in {+1, -1}
    w = np.where(f_bits == 0, 1, -1).astype(np.int32)
    n_size = w.shape[0]
    h = 1
    while h < n_size:
        w = w.reshape(-1, 2 * h)
        left = w[:, :h].copy()
        right = w[:, h:]
        w[:, :h] = left + right
        w[:, h:] = left - right
        h <<= 1
    return w.reshape(n_size)


def fwht_modular(f_bits: np.ndarray, p: int) -> np.ndarray:
    """
    Compute W_f(b) mod p using overflow-free compact unsigned integer arithmetic.

    Uses uint8 (1 byte/amplitude) whenever 2*p <= 255 (p <= 127, covering
    n=16 p=9, n=20 p=19, n=24 p=35, n=28 p=69), and uint16 otherwise (p <= 32767).
    """
    if p <= 1:
        raise ValueError("Modulus p must be > 1.")
    dtype = np.uint8 if (2 * p <= 255) else np.uint16
    p_val = dtype(p)
    w = np.where(f_bits == 0, dtype(1), dtype(p - 1))
    n_size = w.shape[0]
    h = 1
    while h < n_size:
        w = w.reshape(-1, 2 * h)
        left = w[:, :h].copy()
        right = w[:, h:]
        # Addition mod p: left + right in [0, 2p-2] <= 255 (no overflow)
        s = left + right
        s[s >= p_val] -= p_val
        # Subtraction mod p: left + (p - right) in [1, 2p-1] <= 255 (no underflow)
        d = left + (p_val - right)
        d[d >= p_val] -= p_val
        w[:, :h] = s
        w[:, h:] = d
        h <<= 1
    return w.reshape(n_size)


def verify_flat_exact(f_bits: np.ndarray, n_qubits: int) -> tuple[bool, int]:
    """
    Exact check whether |psi_f> is a flat (bent) hypergraph state:
    |W_f(b)| == 2^(n/2) for all b.
    Returns (is_flat, num_violations).
    """
    if n_qubits % 2 != 0:
        raise ValueError("Flat (bent) binary phase states require even n_qubits.")
    target = 1 << (n_qubits // 2)
    w = fwht_exact(f_bits)
    violations = int(np.count_nonzero(np.abs(w) != target))
    return (violations == 0), violations


def verify_flat_modular(f_bits: np.ndarray, n_qubits: int, p: int) -> tuple[bool, int]:
    """
    Modular flatness verifier:
    Checks W_f(b) == +2^(n/2) or -2^(n/2) (mod p) for all b.
    By the 8/15 uniform rigidity theorem (m >= 4, n = 4m), passing this test
    with odd p >= (8/15)*2^m mathematically certifies exact flatness (bentness).
    Returns (passes_modular_test, num_violations).
    """
    if n_qubits % 2 != 0:
        raise ValueError("Flat (bent) binary phase states require even n_qubits.")
    target = 1 << (n_qubits // 2)
    t_pos = target % p
    t_neg = (-target) % p
    w_mod = fwht_modular(f_bits, p)
    ok = (w_mod == t_pos) | (w_mod == t_neg)
    violations = int(np.count_nonzero(~ok))
    return (violations == 0), violations


def minimal_rigidity_modulus(n_qubits: int, ratio: float = 8.0 / 15.0) -> int:
    """Smallest odd integer p >= ratio * 2^(n/4) for n = 4m qubits."""
    if n_qubits % 4 != 0:
        raise ValueError("n_qubits must be a multiple of 4 (n = 4m).")
    m = n_qubits // 4
    M = 1 << m
    p = math.ceil(ratio * M)
    if p % 2 == 0:
        p += 1
    return p


def build_mm_quartic_bent_circuit(
    n_qubits: int, extra_y_gates: Sequence[Gate]
) -> HypergraphCircuit:
    """
    Build a Maiorana-McFarland bent hypergraph circuit on n = 2k qubits:
        f(x, y) = sum_{i=0}^{k-1} x_i y_i + g(y)
    where g(y) is any degree <= 4 polynomial on the second half of qubits
    (qubits k .. 2k-1). Always produces an exact bent (flat) state.
    """
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even.")
    k = n_qubits // 2
    gates: list[Gate] = [(i, k + i) for i in range(k)]
    for g in extra_y_gates:
        if any(q < k or q >= n_qubits for q in g):
            raise ValueError(f"Gate {g} must act only on y-register [{k}, {n_qubits - 1}].")
        gates.append(tuple(sorted(g)))
    return HypergraphCircuit(n_qubits=n_qubits, gates=tuple(gates))


def rotation_orbit(n_qubits: int, seed_gate: Gate) -> tuple[Gate, ...]:
    """Generate the full cyclic rotation orbit of a gate on an n-qubit ring."""
    seen: set[Gate] = set()
    orbit: list[Gate] = []
    for shift in range(n_qubits):
        shifted = tuple(sorted((q + shift) % n_qubits for q in seed_gate))
        if shifted not in seen:
            seen.add(shifted)
            orbit.append(shifted)
    return tuple(orbit)


def build_rotation_symmetric_circuit(
    n_qubits: int, orbit_seeds: Iterable[Gate]
) -> HypergraphCircuit:
    """Build a rotation-symmetric hypergraph circuit from orbit representatives."""
    gate_set: set[Gate] = set()
    for seed in orbit_seeds:
        for g in rotation_orbit(n_qubits, seed):
            if g in gate_set:
                gate_set.remove(g)  # mod-2 cancellation
            else:
                gate_set.add(g)
    return HypergraphCircuit(n_qubits=n_qubits, gates=tuple(sorted(gate_set)))


def ring_distance(n_qubits: int, i: int, j: int) -> int:
    """Shortest path distance between qubits i and j on a 1D ring C_n."""
    d = abs(i - j) % n_qubits
    return min(d, n_qubits - d)


def stabilizer_renyi_entropy_m2(f_bits: np.ndarray, n_qubits: int, batch_size: int = 256) -> float:
    """
    Compute the exact 2-Renyi Stabilizer Entropy M_2(|psi_f>) (Leone et al., PRL 2022)
    using batched vectorized FWHT over directional derivatives D_a f(x) = f(x) xor f(x xor a).

    For a binary phase state |psi_f> = 2^(-n/2) sum_x (-1)^f(x) |x>:
        |<psi_f| P_{a,b} |psi_f>| = |W_{D_a f}(b)| / 2^n.
    M_2 = 0 iff |psi_f> is a stabilizer (Clifford / degree <= 2) state.
    """
    size = 1 << n_qubits
    idx = np.arange(size, dtype=np.uint32)
    total_fourth = 0.0
    bs = min(batch_size, size)

    for start in range(0, size, bs):
        end = min(start + bs, size)
        a_vec = np.arange(start, end, dtype=np.uint32)[:, None]
        # Shape: (batch, size)
        da = f_bits[None, :] ^ f_bits[idx[None, :] ^ a_vec]
        w = np.where(da == 0, 1, -1).astype(np.int32)
        h = 1
        while h < size:
            w = w.reshape(end - start, -1, 2 * h)
            left = w[:, :, :h].copy()
            right = w[:, :, h:]
            w[:, :, :h] = left + right
            w[:, :, h:] = left - right
            h <<= 1
        w = w.reshape(end - start, size).astype(np.float64)
        total_fourth += float(np.sum(w ** 4))

    avg = total_fourth / (2.0 ** (5 * n_qubits))
    avg = min(1.0, max(avg, 1e-300))
    val = -math.log2(avg)
    return 0.0 if abs(val) < 1e-12 else val
