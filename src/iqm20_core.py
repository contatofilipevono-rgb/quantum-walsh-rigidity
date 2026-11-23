"""
IQM Garnet 20-Qubit Non-Clifford Walsh Rigidity, Exterior-Rank &
OpenAI-#274 Antipodal Felinity Benchmark Suite (v4).

New in v4 (Integrating OpenAI #274 Corollaries 8 & 9 + #112 Depth-3 Duality):
1. OpenAI #274 Corollary 8 Antipodal Felinity Estimator:
   F_n(rho) = 2 * sum_{y in {0,1}^n} p_y * p_{~y}, where ~y = y XOR 1^n.
2. Self-Cancelling Readout-Bias Estimator (C_self + C_ant) / 2:
   - Under asymmetric readout bias p(0|1) != p(1|0) (with bias delta = (p01 - p10)/2),
     Self-Collisions drift positively by +4*n*delta^2:
         E[C_self] = binom(S,2) * 2^{-n} * (1 + 4*delta^2)^n
     while OpenAI #274 Antipodal Collisions (Felinity) drift negatively by -4*n*delta^2:
         E[C_ant]  = binom(S,2) * 2^{-n} * (1 - 4*delta^2)^n.
   - Their average C_combined = (C_self + C_ant) / 2 cancels the first-order asymmetric
     readout bias to O(n^2 * delta^4) WITHOUT needing a separate calibration circuit!
3. Disjoint-Layer QAC^0 Entangling Depth (OpenAI #274 Section 2):
   - Computes exact minimum parallel disjoint-support CZ layer depth on IQM Garnet 20.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

Gate = tuple[int, ...]
Graph = dict[int, set[int]]


@dataclass(frozen=True)
class HypergraphCircuit:
    n_qubits: int
    gates: tuple[Gate, ...]
    name: str = ""

    @property
    def degree(self) -> int:
        return max((len(g) for g in self.gates), default=0)

    def gate_counts(self) -> dict[str, int]:
        c = {"Z": 0, "CZ": 0, "CCZ": 0, "CCCZ": 0}
        for g in self.gates:
            if len(g) == 1:
                c["Z"] += 1
            elif len(g) == 2:
                c["CZ"] += 1
            elif len(g) == 3:
                c["CCZ"] += 1
            elif len(g) == 4:
                c["CCCZ"] += 1
        return c

    def truth_table(self) -> np.ndarray:
        size = 1 << self.n_qubits
        idx = np.arange(size, dtype=np.uint32)
        f = np.zeros(size, dtype=np.uint8)
        for gate in self.gates:
            mask = 0
            for q in gate:
                mask |= 1 << q
            f ^= ((idx & mask) == mask).astype(np.uint8)
        return f


# ---------------------------------------------------------------------------
# 1. Official IQM Garnet (Crystal-20) Coupling Graph & QB1..QB20 Labels
# ---------------------------------------------------------------------------

def build_iqm_garnet_20() -> tuple[Graph, dict[int, tuple[int, int]], dict[int, str]]:
    coords_list = [
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 2),
    ]
    coord_to_idx = {pt: i for i, pt in enumerate(coords_list)}
    coords = {i: pt for i, pt in enumerate(coords_list)}
    qb_labels = {i: f"QB{i + 1}" for i in range(20)}
    g: Graph = {i: set() for i in range(20)}
    for i, (r, c) in enumerate(coords_list):
        for dr, dc in ((0, 1), (1, 0)):
            nxt = (r + dr, c + dc)
            if nxt in coord_to_idx:
                j = coord_to_idx[nxt]
                g[i].add(j)
                g[j].add(i)
    assert sum(len(v) for v in g.values()) // 2 == 30
    return g, coords, qb_labels


def build_1d_ring(n: int = 20) -> Graph:
    g: Graph = {i: set() for i in range(n)}
    for i in range(n):
        j = (i + 1) % n
        g[i].add(j)
        g[j].add(i)
    return g


def build_grid_4x5() -> Graph:
    g: Graph = {i: set() for i in range(20)}
    for r in range(4):
        for c in range(5):
            u = r * 5 + c
            if c + 1 < 5:
                v = r * 5 + (c + 1)
                g[u].add(v)
                g[v].add(u)
            if r + 1 < 4:
                v = (r + 1) * 5 + c
                g[u].add(v)
                g[v].add(u)
    return g


def build_heavy_hex_20() -> Graph:
    edges_raw: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(5):
        for c in range(8):
            u = (2 * r, 2 * c)
            if c + 1 < 8:
                mid = (2 * r, 2 * c + 1)
                v = (2 * r, 2 * c + 2)
                edges_raw.append((u, mid))
                edges_raw.append((mid, v))
            if r + 1 < 5 and ((r + c) % 2 == 0):
                mid = (2 * r + 1, 2 * c)
                v = (2 * r + 2, 2 * c)
                edges_raw.append((u, mid))
                edges_raw.append((mid, v))
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for a, b in edges_raw:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    order: list[tuple[int, int]] = []
    seen = {(0, 0)}
    q = deque([(0, 0)])
    while q and len(order) < 20:
        cur = q.popleft()
        order.append(cur)
        for nxt in sorted(adj.get(cur, ())):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    idx_map = {pt: i for i, pt in enumerate(order)}
    g: Graph = {i: set() for i in range(20)}
    for pt, i in idx_map.items():
        for nxt in adj.get(pt, ()):
            if nxt in idx_map:
                j = idx_map[nxt]
                g[i].add(j)
                g[j].add(i)
    return g


def iqm_garnet_20_mm_matching() -> tuple[list[tuple[int, int]], list[int], list[int]]:
    _, coords, _ = build_iqm_garnet_20()
    c2i = {pt: i for i, pt in coords.items()}
    pairs_xy_coords = [
        ((0, 1), (0, 2)),  # 0: QB1-QB2   (y0=QB2)
        ((0, 3), (1, 3)),  # 1: QB3-QB7   (y1=QB7)
        ((1, 0), (1, 1)),  # 2: QB4-QB5   (y2=QB5)
        ((1, 2), (2, 2)),  # 3: QB6-QB11  (y3=QB11)
        ((1, 4), (2, 4)),  # 4: QB8-QB13  (y4=QB13)
        ((3, 0), (2, 0)),  # 5: QB14-QB9  (y5=QB9)
        ((2, 1), (3, 1)),  # 6: QB10-QB15 (y6=QB15)
        ((2, 3), (3, 3)),  # 7: QB12-QB17 (y7=QB17)
        ((4, 1), (4, 2)),  # 8: QB19-QB20 (y8=QB20)
        ((3, 4), (3, 2)),  # 9: QB18-QB16 (y9=QB16)
    ]
    pairs = [(c2i[a], c2i[b]) for a, b in pairs_xy_coords]
    return pairs, [a for a, _ in pairs], [b for _, b in pairs]


# ---------------------------------------------------------------------------
# 2. Fast Walsh-Hadamard Transforms & Magic M_2
# ---------------------------------------------------------------------------

def fwht_exact(f_bits: np.ndarray) -> np.ndarray:
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


def fwht_modular_u8(f_bits: np.ndarray, p: int) -> np.ndarray:
    assert 1 < p <= 127
    p_val = np.uint8(p)
    w = np.where(f_bits == 0, np.uint8(1), np.uint8(p - 1))
    n_size = w.shape[0]
    h = 1
    while h < n_size:
        w = w.reshape(-1, 2 * h)
        left = w[:, :h].copy()
        right = w[:, h:]
        s = left + right
        s[s >= p_val] -= p_val
        d = left + (p_val - right)
        d[d >= p_val] -= p_val
        w[:, :h] = s
        w[:, h:] = d
        h <<= 1
    return w.reshape(n_size)


def verify_flat_exact(f_bits: np.ndarray, n_qubits: int) -> tuple[bool, int, np.ndarray]:
    target = 1 << (n_qubits // 2)
    w = fwht_exact(f_bits)
    viol = int(np.count_nonzero(np.abs(w) != target))
    return (viol == 0), viol, w


def verify_flat_modular(f_bits: np.ndarray, n_qubits: int, p: int) -> tuple[bool, int]:
    target = 1 << (n_qubits // 2)
    t_pos = target % p
    t_neg = (-target) % p
    w_mod = fwht_modular_u8(f_bits, p)
    ok = (w_mod == t_pos) | (w_mod == t_neg)
    viol = int(np.count_nonzero(~ok))
    return (viol == 0), viol


def stabilizer_entropy_m2_small(f_bits: np.ndarray, n_qubits: int) -> float:
    size = 1 << n_qubits
    idx = np.arange(size, dtype=np.uint32)
    total_p4 = 0.0
    norm = float(size)
    for a in range(size):
        deriv = f_bits ^ f_bits[idx ^ a]
        w = fwht_exact(deriv).astype(np.float64) / norm
        total_p4 += float(np.sum(w ** 4))
    return -math.log2(total_p4 / norm)


# ---------------------------------------------------------------------------
# 3. Steiner-Star (16-CZ CCCZ) & Steiner-Path (7-CZ CCZ) Physical Compiler
#    + Disjoint-Support QAC^0 Parallel Layer Depth (OpenAI #274 Section 2)
# ---------------------------------------------------------------------------

def shortest_path(g: Graph, src: int, dst: int) -> list[int]:
    if src == dst:
        return [src]
    parent = {src: -1}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in sorted(g[u]):
            if v not in parent:
                parent[v] = u
                if v == dst:
                    path = [v]
                    while parent[path[-1]] != -1:
                        path.append(parent[path[-1]])
                    return list(reversed(path))
                q.append(v)
    raise RuntimeError(f"Disconnected graph: {src} -> {dst}")


def route_cz_pairs(g: Graph, cz_pairs: Sequence[tuple[int, int]], p_2q: float = 0.005) -> dict:
    n = len(g)
    log_to_phys = list(range(n))
    phys_to_log = list(range(n))
    qubit_depth = [0] * n
    swaps = 0
    remaining = list(cz_pairs)
    while remaining:
        best_idx = 0
        best_path = None
        for idx, (la, lb) in enumerate(remaining):
            path = shortest_path(g, log_to_phys[la], log_to_phys[lb])
            if best_path is None or len(path) < len(best_path):
                best_path = path
                best_idx = idx
        la, lb = remaining.pop(best_idx)
        assert best_path is not None
        for step in range(len(best_path) - 2):
            p1, p2 = best_path[step], best_path[step + 1]
            l1, l2 = phys_to_log[p1], phys_to_log[p2]
            phys_to_log[p1], phys_to_log[p2] = l2, l1
            log_to_phys[l1], log_to_phys[l2] = p2, p1
            d = max(qubit_depth[p1], qubit_depth[p2]) + 3
            qubit_depth[p1] = d
            qubit_depth[p2] = d
            swaps += 1
        pa, pb = log_to_phys[la], log_to_phys[lb]
        d = max(qubit_depth[pa], qubit_depth[pb]) + 1
        qubit_depth[pa] = d
        qubit_depth[pb] = d
    total_cz = 3 * swaps + len(cz_pairs)
    return {
        "logical_cz": len(cz_pairs),
        "swaps": swaps,
        "total_native_cz": total_cz,
        "depth_2q": max(qubit_depth, default=0),
        "est_fidelity_p005": round((1.0 - p_2q) ** total_cz, 4),
    }


def _steiner_path_ccz_7cz_phys(p_c0: int, p_c1: int, p_hub: int) -> tuple[list[tuple], dict[int, int]]:
    ang = math.pi / 4.0
    ops = [
        ("rz", p_c0, +ang), ("rz", p_c1, +ang), ("rz", p_hub, +ang),
        ("cnot", p_c0, p_hub), ("rz", p_hub, -ang),
        ("cnot", p_c1, p_hub), ("rz", p_hub, +ang),
        ("cnot", p_c0, p_hub), ("rz", p_hub, -ang),
        ("cnot", p_hub, p_c1),
        ("cnot", p_c1, p_hub),
        ("cnot", p_c0, p_hub), ("rz", p_hub, -ang),
        ("cnot", p_c0, p_hub),
    ]
    return ops, {p_c0: p_c0, p_c1: p_hub, p_hub: p_c1}


def _steiner_star_cccz_16cz_phys(
    p_c0: int, p_c1: int, p_c2: int, p_hub: int
) -> tuple[list[tuple], dict[int, int]]:
    ang = math.pi / 8.0
    ops = [
        ("rz", p_c0, +ang), ("rz", p_c1, +ang), ("rz", p_c2, +ang), ("rz", p_hub, +ang),
        ("cnot", p_c0, p_hub), ("rz", p_hub, -ang),
        ("cnot", p_c1, p_hub), ("rz", p_hub, +ang),
        ("cnot", p_c0, p_hub), ("rz", p_hub, -ang),
        ("cnot", p_c2, p_hub), ("rz", p_hub, +ang),
        ("cnot", p_c0, p_hub), ("rz", p_hub, -ang),
        ("cnot", p_c1, p_hub), ("rz", p_hub, +ang),
        ("cnot", p_c0, p_hub), ("rz", p_hub, -ang),
        ("cnot", p_hub, p_c2),
        ("cnot", p_c2, p_hub),
        ("cnot", p_c0, p_hub), ("rz", p_hub, -ang),
        ("cnot", p_c1, p_hub), ("rz", p_hub, +ang),
        ("cnot", p_c0, p_hub), ("rz", p_hub, -ang),
        ("cnot", p_hub, p_c1),
        ("cnot", p_c1, p_hub),
        ("cnot", p_c0, p_hub), ("rz", p_hub, -ang),
        ("cnot", p_c0, p_hub),
    ]
    return ops, {p_c0: p_c0, p_c1: p_c2, p_c2: p_hub, p_hub: p_c1}


def compute_qac0_disjoint_layer_depth(n_qubits: int, gates_or_czs: Sequence[tuple[int, ...]]) -> int:
    """
    Compute the exact disjoint-support reflection/entangling depth d_QAC
    (OpenAI #274 Section 2, Eq. 2.2) of a sequence of multi-qubit gates.
    """
    qubit_layer = [0] * n_qubits
    for g in gates_or_czs:
        if len(g) <= 1:
            continue
        d = max(qubit_layer[q] for q in g) + 1
        for q in g:
            qubit_layer[q] = d
    return max(qubit_layer, default=0)


def compile_physically_routed_iqm_qasm(
    circuit: HypergraphCircuit,
    g_hw: Graph,
    qb_labels: dict[int, str],
    mirror: bool = False,
) -> tuple[str, dict]:
    n = circuit.n_qubits
    log_to_phys = list(range(n))
    phys_to_log = list(range(n))
    body_ops: list[tuple] = []
    swaps_inserted = 0
    native_cz_count = 0

    def emit_phys_cnot(p_ctrl: int, p_tgt: int) -> None:
        nonlocal native_cz_count
        assert p_tgt in g_hw[p_ctrl], f"Non-adjacent physical CZ ({p_ctrl}, {p_tgt})!"
        body_ops.append(("ry", p_tgt, math.pi / 2.0))
        body_ops.append(("cz", p_ctrl, p_tgt))
        body_ops.append(("ry", p_tgt, -math.pi / 2.0))
        native_cz_count += 1

    def emit_phys_swap(p1: int, p2: int) -> None:
        nonlocal swaps_inserted
        assert p2 in g_hw[p1], f"Non-adjacent physical SWAP ({p1}, {p2})!"
        emit_phys_cnot(p1, p2)
        emit_phys_cnot(p2, p1)
        emit_phys_cnot(p1, p2)
        l1, l2 = phys_to_log[p1], phys_to_log[p2]
        phys_to_log[p1], phys_to_log[p2] = l2, l1
        log_to_phys[l1], log_to_phys[l2] = p2, p1
        swaps_inserted += 1

    def move_logical_to_target_neighbor(l_mov: int, p_hub: int, forbidden: set[int]) -> int:
        p_cur = log_to_phys[l_mov]
        if p_cur in g_hw[p_hub]:
            return p_cur
        best_path = None
        for nbr in sorted(g_hw[p_hub]):
            if nbr in forbidden and nbr != p_cur:
                continue
            path = shortest_path(g_hw, p_cur, nbr)
            if best_path is None or len(path) < len(best_path):
                best_path = path
        if best_path is None:
            for nbr in sorted(g_hw[p_hub]):
                path = shortest_path(g_hw, p_cur, nbr)
                if best_path is None or len(path) < len(best_path):
                    best_path = path
        assert best_path is not None
        for step in range(len(best_path) - 1):
            emit_phys_swap(best_path[step], best_path[step + 1])
        assert log_to_phys[l_mov] in g_hw[p_hub]
        return log_to_phys[l_mov]

    def apply_wire_content_perm(wire_perm: dict[int, int]) -> None:
        old_p2l = {p: phys_to_log[p] for p in wire_perm}
        for p_wire, old_p in wire_perm.items():
            l_q = old_p2l[old_p]
            phys_to_log[p_wire] = l_q
            log_to_phys[l_q] = p_wire

    z_gates = [g for g in circuit.gates if len(g) == 1]
    cz_gates = [g for g in circuit.gates if len(g) == 2]
    mc_gates = [g for g in circuit.gates if len(g) in (3, 4)]

    deferred_cz: list[Gate] = []
    for g in z_gates:
        body_ops.append(("rz", log_to_phys[g[0]], math.pi))
    for g in cz_gates:
        pa, pb = log_to_phys[g[0]], log_to_phys[g[1]]
        if pb in g_hw[pa]:
            body_ops.append(("cz", pa, pb))
            native_cz_count += 1
        else:
            deferred_cz.append(g)

    deferred_partners: dict[int, int] = {}
    for a, b in deferred_cz:
        deferred_partners[a] = b
        deferred_partners[b] = a

    for g in mc_gates:
        best_hub_l = min(
            g,
            key=lambda l_cand: sum(
                max(0, len(shortest_path(g_hw, log_to_phys[l_cand], log_to_phys[l_other])) - 2)
                for l_other in g if l_other != l_cand
            ),
        )
        p_hub = log_to_phys[best_hub_l]
        leaves_l = [lq for lq in g if lq != best_hub_l]

        placed_phys = {p_hub}
        for l_leaf in leaves_l:
            p_leaf = move_logical_to_target_neighbor(l_leaf, p_hub, placed_phys)
            placed_phys.add(p_leaf)
            p_hub = log_to_phys[best_hub_l]

        if best_hub_l in deferred_partners:
            p_partner = log_to_phys[deferred_partners[best_hub_l]]
            leaves_l.sort(
                key=lambda l_leaf: len(shortest_path(g_hw, log_to_phys[l_leaf], p_partner)),
                reverse=True,
            )

        p_leaves = [log_to_phys[lq] for lq in leaves_l]
        p_hub = log_to_phys[best_hub_l]

        if len(g) == 3:
            ops, wperm = _steiner_path_ccz_7cz_phys(p_leaves[0], p_leaves[1], p_hub)
        else:
            ops, wperm = _steiner_star_cccz_16cz_phys(p_leaves[0], p_leaves[1], p_leaves[2], p_hub)

        for op in ops:
            if op[0] == "rz":
                body_ops.append(op)
            elif op[0] == "cnot":
                emit_phys_cnot(op[1], op[2])
        apply_wire_content_perm(wperm)

    for g in deferred_cz:
        path = shortest_path(g_hw, log_to_phys[g[0]], log_to_phys[g[1]])
        for step in range(len(path) - 2):
            emit_phys_swap(path[step], path[step + 1])
        pa, pb = log_to_phys[g[0]], log_to_phys[g[1]]
        assert pb in g_hw[pa]
        body_ops.append(("cz", pa, pb))
        native_cz_count += 1

    tag = "MIRROR (U_f^dag U_f -> |0^20>)" if mirror else "X-BASIS FLATNESS (H U_f H |0^20>)"
    qb_map_comment = ", ".join(f"q[{i}]={qb_labels[i]}" for i in range(n))
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"// IQM Garnet (Crystal-20) Physically Routed v4 | {circuit.name} | {tag}",
        f"// Physical Qubit Map: {qb_map_comment}",
        f"// Steiner-Star/Path Synthesis: {swaps_inserted} SWAPs, {native_cz_count * (2 if mirror else 1)} native CZs (100% adjacent)",
        f"qubit[{n}] q;",
        f"bit[{n}] c;",
        "",
        "// Prepare |+>^{otimes 20}",
    ]
    for p in range(n):
        lines.append(f"ry(1.570796326795) q[{p}]; // {qb_labels[p]}")
    lines.append("")
    lines.append("// Physically routed U_f on IQM Garnet couplers")

    def fmt_op(op: tuple) -> str:
        if op[0] == "ry":
            return f"ry({op[2]:.12f}) q[{op[1]}];"
        if op[0] == "rz":
            return f"rz({op[2]:.12f}) q[{op[1]}];"
        if op[0] == "cz":
            assert op[2] in g_hw[op[1]]
            return f"cz q[{op[1]}], q[{op[2]}]; // {qb_labels[op[1]]}-{qb_labels[op[2]]}"
        raise ValueError(op)

    for op in body_ops:
        lines.append(fmt_op(op))

    phys_cz_seq = [(op[1], op[2]) for op in body_ops if op[0] == "cz"]
    total_cz = native_cz_count
    total_swaps = swaps_inserted

    if mirror:
        lines.append("")
        lines.append("// Mirror inverse U_f^dag (returns |0^20>)")
        for op in reversed(body_ops):
            if op[0] in ("ry", "rz"):
                lines.append(fmt_op((op[0], op[1], -op[2])))
            else:
                lines.append(fmt_op(op))
        total_cz *= 2
        total_swaps *= 2
        phys_cz_seq = phys_cz_seq + list(reversed(phys_cz_seq))

    lines.append("")
    lines.append("// Final Hadamard layer and logical-mapped measurement")
    for p in range(n):
        lines.append(f"ry(-1.570796326795) q[{p}];")
    if mirror:
        for p in range(n):
            lines.append(f"c[{p}] = measure q[{p}]; // {qb_labels[p]}")
    else:
        for l_idx in range(n):
            p_wire = log_to_phys[l_idx]
            lines.append(f"c[{l_idx}] = measure q[{p_wire}]; // logical {l_idx} -> physical {qb_labels[p_wire]}")

    stats = {
        "mirror": mirror,
        "physical_swaps": total_swaps,
        "physical_native_cz": total_cz,
        "qac0_logical_reflection_depth": compute_qac0_disjoint_layer_depth(n, circuit.gates),
        "qac0_physical_cz_layer_depth": compute_qac0_disjoint_layer_depth(n, phys_cz_seq),
        "all_cz_physically_adjacent": True,
        "final_logical_to_QB": {f"L{i}": qb_labels[log_to_phys[i]] for i in range(n)},
        "est_fidelity_p005": round((1.0 - 0.005) ** total_cz, 4),
    }
    return "\n".join(lines) + "\n", stats


def verify_routed_qasm_exact_20q(qasm_text: str, circuit: HypergraphCircuit, g_hw: Graph) -> bool:
    n = circuit.n_qubits
    raw_lines = [ln.split("//")[0].strip() for ln in qasm_text.splitlines()]
    lines = [ln for ln in raw_lines if ln]

    start_idx = 0
    for idx, ln in enumerate(lines):
        if ln.startswith("ry(1.570796326795) q[19];"):
            start_idx = idx + 1
            break
    end_idx = len(lines)
    for idx in range(start_idx, len(lines)):
        if (
            lines[idx].startswith("ry(-1.570796326795) q[0];")
            and idx + 19 < len(lines)
            and lines[idx + 19].startswith("ry(-1.570796326795) q[19];")
        ):
            end_idx = idx
            break

    state = [1 << i for i in range(n)]
    rz_terms: dict[int, float] = {}
    cz_terms: list[tuple[int, int]] = []

    i = start_idx
    while i < end_idx:
        ln = lines[i]
        if (
            ln.startswith("ry(1.570796326795)")
            and i + 2 < end_idx
            and lines[i + 1].startswith("cz ")
            and lines[i + 2].startswith("ry(-1.570796326795)")
        ):
            t1 = int(ln.split("q[")[1].split("]")[0])
            cz_parts = lines[i + 1].replace("cz ", "").replace(";", "").split(",")
            p_c = int(cz_parts[0].strip().split("[")[1].split("]")[0])
            p_t = int(cz_parts[1].strip().split("[")[1].split("]")[0])
            t2 = int(lines[i + 2].split("q[")[1].split("]")[0])
            if t1 == p_t == t2:
                assert p_t in g_hw[p_c]
                state[p_t] ^= state[p_c]
                i += 3
                continue
        if ln.startswith("rz("):
            ang = float(ln.split("(")[1].split(")")[0])
            p_q = int(ln.split("q[")[1].split("]")[0])
            m = state[p_q]
            rz_terms[m] = rz_terms.get(m, 0.0) + ang
            i += 1
        elif ln.startswith("cz "):
            cz_parts = ln.replace("cz ", "").replace(";", "").split(",")
            p_a = int(cz_parts[0].strip().split("[")[1].split("]")[0])
            p_b = int(cz_parts[1].strip().split("[")[1].split("]")[0])
            assert p_b in g_hw[p_a]
            cz_terms.append((state[p_a], state[p_b]))
            i += 1
        else:
            raise RuntimeError(f"Unexpected QASM line: {ln}")

    meas_lines = [ln for ln in lines[end_idx:] if "measure" in ln]
    assert len(meas_lines) == n
    for ml in meas_lines:
        l_idx = int(ml.split("c[")[1].split("]")[0])
        p_idx = int(ml.split("q[")[1].split("]")[0])
        assert state[p_idx] == (1 << l_idx)

    size = 1 << n
    idx_arr = np.arange(size, dtype=np.uint32)
    phases = np.zeros(size, dtype=np.float64)

    def eval_parity(mask: int) -> np.ndarray:
        v = idx_arr & np.uint32(mask)
        v ^= v >> np.uint32(16)
        v ^= v >> np.uint32(8)
        v ^= v >> np.uint32(4)
        v ^= v >> np.uint32(2)
        v ^= v >> np.uint32(1)
        return (v & np.uint32(1)).astype(np.float64)

    for m, ang in rz_terms.items():
        if abs(ang) > 1e-12:
            phases += ang * eval_parity(m)
    for ma, mb in cz_terms:
        phases += math.pi * (eval_parity(ma) * eval_parity(mb))

    f_target = circuit.truth_table()
    err = float(np.max(np.abs(np.exp(1j * phases) - np.where(f_target == 0, 1.0, -1.0))))
    assert err < 1e-9
    return True


# ---------------------------------------------------------------------------
# 4. OpenAI #274 Antipodal Felinity & Self-Cancelling Readout-Bias Protocol
# ---------------------------------------------------------------------------

def exact_openai274_felinity_and_l2(w_exact: np.ndarray, n_qubits: int) -> dict:
    """
    Compute exact ideal X-basis second moments:
      - L2 collision probability I_self = sum_y p_y^2 (flat => 2^{-n})
      - OpenAI #274 Corollary 8 Antipodal Felinity F_n(rho) = 2 * sum_y p_y * p_{~y}
        where ~y = y XOR (2^n - 1) (flat => 2 * 2^{-n}, so I_ant = F_n/2 = 2^{-n})
      - High-Hamming-weight Walsh-Fourier tail W_{>= n/2} = sum_{|y| >= n/2} p_y
        (OpenAI #274 Corollary 9 comparison!)
    """
    dim = 1 << n_qubits
    p_ideal = (w_exact.astype(np.float64) / float(dim)) ** 2
    p_ideal /= p_ideal.sum()

    i_self = float(np.sum(p_ideal ** 2))
    # Bitwise complement ~y reverses the array p_ideal[::-1]!
    i_ant = float(np.sum(p_ideal * p_ideal[::-1]))
    felinity_fn = 2.0 * i_ant

    # High-weight Fourier tail |y| >= n/2
    idx = np.arange(dim, dtype=np.uint32)
    # Popcount of idx
    v = idx - ((idx >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    pop = (((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
    tail_half = float(np.sum(p_ideal[pop >= (n_qubits // 2)]))

    return {
        "i_self_times_2n": round(i_self * dim, 4),
        "i_ant_times_2n": round(i_ant * dim, 4),
        "openai274_felinity_times_2n": round(felinity_fn * dim, 4),
        "fourier_tail_weight_ge_half": round(tail_half, 4),
    }


def realistic_nisq_triple_protocol_sim(
    w_exact: np.ndarray,
    n_qubits: int,
    physical_cz_forward: int,
    p_2q: float,
    shots: int,
    n_trials: int,
    rng: np.random.Generator,
    p_ro_0given1: float = 0.030,  # Stress-test strong asymmetric readout (3.0% vs 0.5%)!
    p_ro_1given0: float = 0.005,
    coherent_overrot_rad: float = 0.015,
) -> dict:
    """
    Simulates the Triple Protocol (Self-Collisions + OpenAI #274 Antipodal Felinity + Mirror)
    under STRONG asymmetric readout bias (p(0|1)=3.0%, p(1|0)=0.5%) so we can observe:
      1. C_self alone drifts above ideal 128.0 due to +4*n*delta^2 readout bias.
      2. C_ant (OpenAI #274 Felinity) drifts below ideal 128.0 by -4*n*delta^2.
      3. C_combined = (C_self + C_ant) / 2 AUTOMATICALLY SELF-CANCELS the readout-bias
         drift back to 128.0 WITHOUT needing an extra readout calibration circuit!
    """
    dim = 1 << n_qubits
    mask_all = np.uint32(dim - 1)
    p_ideal = (w_exact.astype(np.float64) / float(dim)) ** 2
    p_ideal /= p_ideal.sum()

    p_coh = math.sin(coherent_overrot_rad / 2.0) ** 2
    p_eff_cz = min(0.5, p_2q + p_coh)
    f_forward = (1.0 - p_eff_cz) ** physical_cz_forward
    p_pre_ro = f_forward * p_ideal + (1.0 - f_forward) * (1.0 / dim)

    cdf = np.cumsum(p_pre_ro)
    cdf[-1] = 1.0
    pairs = shots * (shots - 1) / 2.0
    mu_flat_ideal = pairs / float(dim)  # 128.0 for S=16384, n=20
    sigma_single = math.sqrt(mu_flat_ideal)
    # For the average (C_self + C_ant)/2 of two independent Poisson(128) counts,
    # Var((C_self + C_ant)/2) = (128 + 128)/4 = 64 => sigma_comb = sqrt(64) = 8.0!
    sigma_combined = math.sqrt(mu_flat_ideal / 2.0)

    c_self_list = []
    c_ant_list = []
    c_comb_list = []
    z_uncal_self = []
    z_uncal_comb = []

    for _ in range(n_trials):
        u = rng.random(shots)
        samples = np.searchsorted(cdf, u).astype(np.uint32)
        noisy_samples = samples.copy()
        for b in range(n_qubits):
            bit_mask = np.uint32(1 << b)
            is_one = (samples & bit_mask) != 0
            r = rng.random(shots)
            flip_1_to_0 = is_one & (r < p_ro_0given1)
            flip_0_to_1 = (~is_one) & (r < p_ro_1given0)
            noisy_samples ^= np.where(flip_1_to_0 | flip_0_to_1, bit_mask, np.uint32(0))

        vals, counts = np.unique(noisy_samples, return_counts=True)
        c_self = int(np.sum(counts * (counts - 1) // 2))

        # Fast exact Antipodal Collision count C_ant = (1/2) * sum_y c_y * c_{~y}
        # Find where ~vals intersects vals using sorted searchsorted!
        comp_vals = vals ^ mask_all
        pos = np.searchsorted(vals, comp_vals)
        valid = (pos < len(vals)) & (vals[np.minimum(pos, len(vals) - 1)] == comp_vals)
        c_ant = int(np.sum(counts[valid] * counts[pos[valid]])) // 2

        c_comb = 0.5 * (c_self + c_ant)
        c_self_list.append(c_self)
        c_ant_list.append(c_ant)
        c_comb_list.append(c_comb)
        z_uncal_self.append((c_self - mu_flat_ideal) / sigma_single)
        z_uncal_comb.append((c_comb - mu_flat_ideal) / sigma_combined)

    p_mirror_0 = ((1.0 - p_eff_cz) ** (2 * physical_cz_forward)) * ((1.0 - p_ro_1given0) ** n_qubits)
    p_white_noise = 1.0 / float(dim)

    return {
        "shots": shots,
        "p_2q": p_2q,
        "p_ro_0given1": p_ro_0given1,
        "p_ro_1given0": p_ro_1given0,
        "expected_ideal_128": round(mu_flat_ideal, 2),
        "mean_c_self": round(float(np.mean(c_self_list)), 2),
        "mean_c_ant_felinity": round(float(np.mean(c_ant_list)), 2),
        "mean_c_combined_self_cancelling": round(float(np.mean(c_comb_list)), 2),
        "mean_z_uncalibrated_self": round(float(np.mean(z_uncal_self)), 2),
        "mean_z_uncalibrated_combined": round(float(np.mean(z_uncal_comb)), 2),
        "detect_5sigma_combined_no_cal": round(float(np.mean([abs(z) > 5.0 for z in z_uncal_comb])), 3),
        "mirror_survival_prob_0n": round(p_mirror_0, 4),
        "mirror_snr_vs_whitenoise": round(p_mirror_0 / p_white_noise, 1),
    }


def verify_halfspace_collapse_h20_4() -> dict:
    n, m = 20, 10
    seen: set[tuple[int, int, int, int]] = set()
    orbits: list[list[tuple[int, int, int, int]]] = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    t = (a, b, c, d)
                    if t in seen:
                        continue
                    orb = []
                    orb_set = set()
                    for s in range(n):
                        sh = tuple(sorted(((a + s) % n, (b + s) % n, (c + s) % n, (d + s) % n)))
                        if sh not in orb_set:
                            orb_set.add(sh)
                            orb.append(sh)
                            seen.add(sh)
                    orbits.append(orb)

    full_orbits = sum(1 for o in orbits if len(o) == n)
    short_orbits = len(orbits) - full_orbits
    max_deg_on_v10 = 0
    surviving_on_v10 = 0
    for orb in orbits:
        monomial_parity: dict[tuple[int, ...], int] = {}
        for t in orb:
            reduced = tuple(sorted({q % m for q in t}))
            monomial_parity[reduced] = monomial_parity.get(reduced, 0) ^ 1
        surviving = [mon for mon, par in monomial_parity.items() if par == 1]
        if surviving:
            surviving_on_v10 += 1
            max_deg_on_v10 = max(max_deg_on_v10, max(len(mon) for mon in surviving))
    assert max_deg_on_v10 <= 2
    return {
        "n_qubits": n,
        "degree": 4,
        "total_quartic_rotation_orbits": len(orbits),
        "full_length_20_orbits": full_orbits,
        "short_orbits": short_orbits,
        "orbits_surviving_on_V10": surviving_on_v10,
        "max_degree_on_V10": max_deg_on_v10,
        "clifford_collapse_verified": True,
    }
