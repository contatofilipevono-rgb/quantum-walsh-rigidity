"""
Pillar 3: Hardware topology SWAP router for the mandatory antipodal CZ orbit
O_ant = {(i, i+H) : 0 <= i < H} across:
  1. 1D Ring (C_n)
  2. 2D Square Grid (ceil(sqrt(n)) x ceil(sqrt(n)))
  3. IBM Heavy-Hex lattice subgraph of size n
"""

from __future__ import annotations

from collections import deque
import math


Graph = dict[int, set[int]]


def build_1d_ring(n: int) -> Graph:
    g: Graph = {i: set() for i in range(n)}
    for i in range(n):
        j = (i + 1) % n
        g[i].add(j)
        g[j].add(i)
    return g


def build_2d_grid(n: int) -> Graph:
    cols = math.ceil(math.sqrt(n))
    g: Graph = {i: set() for i in range(n)}
    for i in range(n):
        r, c = divmod(i, cols)
        if c + 1 < cols and i + 1 < n:
            g[i].add(i + 1)
            g[i + 1].add(i)
        if i + cols < n:
            g[i].add(i + cols)
            g[i + cols].add(i)
    return g


def build_heavy_hex(n: int) -> Graph:
    """
    Build a connected heavy-hex-like degree-<=3 lattice subgraph of n qubits
    (alternating degree-2 bridge qubits and degree-3 vertices, matching IBM Eagle/Heron).
    """
    # Construct a 2D brickwall/hexagonal grid with edge-subdivision (degree <= 3)
    # and take a BFS-connected subgraph of exact size n.
    coords: list[tuple[int, int]] = []
    edges_raw: list[tuple[tuple[int, int], tuple[int, int]]] = []
    # Base brickwall of size 6x6 with subdivided horizontal/vertical bridges
    for r in range(6):
        for c in range(10):
            u = (2 * r, 2 * c)
            coords.append(u)
            if c + 1 < 10:
                mid = (2 * r, 2 * c + 1)
                v = (2 * r, 2 * c + 2)
                coords.append(mid)
                edges_raw.append((u, mid))
                edges_raw.append((mid, v))
            if r + 1 < 6 and ((r + c) % 2 == 0):
                mid = (2 * r + 1, 2 * c)
                v = (2 * r + 2, 2 * c)
                coords.append(mid)
                edges_raw.append((u, mid))
                edges_raw.append((mid, v))

    adj_coord: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for a, b in edges_raw:
        adj_coord.setdefault(a, set()).add(b)
        adj_coord.setdefault(b, set()).add(a)

    # BFS from (0, 0) to pick exactly n connected qubits
    order: list[tuple[int, int]] = []
    seen = {(0, 0)}
    q = deque([(0, 0)])
    while q and len(order) < n:
        cur = q.popleft()
        order.append(cur)
        for nxt in sorted(adj_coord.get(cur, ())):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)

    idx_map = {pt: i for i, pt in enumerate(order)}
    g: Graph = {i: set() for i in range(n)}
    for pt, i in idx_map.items():
        for nxt in adj_coord.get(pt, ()):
            if nxt in idx_map:
                j = idx_map[nxt]
                g[i].add(j)
                g[j].add(i)
    return g


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
    raise RuntimeError(f"Disconnected graph: no path {src} -> {dst}")


def route_cz_layer(g: Graph, cz_pairs: list[tuple[int, int]], p_2q_error: float = 1e-3) -> dict:
    """
    Greedy shortest-path SWAP router for a layer of CZ gates on hardware graph `g`.
    Tracks logical->physical qubit mapping, exact SWAP count, total 2q gates
    (3*SWAPs + len(cz_pairs)), 2q circuit depth, and estimated fidelity.
    """
    n = len(g)
    log_to_phys = list(range(n))
    phys_to_log = list(range(n))
    qubit_depth = [0] * n
    swaps = 0
    remaining = list(cz_pairs)

    while remaining:
        # Pick pair with shortest current physical distance
        best_idx = 0
        best_path = None
        for idx, (la, lb) in enumerate(remaining):
            path = shortest_path(g, log_to_phys[la], log_to_phys[lb])
            if best_path is None or len(path) < len(best_path):
                best_path = path
                best_idx = idx

        la, lb = remaining.pop(best_idx)
        assert best_path is not None
        # Route la along best_path[:-1] until adjacent to lb
        for step in range(len(best_path) - 2):
            p1, p2 = best_path[step], best_path[step + 1]
            l1, l2 = phys_to_log[p1], phys_to_log[p2]
            phys_to_log[p1], phys_to_log[p2] = l2, l1
            log_to_phys[l1], log_to_phys[l2] = p2, p1
            d = max(qubit_depth[p1], qubit_depth[p2]) + 3  # 1 SWAP = 3 CNOT layers
            qubit_depth[p1] = d
            qubit_depth[p2] = d
            swaps += 1

        # Apply CZ on adjacent physical qubits
        pa, pb = log_to_phys[la], log_to_phys[lb]
        d = max(qubit_depth[pa], qubit_depth[pb]) + 1
        qubit_depth[pa] = d
        qubit_depth[pb] = d

    total_2q = 3 * swaps + len(cz_pairs)
    est_fidelity = (1.0 - p_2q_error) ** total_2q
    return {
        "swaps": swaps,
        "total_2q_gates": total_2q,
        "depth_2q": max(qubit_depth, default=0),
        "est_fidelity_1e3": round(est_fidelity, 4),
    }
