"""Independent-cascade (IC) influence simulation and classical centrality baselines.

IC model (Kempe, Kleinberg & Tardos, KDD 2003): the seed set is active at t=0; each newly active
node u gets ONE chance to activate each inactive out-neighbour v, succeeding independently with
probability p.  Edge direction: the symmetrised graph is treated as two directed arcs per
undirected edge, each tested independently.  ``spread`` = expected number of active nodes at the end
(seeds included).

Randomness is *counter based* (SplitMix64 hash of (seed, cascade id, arc id)), so results do not
depend on the number of threads and are exactly reproducible.  This is a SIMULATION PROXY for
influence; it is never observed real-world influence.
"""
from __future__ import annotations

import numpy as np
import numba as nb
import scipy.sparse as sp

_U64 = np.uint64


@nb.njit(inline="always")
def _mix(z):
    z = z + _U64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
    return z ^ (z >> _U64(31))


@nb.njit(inline="always")
def _uniform(seed, cid, arc):
    z = _mix(_U64(seed) ^ _mix(_U64(cid) + _U64(0x1234567)) ^ _mix(_U64(arc) * _U64(0x2545F4914F6CDD1D)))
    return (z >> _U64(11)) * (1.0 / 9007199254740992.0)


@nb.njit
def _cascade(indptr, indices, p, seeds, cid, rseed, stamp, stampval, queue):
    """One IC cascade; returns number of active nodes. ``stamp`` marks activation (no clearing needed)."""
    head = 0
    tail = 0
    for s in seeds:
        if stamp[s] != stampval:
            stamp[s] = stampval
            queue[tail] = s
            tail += 1
    while head < tail:
        u = queue[head]
        head += 1
        for a in range(indptr[u], indptr[u + 1]):
            v = indices[a]
            if stamp[v] != stampval:
                if _uniform(rseed, cid, a) < p:
                    stamp[v] = stampval
                    queue[tail] = v
                    tail += 1
    return tail


@nb.njit(parallel=True)
def _single_spread(indptr, indices, p, nodes, reps, rseed):
    n = indptr.shape[0] - 1
    m = nodes.shape[0]
    mean = np.zeros(m)
    sd = np.zeros(m)
    for k in nb.prange(m):
        stamp = np.zeros(n, np.int32)
        queue = np.empty(n, np.int32)
        seeds = np.empty(1, np.int64)
        seeds[0] = nodes[k]
        s1 = 0.0
        s2 = 0.0
        for r in range(reps):
            cid = np.int64(nodes[k]) * 1000003 + r
            c = _cascade(indptr, indices, p, seeds, cid, rseed, stamp, r + 1, queue)
            s1 += c
            s2 += c * c
        mu = s1 / reps
        mean[k] = mu
        var = (s2 / reps - mu * mu) * reps / max(reps - 1, 1)
        sd[k] = np.sqrt(max(var, 0.0))
    return mean, sd


@nb.njit(parallel=True)
def _set_spread_chunks(indptr, indices, p, seedset, reps, rseed, nchunk):
    n = indptr.shape[0] - 1
    out = np.zeros(reps)
    per = (reps + nchunk - 1) // nchunk
    for ch in nb.prange(nchunk):
        stamp = np.zeros(n, np.int32)
        queue = np.empty(n, np.int32)
        lo = ch * per
        hi = min(reps, lo + per)
        for r in range(lo, hi):
            out[r] = _cascade(indptr, indices, p, seedset, r, rseed, stamp, r - lo + 1, queue)
    return out


def to_csr(edge_index: np.ndarray, n: int):
    a = sp.csr_matrix((np.ones(edge_index.shape[1], dtype=np.int8), (edge_index[0], edge_index[1])),
                      shape=(n, n))
    a.sort_indices()
    return a.indptr.astype(np.int64), a.indices.astype(np.int64)


def single_node_spread(edge_index, n, p, nodes, reps, seed):
    """Monte-Carlo expected spread of each node as a lone seed. Returns (mean, std-of-cascade-size)."""
    indptr, indices = to_csr(edge_index, n)
    return _single_spread(indptr, indices, float(p), np.asarray(nodes, np.int64), int(reps), int(seed))


def seed_set_spread(edge_index, n, p, seed_set, reps, seed):
    """Spread of a *set* of seeds. Returns (mean, standard error, per-cascade sizes)."""
    indptr, indices = to_csr(edge_index, n)
    sizes = _set_spread_chunks(indptr, indices, float(p), np.asarray(seed_set, np.int64), int(reps),
                               int(seed), 8)
    return float(sizes.mean()), float(sizes.std(ddof=1) / np.sqrt(len(sizes))), sizes


def simulate_targets(edge_index, n, p_grid, splits, reps, seed_base):
    """Spread targets for every node, using DIFFERENT simulations for train / val / test nodes.

    reps = {"train": R_tr, "val": R_va, "test": R_te}; stream seeds are derived from ``seed_base`` and the
    split name, so training targets and held-out evaluation targets come from independent random
    streams.  Returns dict p -> {"mean": [n], "se": [n]}.
    """
    out = {}
    for pi, p in enumerate(p_grid):
        mean = np.zeros(n)
        se = np.zeros(n)
        for si, name in enumerate(("train", "val", "test")):
            idx = np.where(splits[name])[0]
            if len(idx) == 0:
                continue
            R = reps[name]
            m, sd = single_node_spread(edge_index, n, p, idx, R, seed_base * 100 + si * 10 + pi)
            mean[idx] = m
            se[idx] = sd / np.sqrt(R)
        out[float(p)] = {"mean": mean, "se": se}
    return out


# ----------------------------------------------------------------------------------------------
# Classical centrality baselines
# ----------------------------------------------------------------------------------------------
def degree_centrality(edge_index, n):
    return np.bincount(edge_index[0], minlength=n).astype(float)


def pagerank(edge_index, n, alpha=0.85, iters=100, tol=1e-10):
    a = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n))
    deg = np.asarray(a.sum(1)).ravel()
    inv = 1.0 / np.maximum(deg, 1)
    pr = np.full(n, 1.0 / n)
    for _ in range(iters):
        new = (1 - alpha) / n + alpha * (a.T @ (pr * inv) + pr[deg == 0].sum() / n)
        if np.abs(new - pr).sum() < tol:
            pr = new
            break
        pr = new
    return pr


def betweenness(edge_index, n, k_pivots=None, seed=0):
    """Betweenness centrality via networkx.  Exact when ``k_pivots`` is None or >= n; otherwise
    the sampled-pivot approximation of Brandes & Pich (2007) with ``k_pivots`` random sources."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    k = None if (k_pivots is None or k_pivots >= n) else int(k_pivots)
    bc = nx.betweenness_centrality(g, k=k, seed=seed, normalized=True)
    return np.array([bc[i] for i in range(n)])


def uniform_received_attention(edge_index, n):
    """Non-learned control for received attention: every target splits its mass uniformly over
    incoming neighbours + self loop, so a source j receives sum_i 1/(deg_i+1) over its targets i."""
    deg = np.bincount(edge_index[1], minlength=n).astype(float)
    w = 1.0 / (deg[edge_index[1]] + 1.0)
    r = np.zeros(n)
    np.add.at(r, edge_index[0], w)
    return r
