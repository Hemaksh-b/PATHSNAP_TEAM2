"""Road-network-aware HMM transition scoring (Issue #8).

Transition score between candidate i at observation t and candidate j at t'
(Newson & Krumm, 2009)::

    logA(i -> j) = -beta * | d_route(p_i, p_j) - d_gps(o_t, o_t') |

``d_route`` is the driving distance along the directed road graph between the
two candidates' projection points; ``d_gps`` is the straight-line distance
between the two GPS observations. Both are in **metres** (projected coords).

A transition is impossible (``min_logp``) when the target cannot be reached
within ``max(min_cutoff_m, max_speed_mps * dt)`` metres -- this covers
disconnected components and physically implausible jumps. Shortest paths use
bounded Dijkstra and are cached per (source node, cutoff).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import networkx as nx
import numpy as np

TransitionFn = Callable[[int, int], np.ndarray]


@dataclass(frozen=True)
class TransitionConfig:
    """Tune only on training / student-test trips."""

    beta: float = 0.1  # 1/metres; larger = stricter agreement with GPS displacement
    max_speed_mps: float = 50.0  # 180 km/h upper bound on implied speed
    min_cutoff_m: float = 200.0  # search radius floor (GPS noise at low dt)
    backtrack_tolerance_m: float = 15.0  # allowed backward jitter on the same arc
    min_logp: float = -1e4


class RoadGraph:
    """Directed road graph keyed by original node ids, with arc lookup arrays."""

    def __init__(self, node_xy: np.ndarray, arcs_uv: np.ndarray) -> None:
        node_xy = np.asarray(node_xy, dtype=float)
        arcs_uv = np.asarray(arcs_uv, dtype=int)
        if node_xy.ndim != 2 or node_xy.shape[1] != 2:
            raise ValueError("node_xy must have shape (N, 2) in metres")
        if arcs_uv.ndim != 2 or arcs_uv.shape[1] != 2:
            raise ValueError("arcs_uv must have shape (M, 2)")
        if arcs_uv.size and (arcs_uv.min() < 0 or arcs_uv.max() >= len(node_xy)):
            raise ValueError("arc endpoints must be valid node indices")

        self.node_xy = node_xy
        self.arc_u = arcs_uv[:, 0]
        self.arc_v = arcs_uv[:, 1]
        self.arc_len = np.linalg.norm(node_xy[self.arc_v] - node_xy[self.arc_u], axis=1)

        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(node_xy)))
        for arc_id, (u, v) in enumerate(arcs_uv):
            length = float(self.arc_len[arc_id])
            # parallel arcs between the same nodes: keep the shortest for routing
            if not graph.has_edge(u, v) or graph[u][v]["length"] > length:
                graph.add_edge(int(u), int(v), length=length, arc_id=arc_id)
        self.graph = graph
        self._sp_cache: dict[tuple[int, float], dict[int, float]] = {}

    def shortest_from(self, source: int, cutoff: float) -> dict[int, float]:
        """Node -> distance (m) for all nodes reachable from ``source`` within ``cutoff``."""
        key = (int(source), float(cutoff))
        hit = self._sp_cache.get(key)
        if hit is None:
            hit = nx.single_source_dijkstra_path_length(
                self.graph, int(source), cutoff=cutoff, weight="length"
            )
            self._sp_cache[key] = hit
        return hit

    def route_distance(
        self, arc_a: int, frac_a: float, arc_b: int, frac_b: float, cutoff: float, tol: float
    ) -> float:
        """Driving distance (m) from a point on ``arc_a`` to a point on ``arc_b``.

        ``frac_*`` is the position along the arc in [0, 1]. Returns ``inf`` when
        unreachable within ``cutoff`` metres of graph search.
        """
        len_a, len_b = float(self.arc_len[arc_a]), float(self.arc_len[arc_b])
        if arc_a == arc_b:
            delta = (frac_b - frac_a) * len_a
            if delta >= -tol:  # forward, or small backward GPS jitter
                return abs(delta)
        sp = self.shortest_from(int(self.arc_v[arc_a]), cutoff).get(int(self.arc_u[arc_b]))
        if sp is None:
            return math.inf
        return (1.0 - frac_a) * len_a + sp + frac_b * len_b


def make_transition_fn(
    road: RoadGraph,
    cand_arcs: Sequence[Sequence[int]],
    cand_frac: Sequence[Sequence[float]],
    obs_xy: np.ndarray,
    timestamps: np.ndarray,
    config: TransitionConfig = TransitionConfig(),
) -> TransitionFn:
    """Build ``transition_logp(t_prev, t_next)`` for :func:`hmm_viterbi.decode`.

    Args:
        road: road graph in projected metres.
        cand_arcs: candidate arc_ids per observation (same as passed to decode).
        cand_frac: along-arc fraction of each candidate's projection point.
        obs_xy: (T, 2) projected GPS positions in metres.
        timestamps: (T,) seconds.
    """
    obs_xy = np.asarray(obs_xy, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    def transition_logp(t_prev: int, t_next: int) -> np.ndarray:
        dt = max(float(timestamps[t_next] - timestamps[t_prev]), 1.0)
        cutoff = max(config.min_cutoff_m, config.max_speed_mps * dt)
        d_gps = float(np.linalg.norm(obs_xy[t_next] - obs_xy[t_prev]))
        prev_arcs, next_arcs = cand_arcs[t_prev], cand_arcs[t_next]
        out = np.full((len(prev_arcs), len(next_arcs)), config.min_logp)
        for i, (a, fa) in enumerate(zip(prev_arcs, cand_frac[t_prev])):
            for j, (b, fb) in enumerate(zip(next_arcs, cand_frac[t_next])):
                d_route = road.route_distance(
                    a, fa, b, fb, cutoff, config.backtrack_tolerance_m
                )
                if math.isfinite(d_route):
                    out[i, j] = max(-config.beta * abs(d_route - d_gps), config.min_logp)
        return out

    return transition_logp
