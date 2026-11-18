"""Synthetic-graph tests for road-network transition scoring (Issue #8).

Run from the repo root:  python -m pytest hmm_viterbi
"""
import math

import numpy as np
import pytest

from hmm_viterbi import ViterbiConfig, decode, gaussian_emission_logp
from hmm_viterbi.transitions import RoadGraph, TransitionConfig, make_transition_fn

CFG = TransitionConfig()
MIN = CFG.min_logp


def chain_graph():
    """One-way road 0 -> 1 -> 2 -> 3 along the x axis, 100 m per arc."""
    nodes = np.array([[0, 0], [100, 0], [200, 0], [300, 0]], dtype=float)
    arcs = np.array([[0, 1], [1, 2], [2, 3]])
    return RoadGraph(nodes, arcs)


def two_obs_fn(road, prev, nxt, obs_xy, dt=1.0, cfg=CFG):
    """prev/nxt: lists of (arc_id, frac) candidates for two observations."""
    return make_transition_fn(
        road,
        cand_arcs=[[a for a, _ in prev], [a for a, _ in nxt]],
        cand_frac=[[f for _, f in prev], [f for _, f in nxt]],
        obs_xy=np.asarray(obs_xy, dtype=float),
        timestamps=np.array([0.0, dt]),
        config=cfg,
    )(0, 1)


# ---------------------------------------------------------------- graph

def test_arc_lengths_are_metres_from_projected_coordinates():
    road = chain_graph()
    assert road.arc_len.tolist() == [100.0, 100.0, 100.0]
    assert road.route_distance(0, 0.5, 1, 0.5, cutoff=1000, tol=15) == pytest.approx(100.0)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        RoadGraph(np.zeros((3, 3)), np.array([[0, 1]]))
    with pytest.raises(ValueError):
        RoadGraph(np.zeros((2, 2)), np.array([[0, 5]]))


def test_node_and_arc_ids_preserved():
    road = chain_graph()
    assert road.graph[1][2]["arc_id"] == 1
    assert (road.arc_u[2], road.arc_v[2]) == (2, 3)


# ---------------------------------------------------------------- scoring

def test_adjacent_beats_skip_beats_unreachable():
    road = chain_graph()
    # GPS moved 100 m; candidates at t+1: next arc (100 m route), skip arc (200 m),
    # previous arc reversed (unreachable on a one-way road).
    log_a = two_obs_fn(road, [(0, 0.5)], [(1, 0.5), (2, 0.5), (0, 0.0)],
                       obs_xy=[[50, 0], [150, 0]], dt=10.0)
    adjacent, skip, unreachable = log_a[0]
    assert adjacent == pytest.approx(0.0)
    assert skip == pytest.approx(-CFG.beta * 100)
    assert unreachable == MIN
    assert adjacent > skip > unreachable


def test_same_arc_forward_and_backward_on_one_way_road():
    road = chain_graph()
    fwd = road.route_distance(0, 0.2, 0, 0.8, cutoff=1000, tol=15)
    jitter = road.route_distance(0, 0.5, 0, 0.45, cutoff=1000, tol=15)
    backward = road.route_distance(0, 0.8, 0, 0.2, cutoff=1000, tol=15)
    assert fwd == pytest.approx(60.0)
    assert jitter == pytest.approx(5.0)  # within tolerance: GPS noise, not a U-turn
    assert math.isinf(backward)  # one-way: would need to route around, impossible here


def test_disconnected_components_return_min_logp_without_error():
    nodes = np.array([[0, 0], [100, 0], [0, 500], [100, 500]], dtype=float)
    road = RoadGraph(nodes, np.array([[0, 1], [2, 3]]))
    log_a = two_obs_fn(road, [(0, 0.5)], [(1, 0.5)], obs_xy=[[50, 0], [50, 500]], dt=100.0)
    assert log_a.tolist() == [[MIN]]


def test_implausible_speed_is_impossible():
    road = chain_graph()
    # search radius = max(50 m, 10 m/s * 1 s) = 50 m, but arc 2 starts 100 m
    # beyond the end of arc 0 -> implied speed too high, transition impossible
    cfg = TransitionConfig(min_cutoff_m=50.0, max_speed_mps=10.0)
    log_a = two_obs_fn(road, [(0, 0.0)], [(2, 1.0)], obs_xy=[[0, 0], [300, 0]], dt=1.0, cfg=cfg)
    assert log_a.tolist() == [[MIN]]


def test_shortest_path_cache_matches_fresh_graph():
    road = chain_graph()
    first = road.shortest_from(0, 1000.0)
    again = road.shortest_from(0, 1000.0)
    fresh = chain_graph().shortest_from(0, 1000.0)
    assert first is again  # served from cache
    assert first == fresh == {0: 0.0, 1: 100.0, 2: 200.0, 3: 300.0}


# ---------------------------------------------------------------- end to end

def parallel_roads():
    """Main one-way road at y=0 (6 arcs of 50 m) and a disconnected parallel road at y=30."""
    xs = np.arange(0, 301, 50, dtype=float)
    main = np.column_stack([xs, np.zeros_like(xs)])
    side = np.column_stack([xs, np.full_like(xs, 30.0)])
    nodes = np.vstack([main, side])
    n = len(xs)
    arcs = [[i, i + 1] for i in range(n - 1)] + [[n + i, n + i + 1] for i in range(n - 1)]
    return RoadGraph(nodes, np.array(arcs)), n - 1  # n-1 arcs per road


def test_viterbi_with_road_transitions_rejects_parallel_road_jump():
    road, per_road = parallel_roads()
    obs = np.array([[25, 2], [75, -3], [125, 4], [175, 21], [225, 1], [275, -2]], dtype=float)
    ts = np.arange(len(obs), dtype=float) * 5.0
    cand_arcs, cand_frac, emis = [], [], []
    for x, y in obs:
        k = int(x // 50)
        main_arc, side_arc = k, per_road + k
        cand_arcs.append([main_arc, side_arc])
        cand_frac.append([(x % 50) / 50, (x % 50) / 50])
        emis.append(gaussian_emission_logp(np.array([abs(y), abs(y - 30)]), sigma_m=5.0))

    greedy = [c[int(np.argmax(e))] for c, e in zip(cand_arcs, emis)]
    trans = make_transition_fn(road, cand_arcs, cand_frac, obs, ts)
    res = decode(cand_arcs, emis, trans, timestamps=ts, config=ViterbiConfig(gap_break_s=30.0))

    truth = tuple(range(per_road))
    assert greedy[3] == per_road + 3  # noisy point snaps to the parallel road
    assert res.arc_ids == truth  # HMM keeps the connected route
    assert res.breaks == ()
