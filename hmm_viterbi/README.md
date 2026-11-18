# hmm_viterbi — HMM / Viterbi route reconstruction (Member 3)

Tracks issues #7 (decoder) and #8 (road-graph transitions).

## Model (report §5)

| Question | Answer in this implementation |
|---|---|
| Hidden states | For GPS observation *t*, the candidate road arcs `C_t` (original `arc_id`s from Member 1's candidate generator). |
| Emissions | `log P(o_t \| c)` = `log(clip(p, 1e-9, 1))` of Member 2's classifier probability (`classifier_emission_logp`). Gaussian on point-to-arc distance (`gaussian_emission_logp`) is the fallback / ablation baseline. |
| Transitions | Pluggable `transition_logp(t_prev, t_next) -> (K_prev, K_next)` matrix. Road-network scoring (shortest-path vs GPS displacement) is implemented in #8. |
| Decoding | Log-space Viterbi: `δ_t(j) = max_i[δ_{t-1}(i) + logA_t(i,j)] + logB_t(j)`, back-pointers `ψ_t(j)`, backtrack from `argmax δ_T`. |
| Missing candidates | Observation is skipped (predicted `None`); the transition bridges the previous observation that had candidates to the next one. |
| Large GPS gaps | `dt > gap_break_s` (default 30 s) starts a new independent segment. |
| Disconnected segments | If every transition of a step is ≤ `min_logp` (no valid move), the chain is broken and a new segment starts from that observation's emissions; the break is reported in `DecodeResult.breaks`. |

All log-probabilities are floored at `min_logp` (default −1e4) so `-inf` never
propagates; NaN inputs raise `ValueError`. Ties resolve to the lower candidate index.

## Usage

```python
from hmm_viterbi import decode, ViterbiConfig, classifier_emission_logp

res = decode(
    candidates=[[12, 40], [40, 41], []],                 # arc_ids per observation
    emission_logp=[classifier_emission_logp(p) for p in probs],
    transition_logp=lambda t_prev, t_next: logA[(t_prev, t_next)],
    timestamps=track["timestamp"].to_numpy(),
    config=ViterbiConfig(gap_break_s=30.0),
)
res.arc_ids    # (40, 40, None)
res.segments   # ((0, 1),)
res.breaks     # ()
```

The decoder never receives reference routes; labels are only used later, outside
this package, to score the decoded output.

## Road-graph transitions (#8)

`transitions.py` builds a directed `networkx.DiGraph` from projected node
coordinates (metres) and `.arcs`, keeping original node ids and `arc_id`s.

```text
logA(i -> j) = -beta * | d_route(p_i, p_j) - d_gps(o_t, o_t') |
d_route      = (1 - f_i)·len(a_i) + SP(v(a_i), u(a_j)) + f_j·len(a_j)
             = (f_j - f_i)·len(a)      if same arc and moving forward
                                        (backward jitter <= 15 m allowed)
```

* Shortest paths: bounded Dijkstra with cutoff `max(min_cutoff_m, max_speed_mps·dt)`,
  cached per (source node, cutoff). Unreachable within the cutoff (disconnected
  component, one-way violation, implausible speed) → `min_logp`, which the
  decoder turns into a segment break if no candidate pair is reachable.
* `make_transition_fn(road, cand_arcs, cand_frac, obs_xy, timestamps, config)`
  returns the callable `decode` expects.

Measured on training trip `00000001` (largest network seen: 88 809 nodes,
165 790 arcs, UTM EPSG:32632): graph build 0.51 s; bounded Dijkstra (200 m)
0.040 ms per source; cache hit 0.0004 ms.

Default parameters (`beta=0.1`, `max_speed_mps=50`, `min_cutoff_m=200`) are
initial guesses; they must be tuned on training / student-test trips only once
real candidates (#2) and classifier scores (#5) are available.

## Tests

```bash
python -m pytest hmm_viterbi -q
```

Synthetic known-answer cases, a greedy-vs-Viterbi case, brute-force comparison
on 25 random small problems, empty candidates, impossible transitions, time
gaps, 10 000-step numerical stability, arc-ID preservation and input validation.
