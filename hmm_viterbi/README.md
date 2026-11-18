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

## Tests

```bash
python -m pytest hmm_viterbi -q
```

Synthetic known-answer cases, a greedy-vs-Viterbi case, brute-force comparison
on 25 random small problems, empty candidates, impossible transitions, time
gaps, 10 000-step numerical stability, arc-ID preservation and input validation.
