# visualisations — Folium maps and comparison plots (Member 3)

Tracks issue #9.

## Route maps

`route_map.build_route_map(trip, partition, predicted_arc_ids=None, reference_arc_ids=None, classifier_arc_ids=None)`
returns a `folium.Map` with toggleable layers and a legend:

| Layer | Colour |
|---|---|
| GPS observations | blue points (tooltip: obs index, timestamp) |
| Road network (clipped to the trip extent + ~200 m) | grey |
| Reference route — **training / student-test only** | green |
| Classifier-only route (optional) | orange, dashed |
| Predicted HMM route | red |

`trip` is the dict returned by `data_processing.loader.load_trip`. Passing a
reference route with any other partition (e.g. `evaluator_test`) raises
`ValueError`, so evaluator ground truth can never be drawn.

### Regenerate

```bash
python -m visualisations.route_map --trip 00000005 --partition train --reference
python -m visualisations.route_map --trip 00000000 --partition student_test --reference
```

Output: `visualisations/maps/<trip>_<partition>.html`.
Committed example (real data, training trip 00000005, GPS + road network +
reference route): `maps/00000005_train.html`, screenshot `maps/00000005_train.png`.

### For Member 4 (UI, #11)

```python
from visualisations.route_map import build_route_map
fmap = build_route_map(trip, "evaluator_test", predicted_arc_ids=api_response_arc_ids)
st.components.v1.html(fmap.get_root().render(), height=600)   # or streamlit_folium
```

Never pass `reference_arc_ids` for evaluator trips.

## Still to do (#9)

Classifier-only vs HMM route comparison, route-continuity metric and charts —
needs real decoded routes (#7 Phase B) and classifier scores (#5).

## Tests

```bash
python -m pytest visualisations -q
```
