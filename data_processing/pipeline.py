
from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import pandas as pd

try:
    from pyproj import CRS, Transformer

    _HAS_PYPROJ = True
except ImportError:  # pragma: no cover
    CRS = None  # type: ignore
    Transformer = None  # type: ignore
    _HAS_PYPROJ = False


class _SimpleCRS:
    def __init__(self, epsg: int) -> None:
        self.epsg = epsg

    def to_string(self) -> str:
        return f"EPSG:{self.epsg}"


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: list | None
    partition: str
    crs: str


def choose_utm_crs(lon, lat):
    zone = int(np.floor((np.median(lon) + 180) / 6) + 1)
    zone = max(1, min(60, zone))
    epsg = (32600 if np.median(lat) >= 0 else 32700) + zone
    if _HAS_PYPROJ:
        return CRS.from_epsg(epsg)
    return _SimpleCRS(epsg)


def project_lonlat(lon, lat, crs):
    if _HAS_PYPROJ:
        transformer = Transformer.from_crs(
            "EPSG:4326", crs, always_xy=True
        )
        return transformer.transform(
            np.asarray(lon), np.asarray(lat)
        )
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    return lon_arr * 111_320.0, lat_arr * 111_132.0


def inverse_project_xy(x, y, crs):
    if _HAS_PYPROJ:
        transformer = Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        )
        return transformer.transform(
            np.asarray(x), np.asarray(y)
        )
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    return x_arr / 111_320.0, y_arr / 111_132.0


def _read(path, columns, dtypes):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        dtype=dtypes
    )
    if df.shape[1] != len(columns):
        raise ValueError(
            f"{path}: expected {len(columns)} columns, "
            f"found {df.shape[1]}"
        )
    df.columns = columns
    return df


def _validate_coordinates(df, name):
    if df[["lon", "lat"]].isna().any().any():
        raise ValueError(f"{name}: missing coordinates")
    if not df.lon.between(-180, 180).all():
        raise ValueError(f"{name}: invalid longitude")
    if not df.lat.between(-90, 90).all():
        raise ValueError(f"{name}: invalid latitude")


def load_trip(trip_dir, *, partition=None, allow_route=True):
    trip_dir = Path(trip_dir).resolve()

    if not trip_dir.is_dir():
        raise FileNotFoundError(trip_dir)

    name = str(trip_dir).lower()

    if partition is None:
        if "evaluator" in name:
            partition = "evaluator_test"
        elif "student" in name:
            partition = "student_test"
        elif "train" in name:
            partition = "train"
        else:
            partition = os.getenv("PATHSNAP_PARTITION", "unknown")

    if partition == "evaluator_test" and allow_route:
        raise PermissionError(
            "Evaluator reference routes are protected."
        )

    def get_file(suffix):
        files = list(trip_dir.glob(f"*{suffix}"))
        if len(files) != 1:
            raise ValueError(
                f"Expected exactly one {suffix} file in {trip_dir}"
            )
        return files[0]

    track = _read(
        get_file(".track"),
        ["lon", "lat", "t"],
        {"lon": "float64", "lat": "float64", "t": "float64"}
    )
    nodes = _read(
        get_file(".nodes"),
        ["lon", "lat"],
        {"lon": "float64", "lat": "float64"}
    )
    arcs = _read(
        get_file(".arcs"),
        ["u", "v"],
        {"u": "int64", "v": "int64"}
    )

    track.insert(0, "obs_idx", np.arange(len(track)))
    nodes.insert(0, "node_id", np.arange(len(nodes)))
    arcs.insert(0, "arc_id", np.arange(len(arcs)))

    _validate_coordinates(track, "track")
    _validate_coordinates(nodes, "nodes")

    if not np.isfinite(track.t).all():
        raise ValueError("Invalid timestamps")

    if (track.t.diff().dropna() < 0).any():
        raise ValueError("Timestamps are out of order")

    if not arcs.u.between(0, len(nodes)-1).all():
        raise ValueError("Invalid arc source node")
    if not arcs.v.between(0, len(nodes)-1).all():
        raise ValueError("Invalid arc destination node")

    crs = choose_utm_crs(
        pd.concat([track.lon, nodes.lon]),
        pd.concat([track.lat, nodes.lat])
    )

    track["x"], track["y"] = project_lonlat(
        track.lon, track.lat, crs
    )
    nodes["x"], nodes["y"] = project_lonlat(
        nodes.lon, nodes.lat, crs
    )

    xy = nodes.set_index("node_id")[["x", "y"]]
    start = xy.loc[arcs.u].to_numpy()
    end = xy.loc[arcs.v].to_numpy()
    arcs["length_m"] = np.linalg.norm(start-end, axis=1)

    route = None
    if partition in ("train", "student_test") and allow_route:
        route_file = list(trip_dir.glob("*.route"))
        if len(route_file) > 1:
            raise ValueError("Multiple route files")
        if route_file:
            values = np.loadtxt(route_file[0], dtype=int)
            route = np.atleast_1d(values).tolist()
            if any(a < 0 or a >= len(arcs) for a in route):
                raise ValueError("Invalid route arc ID")

    return Trip(
        trip_id=trip_dir.name,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
        partition=partition,
        crs=crs.to_string()
    )


def list_trips(partition):
    if partition not in ("train", "student_test"):
        raise ValueError("Use train or student_test")

    root = Path(os.getenv("PATHSNAP_DATA_ROOT", "data")).resolve()
    result = []

    for f in root.rglob("*.track"):
        folder = str(f.parent).lower()
        if partition == "train" and "train" in folder and "student" not in folder:
            result.append(f.parent)
        elif partition == "student_test" and "student" in folder:
            result.append(f.parent)

    return sorted(set(result))


def load_evaluator_trip(trip_dir):
    return load_trip(
        trip_dir,
        partition="evaluator_test",
        allow_route=False
    )


def inspect_partition(partition):
    rows = []

    for folder in list_trips(partition):
        row = {
            "trip_id": folder.name,
            "partition": partition
        }

        for suffix in (".track", ".nodes", ".arcs", ".route"):
            files = list(folder.glob(f"*{suffix}"))
            row[f"{suffix}_exists"] = bool(files)
            row[f"{suffix}_bytes"] = (
                files[0].stat().st_size if files else 0
            )

        try:
            trip = load_trip(folder, partition=partition)
            dt = trip.track.t.diff()

            row.update({
                "track_rows": len(trip.track),
                "nodes_rows": len(trip.nodes),
                "arcs_rows": len(trip.arcs),
                "route_length": len(trip.route or []),
                "duplicate_gps": int(
                    trip.track.duplicated(["lon", "lat"]).sum()
                ),
                "repeated_timestamps": int(
                    trip.track.t.duplicated().sum()
                ),
                "large_gaps_gt_5s": int((dt > 5).sum()),
                "crs": trip.crs
            })
        except Exception as e:
            row["error"] = str(e)

        rows.append(row)

    return pd.DataFrame(rows)
