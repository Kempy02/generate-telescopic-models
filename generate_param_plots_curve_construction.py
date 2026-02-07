"""Create staged frames illustrating cross-section construction from control points.

This script mirrors the geometry build path used by `generate_models.py`, then
produces an animation-friendly sequence that visualises how the zero-angle
cross-section is assembled:

1. Control points appear one by one.
2. Line segments connect the control points to form the polyline.
3. Circles are added at each point (or every sampled point) to demonstrate the
   union-of-circles construction strategy along the 1D outline.
4. The true 2D cross-section outline is traced from low to high X before the
   framing eases out to the final view.

Example:

python generate_param_plots_curve_construction.py \
  --csv datasets/ParamPlots/test.csv \
  --prototype-id bend_collapse5 \
  --export-formats png \
  --max-control-point-frames 20 \
  --max-polyline-frames 48 \
  --max-circle-frames 192 \
  --start-x-limits -40 5 \
  --start-z-limits -5 30 \
  --axis-transition-frames 36 \
  --circle-radius 1.0 \
  --union-fill-frames 96 \
  --shade-final-outline \
  --final-outline-alpha 0.5

# Combine frames into a video using ffmpeg:
ffmpeg \
  -framerate 30 \
  -pattern_type glob \
  -i "prototype_plots/curve_construction_frames/*.png" \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -pix_fmt yuv420p \
  -c:v libx264 \
  -r 30 \
  report/videos/curve_construction.mp4



"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import art3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "numpy is required for plotting. Install it with `pip install numpy`."
    ) from exc

from builders.two_d_build import _build_thickness_factors
from builders.build_modules.twoD_helpers import (
    process_outer_points,
    generate_vt_control_points,
    apply_thickness,
)
from core.config import BaselineGeometryConfig, BendSettings
from core.generate_geometry import generate_geometry, generate_geometry_bend
from core.param_builder import build_params_from_config_csv
from io_modules.read_csv import read_param_rows_csv

CSV_PATH = "datasets/ParamPlots/ParamPlots.csv"
DEFAULT_OUTPUT_FOLDER = "prototype_plots/curve_construction_frames"
DEFAULT_FILENAME_SUFFIX = "curve_build"
DEFAULT_CMAP = "viridis"
COLOR_PALETTE = ["#BC4A87", "#0a75a3", "#F08A24", "#30855A", "#9F3ED5"]
DIM_LINESTYLE = (0, (5, 4))
DEFAULT_DIM_ALPHA = 0.75
DEFAULT_DIM_LINEWIDTH_FACTOR = 0.5
DEFAULT_2D_SHADE_ALPHA = 0.12
FIG_BORDER_COLOR = "#F8F8F8"
FIG_BORDER_ALPHA = 0.8
FIG_BORDER_LINEWIDTH = 1.2
FIG_BORDER_MARGIN = 0.0
RC_PARAMS = {
    "font.size": 15,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
}

COLUMN_WIDTH = 7.0  # inches
ROW_HEIGHT = 6.0   # inches

STATIC_X_LIMITS: Tuple[float, float] | None = (-31.0, 31.0)
STATIC_Y_LIMITS: Tuple[float, float] | None = (-31.0, 31.0)
STATIC_Z_LIMITS: Tuple[float, float] | None = (-23.0, 39.0)

GLOBAL_LIMIT_MARGIN = 0.05
VIEW_DISTANCE = 9.0
USE_ORTHOGRAPHIC_PROJECTION = True

TITLE_BASE = "C: 3D Model Generation"
CONTROL_POINT_COLOR = "#65324E"
CONTROL_POINT_SIZE = 26
CONTROL_POINT_EDGE = "#A85E85"
LINE_COLOR = "#BC4A87"
LINE_WIDTH = 1.6
CIRCLE_EDGE_COLOR = "#371F35"
CIRCLE_FACE_COLOR = "#CC9ADD"
CIRCLE_EDGE_WIDTH = 1.0
CIRCLE_ALPHA = 0.25
UNION_FILL_COLOR = "#BC4A87"
ONED_TWOD_COLOR = "#BC4A87"


def _normalize_points(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """Return a clean list of (x, y) tuples extracted from geometry output."""
    norm: List[Tuple[float, float]] = []

    def _is_scalar(val) -> bool:
        scalar_types = (int, float, np.generic)
        return isinstance(val, scalar_types)

    def iter_points(obj: Any):
        if obj is None:
            return
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                yield obj.tolist()
            else:
                for sub in obj:
                    yield from iter_points(sub)
            return
        if isinstance(obj, (list, tuple)):
            if obj and _is_scalar(obj[0]):
                yield obj
            else:
                for sub in obj:
                    yield from iter_points(sub)
            return
        if hasattr(obj, "x") and hasattr(obj, "y"):
            yield (float(getattr(obj, "x")), float(getattr(obj, "y")))
            return
        try:
            seq = list(obj)  # type: ignore
        except TypeError as exc:
            raise TypeError(f"Unsupported point format: {obj!r}") from exc
        if seq and _is_scalar(seq[0]):
            yield seq
        else:
            for sub in seq:
                yield from iter_points(sub)

    for item in iter_points(points):
        if item is None or len(item) < 2:
            continue
        norm.append((float(item[0]), float(item[1])))

    if not norm:
        raise TypeError(f"Unsupported point format: {points!r}")

    return norm


def _ensure_closed(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Ensure the polyline is explicitly closed."""
    if not points:
        return []
    first_x, first_y = points[0]
    last_x, last_y = points[-1]
    if math.isclose(first_x, last_x, rel_tol=1e-9, abs_tol=1e-9) and math.isclose(
        first_y, last_y, rel_tol=1e-9, abs_tol=1e-9
    ):
        return list(points)
    closed = list(points)
    closed.append(points[0])
    return closed


def _rotate_section(points: Sequence[Tuple[float, float]], angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate a 2D cross-section around the Z axis by the specified angle."""
    arr = np.asarray(points, dtype=float)
    xs = arr[:, 0]
    ys = arr[:, 1]
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x3d = xs * cos_t
    y3d = xs * sin_t
    z3d = ys
    return x3d, y3d, z3d


def _add_figure_border(
    fig,
    ax,
    *,
    margin: float = FIG_BORDER_MARGIN,
    color: str = FIG_BORDER_COLOR,
    linewidth: float = FIG_BORDER_LINEWIDTH,
    alpha: float = FIG_BORDER_ALPHA,
) -> None:
    """Add a rectangular border around the main axes area, excluding title whitespace."""
    bbox = ax.get_position()
    x0 = max(bbox.x0 - margin, 0.0)
    y0 = max(bbox.y0 - margin, 0.0)
    x1 = min(bbox.x1 + margin, 1.0)
    y1 = min(bbox.y1 + margin, 1.0)
    width = max(x1 - x0, 0.0) + 0.1
    height = max(y1 - y0, 0.0)
    rect = matplotlib.patches.Rectangle(
        (x0, y0),
        width,
        height,
        transform=fig.transFigure,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        joinstyle="round",
    )
    fig.add_artist(rect)


def _angles_for_sections(params, n_sections: int) -> List[float]:
    """Compute bending angles for each section index."""
    if n_sections <= 0:
        return []
    if n_sections == 1:
        return [0.0]
    total = float(getattr(params, "angular_section", 0.0) or BendSettings().total_angular_section)
    step = total / max(n_sections - 1, 1)
    return [idx * step for idx in range(n_sections)]


def _select_row(
    rows: Sequence[Dict[str, Any]], prototype_id: str | None, row_index: int | None
) -> Dict[str, Any]:
    """Select a CSV row by prototype ID or index."""
    if prototype_id:
        for row in rows:
            if str(row.get("Prototype ID", "")).strip().lower() == prototype_id.strip().lower():
                return row
        raise ValueError(f"Prototype ID {prototype_id!r} not found in CSV.")
    if row_index is not None:
        if not (0 <= row_index < len(rows)):
            raise IndexError(f"Row index {row_index} out of range (0..{len(rows)-1}).")
        return rows[row_index]
    if len(rows) != 1:
        raise ValueError("CSV contains multiple rows; specify --prototype-id or --row-index.")
    return rows[0]


def _collect_cross_sections(report, params) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """Return ordered cross-sections and their angles for a single prototype."""
    if getattr(params, "bending_enabled", False):
        sections_raw = [xsec.twoD_cross_section for xsec in getattr(report, "xsections2d_list", [])]
        if not sections_raw:
            raise ValueError("Bending run did not return any 2D cross-sections.")
    else:
        single = getattr(report, "xsections2d", None)
        if single is None or not getattr(single, "twoD_cross_section", None):
            raise ValueError("Linear run did not return a 2D cross-section.")
        sections_raw = [single.twoD_cross_section]

    sections = [_ensure_closed(_normalize_points(section)) for section in sections_raw]
    angles = _angles_for_sections(params, len(sections))
    return sections, angles


def _compute_global_limits(
    sections: Sequence[List[Tuple[float, float]]],
    angles: Sequence[float],
    *,
    margin_fraction: float = GLOBAL_LIMIT_MARGIN,
) -> Dict[str, Tuple[float, float]]:
    """Return axis-aligned bounds that encapsulate all rotated sections."""
    if not sections or not angles:
        return {}

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    for points, angle in zip(sections, angles):
        x3d, y3d, z3d = _rotate_section(points, angle)
        min_x = min(min_x, float(np.min(x3d)))
        max_x = max(max_x, float(np.max(x3d)))
        min_y = min(min_y, float(np.min(y3d)))
        max_y = max(max_y, float(np.max(y3d)))
        min_z = min(min_z, float(np.min(z3d)))
        max_z = max(max_z, float(np.max(z3d)))

    bounds: Dict[str, Tuple[float, float]] = {}
    for axis, lower, upper in (
        ("x", min_x, max_x),
        ("y", min_y, max_y),
        ("z", min_z, max_z),
    ):
        if not math.isfinite(lower) or not math.isfinite(upper):
            continue
        span = max(upper - lower, 1e-9)
        pad = span * max(margin_fraction, 0.0)
        bounds[axis] = (lower - pad, upper + pad)

    return bounds


def _resolve_axes_limits(auto_limits: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Blend optional static overrides with auto-computed limits."""
    resolved: Dict[str, Tuple[float, float]] = {}
    static_map = {
        "x": STATIC_X_LIMITS,
        "y": STATIC_Y_LIMITS,
        "z": STATIC_Z_LIMITS,
    }
    for axis in ("x", "y", "z"):
        static_limits = static_map[axis]
        if static_limits is not None:
            resolved[axis] = static_limits
        else:
            auto = auto_limits.get(axis)
            if auto is not None:
                resolved[axis] = auto
    return resolved


def _sample_counts(total: int, max_frames: int | None) -> List[int]:
    """Return monotonically increasing counts up to `total`, capped by `max_frames`."""
    if total <= 0:
        return []
    if max_frames is None or max_frames <= 0 or max_frames >= total:
        return list(range(1, total + 1))
    targets = np.linspace(1, total, num=max_frames, dtype=int)
    counts: List[int] = []
    seen = set()
    for value in targets:
        value = int(max(1, min(total, value)))
        if value not in seen:
            counts.append(value)
            seen.add(value)
    if counts[-1] != total:
        counts.append(total)
    return counts


def _sample_counts_start(total: int, max_frames: int | None, start: int) -> List[int]:
    """Variant of `_sample_counts` beginning at a specified minimum count."""
    if total < start:
        return []
    counts = _sample_counts(total, max_frames)
    return [c for c in counts if c >= start]


def _sample_indices(indices: Sequence[int], max_frames: int | None) -> List[int]:
    """Return subsequence of `indices` no longer than `max_frames`, preserving order."""
    if not indices:
        return []
    if max_frames is None or max_frames <= 0 or max_frames >= len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, num=max_frames, dtype=int)
    selected: List[int] = []
    last = None
    for pos in positions:
        idx = indices[int(pos)]
        if idx != last:
            selected.append(idx)
            last = idx
    if selected[-1] != indices[-1]:
        selected.append(indices[-1])
    return selected


def _normalize_limit_pair(pair: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    if pair is None:
        return None
    if len(pair) != 2:
        raise ValueError("Axis limit pairs must contain exactly two values.")
    low, high = float(pair[0]), float(pair[1])
    if low > high:
        low, high = high, low
    return (low, high)


def _clean_limits_dict(limits: Dict[str, Optional[Tuple[float, float]]]) -> Optional[Dict[str, Tuple[float, float]]]:
    cleaned = {axis: limit for axis, limit in limits.items() if limit is not None}
    return cleaned if cleaned else None


def _limits_equal(
    limits_a: Optional[Tuple[float, float]],
    limits_b: Optional[Tuple[float, float]],
    *,
    tol: float = 1e-6,
) -> bool:
    if limits_a is None and limits_b is None:
        return True
    if limits_a is None or limits_b is None:
        return False
    return all(abs(a - b) <= tol for a, b in zip(limits_a, limits_b))


def _interpolate_limits(
    start_limits: Dict[str, Optional[Tuple[float, float]]],
    end_limits: Dict[str, Optional[Tuple[float, float]]],
    fraction: float,
) -> Dict[str, Optional[Tuple[float, float]]]:
    fraction = max(0.0, min(1.0, fraction))
    blended: Dict[str, Optional[Tuple[float, float]]] = {}
    for axis in ("x", "y", "z"):
        start = start_limits.get(axis)
        end = end_limits.get(axis)
        if start is None and end is None:
            blended[axis] = None
        elif start is None:
            blended[axis] = end
        elif end is None:
            blended[axis] = start
        else:
            blended[axis] = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
    return blended


def _apply_axes_limits(ax, *, axes_limits: Dict[str, Tuple[float, float]] | None) -> None:
    """Set axis limits consistently across frames."""
    if axes_limits:
        setters = {
            "x": ax.set_xlim3d,
            "y": ax.set_ylim3d,
            "z": ax.set_zlim3d,
        }
        for axis, setter in setters.items():
            limits = axes_limits.get(axis)
            if not limits:
                continue
            setter(*limits)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))
    else:
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        span = limits[:, 1] - limits[:, 0]
        half = max(span) / 2.0 if span.size else 1.0
        centers = np.mean(limits, axis=1)
        ax.set_xlim3d(centers[0] - half, centers[0] + half)
        ax.set_ylim3d(centers[1] - half, centers[1] + half)
        ax.set_zlim3d(centers[2] - half, centers[2] + half)


def _create_3d_axes(
    *,
    title: str,
    view_elev: float,
    view_azim: float,
    axes_limits: Dict[str, Tuple[float, float]] | None,
) -> Tuple[plt.Figure, Any]:
    fig = plt.figure(figsize=(COLUMN_WIDTH, ROW_HEIGHT))
    ax = fig.add_subplot(111, projection="3d")
    if USE_ORTHOGRAPHIC_PROJECTION and hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")
    ax.dist = VIEW_DISTANCE
    # ax.set_title(title, pad=10, fontweight="bold", loc="left")
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, alpha=0.35, linestyle=":", linewidth=0.8)
    ax.tick_params(axis="both", which="major", length=5, width=1.1)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)
    ax.minorticks_on()
    ax.set_xlabel("X [mm]", fontweight="medium", labelpad=4)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_zlabel("Z [mm]", fontweight="medium", labelpad=4)
    _apply_axes_limits(ax, axes_limits=axes_limits)
    return fig, ax


def _scatter_control_points(
    ax,
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
    count: int,
    *,
    facecolor: str = CONTROL_POINT_COLOR,
    edgecolor: str = CONTROL_POINT_EDGE,
    size: float = CONTROL_POINT_SIZE,
    alpha: float = 1.0,
) -> None:
    if count <= 0:
        return
    ax.scatter(
        x3d[:count],
        y3d[:count],
        z3d[:count],
        s=size,
        color=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        alpha=alpha,
        depthshade=False,
    )


def _plot_polyline(
    ax,
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
    *,
    upto: int,
    close: bool,
    color: str = LINE_COLOR,
    linewidth: float = LINE_WIDTH,
    alpha: float = 1.0,
) -> None:
    if upto < 2:
        return
    ax.plot(
        x3d[:upto],
        y3d[:upto],
        z3d[:upto],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        # solid_capstyle="round",
    )
    if close and upto == len(x3d):
        ax.plot(
            [x3d[-1], x3d[0]],
            [y3d[-1], y3d[0]],
            [z3d[-1], z3d[0]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
        )


def _plot_outline_progress(
    ax,
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
    arc_lengths: np.ndarray,
    threshold: float,
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    n = len(x3d)
    if n < 2 or arc_lengths.size < n + 1:
        return
    for i in range(n):
        j = (i + 1) % n
        x0, y0, z0 = x3d[i], y3d[i], z3d[i]
        x1, y1, z1 = x3d[j], y3d[j], z3d[j]
        start_len = arc_lengths[i]
        end_len = arc_lengths[i + 1]

        if threshold >= end_len:
            xs_seg = [x0, x1]
            ys_seg = [y0, y1]
            zs_seg = [z0, z1]
        elif threshold <= start_len:
            continue
        else:
            segment_span = end_len - start_len
            if segment_span <= 1e-12:
                continue
            t = (threshold - start_len) / segment_span
            t = max(0.0, min(1.0, t))
            x_clip = x0 + (x1 - x0) * t
            y_clip = y0 + (y1 - y0) * t
            z_clip = z0 + (z1 - z0) * t
            xs_seg = [x0, x_clip]
            ys_seg = [y0, y_clip]
            zs_seg = [z0, z_clip]

        ax.plot(
            xs_seg,
            ys_seg,
            zs_seg,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
        )


def _compute_arc_lengths(
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
    *,
    closed: bool,
) -> np.ndarray:
    coords = np.column_stack((x3d, y3d, z3d))
    if coords.size == 0:
        return np.zeros(1, dtype=float)
    arc = [0.0]
    for i in range(1, len(coords)):
        arc.append(arc[-1] + float(np.linalg.norm(coords[i] - coords[i - 1])))
    if closed and len(coords) > 1:
        arc.append(arc[-1] + float(np.linalg.norm(coords[0] - coords[-1])))
    else:
        arc.append(arc[-1])
    return np.asarray(arc, dtype=float)


def _add_circle_patch(ax, center_x: float, center_z: float, radius: float) -> Circle:
    circle = Circle(
        (center_x, center_z),
        radius=radius,
        facecolor=(*mcolors.to_rgb(CIRCLE_FACE_COLOR), CIRCLE_ALPHA),
        edgecolor=CIRCLE_EDGE_COLOR,
        linewidth=CIRCLE_EDGE_WIDTH,
    )
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=0.0, zdir="y")
    return circle


def _add_outline_fill(
    ax,
    x3d: np.ndarray,
    y3d: np.ndarray,
    z3d: np.ndarray,
    *,
    alpha: float,
) -> None:
    if x3d.size < 3:
        return
    polygon = list(zip(x3d, y3d, z3d))
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    collection = Poly3DCollection(
        [polygon],
        facecolors=[(*mcolors.to_rgb(UNION_FILL_COLOR), alpha)],
        linewidths=0.0,
        edgecolors="none",
    )
    ax.add_collection3d(collection)


def generate_curve_construction_frames_for_configuration(
    csv_path: str,
    prototype_id: str | None = None,
    row_index: int | None = None,
    *,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    output_name: str | None = None,
    overwrite: bool = True,
    view_elev: float = 0.0,
    view_azim: float = -90.0,
    max_control_point_frames: int | None = None,
    max_polyline_frames: int | None = None,
    max_circle_frames: int | None = None,
    circle_radius: float = 1.0,
    start_x_limits: Optional[Sequence[float]] = None,
    start_y_limits: Optional[Sequence[float]] = None,
    start_z_limits: Optional[Sequence[float]] = None,
    axis_transition_frames: int = 12,
    union_fill_frames: int = 8,
    shade_final_outline: bool = False,
    final_outline_alpha: float = 0.12,
    export_formats: Sequence[str] = ("png",),
) -> List[str]:
    """Generate per-frame images illustrating the cross-section construction."""
    rows = read_param_rows_csv(csv_path)
    selected_row = _select_row(rows, prototype_id, row_index)

    baseline = BaselineGeometryConfig()
    params, config_csv_path, use_linear_fast = build_params_from_config_csv(selected_row, baseline)
    proto_id = params.export_filename or selected_row.get("Prototype ID") or "prototype"

    if use_linear_fast:
        params.bending_enabled = False
        report = generate_geometry(params)
    else:
        params.bending_enabled = True
        report = generate_geometry_bend(params, config_csv_path, testing_mode=False)

    sections, angles = _collect_cross_sections(report, params)
    if not sections:
        raise ValueError("No cross-sections returned; cannot create frames.")

    first_section = sections[0]
    first_angle = angles[0] if angles else 0.0

    if hasattr(report, "curves1d_list") and getattr(report, "curves1d_list", None):
        curves_source = report.curves1d_list[0]
    else:
        curves_source = getattr(report, "curves1d", None)
    if curves_source is None:
        raise ValueError("Curves data unavailable for control-point construction frames.")

    outer_points_raw = process_outer_points(curves_source.curve_points)
    if not outer_points_raw:
        raise ValueError("Unable to derive 1D outer curve points for construction frames.")
    control_points_raw = process_outer_points(curves_source.control_points)
    if not control_points_raw:
        raise ValueError("Unable to derive control points for construction frames.")
    vt_control_points = generate_vt_control_points(curves_source.control_points, curves_source.cp_idx)
    thickness_factors = _build_thickness_factors(params)
    thickness_result = apply_thickness(
        outer_points=outer_points_raw,
        mode=params.thickness_mode,
        vt_control_points=vt_control_points,
        all_thicknesses=thickness_factors,
        thickness_factor1=params.thickness_factor,
        thickness_factor2=params.thickness_factor2,
        baseline_thickness=params.thickness,
    )
    if isinstance(thickness_result, tuple):
        raw_thicknesses, _cap_thickness = thickness_result
    else:
        raw_thicknesses = thickness_result

    outer_points_seq = list(outer_points_raw)
    if outer_points_seq and outer_points_seq[0] == outer_points_seq[-1]:
        outer_points_seq = outer_points_seq[:-1]
    control_points_seq = list(control_points_raw)
    if control_points_seq and control_points_seq[0] == control_points_seq[-1]:
        control_points_seq = control_points_seq[:-1]

    if raw_thicknesses:
        if len(raw_thicknesses) == len(outer_points_seq):
            point_thicknesses = list(raw_thicknesses)
        else:
            # Resample thicknesses to match the trimmed outer-point count.
            point_thicknesses = list(
                np.interp(
                    np.linspace(0, len(raw_thicknesses) - 1, num=len(outer_points_seq)),
                    np.arange(len(raw_thicknesses)),
                    raw_thicknesses,
                )
            )
    else:
        point_thicknesses = [params.thickness] * len(outer_points_seq)

    unique_section = first_section[:-1] if len(first_section) > 1 and first_section[0] == first_section[-1] else first_section
    if len(unique_section) < 3:
        raise ValueError("Cross-section must contain at least three points for construction visuals.")

    control_x3d, control_y3d, control_z3d = _rotate_section(control_points_seq, first_angle)
    outer_x3d, outer_y3d, outer_z3d = _rotate_section(outer_points_seq, first_angle)
    x3d_full, y3d_full, z3d_full = _rotate_section(unique_section, first_angle)
    outer_arc_lengths = _compute_arc_lengths(outer_x3d, outer_y3d, outer_z3d, closed=False)
    outline_arc_lengths = _compute_arc_lengths(x3d_full, y3d_full, z3d_full, closed=True)
    outline_arc_points = outline_arc_lengths[:-1] if outline_arc_lengths.size else np.array([0.0])
    total_outline_length = outline_arc_lengths[-1] if outline_arc_lengths.size else 0.0

    if not output_name:
        output_name = f"{proto_id}_{DEFAULT_FILENAME_SUFFIX}"

    plt.rcParams.update(RC_PARAMS)

    auto_limits = _compute_global_limits([first_section], [first_angle])
    axes_limits = _resolve_axes_limits(auto_limits)
    if not axes_limits:
        axes_limits = None

    end_limits_dict = axes_limits or {}
    start_limits_input = {
        "x": _normalize_limit_pair(start_x_limits),
        "y": _normalize_limit_pair(start_y_limits),
        "z": _normalize_limit_pair(start_z_limits),
    }
    stage_axes_limits = {
        axis: start_limits_input[axis] if start_limits_input[axis] is not None else end_limits_dict.get(axis)
        for axis in ("x", "y", "z")
    }

    export_formats = tuple(dict.fromkeys(str(ext).lower().lstrip(".") for ext in export_formats if ext))
    output_dir = os.path.join(os.getcwd(), output_folder)
    os.makedirs(output_dir, exist_ok=True)

    saved_paths: List[str] = []
    frame_counter = 0

    def _finalize_and_save(fig: plt.Figure) -> None:
        nonlocal frame_counter
        frame_counter += 1
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=0.9)
        _add_figure_border(fig, fig.axes[0])
        for ext in export_formats:
            if ext not in {"png", "svg", "pdf"}:
                continue
            filename = f"{output_name}_{frame_counter:04d}.{ext}"
            path = os.path.join(output_dir, filename)
            if not overwrite and os.path.exists(path):
                raise FileExistsError(f"{path} already exists; rerun with --overwrite to replace it.")
            fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
            saved_paths.append(path)
        plt.close(fig)

    total_control_points = len(control_points_seq)
    total_outer_points = len(outer_points_seq)
    if max_control_point_frames and max_control_point_frames > 0:
        control_end = min(total_control_points, int(max_control_point_frames))
    else:
        control_end = total_control_points
    control_counts = list(range(1, control_end + 1))
    stage_limits_clean = _clean_limits_dict(stage_axes_limits)
    end_limits_all = {axis: end_limits_dict.get(axis) for axis in ("x", "y", "z")}
    end_limits_clean = _clean_limits_dict(end_limits_all)

    for idx, count in enumerate(control_counts, start=1):
        fig, ax = _create_3d_axes(
            title=f"{TITLE_BASE} – Control Points",
            view_elev=view_elev,
            view_azim=view_azim,
            axes_limits=stage_limits_clean,
        )
        _scatter_control_points(ax, control_x3d, control_y3d, control_z3d, count)
        _finalize_and_save(fig)

    # Stage 2: connect points into polyline
    polyline_counts = _sample_counts_start(total_outer_points, max_polyline_frames, start=2)
    for count in polyline_counts:
        fig, ax = _create_3d_axes(
            title=f"{TITLE_BASE} – Linking Points",
            view_elev=view_elev,
            view_azim=view_azim,
            axes_limits=stage_limits_clean,
        )
        _scatter_control_points(ax, control_x3d, control_y3d, control_z3d, total_control_points)
        _plot_polyline(ax, outer_x3d, outer_y3d, outer_z3d, upto=count, close=False)
        _finalize_and_save(fig)

    # Add one frame with closing segment.
    fig, ax = _create_3d_axes(
        title=f"{TITLE_BASE} – Complete Polyline",
        view_elev=view_elev,
        view_azim=view_azim,
        axes_limits=stage_limits_clean,
    )
    _scatter_control_points(ax, control_x3d, control_y3d, control_z3d, total_control_points)
    _plot_polyline(
        ax,
        outer_x3d,
        outer_y3d,
        outer_z3d,
        upto=total_outer_points,
        close=False,
    )
    _finalize_and_save(fig)

    # Stage 3: circles accumulate.
    circle_indices_all = list(range(total_outer_points))
    circle_indices = _sample_indices(circle_indices_all, max_circle_frames)
    circle_centers: List[Tuple[float, float, float, float]] = []
    for point_idx in circle_indices:
        fig, ax = _create_3d_axes(
            title=f"{TITLE_BASE} – Circle Placement",
            view_elev=view_elev,
            view_azim=view_azim,
            axes_limits=stage_limits_clean,
        )
        _plot_polyline(
            ax,
            outer_x3d,
            outer_y3d,
            outer_z3d,
            upto=total_outer_points,
            close=False,
            alpha=0.6,
            color=ONED_TWOD_COLOR,
        )
        for cx, cz, radius, _ in circle_centers:
            _add_circle_patch(ax, cx, cz, radius)

        base_radius = point_thicknesses[point_idx] if point_idx < len(point_thicknesses) else params.thickness
        radius = max(0.0, float(base_radius) * float(circle_radius))
        center_x, center_z = outer_x3d[point_idx], outer_z3d[point_idx]
        _add_circle_patch(ax, center_x, center_z, radius)
        arc_len = outer_arc_lengths[point_idx] if point_idx < outer_arc_lengths.size else outer_arc_lengths[-1]
        circle_centers.append((center_x, center_z, radius, arc_len))
        _finalize_and_save(fig)

    # Stage 4: union fill fade in.
    # Stage 4: 2D outline reveal (uses the open 1D polyline)
    outline_steps = max(int(union_fill_frames), 1)
    outline_color = UNION_FILL_COLOR
    for step in range(1, outline_steps + 1):
        fraction = step / outline_steps
        threshold = total_outline_length * fraction
        fig, ax = _create_3d_axes(
            title=f"{TITLE_BASE} – 2D Outline Reveal",
            view_elev=view_elev,
            view_azim=view_azim,
            axes_limits=stage_limits_clean,
        )
        _plot_polyline(
            ax,
            outer_x3d,
            outer_y3d,
            outer_z3d,
            upto=total_outer_points,
            close=False,
            alpha=0.2,
            color=ONED_TWOD_COLOR,
        )
        for cx, cz, radius, _ in circle_centers:
            _add_circle_patch(ax, cx, cz, radius)
        _plot_outline_progress(
            ax,
            x3d_full,
            y3d_full,
            z3d_full,
            arc_lengths=outline_arc_lengths,
            threshold=threshold,
            color=outline_color,
            linewidth=LINE_WIDTH*1.5,
            alpha=0.9,
        )
        _finalize_and_save(fig)

    if shade_final_outline:
        fig, ax = _create_3d_axes(
            title=f"{TITLE_BASE} – Shaded Outline",
            view_elev=view_elev,
            view_azim=view_azim,
            axes_limits=stage_limits_clean,
        )
        _plot_polyline(
            ax,
            x3d_full,
            y3d_full,
            z3d_full,
            upto=len(unique_section),
            close=True,
            alpha=0.85,
            color=outline_color,
        )
        _add_outline_fill(ax, x3d_full, y3d_full, z3d_full, alpha=final_outline_alpha)
        _finalize_and_save(fig)

    # Stage 5: axis transition to final limits.
    transition_frames = max(int(axis_transition_frames or 0), 0)
    end_limits_for_transition = {
        axis: end_limits_all.get(axis) if end_limits_all.get(axis) is not None else stage_axes_limits.get(axis)
        for axis in ("x", "y", "z")
    }
    needs_transition = transition_frames > 0 and any(
        not _limits_equal(stage_axes_limits.get(axis), end_limits_for_transition.get(axis))
        for axis in ("x", "y", "z")
    )
    if needs_transition:
        for step in range(1, transition_frames + 1):
            fraction = step / transition_frames
            blended_limits = _interpolate_limits(stage_axes_limits, end_limits_for_transition, fraction)
            blended_clean = _clean_limits_dict(blended_limits)
            fig, ax = _create_3d_axes(
                title=f"{TITLE_BASE} – Axis Expansion",
                view_elev=view_elev,
                view_azim=view_azim,
                axes_limits=blended_clean,
            )
            _plot_polyline(
                ax,
                x3d_full,
                y3d_full,
                z3d_full,
                upto=len(unique_section),
                close=True,
                alpha=0.85,
                color=outline_color,
            )
            if shade_final_outline:
                _add_outline_fill(ax, x3d_full, y3d_full, z3d_full, alpha=final_outline_alpha)
            _finalize_and_save(fig)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate frames illustrating the curve construction process for a cross-section."
    )
    parser.add_argument("--csv", default=CSV_PATH, help="Path to the CSV listing prototype configurations.")
    parser.add_argument(
        "--prototype-id",
        help="Prototype ID to plot (matches the CSV 'Prototype ID' column).",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        help="Row index to plot if no prototype ID is supplied.",
    )
    parser.add_argument(
        "--output-folder",
        default=DEFAULT_OUTPUT_FOLDER,
        help="Directory that receives the generated frames.",
    )
    parser.add_argument(
        "--output-name",
        help="Filename stem for exported frames (defaults to '<prototype>_curve_build').",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing files.",
    )
    parser.add_argument(
        "--view-elev",
        type=float,
        default=0.0,
        help="Elevation angle for the 3D view (default keeps the y-axis head-on).",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=-90.0,
        help="Azimuth angle for the 3D view (default aligns with the y=0 plane).",
    )
    parser.add_argument(
        "--max-control-point-frames",
        type=int,
        default=0,
        help="Maximum frames used to reveal control points (0 keeps one per point).",
    )
    parser.add_argument(
        "--max-polyline-frames",
        type=int,
        default=0,
        help="Maximum frames used while connecting the polyline (0 keeps one per segment).",
    )
    parser.add_argument(
        "--max-circle-frames",
        type=int,
        default=0,
        help="Maximum frames used when adding circles (0 keeps one per curve point).",
    )
    parser.add_argument(
        "--circle-radius",
        type=float,
        default=1.0,
        help="Scale factor applied to each point's thickness when drawing circles.",
    )
    parser.add_argument(
        "--start-x-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Initial X-axis limits before the axis expansion.",
    )
    parser.add_argument(
        "--start-y-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Initial Y-axis limits before the axis expansion.",
    )
    parser.add_argument(
        "--start-z-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Initial Z-axis limits before the axis expansion.",
    )
    parser.add_argument(
        "--axis-transition-frames",
        type=int,
        default=12,
        help="Number of frames to transition from the initial limits to the final limits (0 disables).",
    )
    parser.add_argument(
        "--union-fill-frames",
        type=int,
        default=8,
        help="Number of frames used to sweep the 2D outline along the X axis.",
    )
    parser.add_argument(
        "--shade-final-outline",
        action="store_true",
        help="Add a final frame (and transition frames) with the 2D outline shaded.",
    )
    parser.add_argument(
        "--final-outline-alpha",
        type=float,
        default=0.12,
        help="Alpha used when shading the final 2D outline (ignored if shading disabled).",
    )
    parser.add_argument(
        "--export-formats",
        nargs="+",
        default=("png",),
        help="File extensions to export (subset of png/svg/pdf).",
    )

    args = parser.parse_args()
    generate_curve_construction_frames_for_configuration(
        csv_path=args.csv,
        prototype_id=args.prototype_id,
        row_index=args.row_index,
        output_folder=args.output_folder,
        output_name=args.output_name,
        overwrite=not args.no_overwrite,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
        max_control_point_frames=args.max_control_point_frames,
        max_polyline_frames=args.max_polyline_frames,
        max_circle_frames=args.max_circle_frames,
        circle_radius=args.circle_radius,
        start_x_limits=args.start_x_limits,
        start_y_limits=args.start_y_limits,
        start_z_limits=args.start_z_limits,
        axis_transition_frames=args.axis_transition_frames,
        union_fill_frames=args.union_fill_frames,
        shade_final_outline=args.shade_final_outline,
        final_outline_alpha=args.final_outline_alpha,
        export_formats=args.export_formats,
    )


if __name__ == "__main__":
    main()
