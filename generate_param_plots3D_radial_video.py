"""Create staged 3D radial plots for assembling video frames.

This script mirrors the geometry build path used by `generate_models.py` for a
single CSV row, then exports one figure per interpolated angular cross-section.
The frames show the geometry accumulating stage by stage, making it easy to
stitch the output into an animation or video timeline.

python generate_param_plots3D_radial_video.py \
  --csv datasets/ParamPlots/ParamPlots.csv \
  --prototype-id Baseline \
  --highlight-angles 0 180 \
  --shade-2d-highlight \
  --export-formats png \
  --zoom 1.4

python generate_param_plots3D_radial_video.py \
  --csv datasets/ParamPlots/test.csv \
  --prototype-id bend_collapse5 \
  --highlight-angles 0 180 \
  --hide-line-angles 5 10 15 20 25 35 40 45 50 55 65 70 75 80 85 95 100 105 110 115 125 130 135 140 145 155 160 165 170 175 185 190 195 200 205 215 220 225 230 235 245 250 255 260 265 275 280 285 290 295 305 310 315 320 325 335 340 345 350 355\
  --shade-2d-highlight \
  --shade-2d-alpha 0.5 \
  --hold-surface-colors \
  --surface-alpha 0.3 \
  --export-formats png

ffmpeg \
  -framerate 15 \
  -pattern_type glob \
  -i "prototype_plots/video_frames/*.png" \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -pix_fmt yuv420p \
  -c:v libx264 \
  -r 15 \
  report/videos/3d_generation.mp4

ffmpeg -f concat -safe 0 -i report/videos/video_order.txt -c copy final_video.mp4

"""

# generate_param_plots3D_radial_video.py --csv <path> --prototype-id <id> --export-formats png

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "numpy is required for 3D plotting. Install it with `pip install numpy`."
    ) from exc

from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

from core.config import BaselineGeometryConfig, BendSettings
from core.generate_geometry import generate_geometry, generate_geometry_bend
from core.param_builder import build_params_from_config_csv
from io_modules.read_csv import read_param_rows_csv

CSV_PATH = "datasets/ParamPlots/ParamPlots.csv"
DEFAULT_OUTPUT_FOLDER = "prototype_plots/video_frames"
DEFAULT_FRAME_FILENAME_SUFFIX = "radial3d_stage"
DEFAULT_CMAP = "viridis"
COLOR_PALETTE = ["#BC4A87", "#b22f82", "#9836A5", "#732EAF", "#4C2BC1", "#233196", "#205097", "#233196", "#4C2BC1", "#732EAF", "#9836A5", "#b22f82", "#BC4A87"]
DEFAULT_DIM_ALPHA = 0.75
DEFAULT_DIM_LINEWIDTH_FACTOR = 0.5
DEFAULT_2D_SHADE_ALPHA = 0.12
DIM_LINESTYLE = (0, (5, 4))
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

# Optional manual axis bounds for consistent video framing.
# Set each to a (min, max) tuple in mm, or leave as None to auto-compute.
STATIC_X_LIMITS: Tuple[float, float] | None = (-31.0, 31.0)
STATIC_Y_LIMITS: Tuple[float, float] | None = (-31.0, 31.0)
STATIC_Z_LIMITS: Tuple[float, float] | None = (-23.0, 39.0)

# Extra padding (fraction of the span) added when auto-calculating bounds.
GLOBAL_LIMIT_MARGIN = 0.05

VIEW_DISTANCE = 9.0


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


def _set_axes_equal(ax, zoom: float = 1.0) -> None:
    """Force an equal aspect ratio for a 3D axis with optional zoom."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    span = limits[:, 1] - limits[:, 0]
    zoom = max(float(zoom), 1e-6)
    half = (max(span) / 2.0 if span.size else 1.0) / zoom
    centers = np.mean(limits, axis=1)
    ax.set_xlim3d(centers[0] - half, centers[0] + half)
    ax.set_ylim3d(centers[1] - half, centers[1] + half)
    ax.set_zlim3d(centers[2] - half, centers[2] + half)


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
    rect = Rectangle(
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


def _apply_axes_limits(ax, *, zoom: float, axes_limits: Dict[str, Tuple[float, float]] | None) -> None:
    """Set axis limits consistently across frames, honouring optional zoom."""
    zoom = max(float(zoom), 1e-6)
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
            lower, upper = limits
            if not math.isfinite(lower) or not math.isfinite(upper):
                continue
            center = 0.5 * (lower + upper)
            half = max((upper - lower) / 2.0, 1e-6) / zoom
            setter(center - half, center + half)

        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))
        else:  # pragma: no cover - legacy Matplotlib fallback
            current_limits = {
                "x": ax.get_xlim3d(),
                "y": ax.get_ylim3d(),
                "z": ax.get_zlim3d(),
            }
            half = max((lim[1] - lim[0]) / 2.0 for lim in current_limits.values())
            for axis in ("x", "y", "z"):
                limits = current_limits[axis]
                center = 0.5 * (limits[0] + limits[1])
                setters[axis](center - half, center + half)
        return

    _set_axes_equal(ax, zoom=zoom)


def _plot_sections_radially(
    ax,
    sections: Sequence[List[Tuple[float, float]]],
    angles: Sequence[float],
    cmap_name: str,
    fill_surfaces: bool,
    surface_alpha: float,
    line_alpha: float,
    linewidth: float,
    highlight_angles: Sequence[float],
    angle_tolerance: float,
    hide_line_angles: Sequence[float],
    dim_alpha: float,
    dim_linewidth_factor: float,
    shade_2d_all: bool,
    shade_2d_highlight: bool,
    shade_2d_alpha: float,
    hold_surface_colors: bool,
    zoom: float,
    axes_limits: Dict[str, Tuple[float, float]] | None = None,
) -> None:
    """Plot each cross-section at its angular placement and optionally shade surfaces."""
    if not sections:
        raise ValueError("No cross-sections supplied for plotting.")

    rotated_sections: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    highlight_handles = []

    cmap = plt.get_cmap(cmap_name)
    highlight_flags = [False] * len(angles)
    hide_line_flags = [False] * len(angles)
    highlight_angles = [float(a) for a in highlight_angles]
    hide_line_angles = [float(a) for a in hide_line_angles]
    angle_tolerance = max(angle_tolerance, 0.0)

    for idx, angle in enumerate(angles):
        highlight_flags[idx] = any(abs(angle - target) <= angle_tolerance for target in highlight_angles)
        hide_line_flags[idx] = any(abs(angle - target) <= angle_tolerance for target in hide_line_angles)

    total_sections = len(sections)

    color_indices: List[int] = []
    current_color_idx = -1
    next_color_idx = 0

    for idx in range(total_sections):
        hide_line = hide_line_flags[idx]
        if not hold_surface_colors:
            color_idx = idx
        else:
            if current_color_idx < 0:
                current_color_idx = next_color_idx
            if not hide_line:
                current_color_idx = next_color_idx
                next_color_idx += 1
            color_idx = current_color_idx
        color_indices.append(color_idx)

    color_count = max(color_indices) + 1 if color_indices else 0

    def color_for_index(order: int) -> Tuple[float, float, float, float]:
        if COLOR_PALETTE:
            base = COLOR_PALETTE[order % len(COLOR_PALETTE)] if COLOR_PALETTE else "#999999"
            rgb = np.array(mcolors.to_rgb(base))
        else:
            denom = max(color_count - 1, 1)
            rgb = np.array(cmap(order / denom)[:3])
        return (*rgb, 1.0)

    for idx, (points, angle) in enumerate(zip(sections, angles)):
        color_idx = color_indices[idx]
        color = color_for_index(color_idx if color_idx >= 0 else 0)
        x3d, y3d, z3d = _rotate_section(points, angle)
        rotated_sections.append((x3d, y3d, z3d))
        is_highlight = highlight_flags[idx]
        hide_line = hide_line_flags[idx]

        if is_highlight and not hide_line:
            line_style = "-"
            lw = linewidth
            alpha = line_alpha
            label = f"{angle:.1f}°"
        else:
            line_style = DIM_LINESTYLE
            lw = max(0.1, linewidth * dim_linewidth_factor)
            alpha = max(dim_alpha, 0.05)
            label = "_nolegend_"

        line_list = None
        if not hide_line:
            line_list = ax.plot(
                x3d,
                y3d,
                z3d,
                color=color,
                linestyle=line_style,
                linewidth=lw,
                alpha=alpha,
                label=label,
            )
        if is_highlight and line_list:
            highlight_handles.append(line_list[0])

        shade_this_section = shade_2d_all or (shade_2d_highlight and is_highlight)
        if shade_this_section:
            polygon = list(zip(x3d, y3d, z3d))
            poly = Poly3DCollection(
                [polygon],
                facecolors=[(*color[:3], shade_2d_alpha)],
                linewidths=0.0,
                edgecolors="none",
                alpha=shade_2d_alpha,
            )
            ax.add_collection3d(poly)

    if fill_surfaces and len(rotated_sections) >= 2:
        faces: List[List[Tuple[float, float, float]]] = []
        face_colors: List[Tuple[float, float, float, float]] = []
        for idx in range(len(rotated_sections) - 1):
            x_a, y_a, z_a = rotated_sections[idx]
            x_b, y_b, z_b = rotated_sections[idx + 1]

            if len(x_a) != len(x_b):
                continue  # Skip mismatched discretisation

            color_idx = color_indices[idx] if idx < len(color_indices) else idx
            base_color = color_for_index(color_idx)[:3]
            color = (*base_color, surface_alpha)
            for pt_idx in range(len(x_a) - 1):
                quad = [
                    (x_a[pt_idx], y_a[pt_idx], z_a[pt_idx]),
                    (x_a[pt_idx + 1], y_a[pt_idx + 1], z_a[pt_idx + 1]),
                    (x_b[pt_idx + 1], y_b[pt_idx + 1], z_b[pt_idx + 1]),
                    (x_b[pt_idx], y_b[pt_idx], z_b[pt_idx]),
                ]
                faces.append(quad)
                face_colors.append(color)

        if faces:
            poly = Poly3DCollection(faces, facecolors=face_colors, linewidths=0.0)
            ax.add_collection3d(poly)

    legend = None
    if highlight_handles:
        legend = ax.legend(
            handles=highlight_handles,
            labels=[h.get_label() for h in highlight_handles],
            loc="upper right",
            fontsize=11,
            frameon=True,
            framealpha=0.9,
            edgecolor="#4A4A4A",
            facecolor="#FFFFFF",
        )
    if legend:
        legend.get_frame().set_linewidth(0.8)

    ax.set_facecolor("#FAFAFA")
    _apply_axes_limits(ax, zoom=zoom, axes_limits=axes_limits)
    ax.grid(True, alpha=0.35, linestyle=":", linewidth=0.8)
    ax.tick_params(axis="both", which="major", length=5, width=1.2)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)
    ax.minorticks_on()
    if hasattr(ax, "spines"):
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#333333")
    ax.set_xlabel("X [mm]", fontweight="medium", labelpad=4)
    ax.set_ylabel("Y [mm]", fontweight="medium", labelpad=4)
    ax.set_zlabel("Z [mm]", fontweight="medium", labelpad=4)


def generate_radial_video_frames_for_configuration(
    csv_path: str,
    prototype_id: str | None = None,
    row_index: int | None = None,
    *,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    output_name: str | None = None,
    overwrite: bool = True,
    frame_padding: int | None = None,
    cmap: str = DEFAULT_CMAP,
    fill_surfaces: bool = True,
    surface_alpha: float = 0.25,
    line_alpha: float = 0.95,
    linewidth: float = 1.6,
    view_elev: float = 25.0,
    view_azim: float = -45.0,
    export_formats: Sequence[str] = ("png",),
    show: bool = False,
    highlight_angles: Sequence[float] = (),
    angle_tolerance: float = 1e-3,
    hide_line_angles: Sequence[float] = (),
    dim_alpha: float = DEFAULT_DIM_ALPHA,
    dim_linewidth_factor: float = DEFAULT_DIM_LINEWIDTH_FACTOR,
    shade_2d_all: bool = False,
    shade_2d_highlight: bool = False,
    shade_2d_alpha: float = DEFAULT_2D_SHADE_ALPHA,
    hold_surface_colors: bool = False,
    zoom: float = 1.0,
) -> List[str]:
    """Load a single configuration, build its geometry, and emit staged radial plots."""
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

    if not output_name:
        output_name = f"{proto_id}_{DEFAULT_FRAME_FILENAME_SUFFIX}"

    plt.rcParams.update(RC_PARAMS)

    auto_limits = _compute_global_limits(sections, angles)
    axes_limits = _resolve_axes_limits(auto_limits)
    if not axes_limits:
        axes_limits = None

    export_formats = tuple(dict.fromkeys(str(ext).lower().lstrip(".") for ext in export_formats if ext))
    output_dir = os.path.join(os.getcwd(), output_folder)
    os.makedirs(output_dir, exist_ok=True)

    total_frames = len(sections)
    if total_frames == 0:
        return []

    padding = frame_padding if frame_padding and frame_padding > 0 else max(3, len(str(total_frames)))
    saved_paths: List[str] = []

    for stage_idx in range(1, total_frames + 1):
        fig = plt.figure(figsize=(COLUMN_WIDTH, ROW_HEIGHT))
        ax = fig.add_subplot(111, projection="3d")
        ax.dist = VIEW_DISTANCE
        # ax.set_title("C: 3D Model Generation", pad=10, fontweight="bold", loc="left")
        ax.view_init(elev=view_elev, azim=view_azim)

        stage_sections = sections[:stage_idx]
        stage_angles = angles[:stage_idx]

        _plot_sections_radially(
            ax,
            stage_sections,
            stage_angles,
            cmap_name=cmap,
            fill_surfaces=fill_surfaces,
            surface_alpha=surface_alpha,
            line_alpha=line_alpha,
            linewidth=linewidth,
            highlight_angles=highlight_angles,
            angle_tolerance=angle_tolerance,
            hide_line_angles=hide_line_angles,
            dim_alpha=dim_alpha,
            dim_linewidth_factor=dim_linewidth_factor,
            shade_2d_all=shade_2d_all,
            shade_2d_highlight=shade_2d_highlight,
            shade_2d_alpha=shade_2d_alpha,
            hold_surface_colors=hold_surface_colors,
            zoom=zoom,
            axes_limits=axes_limits,
        )

        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=0.9)
        _add_figure_border(fig, ax)

        for ext in export_formats:
            if ext not in {"png", "svg", "pdf"}:
                continue
            filename = f"{output_name}_{stage_idx:0{padding}d}.{ext}"
            path = os.path.join(output_dir, filename)
            if not overwrite and os.path.exists(path):
                raise FileExistsError(f"{path} already exists; rerun with --overwrite to replace it.")
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            angle_label = stage_angles[-1] if stage_angles else 0.0
            print(f"Saved frame {stage_idx}/{total_frames} ({angle_label:.1f}°) to: {path}")
            saved_paths.append(path)

        if show and stage_idx == total_frames:
            plt.show()
        plt.close(fig)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate staged 3D radial plots for a single prototype configuration."
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
        help="Filename stem for exported frames (defaults to '<prototype>_radial3d_stage').",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing files.",
    )
    parser.add_argument(
        "--frame-padding",
        type=int,
        help="Zero padding width for frame numbering (default scales with number of sections).",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help="Matplotlib colormap for angular slices (default: %(default)s).",
    )
    parser.add_argument(
        "--no-fill",
        action="store_true",
        help="Disable surface shading between neighbouring cross-sections.",
    )
    parser.add_argument(
        "--surface-alpha",
        type=float,
        default=0.15,
        help="Opacity for the optional shaded surfaces.",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=0.95,
        help="Opacity for cross-section outlines.",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=1.6,
        help="Line width for cross-section outlines.",
    )
    parser.add_argument(
        "--view-elev",
        type=float,
        default=25.0,
        help="Elevation angle for the 3D view.",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=-45.0,
        help="Azimuth angle for the 3D view.",
    )
    parser.add_argument(
        "--export-formats",
        nargs="+",
        default=("png",),
        help="File extensions to export (subset of png/svg/pdf).",
    )
    parser.add_argument(
        "--shade-2d-all",
        action="store_true",
        help="Fill each cross-section polygon with a translucent color.",
    )
    parser.add_argument(
        "--shade-2d-highlight",
        action="store_true",
        help="Fill only highlighted cross-sections.",
    )
    parser.add_argument(
        "--shade-2d-alpha",
        type=float,
        default=DEFAULT_2D_SHADE_ALPHA,
        help="Opacity for the per-section fills.",
    )
    parser.add_argument(
        "--hold-surface-colors",
        action="store_true",
        help="Keep the same surface band color across hidden angular sections.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Zoom factor for the 3D axes (>1 zooms in, <1 zooms out).",
    )
    parser.add_argument(
        "--highlight-angles",
        nargs="+",
        type=float,
        default=(),
        help="Angles (deg) to highlight with solid lines.",
    )
    parser.add_argument(
        "--hide-line-angles",
        nargs="+",
        type=float,
        default=(),
        help="Angles (deg) whose outlines should be hidden while keeping surfaces.",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance (deg) when matching highlight angles to available sections.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the final plot interactively after rendering (requires GUI backend).",
    )

    args = parser.parse_args()
    generate_radial_video_frames_for_configuration(
        csv_path=args.csv,
        prototype_id=args.prototype_id,
        row_index=args.row_index,
        output_folder=args.output_folder,
        output_name=args.output_name,
        overwrite=not args.no_overwrite,
        frame_padding=args.frame_padding,
        cmap=args.cmap,
        fill_surfaces=not args.no_fill,
        surface_alpha=args.surface_alpha,
        line_alpha=args.line_alpha,
        linewidth=args.linewidth,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
        export_formats=args.export_formats,
        show=args.show,
        highlight_angles=args.highlight_angles,
        angle_tolerance=args.angle_tolerance,
        hide_line_angles=args.hide_line_angles,
        shade_2d_all=args.shade_2d_all,
        shade_2d_highlight=args.shade_2d_highlight,
        shade_2d_alpha=args.shade_2d_alpha,
        hold_surface_colors=args.hold_surface_colors,
        zoom=args.zoom,
    )


if __name__ == "__main__":
    main()
