# 2D cross-sections with shading
# python generate_param_plots.py --csv datasets/ParamPlots/ParamPlots2D.csv --plot-dimension 2d --shade-2d

# 1D curve comparison
# python generate_param_plots1D.py --csv datasets/ParamPlots/ParamPlots1D.csv --plot-dimension 1d --show-control-points --stack-variants

from __future__ import annotations

import math
import re
import ast
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc

from core.config import BaselineGeometryConfig
from core.config import BaselineGeometryConfigPlot7
from core.generate_geometry import generate_geometry, generate_geometry_bend
from core.param_builder import build_params_from_config_csv
from io_modules.exporting import export_plot
from io_modules.read_csv import read_param_rows_csv


CSV_PATH = "datasets/ParamPlots/ParamPlots.csv"
DEFAULT_PLOT_FOLDER = "prototype_plots"
DEFAULT_PLOT_FILENAME = "param_variants"

TITLE = "Geometric Impacts of 1D Parameters"

_GROUP_RE = re.compile(r"^(?P<name>[A-Za-z]+?)(?P<idx>\d+)$")
_LOW_LABEL = {1: "low", 2: "high"}
PARAM_LABELS = {
    "AMP": "Amplitude (AMP)",
    "RAD": "Radius (RAD)",
    "NOC": "Number of Curves (NOC)",
    "MYP": "Amplitude Scaling (MYP)",
    "PRD": "Period Scaling (PRD)",
    "XOF": "Peak/Valley CP X-Offset (XOF)",
    "XMF": "Centre CP X-Offset (XMF)",
    "CWT": "Curve Weight (CWT)",
    "THF": "Thickness Mode: Variable (THF)",
    "CPS": "Thickness Mode: Collapse (CPS)",
    "SBD": "Thickness Mode: S-Bending (SBD)",
    "THV": "Thickness Value (THV)",
}
# COLOR_PALETTE = [ '#A23B72',"#021720",]  # Blue and Rose colors
# COLOR_PALETTE = ["#BC4A87", "#0a75a3"]
COLOR_PALETTE = ['#A4C8E1' , '#145DA0']
# COLOR_PALETTE = ['#FF12F3', '#1241ff']
LEGEND_PAD = 0.3  # additional headroom (fraction of y-range) reserved for subplot legends

COLUMNS = 4
columns = COLUMNS

THICKNESS_FACTOR = 1.2
CONTROL_POINT_MARKER_SIZE = 40
CONTROL_POINT_BASELINE_EDGE_COLOR = "#FFFFFF"
CONTROL_POINT_VARIANT_EDGE_COLOR = "#0E0E0E"
CONTROL_POINT_BASE_COLOR = "#505050"
CONTROL_POINT_MARKER_BASELINE = "o"
CONTROL_POINT_MARKER_VARIANT = "o"
CONTROL_POINTS_PER_SECTION = 5
CAP_CONTROL_POINTS = 3  # number of initial cap control points to skip when plotting
STACKED_GROUP_GAP_RATIO = 0.25

INCLUDE_BASELINE = True


@dataclass
class ConfigPlotData:
    """Container for the sampled geometry points of a single config."""
    name: str
    points: List[Tuple[float, float]]
    control_points: List[Tuple[float, float]] | None = None
    highlight_control_points: List[int] | None = None
    first_and_last_only: bool = False


def _normalize_highlight_indices(raw_value) -> List[int] | None:
    """Return a sanitized list of 1-based control point indices to highlight."""
    if raw_value is None:
        return None
    candidates: Iterable = ()
    if isinstance(raw_value, (list, tuple, set)):
        candidates = raw_value
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return None
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            candidates = parsed
        else:
            candidates = (piece.strip() for piece in stripped.split(",") if piece.strip())
    else:
        candidates = (raw_value,)

    normalized: List[int] = []
    for candidate in candidates:
        try:
            idx = int(candidate)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= CONTROL_POINTS_PER_SECTION:
            normalized.append(idx)
    if not normalized:
        return []
    # Deduplicate while preserving order
    seen = set()
    deduped: List[int] = []
    for idx in normalized:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def _normalize_points(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """Return a clean list of (x, y) tuples extracted from geometry output."""
    norm: List[Tuple[float, float]] = []

    try:
        import numpy as np  # type: ignore
    except ImportError:  # pragma: no cover
        np = None  # type: ignore

    def _is_scalar(val) -> bool:
        scalar_types = (int, float)
        if np is not None:
            scalar_types = scalar_types + (np.generic,)  # type: ignore
        return isinstance(val, scalar_types)

    def iter_points(obj):
        if obj is None:
            return
        if np is not None and isinstance(obj, np.ndarray):
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
            yield (getattr(obj, "x"), getattr(obj, "y"))
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
        if item is None:
            continue
        if len(item) < 2:
            continue
        x, y = item[0], item[1]
        norm.append((float(x), float(y)))

    if not norm:
        raise TypeError(f"Unsupported point format: {points!r}")

    return norm


def _ensure_closed(
    points: Iterable[Tuple[float, float]],
    *,
    close: bool = True,
) -> Tuple[List[float], List[float]]:
    """Return X, Y coordinate lists, optionally closing the polygon."""
    xs, ys = [], []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    if not xs:
        return xs, ys
    if close and (not math.isclose(xs[0], xs[-1], rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(ys[0], ys[-1], rel_tol=1e-9, abs_tol=1e-9)):
        xs.append(xs[0])
        ys.append(ys[0])
    return xs, ys


def _max_x_coordinate(points: Sequence[Tuple[float, float]]) -> float:
    """Return the maximum X coordinate from a set of points."""
    if not points:
        raise ValueError("Cannot determine X-extent from an empty point set.")
    return max(x for x, _ in points)


def _align_entry_to_xmax(entry: ConfigPlotData, target_max_x: float) -> ConfigPlotData:
    """Translate an entry so its rightmost X matches the target maximum."""
    if not entry.points:
        return ConfigPlotData(
            entry.name,
            [],
            entry.control_points,
            entry.highlight_control_points,
        )

    current_max = max(x for x, _ in entry.points)
    shift = target_max_x - current_max
    if math.isclose(shift, 0.0, abs_tol=1e-9):
        shift = 0.0

    if shift == 0.0:
        shifted_points = entry.points
        shifted_control = entry.control_points
    else:
        shifted_points = [(x + shift, y) for x, y in entry.points]
        shifted_control = (
            [(x + shift, y) for x, y in entry.control_points]
            if entry.control_points
            else None
        )

    return ConfigPlotData(
        entry.name,
        shifted_points,
        shifted_control,
        entry.highlight_control_points,
        entry.first_and_last_only,
    )


def _build_stacked_height_ratios(
    group_rows: int,
    stacked_total: int,
    gap_ratio: float,
) -> Tuple[List[float], Set[int]]:
    """Create per-row height ratios and record spacer rows for stacked layouts."""
    ratios: List[float] = []
    gap_rows: Set[int] = set()
    if group_rows <= 0 or stacked_total <= 0:
        return ratios, gap_rows

    for block_idx in range(group_rows):
        ratios.extend([1.0] * stacked_total)
        if block_idx < group_rows - 1:
            gap_row_idx = len(ratios)
            ratios.append(gap_ratio)
            gap_rows.add(gap_row_idx)
    return ratios, gap_rows


def _plot_control_points(
    ax,
    points: List[Tuple[float, float]] | None,
    highlight_indices: List[int] | None,
    *,
    marker: str,
    highlight_marker: str | None,
    highlight_color: str,
    base_color: str,
    edge_color: str = CONTROL_POINT_VARIANT_EDGE_COLOR,
    base_size: float = CONTROL_POINT_MARKER_SIZE * 0.7,
    highlight_size: float = CONTROL_POINT_MARKER_SIZE,
    base_alpha: float = 0.2,
    highlight_alpha: float = 0.9,
    edge_width: float = 0.6,
    zorder: float = 4.0,
    first_and_last_only: bool = False,
) -> None:
    """Scatter control points, emphasizing selected indices per 5-point section."""
    if not points:
        return

    if first_and_last_only:
        first_idx = 0
        last_idx = len(points) - 1

        base_x: List[float] = []
        base_y: List[float] = []
        highlight_x: List[float] = []
        highlight_y: List[float] = []

        for idx, (x, y) in enumerate(points):
            if idx == first_idx or idx == last_idx:
                highlight_x.append(x)
                highlight_y.append(y)
            else:
                base_x.append(x)
                base_y.append(y)

        if base_x:
            ax.scatter(
                base_x,
                base_y,
                s=base_size,
                marker=marker,
                facecolors=base_color,
                edgecolors=edge_color,
                linewidths=edge_width,
                alpha=base_alpha,
                zorder=zorder,
                label=None,
            )
        if highlight_x:
            ax.scatter(
                highlight_x,
                highlight_y,
                s=highlight_size,
                marker=highlight_marker or marker,
                facecolors=highlight_color,
                edgecolors=edge_color,
                linewidths=edge_width,
                alpha=highlight_alpha,
                zorder=zorder + 0.5,
                label=None,
            )
        return

    processed_points: List[Tuple[float, float]] = []
    for idx, point in enumerate(points):
        if idx < CAP_CONTROL_POINTS:
            continue
        processed_points.append(point)
    points = processed_points

    highlight_mask = None
    if highlight_indices is not None:
        highlight_mask = {
            idx - 1 for idx in highlight_indices if 1 <= idx <= CONTROL_POINTS_PER_SECTION
        }

    base_x: List[float] = []
    base_y: List[float] = []
    highlight_x: List[float] = []
    highlight_y: List[float] = []

    for idx, (x, y) in enumerate(points):
        pos = idx % CONTROL_POINTS_PER_SECTION
        is_highlight = True if highlight_mask is None else pos in highlight_mask
        if is_highlight:
            highlight_x.append(x)
            highlight_y.append(y)
        else:
            base_x.append(x)
            base_y.append(y)

    if base_x:
        ax.scatter(
            base_x,
            base_y,
            s=base_size,
            marker=marker,
            facecolors=base_color,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=base_alpha,
            zorder=zorder,
            label=None,
        )
    if highlight_x:
        ax.scatter(
            highlight_x,
            highlight_y,
            s=highlight_size,
            marker=highlight_marker or marker,
            facecolors=highlight_color,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=highlight_alpha,
            zorder=zorder + 0.5,
            label=None,
        )


def _extract_plot_points(
    top_row: Dict[str, object],
    baseline: BaselineGeometryConfig,
    plot_dimension: str,
) -> ConfigPlotData:
    """Build geometry for a single row and return the 0° data for the requested dimension."""
    params, config_csv_path, use_linear_fast = build_params_from_config_csv(top_row, baseline)
    proto_id = params.export_filename or "unnamed"

    if use_linear_fast:
        params.bending_enabled = False
        report = generate_geometry(params)
        if plot_dimension == "2d":
            raw_points = report.xsections2d.twoD_cross_section
            control_raw = getattr(report.curves1d, "control_points", None)
        elif plot_dimension == "1d":
            raw_points = report.curves1d.curve_points
            control_raw = getattr(report.curves1d, "control_points", None)
        else:
            raise ValueError(f"Unsupported plot dimension: {plot_dimension}")
    else:
        params.bending_enabled = True
        report = generate_geometry_bend(params, config_csv_path, testing_mode=False)
        if plot_dimension == "2d":
            if not report.xsections2d_list:
                raise ValueError(f"No 2D cross-section data for {proto_id}")
            raw_points = report.xsections2d_list[0].twoD_cross_section
            control_raw = getattr(report.curves1d_list[0], "control_points", None) if report.curves1d_list else None
        elif plot_dimension == "1d":
            if not report.curves1d_list:
                raise ValueError(f"No 1D curve data for {proto_id}")
            raw_points = report.curves1d_list[0].curve_points
            control_raw = getattr(report.curves1d_list[0], "control_points", None)
        else:
            raise ValueError(f"Unsupported plot dimension: {plot_dimension}")

    control_points: List[Tuple[float, float]] | None = None
    if control_raw is not None:
        try:
            control_points = _normalize_points(control_raw)
        except TypeError:
            control_points = None

    highlight_indices = _normalize_highlight_indices(top_row.get("highlight_control_points"))
    raw_first_last = top_row.get("first_and_last_only")
    if isinstance(raw_first_last, bool):
        first_and_last_only = raw_first_last
    elif raw_first_last is None:
        first_and_last_only = False
    else:
        first_and_last_only = str(raw_first_last).strip().lower() in {"1", "true", "yes", "y"}

    return ConfigPlotData(
        proto_id,
        _normalize_points(raw_points),
        control_points,
        highlight_indices,
        first_and_last_only,
    )


def _group_variants(config_data: List[ConfigPlotData]) -> Tuple[ConfigPlotData, "OrderedDict[str, List[ConfigPlotData]]"]:
    """Return the baseline config and an ordered mapping of parameter family -> configs."""
    baseline_entry = None
    baseline_entry7 = None
    grouped: "OrderedDict[str, List[ConfigPlotData]]" = OrderedDict()

    for entry in config_data:
        key = entry.name.strip()
        if key.lower() == "baseline":
            baseline_entry = entry
            continue
        elif key.lower() == "baseline7":
            baseline_entry7 = entry
            continue

        match = _GROUP_RE.match(key)
        if not match:
            continue

        group = match.group("name")
        grouped.setdefault(group, [])
        grouped[group].append(entry)

    if baseline_entry is None:
        raise ValueError("Baseline configuration not found in CSV.")

    # sort each group's entries by numeric suffix so '1' precedes '2'
    for entries in grouped.values():
        entries.sort(
            key=lambda e: int(_GROUP_RE.match(e.name).group("idx")) if _GROUP_RE.match(e.name) else e.name
        )

    # return baseline_entry, grouped
    return baseline_entry, grouped, baseline_entry7 

def _format_group_title(group_name: str, subplot_idx: int) -> str:
    """Create a subplot title using alphabetic labels and full parameter names when possible."""
    label_char = chr(ord("A") + subplot_idx)
    display_name = PARAM_LABELS.get(group_name.upper(), group_name)
    return f"{label_char}: {display_name}"


def _compute_local_limits(
    baseline: ConfigPlotData,
    variants: List[ConfigPlotData],
    plot_dimension: str,
    include_control_points: bool,
    include_baseline: bool = True,
    stack_mode: bool = False,
    group_name: str = "",
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute per-subplot axis limits."""
    close_polygons = plot_dimension == "2d"
    xs_all: List[float] = []
    ys_all: List[float] = []
    target_max_x = _max_x_coordinate(baseline.points)

    def extend_entry(entry: ConfigPlotData) -> None:
        """Add points from a single entry to the overall limits calculation."""
        xs, ys = _ensure_closed(entry.points, close=close_polygons)
        xs_all.extend(xs)
        ys_all.extend(ys)
        if include_control_points and entry.control_points:
            for x, y in entry.control_points:
                xs_all.append(x)
                ys_all.append(y)

    aligned_baseline = _align_entry_to_xmax(baseline, target_max_x)
    if include_baseline:
        extend_entry(aligned_baseline)
    for entry in variants:
        extend_entry(_align_entry_to_xmax(entry, target_max_x))

    if not xs_all or not ys_all:
        raise ValueError("Unable to compute axis limits; geometry data is empty.")

    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)

    base_pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    base_pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    pad_x = base_pad_x
    if not stack_mode:
        pad_y = base_pad_y + LEGEND_PAD * (y_max - y_min if y_max > y_min else 1.0)
        pad_y_lower = base_pad_y + 0.1 * (y_max - y_min if y_max > y_min else 1.0)
    else:
        if group_name == "AMP":
            pad_y = base_pad_y + LEGEND_PAD/2 * (y_max - y_min if y_max > y_min else 1.0)
            pad_y_lower = base_pad_y
        else:
            pad_y = base_pad_y + LEGEND_PAD*1.5 * (y_max - y_min if y_max > y_min else 1.0)
            pad_y_lower = base_pad_y + 0.05 * (y_max - y_min if y_max > y_min else 1.0)
       

    return (x_min - pad_x, x_max + pad_x), (y_min - pad_y_lower, y_max + pad_y)


def _plot_group_axes(
    ax,
    baseline: ConfigPlotData,
    variants: List[ConfigPlotData],
    group_name: str,
    subplot_idx: int,
    plot_dimension: str,
    shade_2d: bool,
    show_control_points: bool,
    stacked_mode: bool,
    stacked_position: int,
    stacked_total: int,
    shared_limits: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
) -> None:
    """Render a single subplot comparing baseline vs. parameter variants."""
    close_polygons = plot_dimension == "2d"
    target_max_x = _max_x_coordinate(baseline.points)
    baseline_aligned = _align_entry_to_xmax(baseline, target_max_x)
    aligned_variants = [
        _align_entry_to_xmax(variant, target_max_x) for variant in variants
    ]

    xs_base, ys_base = _ensure_closed(baseline_aligned.points, close=close_polygons)
    if not xs_base:
        raise ValueError("Baseline geometry is empty.")

    if INCLUDE_BASELINE:
        # Baseline in dashed black so variants always have a reference
        ax.plot(
            xs_base,
            ys_base,
            # color="#777777",
            color='#021720',
                linestyle=(0, (6, 4)),
                linewidth=1.2,
                label="baseline",
            )

    baseline_first_last = baseline_aligned.first_and_last_only
    if not baseline_first_last:
        baseline_first_last = any(
            variant_aligned.first_and_last_only for variant_aligned in aligned_variants
        )
    baseline_highlights = (
        None if baseline_first_last else baseline_aligned.highlight_control_points
    )
    if baseline_highlights is None and not baseline_first_last:
        for variant_aligned in aligned_variants:
            if variant_aligned.first_and_last_only:
                continue
            if variant_aligned.highlight_control_points is not None:
                baseline_highlights = variant_aligned.highlight_control_points
                break

    color_cycle = COLOR_PALETTE

    for idx, variant_aligned in enumerate(aligned_variants):
        xs_var, ys_var = _ensure_closed(variant_aligned.points, close=close_polygons)
        if not xs_var:
            continue
        match = _GROUP_RE.match(variant_aligned.name)
        label_suffix = None
        if match:
            order = int(match.group("idx"))
            label_suffix = _LOW_LABEL.get(order)
        if not group_name == "MYP":
            label = f"{group_name} {label_suffix}" if label_suffix else variant_aligned.name
        else:
            if label_suffix == "low":
                label_suffix = "med"
            label = f"{group_name} {label_suffix}" if label_suffix else variant_aligned.name
        # param_title,   = _format_group_title(group_name, subplot_idx)
        if stacked_mode:
            color_index = stacked_position % len(color_cycle)
        else:
            color_index = idx % len(color_cycle)
        color = color_cycle[color_index]
        ax.plot(xs_var, ys_var, color=color, linewidth=1.8*THICKNESS_FACTOR, label=label)
        if plot_dimension == "2d" and shade_2d:
            ax.fill(xs_var, ys_var, color=color, alpha=0.15, linewidth=0.0)
        if show_control_points:
            # Variant control points
            _plot_control_points(
                ax,
                variant_aligned.control_points,
                variant_aligned.highlight_control_points,
                marker=CONTROL_POINT_MARKER_VARIANT,
                highlight_marker=CONTROL_POINT_MARKER_VARIANT,
                highlight_color=color,
                base_color=color,
                edge_color=CONTROL_POINT_VARIANT_EDGE_COLOR,
                base_alpha=0.28,
                highlight_alpha=0.95,
                zorder=4.5,
                first_and_last_only=variant_aligned.first_and_last_only,
            )
            # Baseline control points
            _plot_control_points(
                ax,
                baseline_aligned.control_points,
                variant_aligned.highlight_control_points,
                marker=CONTROL_POINT_MARKER_BASELINE,
                highlight_marker=CONTROL_POINT_MARKER_BASELINE,
                highlight_color=CONTROL_POINT_BASE_COLOR,
                base_color=CONTROL_POINT_BASE_COLOR,
                edge_color=CONTROL_POINT_BASELINE_EDGE_COLOR,
                base_size=CONTROL_POINT_MARKER_SIZE * 0.5,
                highlight_size=CONTROL_POINT_MARKER_SIZE * 0.75,
                base_alpha=0.01,
                highlight_alpha=0.75,
                zorder=4.0,
                first_and_last_only=baseline_first_last,
            )


    if not stacked_mode or stacked_position == 0:
        ax.set_title(_format_group_title(group_name, subplot_idx), fontsize=15, fontweight="bold", loc="left", pad=10)
    else:
        ax.set_title("")
    if not stacked_mode or stacked_position == stacked_total - 1:
        ax.set_xlabel("X [mm]", fontsize=14, fontweight="medium")
    else:
        ax.set_xlabel("")
    if stacked_mode and stacked_position > 0:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Y [mm]", fontsize=14, fontweight="medium", loc="bottom")
    if shared_limits is not None:
        x_limits, y_limits = shared_limits
    else:
        x_limits, y_limits = _compute_local_limits(
            baseline,
            variants,
            plot_dimension,
            show_control_points,
            include_baseline=INCLUDE_BASELINE,
            group_name=group_name,
            # stack_variants=stack_variants,
        )
    ax.set_aspect("equal")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_facecolor("#FAFAFA")

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)

    if stacked_mode and stacked_position < stacked_total - 1:
        ax.tick_params(labelbottom=False)

    legend = ax.legend(
        fontsize=11,
        loc="upper right" if not stacked_mode else "upper center",
        bbox_to_anchor=(1.0, 1.0) if not stacked_mode else (0.5, 1.02),
        framealpha=0.95,
        edgecolor="gray",
        frameon=True,
        shadow=False,
        borderpad=0.6,
        handletextpad=0.5,
        columnspacing=1.0,
        borderaxespad=0.6,
        ncol=1 if not stacked_mode else stacked_total,
    )
    legend.get_frame().set_linewidth(0.8)

    ax.tick_params(axis="both", which="major", labelsize=12, length=5, width=1.2)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#333333")


def generate_parameter_plots(
    csv_path: str = CSV_PATH,
    output_folder: str = DEFAULT_PLOT_FOLDER,
    output_filename: str = DEFAULT_PLOT_FILENAME,
    overwrite: bool = True,
    show: bool = False,
    plot_dimension: str = "2d",
    shade_2d: bool = False,
    show_control_points: bool = False,
    stack_variants: bool = False,
) -> None:
    """Generate comparison plots for parameter variants defined in a CSV.

    plot_dimension controls whether 1D curves or 2D cross-sections are plotted.
    Set shade_2d to True to fill the interior of 2D cross-sections.
    Enable show_control_points to overlay available NURBS control points.
    If the CSV row includes `highlight_control_points`, supply 1-based indices (1-5)
    to keep those control points emphasized within each 5-point section.
    Set `first_and_last_only` to True to draw only the first and final control
    points for that configuration.
    Enable stack_variants to split baseline and variants into vertically stacked sub-axes per parameter.
    """
    plot_dimension = plot_dimension.lower()
    if plot_dimension not in {"1d", "2d"}:
        raise ValueError("plot_dimension must be '1d' or '2d'.")
    shade_2d = shade_2d and plot_dimension == "2d"

    rows = read_param_rows_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    plt.rcParams.update({
        "font.size": 16,
        "font.family": "sans-serif",
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 18,
    })

    baseline_config = BaselineGeometryConfig()
    # baseline7_config = BaselineGeometryConfigPlot7()

    data: List[ConfigPlotData] = []
    for row in rows:
        # print(f"Processing row: {row}")
        # baseline_config_to_use = baseline_config if row.get('Prototype ID') != 'SBD1' else baseline7_config
        entry = _extract_plot_points(row, baseline_config, plot_dimension)
        data.append(entry)

    # print(data[3])

    # baseline_entry, grouped = _group_variants(data)
    baseline_entry, grouped, baseline_entry7 = _group_variants(data)

    if not grouped:
        raise ValueError("No parameter variants (suffix 1/2) found in CSV.")

    group_names = list(grouped.keys())
    n_groups = len(group_names)
    if not columns:
        cols = 3 if n_groups > 3 else n_groups
        cols = max(cols, 1)
    else:
        cols = columns

    rows_n = math.ceil(n_groups / cols)


    fig_width = max(11, 5.5 * cols)

    if not stack_variants:
        fig_height = max(10, 5.0 * rows_n)
        fig, axes = plt.subplots(rows_n, cols, figsize=(fig_width, fig_height), squeeze=False)
        axes_flat = axes.ravel()

        for subplot_idx, group_name in enumerate(group_names):
            ax = axes_flat[subplot_idx]
            variants = grouped[group_name]
            if group_name == "SBD":
                print(group_name)
                baseline_entry_to_use = baseline_entry7
            else:
                baseline_entry_to_use = baseline_entry
            _plot_group_axes(
                ax,
                baseline_entry_to_use,
                variants,
                group_name,
                subplot_idx,
                plot_dimension,
                shade_2d,
                show_control_points,
                stacked_mode=False,
                stacked_position=0,
                stacked_total=1,
            )

        # Hide unused axes
        for ax_idx in range(n_groups, len(axes_flat)):
            axes_flat[ax_idx].axis("off")
    # Stacked variant mode
    else:
        max_variants = max((len(v) for v in grouped.values()), default=0)
        stacked_total = max(1, max_variants)
        height_ratios, gap_row_indices = _build_stacked_height_ratios(
            rows_n, stacked_total, STACKED_GROUP_GAP_RATIO
        )
        if not height_ratios:
            height_ratios = [1.0]
        print(f"Height ratios: {height_ratios}")
        rows_total = len(height_ratios)
        fig_height = max(10, 3.6 * rows_total)
        gridspec_kw = {"height_ratios": height_ratios, "wspace": 0.2}
        fig, axes = plt.subplots(
            rows_total,
            cols,
            figsize=(fig_width, fig_height),
            sharex=True,
            squeeze=False,
            gridspec_kw=gridspec_kw,
        )

        def hide_axis(row_idx: int, col_idx: int) -> None:
            axes[row_idx][col_idx].axis("off")

        for gap_row in gap_row_indices:
            if gap_row < rows_total:
                for col_idx in range(cols):
                    hide_axis(gap_row, col_idx)

        rows_per_block = stacked_total + 1

        for group_idx, group_name in enumerate(group_names):
            col_idx = group_idx % cols
            block_number = group_idx // cols
            block_base = block_number * rows_per_block
            variants = grouped[group_name]
            if not variants:
                for offset in range(stacked_total):
                    target_row = block_base + offset
                    if target_row < rows_total:
                        hide_axis(target_row, col_idx)
                continue
            group_total = max(1, len(variants))
            shared_limits = _compute_local_limits(
                baseline_entry,
                variants,
                plot_dimension,
                show_control_points,
                include_baseline=INCLUDE_BASELINE,
                stack_mode=stack_variants,
                group_name=group_name
            )

            for offset, variant in enumerate(variants):
                target_row = block_base + offset
                if target_row >= rows_total:
                    continue
                ax = axes[target_row][col_idx]
                _plot_group_axes(
                    ax,
                    baseline_entry,
                    [variant],
                    group_name,
                    group_idx,
                    plot_dimension,
                    shade_2d,
                    show_control_points,
                    stacked_mode=True,
                    stacked_position=offset,
                    stacked_total=group_total,
                    shared_limits=shared_limits,
                )

            for extra in range(group_total, stacked_total):
                target_row = block_base + extra
                if target_row < rows_total:
                    hide_axis(target_row, col_idx)

        total_slots = rows_n * cols
        for group_idx in range(len(group_names), total_slots):
            col_idx = group_idx % cols
            block_number = group_idx // cols
            block_base = block_number * rows_per_block
            for offset in range(stacked_total):
                target_row = block_base + offset
                if target_row < rows_total:
                    hide_axis(target_row, col_idx)

    # fig.suptitle(TITLE, y=0.925, fontweight="bold", fontsize=20)
    if stack_variants:
        fig.tight_layout(pad=0.9)
        fig.subplots_adjust(hspace=0.0)
    else:
        fig.tight_layout(pad=1.2)

    # export_plot(
    #     fig,
    #     title=output_filename,
    #     export_type="svg",
    #     directory=".",
    #     folder=output_folder,
    #     overwrite=overwrite,
    # )

    save_path = os.path.join(os.getcwd(), 'prototype_plots/param_variants.png')
    plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white', edgecolor='none')
    save_path = os.path.join(os.getcwd(), 'prototype_plots/param_variants.svg')
    plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white', edgecolor='none')
    save_path = os.path.join(os.getcwd(), 'prototype_plots/param_variants.pdf')
    plt.savefig(save_path, dpi=500, bbox_inches='tight', facecolor='white', edgecolor='none')

    print(f"Plot saved to {output_folder}/{output_filename}.png/svg/pdf")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate comparison plots for parameter variants.")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to the CSV listing prototype configurations.")
    parser.add_argument("--output-folder", default=DEFAULT_PLOT_FOLDER, help="Folder to store the generated plot.")
    parser.add_argument("--output-name", default=DEFAULT_PLOT_FILENAME, help="Filename (without extension) for the plot.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite an existing plot file.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively after saving.")
    parser.add_argument(
        "--plot-dimension",
        choices=["1d", "2d"],
        default="2d",
        help="Choose whether to plot 1D curves or 2D cross-sections.",
    )
    parser.add_argument(
        "--shade-2d",
        action="store_true",
        help="Fill the interior of 2D cross-sections.",
    )
    parser.add_argument(
        "--stack-variants",
        action="store_true",
        help="Display baseline and variants for each parameter as vertically stacked axes instead of overlays.",
    )
    parser.add_argument(
        "--show-control-points",
        action="store_true",
        help="Overlay NURBS control points when available.",
    )

    args = parser.parse_args()
    generate_parameter_plots(
        csv_path=args.csv,
        output_folder=args.output_folder,
        output_filename=args.output_name,
        overwrite=not args.no_overwrite,
        show=args.show,
        plot_dimension=args.plot_dimension,
        shade_2d=args.shade_2d,
        show_control_points=args.show_control_points,
        stack_variants=args.stack_variants,
    )


if __name__ == "__main__":
    main()
