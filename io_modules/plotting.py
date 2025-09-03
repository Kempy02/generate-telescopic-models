# geometry/plotting.py

import math
import matplotlib.pyplot as plt
import numpy as np
import os

from io_modules.exporting import export_plot

def plot_curves(all_control_points, all_curve_points):
    """Visualize the 1D cross‐section curves & control points."""
    plt.close('all')
    plt.figure(figsize=(10, 6))
    plt.title('Sequential NURBS Curves')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    colors = ['b','g','r','c','m','y','k']

    for i, curve_points in enumerate(all_curve_points):
        color = colors[i % len(colors)]
        xvals = curve_points[:,0]
        yvals = curve_points[:,1]
        plt.plot(xvals, yvals, label=f'Curve {i+1}', color=color)

        # Control polygon
        control_pts = all_control_points[i]
        cp_x = [pt[0] for pt in control_pts]
        cp_y = [pt[1] for pt in control_pts]
        plt.plot(cp_x, cp_y, 'o--', label=f'Control {i+1}', color=color)

    plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
    # plt.show()


# ...existing code...
def plot_grouped_curves(plots_data):
    """
    Creates a single figure with subplots arranged in a grid.
      - One subplot per distinct variable (key in plots_data).
      - In each subplot, all prototypes that changed that variable are plotted.
      All plots share the same x and y scale.
    """
    if not plots_data:
        print("No plot data available.")
        return

    # Exclude specific variables from plotting
    exclude_keys = {"thickness", "thickness_factor"}
    filtered_data = {k: v for k, v in plots_data.items() if k not in exclude_keys}

    # Check if there's data left to plot
    if not filtered_data:
        print("No plot data available after excluding thickness-related variables.")
        return

    # Precompute global x and y limits
    global_xmin, global_xmax = float('inf'), float('-inf')
    global_ymin, global_ymax = float('inf'), float('-inf')
    for var_name, prototypes_list in filtered_data.items():
        for entry in prototypes_list:
            for pts in entry["curve_points"]:
                xs = pts[:, 0]
                ys = pts[:, 1]
                global_xmin = min(global_xmin, xs.min())
                global_xmax = max(global_xmax, xs.max())
                global_ymin = min(global_ymin, ys.min())
                global_ymax = max(global_ymax, ys.max())

    # Number of distinct variables (plus one extra subplot for layout consistency)
    n_vars = len(filtered_data) + 1

    # Compute rows/columns for a roughly square layout
    nrows = int(math.ceil(n_vars/2))
    ncols = int(math.ceil(n_vars/2))

    # Create the figure & axes array
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6*ncols, 4*nrows),
        squeeze=False
    )

    # Convert filtered_data.items() to a list so we can iterate with index
    items_list = list(filtered_data.items())

    # Define some colors
    colors = ['b','r','c','m','y','k']
    color_count = len(colors)

    for i, (var_name, prototypes_list) in enumerate(items_list):
        row = i // ncols
        col = i % ncols
        ax = axes[row][col]

        if i == 0:
            # Save the baseline (first) plot for reference on all subplots
            baseline_ax = ax

        # Plot each prototype in this variable group
        for idx, entry in enumerate(prototypes_list):
            proto_id = entry["proto_id"]
            val      = entry["value"]
            all_curves = entry["curve_points"]

            color = colors[idx % color_count]
            label_txt = f"{proto_id} (val={val})"

            for j, pts in enumerate(all_curves):
                xvals = pts[:, 0]
                yvals = pts[:, 1]
                if j == 0:
                    ax.plot(xvals, yvals, color=color, label=label_txt)
                else:
                    ax.plot(xvals, yvals, color=color)

        ax.set_title(f"Variable: {var_name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend(loc="best")

        # Set global limits to ensure same scale across all subplots
        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(global_ymin, global_ymax)

    # Hide unused subplots if n_vars < nrows*ncols
    total_subplots = nrows * ncols
    if n_vars < total_subplots:
        for empty_idx in range(n_vars, total_subplots):
            row = empty_idx // ncols
            col = empty_idx % ncols
            axes[row][col].set_visible(False)

    plt.tight_layout()
    # Show the figure
    # plt.show()

    oneD_fig = fig

    return oneD_fig
    


def plot_thickness_and_factors(plot_data):
    """
    Plots cross-section curves for any prototypes that varied the parameter
    'thickness' or 'thickness_factor', all in one figure with two subplots:
      - Subplot (1) for thickness
      - Subplot (2) for thickness_factor"
    """

    # Determine which keys exist
    has_thickness = ("thickness" in plot_data)
    has_thf       = ("thickness_factor" in plot_data)

    # If neither key is present, do nothing
    if not has_thickness and not has_thf:
        print("No thickness or thickness_factor data to plot.")
        return

    # Figure out how many subplots we need
    num_subplots = 0
    if has_thickness:
        num_subplots += 1
    if has_thf:
        num_subplots += 1

    # Create figure with subplots
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=num_subplots,
        figsize=(6*num_subplots, 6),  # 6" wide per subplot
        squeeze=False
    )
    # Because we forced squeeze=False, axes is 2D, so we'll access axes[0][0] or axes[0][1]

    # We'll keep a column index to fill
    col_index = 0

    # A simple color cycle for lines
    colors = ['b','r','c','m','y','k']
    color_count = len(colors)

    # ----- 1) Subplot for thickness -----
    if has_thickness:
        # gather entries from plots_data["thickness"]
        thickness_list = plot_data["thickness"]  # list of dicts
        ax = axes[0][col_index]

        # Plot each entry
        for i, entry in enumerate(thickness_list):
            proto_id = entry["proto_id"]
            val      = entry["value"]
            cross_pts= entry["cross_section_points"]  # list of (x,y)

            c = colors[i % color_count]
            label_txt = f"{proto_id} (val={val})"

            # Extract X,Y
            xvals = [pt[0] for pt in cross_pts]
            yvals = [pt[1] for pt in cross_pts]

            ax.plot(xvals, yvals, '-', color=c, label=label_txt)

        ax.set_title("2D Cross Sections (thickness)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend(loc="best")

        col_index += 1

    # ----- 2) Subplot for thickness_factor -----
    if has_thf:
        # gather entries
        thf_list = plot_data["thickness_factor"]
        ax = axes[0][col_index]

        for i, entry in enumerate(thf_list):
            proto_id = entry["proto_id"]
            val      = entry["value"]
            cross_pts= entry["cross_section_points"]

            c = colors[i % color_count]
            label_txt = f"{proto_id} (val={val})"

            xvals = [pt[0] for pt in cross_pts]
            yvals = [pt[1] for pt in cross_pts]

            ax.plot(xvals, yvals, '-', color=c, label=label_txt)

        ax.set_title("2D Cross Sections (thickness_factor)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend(loc="best")

        col_index += 1

    plt.tight_layout()
    
    # Show the figure
    # plt.show()

    twoD_fig = fig

    return twoD_fig


def plot_twoD_xsection(plot_data, plot_points=False, marker='o', markersize=3):
    """
    DEBUG VIEW: Plot the 2D cross-section for *all* models found in `plot_data`.
    - Accepts your existing `plots_data` dict produced in generate_prototypes().
    - Works whether you bucketed everything under "_all" or under multiple keys.
    - Ignores which parameter varied; just plots cross_section_points for each model.
    - plot_points: if True, draws a marker on each point (default False).
    - marker / markersize: style for the plotted points.
    Returns: matplotlib Figure
    """

    # Flatten all entries into one list
    entries = []

    if "_all" in plot_data:
        entries.extend(plot_data["_all"])
    else:
        for _, lst in plot_data.items():
            if isinstance(lst, list):
                entries.extend(lst)

    if not entries:
        print("No cross-section data found to plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    for entry in entries:
        proto_id = entry.get("proto_id", "unknown")
        cross_pts = entry.get("cross_section_points")

        if not cross_pts:
            # skip models that didn't produce a cross section
            continue

        # Original points for markers
        x_pts = [pt[0] for pt in cross_pts]
        y_pts = [pt[1] for pt in cross_pts]

        # Create closed ring for the line plot if needed (but keep original pts for markers)
        x_line = x_pts[:]
        y_line = y_pts[:]
        if cross_pts[0] != cross_pts[-1]:
            x_line = x_line + [x_line[0]]
            y_line = y_line + [y_line[0]]

        # Draw the polyline and capture the color assigned
        line, = ax.plot(x_line, y_line, '-', linewidth=1.0, label=proto_id)
        line_color = line.get_color()

        # Optionally draw markers on each original point
        if plot_points:
            ax.plot(x_pts, y_pts, linestyle='None', marker=marker, markersize=markersize, color=line_color)

    ax.set_title("2D Cross Sections (all models)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')  # geometry sanity!
    ax.legend(loc="best", fontsize=8, ncol=1)

    fig.tight_layout()
    return fig

# io_modules/plotting.py
import re
import math
import matplotlib.pyplot as plt

_ANG_RE = re.compile(r"^(?P<model>.+?)_ang(?P<idx>\d+)$")

def _split_model_section(proto_id: str):
    """
    Split names like 'BENDING1_ang3' -> ('BENDING1', 3).
    If there is no _ang suffix, treat as base section index 0.
    """
    m = _ANG_RE.match(str(proto_id))
    if m:
        return m.group("model"), int(m.group("idx"))
    return proto_id, 0

def plot_twoD_xsection_by_model(
    plot_data,
    plot_points: bool = False,
    marker: str = 'o',
    markersize: int = 2,
    cols: int = 3,
    share_axes: bool = True,
    resolution: int = 6,      # <- NEW: plot every Nth section (after the first)
    include_last: bool = True # <- ensure we still show the final angle
):
    """
    Grid of subplots: one subplot per model (Prototype ID).
    Each subplot draws only every `resolution`-th angular section after the first (0°),
    to keep plots readable. The first section is always included, and the last
    can be forced via `include_last=True`.

    Expects entries like those produced by your collector:
      - "proto_id", "cross_section_points"
      - optionally "model_id", "section_idx", "angular_section"
      If model_id/section_idx are missing, they are parsed from proto_id "NAME_angN".
    """
    # ---- flatten ----
    entries = []
    if "_all" in plot_data:
        entries.extend(plot_data["_all"])
    else:
        for _, lst in plot_data.items():
            if isinstance(lst, list):
                entries.extend(lst)

    if not entries:
        print("No cross-section data found to plot.")
        return None

    # ---- group by model ----
    grouped = {}
    for e in entries:
        pid = e.get("proto_id", "unknown")
        model_id = e.get("model_id")
        section_idx = e.get("section_idx")
        total_ang = e.get("angular_section")
        if model_id is None or section_idx is None:
            model_id, section_idx = _split_model_section(pid)

        grouped.setdefault(model_id, []).append(
            (section_idx, e)
        )

    # Compute angular increments for each model
    angle_incs = total_ang / (len(grouped[model_id]) - 1)

    # ---- sort within each model (prefer angle, fallback to section_idx, else by insertion) ----
    for model_id, series in grouped.items():
        series.sort(key=lambda t: (
            float('inf') if t[0] is None else t[0],     # angle
            float('inf') if t[1] is None else t[1]      # section_idx
        ))

    # ---- optional shared axes limits ----
    x_min = x_max = y_min = y_max = None
    if share_axes:
        xs, ys = [], []
        for series in grouped.values():
            for _, e in series:
                pts = e.get("cross_section_points")
                if pts:
                    xs.extend(p[0] for p in pts)
                    ys.extend(p[1] for p in pts)
        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

    # ---- layout ----
    model_ids = list(grouped.keys())
    n_models = len(model_ids)
    cols = max(1, int(cols))
    rows = math.ceil(n_models / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), squeeze=False)
    axes_flat = axes.ravel()

    # ---- plot per model (downsample by `resolution`) ----
    resolution = max(1, int(resolution))
    for ax_idx, model_id in enumerate(model_ids):
        ax = axes_flat[ax_idx]
        series = grouped[model_id]

        # pick indices: always include first (idx 0), then every `resolution`th,
        # and (optionally) ensure the last is included
        keep_idx = set()
        for i in range(len(series)):
            if i == 0 or (i % resolution == 0):
                keep_idx.add(i)
        if include_last and len(series) > 0:
            keep_idx.add(len(series) - 1)

        for i, (section_idx, entry) in enumerate(series):
            if i not in keep_idx:
                continue

            cross_pts = entry.get("cross_section_points")
            if not cross_pts:
                continue

            # build line (closed) and point arrays
            x_pts = [p[0] for p in cross_pts]
            y_pts = [p[1] for p in cross_pts]
            closed = (cross_pts[0] == cross_pts[-1])
            x_line = x_pts[:] + ([x_pts[0]] if not closed else [])
            y_line = y_pts[:] + ([y_pts[0]] if not closed else [])

            # label:
            label = f"{section_idx*angle_incs}°" if section_idx is not None else "sec ?"
            line, = ax.plot(x_line, y_line, '-', linewidth=1.0, label=label)
            if plot_points:
                ax.plot(x_pts, y_pts, linestyle='None', marker=marker, markersize=markersize, color=line.get_color())

        ax.set_title(model_id)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc="best", fontsize=8, ncol=1)

        if share_axes and x_min is not None:
            pad_x = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
            pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # hide unused axes
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle("2D Cross Sections by Model", y=0.995)
    fig.tight_layout()
    return fig


def plot_grouped_curves_by_model(
    plots_data,
    cols: int = 3,
    share_axes: bool = True,
    resolution: int = 6,       # <- NEW: plot every Nth section after the first
    include_last: bool = True, # <- ensure final section is shown
):
    """
    Grid of subplots: one subplot per model (Prototype ID).
    Each subplot shows only every `resolution`-th angular section after the first (0°),
    to keep plots readable. The first section is always included; the last can be forced.
    Expects entries like:
      {
        "proto_id": "...",          # e.g. "BENDING1_ang3"
        "model_id": "BENDING1",     # (recommended; else parsed from proto_id)
        "section_idx": 3,           # (recommended; else parsed from proto_id)
        "angular_section": 15,      # optional but preferred for labels
        "curve_points": [np.ndarray, ...]
      }
    """
    # ---- flatten entries ----
    entries = []
    if "_all" in plots_data:
        entries.extend(plots_data["_all"])
    else:
        for _, lst in plots_data.items():
            if isinstance(lst, list):
                entries.extend(lst)

    if not entries:
        print("No 1D curve data to plot.")
        return None

    # ---- group by model_id; keep angle and section_idx if present ----
    grouped = {}
    for e in entries:
        pid = e.get("proto_id", "unknown")
        model_id = e.get("model_id")
        section_idx = e.get("section_idx")
        total_ang = e.get("angular_section")
        if model_id is None or section_idx is None:
            model_id, section_idx = _split_model_section(pid)
        grouped.setdefault(model_id, []).append((section_idx, e))

    # Compute angular increments for each model
    angle_incs = total_ang / (len(grouped[model_id]) - 1)

    # ---- sort within each model: by angle if present, else by section_idx ----
    for model_id, series in grouped.items():
        series.sort(key=lambda t: (
            float('inf') if t[0] is None else t[0],           # angle first
            float('inf') if t[1] is None else t[1]            # then section_idx
        ))

    # ---- global limits if sharing axes ----
    gmin_x = gmin_y = float("inf")
    gmax_x = gmax_y = float("-inf")
    if share_axes:
        for series in grouped.values():
            for _, e in series:
                curves = e.get("curve_points") or []
                for pts in curves:
                    if pts is None or len(pts) == 0:
                        continue
                    xs, ys = pts[:, 0], pts[:, 1]
                    gmin_x, gmax_x = min(gmin_x, xs.min()), max(gmax_x, xs.max())
                    gmin_y, gmax_y = min(gmin_y, ys.min()), max(gmax_y, ys.max())
        if gmin_x == float("inf"):
            share_axes = False  # nothing valid found

    # ---- layout ----
    model_ids = list(grouped.keys())
    n_models = len(model_ids)
    cols = max(1, int(cols))
    rows = math.ceil(n_models / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)
    axes_flat = axes.ravel()

    # ---- plot per model with downsampling ----
    resolution = max(1, int(resolution))
    for ax_idx, model_id in enumerate(model_ids):
        ax = axes_flat[ax_idx]
        series = grouped[model_id]

        # pick indices: include first (0), then every resolution-th; optionally last
        keep_idx = set()
        for i in range(len(series)):
            if i == 0 or (i % resolution == 0):
                keep_idx.add(i)
        if include_last and len(series) > 0:
            keep_idx.add(len(series) - 1)

        # plot each kept section's set of curves
        for i, (section_idx, entry) in enumerate(series):
            if i not in keep_idx:
                continue
            curves = entry.get("curve_points") or []
            # label prefers angle if available
            label = f"{section_idx*angle_incs}°" if section_idx is not None else "sec ?"

            first = True
            line = None
            for pts in curves:
                if pts is None or len(pts) == 0:
                    continue
                xvals, yvals = pts[:, 0], pts[:, 1]
                if first:
                    line, = ax.plot(xvals, yvals, '-', linewidth=1.0, label=label)
                    first = False
                else:
                    ax.plot(xvals, yvals, '-', linewidth=1.0, color=line.get_color())

        ax.set_title(model_id)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)

        if share_axes:
            pad_x = 0.05 * (gmax_x - gmin_x) if gmax_x > gmin_x else 1.0
            pad_y = 0.05 * (gmax_y - gmin_y) if gmax_y > gmin_y else 1.0
            ax.set_xlim(gmin_x - pad_x, gmax_x + pad_x)
            ax.set_ylim(gmin_y - pad_y, gmax_y + pad_y)

        ax.legend(loc="best", fontsize=8, ncol=1)

    # hide unused subplots
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle("1D Curves by Model (downsampled)", y=0.995)
    fig.tight_layout()
    return fig

def plot_twoD_xsection_by_model_3d(
    plot_data,
    plot_points: bool = False,
    marker: str = 'o',
    markersize: int = 2,
    cols: int = 2,
    resolution: int = 6,       # plot every Nth section after the first
    include_last: bool = True, # ensure final section is shown
    projection: str = "persp", # "persp" or "ortho"
    elev: float = 20,
    azim: float = -60,
    zlabel: str = "Angle (deg)",
):
    """
    3D view of 2D cross-sections stacked along Z (angle).
    One subplot per model. Each subplot contains downsampled sections.

    Expects entries like those produced by your collector:
      - "proto_id", "cross_section_points"
      - optionally "model_id", "section_idx", "angular_section"
      If model_id/section_idx are missing, parsed from proto_id "NAME_angN".
    """
    # ---- flatten ----
    entries = []
    if "_all" in plot_data:
        entries.extend(plot_data["_all"])
    else:
        for _, lst in plot_data.items():
            if isinstance(lst, list):
                entries.extend(lst)
    if not entries:
        print("No cross-section data found to plot.")
        return None

    # ---- group by model ----
    grouped = {}
    for e in entries:
        pid = e.get("proto_id", "unknown")
        model_id = e.get("model_id")
        section_idx = e.get("section_idx")
        total_ang = e.get("angular_section")
        if model_id is None or section_idx is None:
            model_id, section_idx = _split_model_section(pid)
        grouped.setdefault(model_id, []).append((section_idx, e))

    # Compute angular increments for each model
    angle_incs = total_ang / (len(grouped[model_id]) - 1)

    # ---- sort within each model (angle first, then index) ----
    for model_id, series in grouped.items():
        series.sort(key=lambda t: (
            float('inf') if t[0] is None else t[0],
            float('inf') if t[1] is None else t[1]
        ))

    # ---- layout ----
    model_ids = list(grouped.keys())
    n_models = len(model_ids)
    cols = max(1, int(cols))
    rows = math.ceil(n_models / cols)

    fig = plt.figure(figsize=(7*cols, 6*rows))
    axes = []

    # ---- plot per model ----
    resolution = max(1, int(resolution))
    for i, model_id in enumerate(model_ids, start=1):
        ax = fig.add_subplot(rows, cols, i, projection='3d')
        axes.append(ax)

        series = grouped[model_id]

        # choose kept sections: first, every Nth, (optionally) last
        keep_idx = set()
        for j in range(len(series)):
            if j == 0 or (j % resolution == 0):
                keep_idx.add(j)
        if include_last and len(series) > 0:
            keep_idx.add(len(series) - 1)

        # if angles missing, infer evenly across 0..360
        if all(t[0] is None for t in series) and len(series) >= 2:
            inferred = np.linspace(0.0, 360.0, len(series))
        else:
            inferred = None

        for j, (section_idx, entry) in enumerate(series):
            if j not in keep_idx:
                continue

            cross_pts = entry.get("cross_section_points")
            if not cross_pts:
                continue

            x = np.array([p[0] for p in cross_pts], dtype=float)
            y = np.array([p[1] for p in cross_pts], dtype=float)
            closed = (cross_pts[0] == cross_pts[-1])
            if not closed:
                x = np.append(x, x[0])
                y = np.append(y, y[0])

            zval = float(section_idx*angle_incs) if (section_idx*angle_incs) is not None else float(inferred[j] if inferred is not None else j)
            z = np.full_like(x, zval, dtype=float)

            # plot line
            ax.plot(x, y, z, linewidth=1.0, label=(f"{int(round(zval))}°"))

            # optional markers
            if plot_points:
                ax.scatter(x, y, z, marker=marker, s=markersize)

        ax.set_title(model_id)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(zlabel)
        ax.grid(True)

        # camera & projection
        try:
            ax.set_proj_type(projection)  # "persp" or "ortho" (Matplotlib ≥3.2)
        except Exception:
            pass
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("3D View: 2D Cross-Sections Stacked by Angle", y=0.98)
    fig.tight_layout()
    return fig