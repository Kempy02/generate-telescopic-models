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
    resolution: int = 6,
    include_last: bool = True
):
    import math
    import matplotlib.pyplot as plt

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

    # ---- group by model (store (section_idx, entry)) ----
    grouped = {}
    for e in entries:
        pid = e.get("proto_id", "unknown")
        model_id = e.get("model_id")
        section_idx = e.get("section_idx")
        if model_id is None or section_idx is None:
            model_id, section_idx = _split_model_section(pid)
        grouped.setdefault(model_id, []).append((section_idx, e))

    # ---- sort within each model by section index ----
    for model_id, series in grouped.items():
        series.sort(key=lambda t: (float('inf') if t[0] is None else t[0]))

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
        n = len(series)

        # per-model angle increment; fallback to 0 (1 section) or 360
        if n <= 1:
            angle_incs = 0.0
        else:
            total_ang = next(
                (e.get("angular_section") for _, e in series if e.get("angular_section") is not None),
                360.0
            )
            angle_incs = float(total_ang) / float(n - 1)

        # keep: first, every Nth, (optionally) last
        keep_idx = set()
        for i in range(n):
            if i == 0 or (i % resolution == 0):
                keep_idx.add(i)
        if include_last and n > 0:
            keep_idx.add(n - 1)

        for i, (section_idx, entry) in enumerate(series):
            if i not in keep_idx:
                continue

            cross_pts = entry.get("cross_section_points")
            if not cross_pts:
                continue

            x_pts = [p[0] for p in cross_pts]
            y_pts = [p[1] for p in cross_pts]
            closed = (cross_pts[0] == cross_pts[-1])
            x_line = x_pts[:] + ([x_pts[0]] if not closed else [])
            y_line = y_pts[:] + ([y_pts[0]] if not closed else [])

            label = f"{section_idx * angle_incs:.0f}°" if section_idx is not None else "sec ?"
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

    # ---- group by model_id; keep section_idx ----
    grouped = {}
    for e in entries:
        pid = e.get("proto_id", "unknown")
        model_id = e.get("model_id")
        section_idx = e.get("section_idx")
        if model_id is None or section_idx is None:
            model_id, section_idx = _split_model_section(pid)
        grouped.setdefault(model_id, []).append((section_idx, e))

    # ---- sort within each model by section_idx (0..N-1) ----
    for model_id, series in grouped.items():
        series.sort(key=lambda t: (float('inf') if t[0] is None else t[0]))

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

        # per-model angle increment (fallbacks: 0 if only one section; else 360)
        n = len(series)
        if n <= 1:
            angle_incs = 0.0
        else:
            # pick the first non-None angular_section in this model's entries, else 360
            total_ang = next(
                (e.get("angular_section") for _, e in series if e.get("angular_section") is not None),
                360.0
            )
            angle_incs = float(total_ang) / float(n - 1)

        # pick indices to keep (downsampling) ...
        keep_idx = set()
        for i in range(n):
            if i == 0 or (i % resolution == 0):
                keep_idx.add(i)
        if include_last and n > 0:
            keep_idx.add(n - 1)

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
    resolution: int = 6,
    include_last: bool = True,
    projection: str = "persp",
    elev: float = 20,
    azim: float = -60,
    zlabel: str = "Angle (deg)",
):
    import math
    import numpy as np
    import matplotlib.pyplot as plt

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
        if model_id is None or section_idx is None:
            model_id, section_idx = _split_model_section(pid)
        grouped.setdefault(model_id, []).append((section_idx, e))

    # ---- sort within each model by section index ----
    for model_id, series in grouped.items():
        series.sort(key=lambda t: (float('inf') if t[0] is None else t[0]))

    # ---- layout ----
    model_ids = list(grouped.keys())
    n_models = len(model_ids)
    cols = max(1, int(cols))
    rows = math.ceil(n_models / cols)

    fig = plt.figure(figsize=(7*cols, 6*rows))

    # ---- plot per model ----
    resolution = max(1, int(resolution))
    for i, model_id in enumerate(model_ids, start=1):
        ax = fig.add_subplot(rows, cols, i, projection='3d')

        series = grouped[model_id]
        n = len(series)

        # per-model angle increment; fallback to 0 (1 section) or 360
        # if n <= 1:
        #     angle_incs = 0.0
        # else:
        #     total_ang = next(
        #         (e.get("angular_section") for _, e in series if e.get("angular_section") is not None),
        #         360.0
        #     )
        #     angle_incs = float(total_ang) / float(n - 1)

        total_ang = next(
            (e.get("angular_section") for _, e in series if e.get("angular_section") is not None),
            360.0
        )
        angle_incs = float(total_ang) / float(n - 1)

        # keep: first, every Nth, (optionally) last
        keep_idx = set()
        for j in range(n):
            if j == 0 or (j % resolution == 0):
                keep_idx.add(j)
        if include_last and n > 0:
            keep_idx.add(n - 1)

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

            zval = float(section_idx * angle_incs) if section_idx is not None else float(j * angle_incs)
            z = np.full_like(x, zval, dtype=float)

            ax.plot(x, y, z, linewidth=1.0, label=f"{int(round(zval))}°")
            if plot_points:
                ax.scatter(x, y, z, marker=marker, s=markersize)

        ax.set_title(model_id)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(zlabel)
        ax.grid(True)

        try:
            ax.set_proj_type(projection)  # "persp" or "ortho"
        except Exception:
            pass
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("3D View: 2D Cross-Sections Stacked by Angle", y=0.98)
    fig.tight_layout()
    return fig
