"""
Interactive 1D curve demo

This script visualizes how the 1D curve (centerline + cap) responds to param
changes. It reuses the existing helpers from builders/build_modules/oneD_helpers.py
and the defaults from core/config.py, and provides Matplotlib sliders to adjust
parameters live.

Usage:
  conda activate generate-geometry
  python testing/oneD_live_demo.py

Requirements:
  - geomdl (installed via pip in the provided conda env)
  - matplotlib, numpy
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from builders.build_modules.oneD_helpers import (
    validate_parameters,
    compute_x_increments_and_y_positions,
    generate_cap_curve,
    generate_curves,
)

from core.config import (
    BaselineGeometryConfig,
    NURBSConfig,
    CurveSettings,
)


# --- Defaults from config ---
baseline = BaselineGeometryConfig()
nurbs = NURBSConfig()
curves = CurveSettings()


def _compute_curves(
    amplitude0: float,
    desired_radius: float,
    n_curves: int,
    period_factors: list[float],
    offset_factor_x: float,
    mid_factor_x: float,
    mid_factor_y: float,
    min_y_positions_frac: list[float],
    curve_weight: float,
):
    """Compute control + curve points for cap + sequential curves.

    min_y_positions in the repo are scaled by amplitude0 in one_d_build.
    We accept fractional sliders (0..~0.5) and scale here to match that behavior.
    """
    # Derived
    weights0 = [1, nurbs.cp1_weight0, 1]
    weights = [1, curve_weight, 1, curve_weight, 1]

    # Scale min_y_positions by amplitude
    min_y_positions = [y * amplitude0 for y in min_y_positions_frac]

    # Ensure lists have sufficient length for the requested n_curves
    # Required lengths:
    #   n_periods = ceil(n_curves/2)
    #   n_descending_curves = floor(n_curves/2)
    import math
    n_periods_req = int(math.ceil(int(n_curves) / 2))
    n_desc_req = int(math.floor(int(n_curves) / 2))

    def _pad_to(lst, length):
        if len(lst) >= length:
            return lst[:length]
        if not lst:
            return [0.0] * length
        return lst + [lst[-1]] * (length - len(lst))

    period_factors = _pad_to(period_factors, n_periods_req)
    # min_y needs length n_desc_req+1
    min_y_positions = _pad_to(min_y_positions, n_desc_req + 1)

    # Validate and compute period values (helper will re-check lengths and scale)
    period_values, min_y_positions, n_periods, n_descending_curves = validate_parameters(
        period_factors     = period_factors,
        min_y_positions    = min_y_positions,
        desired_radius     = desired_radius,
        inside_tolerance   = baseline.inside_tolerance,
        n_curves           = int(n_curves)
    )

    x_increments, y_positions = compute_x_increments_and_y_positions(
        int(n_curves), amplitude0, min_y_positions, period_values, baseline.start_y
    )

    # Cap curve
    (
        all_control_points, all_curve_points,
        control_points_idx, control_points_idx_names,
        curve_points_idx, curve_points_idx_names,
        end_x0, end_y0
    ) = generate_cap_curve(
        baseline.start_x, baseline.start_y,
        baseline.cap_height, baseline.cap_length,
        weights0, nurbs.degree0
    )

    # Main curves
    (
        all_control_points, all_curve_points,
        control_points_idx, control_points_idx_names,
        curve_points_idx, curve_points_idx_names,
    ) = generate_curves(
        int(n_curves),
        x_increments, y_positions,
        all_control_points, all_curve_points,
        control_points_idx, control_points_idx_names,
        curve_points_idx, curve_points_idx_names,
        end_x0, end_y0,
        offset_factor_x, curves.offset_factor_y0,
        mid_factor_x, mid_factor_y,
        curves.true_mid, curves.rel_mid,
        thickness=1.0, inside_tolerance=baseline.inside_tolerance,
        degree=nurbs.degree, order=(nurbs.degree + 1), knot_c=nurbs.knot_c,
        resolution=curves.resolution,
        weights=weights, center_offset=baseline.revolve_offset
    )

    return all_control_points, all_curve_points


def _draw(ax, curve_segments, control_segments=None):
    ax.clear()
    # draw curves
    for seg in curve_segments:
        seg = np.asarray(seg)
        ax.plot(seg[:, 0], seg[:, 1], '-', color='tab:blue', lw=1.2)

    # optional: control points
    if control_segments:
        for cps in control_segments:
            cps = np.asarray(cps)
            ax.plot(cps[:, 0], cps[:, 1], 'o--', ms=3, lw=0.8, color='tab:orange', alpha=0.7)

    ax.set_title("1D Curve (cap + sequential curves)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # auto-limits with a small pad
    all_x = np.concatenate([np.asarray(seg)[:, 0] for seg in curve_segments])
    all_y = np.concatenate([np.asarray(seg)[:, 1] for seg in curve_segments])
    xpad = max(1.0, 0.05 * (all_x.max() - all_x.min() + 1e-6))
    ypad = max(1.0, 0.05 * (all_y.max() - all_y.min() + 1e-6))
    ax.set_xlim(all_x.min() - xpad, all_x.max() + xpad)
    ax.set_ylim(all_y.min() - ypad, all_y.max() + ypad)


def main():
    # Initial values (aligned with dataset examples)
    init = dict(
        amplitude0=20.0,
        desired_radius=25.0,
        n_curves=5,
        period_factors=[0.9, 1.0, 1.1],
        offset_factor_x=0.3,
        mid_factor_x=0.0,
        mid_factor_y=0.0,
        min_y_positions_frac=[0.0, 0.35, 0.35],
        curve_weight=5.0,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.35)

    # Compute initial
    cps, segs = _compute_curves(**init)
    _draw(ax, segs, control_segments=cps)

    # --- Sliders ---
    slider_height = 0.027
    y0 = 0.30
    x0 = 0.12
    w = 0.78

    ax_amp = plt.axes([x0, y0, w, slider_height])
    s_amp = Slider(ax_amp, 'amplitude0', 0.0, 60.0, valinit=init['amplitude0'], valstep=0.1)

    ax_rad = plt.axes([x0, y0 - 1*slider_height, w, slider_height])
    s_rad = Slider(ax_rad, 'desired_radius', 5.0, 80.0, valinit=init['desired_radius'], valstep=0.1)

    ax_nc = plt.axes([x0, y0 - 2*slider_height, w, slider_height])
    s_nc = Slider(ax_nc, 'n_curves', 2, 9, valinit=init['n_curves'], valstep=1)

    ax_ofx = plt.axes([x0, y0 - 3*slider_height, w, slider_height])
    s_ofx = Slider(ax_ofx, 'offset_factor_x', -1.0, 1.0, valinit=init['offset_factor_x'], valstep=0.01)

    ax_mfx = plt.axes([x0, y0 - 4*slider_height, w, slider_height])
    s_mfx = Slider(ax_mfx, 'mid_factor_x', -1.0, 1.0, valinit=init['mid_factor_x'], valstep=0.01)

    ax_mfy = plt.axes([x0, y0 - 5*slider_height, w, slider_height])
    s_mfy = Slider(ax_mfy, 'mid_factor_y', -1.0, 1.0, valinit=init['mid_factor_y'], valstep=0.01)

    ax_w = plt.axes([x0, y0 - 6*slider_height, w, slider_height])
    s_w = Slider(ax_w, 'curve_weight', 0.5, 10.0, valinit=init['curve_weight'], valstep=0.1)

    # Period factors (three sliders)
    ax_pf0 = plt.axes([x0, y0 - 7*slider_height, w, slider_height])
    s_pf0 = Slider(ax_pf0, 'period_f0', 0.1, 2.0, valinit=init['period_factors'][0], valstep=0.01)

    ax_pf1 = plt.axes([x0, y0 - 8*slider_height, w, slider_height])
    s_pf1 = Slider(ax_pf1, 'period_f1', 0.1, 2.0, valinit=init['period_factors'][1], valstep=0.01)

    ax_pf2 = plt.axes([x0, y0 - 9*slider_height, w, slider_height])
    s_pf2 = Slider(ax_pf2, 'period_f2', 0.1, 2.0, valinit=init['period_factors'][2], valstep=0.01)

    # min_y_positions (fractions, scaled by amplitude)
    ax_my0 = plt.axes([x0, y0 - 10*slider_height, w, slider_height])
    s_my0 = Slider(ax_my0, 'min_y0(frac)', 0.0, 0.8, valinit=init['min_y_positions_frac'][0], valstep=0.01)

    ax_my1 = plt.axes([x0, y0 - 11*slider_height, w, slider_height])
    s_my1 = Slider(ax_my1, 'min_y1(frac)', 0.0, 0.8, valinit=init['min_y_positions_frac'][1], valstep=0.01)

    ax_my2 = plt.axes([x0, y0 - 12*slider_height, w, slider_height])
    s_my2 = Slider(ax_my2, 'min_y2(frac)', 0.0, 0.8, valinit=init['min_y_positions_frac'][2], valstep=0.01)

    def on_change(_):
        # period_factors count depends on n_curves; we provide three and the helper trims as needed
        period_factors = [s_pf0.val, s_pf1.val, s_pf2.val]
        min_y_positions_frac = [s_my0.val, s_my1.val, s_my2.val]
        try:
            cps, segs = _compute_curves(
                amplitude0=s_amp.val,
                desired_radius=s_rad.val,
                n_curves=int(s_nc.val),
                period_factors=period_factors,
                offset_factor_x=s_ofx.val,
                mid_factor_x=s_mfx.val,
                mid_factor_y=s_mfy.val,
                min_y_positions_frac=min_y_positions_frac,
                curve_weight=s_w.val,
            )
            _draw(ax, segs, control_segments=cps)
        except Exception as e:
            ax.clear()
            ax.text(0.05, 0.95, f"Error: {e}", transform=ax.transAxes, va='top', ha='left', color='red')
            ax.set_axis_off()
        fig.canvas.draw_idle()

    # Connect updates
    for s in (s_amp, s_rad, s_nc, s_ofx, s_mfx, s_mfy, s_w, s_pf0, s_pf1, s_pf2, s_my0, s_my1, s_my2):
        s.on_changed(on_change)

    plt.show()


if __name__ == "__main__":
    main()
