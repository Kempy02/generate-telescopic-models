
"""
Interactive 1D/2D curve demo (sidebar UI + dynamic sliders)

Usage:
  conda activate generate-geometry
  python oneD_twoD_live_demo.py
"""

from __future__ import annotations

import os, math, time
from types import SimpleNamespace
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

# Preferred high-level builders
from builders.one_d_build import generate_1D_curves as build_one_d
from builders.two_d_build import generate_2D_cross_sections as build_two_d

# Low-level fallbacks
from builders.build_modules.oneD_helpers import (
    validate_parameters,
    compute_x_increments_and_y_positions,
    generate_cap_curve,
    generate_curves,
)

from core.config import BaselineGeometryConfig, NURBSConfig, CurveSettings

baseline = BaselineGeometryConfig()
nurbs = NURBSConfig()
curve_cfg = CurveSettings()

# ---------- utils ----------
def _pad_to(lst: List[float], length: int, fill: float = 0.0) -> List[float]:
    if len(lst) >= length: return lst[:length]
    if not lst: return [fill]*length
    return lst + [lst[-1]]*(length - len(lst))

def _build_params_namespace(
    amplitude0: float,
    desired_radius: float,
    n_curves: int,
    period_factors: List[float],
    offset_factor_x: float,
    mid_factor_x: float,
    mid_factor_y: float,
    # fractional list from the UI
    min_y_positions_frac: List[float],
    curve_weight: float,
    # NEW: extras
    weights: List[float],
    cap_thickness: float,
    center_offset: float,
    bending_enabled: bool,
    angular_section: float,
    # 2D
    thickness: float,
    thickness_mode: str,
    thickness_factor: List[float],
    thickness_factor2: List[float],
) -> SimpleNamespace:

    # Convert fractional UI → absolute mm (matches param_builder expectations)
    min_y_positions_abs = [v * amplitude0 for v in min_y_positions_frac]

    # Ensure list lengths based on n_curves
    n_periods_req = int(math.ceil(n_curves / 2))
    n_desc_req    = int(math.floor(n_curves / 2))
    period_factors = _pad_to(period_factors, n_periods_req, 1.0)
    min_y_positions_abs = _pad_to(min_y_positions_abs, n_desc_req + 1, 0.0)

    return SimpleNamespace(
        # 1D
        amplitude0=float(amplitude0),
        desired_radius=float(desired_radius),
        n_curves=int(n_curves),
        period_factors=list(period_factors),
        offset_factor_x=float(offset_factor_x),
        mid_factor_x=float(mid_factor_x),
        mid_factor_y=float(mid_factor_y),
        # both forms, absolute is the one builders use
        min_y_positions=list(min_y_positions_abs),
        min_y_positions_frac=list(min_y_positions_frac),
        curve_weight=float(curve_weight),
        weights=list(weights),

        # 2D
        thickness=float(thickness),
        thickness_mode=str(thickness_mode),
        thickness_factor=list(thickness_factor),
        thickness_factor2=list(thickness_factor2),

        # extras from param_builder
        cap_thickness=float(cap_thickness),
        center_offset=float(center_offset),
        bending_enabled=bool(bending_enabled),
        angular_section=float(angular_section),

        # export/misc (kept for parity)
        export_filename="live_demo",
        export_folder="prototype_models",
    )


def _compute_curves_fallback(params: SimpleNamespace):
    weights0 = [1, nurbs.cp1_weight0, 1]
    # use the list from params if it looks valid, else derive from curve_weight
    main_weights = params.weights if (isinstance(params.weights, list) and len(params.weights) == 5) \
                   else [1, params.curve_weight, 1, params.curve_weight, 1]

    pv, my, _, _ = validate_parameters(
        period_factors=params.period_factors,
        min_y_positions=params.min_y_positions,    # absolute mm
        desired_radius=params.desired_radius,
        inside_tolerance=baseline.inside_tolerance,
        n_curves=params.n_curves,
    )
    xinc, ypos = compute_x_increments_and_y_positions(
        params.n_curves, params.amplitude0, my, pv, baseline.start_y
    )

    (all_cps, all_curves, cp_idx, _, _, _, ex, ey) = generate_cap_curve(
        baseline.start_x, baseline.start_y,
        baseline.cap_height, baseline.cap_length,
        weights0, nurbs.degree0
    )
    (all_cps, all_curves, cp_idx, _, _, _) = generate_curves(
        int(params.n_curves),
        xinc, ypos,
        all_cps, all_curves,
        cp_idx, None, None, None,
        ex, ey,
        params.offset_factor_x, curve_cfg.offset_factor_y0,
        params.mid_factor_x, params.mid_factor_y,
        curve_cfg.true_mid, curve_cfg.rel_mid,
        thickness=1.0, inside_tolerance=baseline.inside_tolerance,
        degree=nurbs.degree, order=(nurbs.degree + 1), knot_c=nurbs.knot_c,
        resolution=curve_cfg.resolution,
        weights=main_weights,
        center_offset=float(params.center_offset),   # NEW: from defaults
    )
    return SimpleNamespace(curve_points=all_curves, control_points=all_cps, cp_idx=cp_idx)

def _draw_1d(ax, curves1d):
    ax.clear()
    for seg in curves1d.curve_points:
        arr = np.asarray(seg); ax.plot(arr[:,0], arr[:,1], '-', lw=1.4, color='tab:blue')
    for cps in curves1d.control_points:
        arr = np.asarray(cps); ax.plot(arr[:,0], arr[:,1], 'o--', ms=3, lw=0.9, color='tab:orange', alpha=0.7)
    ax.set_title("1D curve (cap + sequential)"); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.set_aspect('equal', adjustable='datalim')
    all_x = np.concatenate([np.asarray(seg)[:,0] for seg in curves1d.curve_points])
    all_y = np.concatenate([np.asarray(seg)[:,1] for seg in curves1d.curve_points])
    xpad = max(1.0, 0.05*(all_x.max()-all_x.min()+1e-6))
    ypad = max(1.0, 0.05*(all_y.max()-all_y.min()+1e-6))
    ax.set_xlim(all_x.min()-xpad, all_x.max()+xpad)
    ax.set_ylim(all_y.min()-ypad, all_y.max()+ypad)

def _get_2d_polygon_from_result(result):
    # Accept object or tuple/list
    if hasattr(result, "twoD_cross_section"):
        return np.asarray(result.twoD_cross_section)
    if isinstance(result, (list, tuple)) and len(result) > 0:
        return np.asarray(result[0])
    return None

def _draw_2d(ax, xsec2d):
    ax.clear()
    pts = _get_2d_polygon_from_result(xsec2d)
    if pts is None or len(pts) == 0:
        ax.text(0.5, 0.5, "No 2D polygon", ha='center', va='center', color='red')
        ax.set_axis_off(); return
    ax.plot(pts[:,0], pts[:,1], '-', lw=1.4, color='tab:green')
    ax.plot(pts[:,0], pts[:,1], 'o', ms=2, color='tab:green', alpha=0.55)
    ax.set_title("2D cross-section"); ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3); ax.set_aspect('equal', adjustable='datalim')
    xpad = max(1.0, 0.05*(pts[:,0].max()-pts[:,0].min()+1e-6))
    ypad = max(1.0, 0.05*(pts[:,1].max()-pts[:,1].min()+1e-6))
    ax.set_xlim(pts[:,0].min()-xpad, pts[:,0].max()+xpad)
    ax.set_ylim(pts[:,1].min()-ypad, pts[:,1].max()+ypad)

# ---------- sidebar layout ----------
class Sidebar:
    def __init__(self, fig, left=0.02, bottom=0.06, width=0.30, height=0.88, row_h=0.035, pad=0.006):
        self.fig, self.left, self.bottom, self.width, self.height = fig, left, bottom, width, height
        self.row_h, self.pad = row_h, pad
        self.cursor = bottom + height
        self.section_gap = 0.012
    def header(self, text, h=None):
        h = h or self.row_h; self.cursor -= (h + self.pad)
        ax = self.fig.add_axes([self.left, self.cursor, self.width, h]); ax.set_axis_off()
        ax.text(0.0, 0.5, text, va='center', ha='left', fontsize=10, weight='bold')
        self.cursor -= self.section_gap; return ax
    def row(self, h=None):
        h = h or self.row_h; self.cursor -= (h + self.pad)
        return self.fig.add_axes([self.left, self.cursor, self.width, h])
    def gap(self, g=None): self.cursor -= (g or self.section_gap)

# ---------- app ----------
def main():
    init = dict(
        # 1D core
        amplitude0=20.0,
        desired_radius=25.0,
        n_curves=BaselineGeometryConfig().n_curves,  # from baseline
        period_factors=[1.0, 1.0, 1.0],
        offset_factor_x=0.0,
        mid_factor_x=0.0,
        mid_factor_y=0.0,

        min_y_positions=[0.0, 0.0, 0.0, 0.0],     # absolute (mm) per param_builder
        min_y_positions_frac=[0.0, 0.0, 0.0, 0.0],# shown in UI (scaled by amplitude0)

        curve_weight=5.0,
        weights=[1.0, 5.0, 1.0, 5.0, 1.0],        # NEW: mirror param_builder

        # 2D / cross-section
        thickness=1.0,
        thickness_factor=[1.0, 1.0, 1.0],
        thickness_factor2=[1.0, 1.0, 1.0],
        thickness_mode="variable",                # param_builder default

        # extra geometry params used by builders
        cap_thickness=baseline.cap_thickness,
        center_offset=baseline.revolve_offset,

        # export / misc
        export_filename="live_demo",
        export_folder="prototype_models",
        bending_enabled=True,                     # param_builder sets (not use_linear_fast)
        angular_section=0.0,
    )
    fig = plt.figure(figsize=(13.5, 7.6))
    ax1d = fig.add_axes([0.36, 0.55, 0.62, 0.40])
    ax2d = fig.add_axes([0.36, 0.08, 0.62, 0.40])

    side = Sidebar(fig, left=0.02, bottom=0.06, width=0.30, height=0.88, row_h=0.035)

    # Mode
    side.header("Modes"); ax_mode = side.row(h=0.060)
    mode_radio = RadioButtons(ax_mode, ("1D", "2D"), active=1)

    # 1D basics
    side.header("1D — Geometry")
    s_amp = Slider(side.row(), 'amplitude0', 0.0, 60.0, valinit=init['amplitude0'], valstep=0.1)
    s_rad = Slider(side.row(), 'desired_radius', 5.0, 80.0, valinit=init['desired_radius'], valstep=0.1)
    s_nc  = Slider(side.row(), 'n_curves', 2, 9, valinit=init['n_curves'], valstep=1)
    s_ofx = Slider(side.row(), 'offset_x', -1.0, 1.0, valinit=init['offset_factor_x'], valstep=0.01)
    s_mfx = Slider(side.row(), 'mid_x', -1.0, 1.0, valinit=init['mid_factor_x'], valstep=0.01)
    s_mfy = Slider(side.row(), 'mid_y', -1.0, 1.0, valinit=init['mid_factor_y'], valstep=0.01)
    s_wt  = Slider(side.row(), 'curve_weight', 0.5, 10.0, valinit=init['curve_weight'], valstep=0.1)

    # 1D advanced (dynamic)
    side.header("1D — Period & min-y(frac)")
    # Create a safe max; n_curves up to 9 ⇒ ceil(9/2)=5 period slots
    max_slots = 6
    pf_sliders, my_sliders = [], []
    for i in range(max_slots):
        pf_sliders.append(Slider(
            side.row(), f'period_f{i}', 0.1, 2.0,
            valinit=init['period_factors'][min(i, len(init['period_factors'])-1)], valstep=0.01
        ))
    for i in range(max_slots):
        my_sliders.append(Slider(
            side.row(), f'min_y{i}(frac)', 0.0, 0.8,
            valinit=init['min_y_positions_frac'][min(i, len(init['min_y_positions_frac'])-1)], valstep=0.01
        ))

    # 2D
    side.header("2D — Thickness mode")
    ax_mode2d = side.row(h=0.085)
    mode2d_radio = RadioButtons(
        ax_mode2d, ("constant", "linear", "variable", "collapsed", "sbend"),
        active=("constant", "linear", "variable", "collapsed", "sbend").index(init["thickness_mode"])
    )

    side.header("2D — Thickness params")
    s_th   = Slider(side.row(), 'thickness', 0.2, 6.0, valinit=init['thickness'], valstep=0.05)
    s_tfa  = Slider(side.row(), 'fac1', 0.0, 2.0, valinit=init['thickness_factor'][0], valstep=0.01)
    s_tfb  = Slider(side.row(), 'fac2', 0.0, 2.0, valinit=init['thickness_factor'][1], valstep=0.01)
    s_tfc  = Slider(side.row(), 'fac3', 0.0, 2.0, valinit=init['thickness_factor'][2], valstep=0.01)
    s_tf2a = Slider(side.row(), 'fac2_1', 0.0, 2.0, valinit=init['thickness_factor2'][0], valstep=0.01)
    s_tf2b = Slider(side.row(), 'fac2_2', 0.0, 2.0, valinit=init['thickness_factor2'][1], valstep=0.01)
    s_tf2c = Slider(side.row(), 'fac2_3', 0.0, 2.0, valinit=init['thickness_factor2'][2], valstep=0.01)

    # Buttons
    side.gap(0.02)
    btn_reset = Button(fig.add_axes([side.left, side.bottom - 0.01, side.width*0.47, 0.045]), "Reset", hovercolor="#ddd")
    btn_save  = Button(fig.add_axes([side.left + side.width*0.53, side.bottom - 0.01, side.width*0.47, 0.045]), "Save PNG", hovercolor="#ddd")

    state = {"_busy": False}

    def _visible_dynamic_sliders():
        ncur = int(s_nc.val)
        n_periods_req = int(math.ceil(ncur/2))
        n_min_y_req   = int(math.floor(ncur/2)) + 1
        for i, s in enumerate(pf_sliders): s.ax.set_visible(i < n_periods_req)
        for i, s in enumerate(my_sliders): s.ax.set_visible(i < n_min_y_req)

    def _collect_params() -> SimpleNamespace:
        pf = [s.val for s in pf_sliders if s.ax.get_visible()]
        my = [s.val for s in my_sliders if s.ax.get_visible()]
        return _build_params_namespace(
            amplitude0=s_amp.val,
            desired_radius=s_rad.val,
            n_curves=int(s_nc.val),
            period_factors=pf,  # or dynamic list if you kept that
            offset_factor_x=s_ofx.val,
            mid_factor_x=s_mfx.val,
            mid_factor_y=s_mfy.val,
            min_y_positions_frac=my,  # extend/truncate as you do now
            curve_weight=s_wt.val,
            # NEW: from defaults (no sliders for these—feel free to add later)
            weights=init["weights"],
            cap_thickness=init["cap_thickness"],
            center_offset=init["center_offset"],
            bending_enabled=init["bending_enabled"],
            angular_section=init["angular_section"],
            # 2D
            thickness=s_th.val,
            thickness_mode=mode2d_radio.value_selected,
            thickness_factor=[s_tfa.val, s_tfb.val, s_tfc.val],
            thickness_factor2=[s_tf2a.val, s_tf2b.val, s_tf2c.val],
        )

    def redraw(_=None):
        if state["_busy"]: return
        state["_busy"] = True
        try:
            _visible_dynamic_sliders()
            params = _collect_params()
            try:
                curves1d = build_one_d(params)
            except Exception:
                curves1d = _compute_curves_fallback(params)
            _draw_1d(ax1d, curves1d)

            if mode_radio.value_selected == "2D":
                try:
                    result2d = build_two_d(curves1d, params)
                    _draw_2d(ax2d, result2d)
                except Exception as e:
                    ax2d.clear()
                    ax2d.text(0.5, 0.5, f"2D error:\n{e}", ha='center', va='center', color='red'); ax2d.set_axis_off()
            else:
                ax2d.clear()
                ax2d.text(0.5, 0.5, "2D disabled (switch to '2D')", ha='center', va='center', color='gray'); ax2d.set_axis_off()

            fig.canvas.draw_idle()
        finally:
            state["_busy"] = False

    # callbacks
    for s in [s_amp, s_rad, s_nc, s_ofx, s_mfx, s_mfy, s_wt,
              *pf_sliders, *my_sliders,
              s_th, s_tfa, s_tfb, s_tfc, s_tf2a, s_tf2b, s_tf2c]:
        s.on_changed(redraw)
    mode_radio.on_clicked(redraw)
    mode2d_radio.on_clicked(redraw)

    def on_reset(_):
        for s in [s_amp, s_rad, s_nc, s_ofx, s_mfx, s_mfy, s_wt,
                  *pf_sliders, *my_sliders,
                  s_th, s_tfa, s_tfb, s_tfc, s_tf2a, s_tf2b, s_tf2c]:
            s.reset()
        mode_radio.set_active(1)  # 2D
        mode2d_radio.set_active(("constant","linear","variable","collapsed","sbend").index(init["thickness_mode"]))
        redraw()

    def on_save(_):
        out = f"oneD_2D_demo_{int(time.time())}.png"
        fig.savefig(out, dpi=200)
        print(f"[saved] {os.path.abspath(out)}")

    btn_reset.on_clicked(on_reset)
    btn_save.on_clicked(on_save)

    _visible_dynamic_sliders()
    redraw()
    plt.show()

if __name__ == "__main__":
    main()
