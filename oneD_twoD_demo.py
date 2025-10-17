"""
Qt-based interactive 1D/2D curve demo (slider UI) — trimmed + CSV export

- Left: 1D (cap + sequential)     Right: 2D cross-section
- Right-side tabbed controls with scrolling
- Numeric inputs are sliders with live value labels (float-friendly)
- Save Image + Export Config CSV buttons in a top bar
"""

from __future__ import annotations
import sys, math, json, csv
from types import SimpleNamespace
from typing import List

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QTabWidget, QGroupBox, QScrollArea, QLabel, QComboBox, QSlider, QPushButton,
    QFileDialog, QToolBar, QDoubleSpinBox, QSpinBox
)

# --------- your project imports ----------
try:
    from builders.one_d_build import generate_1D_curves as build_one_d  # Curves1D
except Exception:
    build_one_d = None

try:
    from builders.two_d_build import generate_2D_cross_sections as build_two_d  # CrossSections2D
except Exception:
    build_two_d = None

from builders.build_modules.oneD_helpers import (
    validate_parameters,
    compute_x_increments_and_y_positions,
    generate_cap_curve,
    generate_curves,
)

from core.config import BaselineGeometryConfig, NURBSConfig, CurveSettings
baseline = BaselineGeometryConfig()
nurbs    = NURBSConfig()
curve_cfg= CurveSettings()


# ---------- helpers ----------
def _pad_to(lst: List[float], length: int, fill: float = 0.0) -> List[float]:
    if len(lst) >= length: return lst[:length]
    return lst + ([lst[-1]] if lst else [fill]) * (length - len(lst))


def _compute_curves_fallback(params: SimpleNamespace):
    """Fallback 1D curve computation via low-level helpers."""
    weights0 = [1, nurbs.cp1_weight0, 1]
    main_weights = params.weights if (isinstance(params.weights, list) and len(params.weights) == 5) \
                   else [1, params.curve_weight, 1, params.curve_weight, 1]

    # convert UI fractions → absolute mm for helper API
    pv, my, _, _ = validate_parameters(
        period_factors=params.period_factors,
        min_y_positions=[v*params.amplitude0 for v in params.min_y_positions],
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

    # center_offset might not be present now → default to config
    center_offset = float(getattr(params, "center_offset", baseline.revolve_offset))

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
        center_offset=center_offset,
    )

    return SimpleNamespace(curve_points=all_curves, control_points=all_cps, cp_idx=cp_idx)


def _draw_1d(ax, curves1d):
    ax.clear()
    for seg in curves1d.curve_points:
        arr = np.asarray(seg)
        ax.plot(arr[:,0], arr[:,1], '-', color='tab:blue', lw=1.2)
    for cps in curves1d.control_points:
        arr = np.asarray(cps)
        ax.plot(arr[:,0], arr[:,1], 'o--', ms=3, lw=0.8, color='tab:orange', alpha=0.7)
    ax.set_title("1D curve (cap + sequential)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    try:
        allx = np.concatenate([np.asarray(seg)[:,0] for seg in curves1d.curve_points])
        ally = np.concatenate([np.asarray(seg)[:,1] for seg in curves1d.curve_points])
        xpad = max(1.0, 0.05*(allx.max()-allx.min()+1e-6))
        ypad = max(1.0, 0.05*(ally.max()-ally.min()+1e-6))
        ax.set_xlim(allx.min()-xpad, allx.max()+xpad)
        ax.set_ylim(ally.min()-ypad, ally.max()+ypad)
    except Exception:
        pass


def _draw_2d(ax, xsec2d):
    ax.clear()
    pts = np.asarray(xsec2d.twoD_cross_section)
    ax.plot(pts[:,0], pts[:,1], '-', color='tab:green', lw=1.2)
    ax.plot(pts[:,0], pts[:,1], 'o', ms=2, color='tab:green', alpha=0.6)
    ax.set_title("2D cross-section")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    xpad = max(1.0, 0.05*(pts[:,0].max()-pts[:,0].min()+1e-6))
    ypad = max(1.0, 0.05*(pts[:,1].max()-pts[:,1].min()+1e-6))
    ax.set_xlim(pts[:,0].min()-xpad, pts[:,0].max()+xpad)
    ax.set_ylim(pts[:,1].min()-ypad, pts[:,1].max()+ypad)


# ---------- slider widgets ----------
def _scale_from_step(step: float) -> int:
    """
    Scale factor to represent float steps as slider integers.
    e.g. step=0.01 -> 100; step=0.05 -> 20; step=0.1 -> 10
    """
    s = str(step)
    if "." in s:
        decimals = len(s.split(".")[1].rstrip("0"))
        # handle odd steps like 0.05 → keep enough precision
        # 0.05 -> 20, 0.025 -> 40, etc.
        base = 10 ** max(decimals, 0)
        # If step*base isn't integer, bump until it is
        scale = base
        while abs(step * scale - round(step * scale)) > 1e-12:
            scale *= 2
        return scale
    return 1


class FloatSlider(QWidget):
    """
    Float slider + spinbox (synced).
    - Drag slider for coarse change
    - Type in spinbox for exact values
    Signals:
      valueChanged(float)
    """
    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, step: float, init: float, suffix: str = ""):
        super().__init__()
        # from PySide6.QtWidgets import QHBoxLayout
        self.suffix = suffix
        self.scale = _scale_from_step(step)

        self._min = int(round(minimum * self.scale))
        self._max = int(round(maximum * self.scale))
        self._step = max(1, int(round(step * self.scale)))

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Slider (integer domain)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(self._min, self._max)
        self.slider.setSingleStep(self._step)
        self.slider.setPageStep(self._step * 5)
        self.slider.setValue(int(round(init * self.scale)))

        # Spinbox (float domain)
        self.spin = QDoubleSpinBox()
        # derive decimals from step
        decimals = 0
        s = str(step)
        if "." in s:
            decimals = len(s.split(".")[1].rstrip("0"))
        # also handle steps like 0.05 cleanly
        while abs(step * (10**decimals) - round(step * (10**decimals))) > 1e-12 and decimals < 6:
            decimals += 1
        self.spin.setDecimals(decimals)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.setValue(init)
        self.spin.setSuffix(f" {self.suffix}".strip())

        lay.addWidget(self.slider, stretch=1)
        lay.addWidget(self.spin, stretch=0)

        # Sync both ways (guard re-entrancy)
        self._updating = False

        def slider_changed(ivalue: int):
            if self._updating:
                return
            self._updating = True
            v = ivalue / float(self.scale)
            self.spin.setValue(v)
            self._updating = False
            self.valueChanged.emit(v)

        def spin_changed(v: float):
            if self._updating:
                return
            self._updating = True
            self.slider.setValue(int(round(v * self.scale)))
            self._updating = False
            self.valueChanged.emit(float(v))

        self.slider.valueChanged.connect(slider_changed)
        self.spin.valueChanged.connect(spin_changed)

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, v: float):
        self.spin.setValue(float(v))


class IntSlider(QWidget):
    """
    Integer slider + spinbox (synced).
    - Same API as FloatSlider: .value(), .setValue(), .valueChanged(float)
    - Emits float for consistency, but holds ints internally.
    """
    valueChanged = Signal(float)

    def __init__(self, minimum: int, maximum: int, step: int, init: int, suffix: str = ""):
        super().__init__()
        self._min, self._max, self._step = int(minimum), int(maximum), max(1, int(step))
        self.suffix = suffix

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(self._min, self._max)
        self.slider.setSingleStep(self._step)
        self.slider.setPageStep(self._step * 5)
        self.slider.setValue(int(init))

        self.spin = QSpinBox()
        self.spin.setRange(self._min, self._max)
        self.spin.setSingleStep(self._step)
        self.spin.setValue(int(init))
        # (QSpinBox has no native suffix like QDoubleSpinBox; skip or add a QLabel if desired.)

        lay.addWidget(self.slider, stretch=1)
        lay.addWidget(self.spin, stretch=0)

        self._updating = False

        def slider_changed(ivalue: int):
            if self._updating: return
            self._updating = True
            self.spin.setValue(ivalue)
            self._updating = False
            self.valueChanged.emit(float(ivalue))

        def spin_changed(v: int):
            if self._updating: return
            self._updating = True
            self.slider.setValue(int(v))
            self._updating = False
            self.valueChanged.emit(float(v))

        self.slider.valueChanged.connect(slider_changed)
        self.spin.valueChanged.connect(spin_changed)

        # Optional: smoother typing UX
        self.spin.setKeyboardTracking(False)
        self.spin.editingFinished.connect(lambda: self.valueChanged.emit(self.value()))

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, v: float):
        self.spin.setValue(int(round(v)))

# ---------- UI ----------
class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("1D/2D Curve Demo (Qt + sliders)")
        self.resize(1480, 900)

        # Defaults (aligned with param_builder.py, minus the extras you removed)
        self.defaults = dict(
            amplitude0=20.0,
            desired_radius=25.0,
            period_factors=[1.0,1.0,1.0],
            offset_factor_x=0.0,
            mid_factor_x=0.0,
            mid_factor_y=0.0,
            min_y_positions=[0.0,0.0,0.0,0.0],  # fractions
            curve_weight=5.0,
            weights=[1.0,5.0,1.0,5.0,1.0],
            thickness=1.0,
            thickness_factor=[1.0,1.0,1.0],
            thickness_factor2=[1.0,1.0,1.0],
            thickness_mode="variable",
            n_curves=baseline.n_curves,
            # "extras" still available as defaults (not shown in UI) so builders don't crash
            cap_thickness=baseline.cap_thickness,
            center_offset=baseline.revolve_offset,
            bending_enabled=True,
            angular_section=0.0,
            export_filename="live_demo",
            export_folder="prototype_models",
        )

        # ---- top toolbar (Save image / Export CSV) ----
        tb = QToolBar("Tools", self); self.addToolBar(Qt.TopToolBarArea, tb)
        btn_save = QPushButton("Save Image (PNG)"); btn_csv = QPushButton("Export Config CSV")
        btn_save.clicked.connect(self._save_image)
        btn_csv.clicked.connect(self._export_csv)
        tb.addWidget(btn_save); tb.addSeparator(); tb.addWidget(btn_csv)

        # ---- central area: figure ----
        central = QWidget(); self.setCentralWidget(central)
        h = QHBoxLayout(central); h.setContentsMargins(8,8,8,8); h.setSpacing(10)

        self.fig = Figure(figsize=(8,5), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax1 = self.fig.add_subplot(1,2,1)
        self.ax2 = self.fig.add_subplot(1,2,2)
        h.addWidget(self.canvas, stretch=3)

        # ---- right sidebar with tabs (scrollable) ----
        right = QTabWidget(); right.setTabPosition(QTabWidget.West)
        right.setMinimumWidth(420)
        right.addTab(self._build_tab_1d(), "1D")
        right.addTab(self._build_tab_2d(), "2D")
        h.addWidget(right, stretch=0)

        # First draw
        self._redraw()

    # 1D Tab
    def _build_tab_1d(self):
        tab = QWidget()
        outer = QVBoxLayout(tab); outer.setContentsMargins(0,0,0,0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); outer.addWidget(scroll)
        w = QWidget(); scroll.setWidget(w)
        lay = QVBoxLayout(w)

        # Group: core
        g_core = QGroupBox("Core")
        f = QFormLayout(g_core)
        self.s_amp = FloatSlider(0.0, 40.0, 0.1, self.defaults["amplitude0"], " mm")
        self.s_rad = FloatSlider(1.0, 50.0, 0.1, self.defaults["desired_radius"], " mm")
        self.s_nc  = IntSlider(3, 7, 2, self.defaults["n_curves"])
        f.addRow("amplitude0", self.s_amp); f.addRow("desired_radius", self.s_rad); f.addRow("n_curves", self.s_nc)

        # Group: layout factors
        g_layout = QGroupBox("Layout factors")
        f2 = QFormLayout(g_layout)
        self.s_ofx = FloatSlider(-2.0, 2.0, 0.01, self.defaults["offset_factor_x"])
        self.s_mfx = FloatSlider(-2.0, 2.0, 0.01, self.defaults["mid_factor_x"])
        self.s_mfy = FloatSlider(-2.0, 2.0, 0.01, self.defaults["mid_factor_y"])
        self.s_wt  = FloatSlider(0.1, 10.0, 0.1, self.defaults["curve_weight"])
        f2.addRow("offset_factor_x", self.s_ofx)
        f2.addRow("mid_factor_x",   self.s_mfx)
        f2.addRow("mid_factor_y",   self.s_mfy)
        f2.addRow("curve_weight",   self.s_wt)

        # Group: period factors (3)
        g_pf = QGroupBox("period_factors")
        f3 = QFormLayout(g_pf)
        self.s_pf0 = FloatSlider(0.01, 3.0, 0.01, self.defaults["period_factors"][0])
        self.s_pf1 = FloatSlider(0.01, 3.0, 0.01, self.defaults["period_factors"][1])
        self.s_pf2 = FloatSlider(0.01, 3.0, 0.01, self.defaults["period_factors"][2])
        f3.addRow("f0", self.s_pf0); f3.addRow("f1", self.s_pf1); f3.addRow("f2", self.s_pf2)

        # Group: min_y_positions (fractions shown in UI)
        g_my = QGroupBox("min_y_positions (fraction of amplitude)")
        f4 = QFormLayout(g_my)
        self.s_my0 = FloatSlider(0.0, 1.0, 0.01, self.defaults["min_y_positions"][0])
        self.s_my1 = FloatSlider(0.0, 1.0, 0.01, self.defaults["min_y_positions"][1])
        self.s_my2 = FloatSlider(0.0, 1.0, 0.01, self.defaults["min_y_positions"][2])
        self.s_my3 = FloatSlider(0.0, 1.0, 0.01, self.defaults["min_y_positions"][3])
        f4.addRow("y0", self.s_my0); f4.addRow("y1", self.s_my1)
        f4.addRow("y2", self.s_my2); f4.addRow("y3", self.s_my3)

        for g in (g_core, g_layout, g_pf, g_my):
            lay.addWidget(g)
        lay.addStretch(1)

        # connect changes
        for w in (self.s_amp, self.s_rad, self.s_nc, self.s_ofx, self.s_mfx, self.s_mfy,
                  self.s_wt, self.s_pf0, self.s_pf1, self.s_pf2, self.s_my0, self.s_my1,
                  self.s_my2, self.s_my3):
            w.valueChanged.connect(self._redraw)

        return tab

    # 2D Tab
    def _build_tab_2d(self):
        tab = QWidget()
        outer = QVBoxLayout(tab); outer.setContentsMargins(0,0,0,0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); outer.addWidget(scroll)
        w = QWidget(); scroll.setWidget(w)
        lay = QVBoxLayout(w)

        g2_core = QGroupBox("Thickness")
        f = QFormLayout(g2_core)
        self.s_th = FloatSlider(0.25, 5.0, 0.01, self.defaults["thickness"], " mm")

        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["constant","linear","variable","collapsed","sbend"])
        self.cb_mode.setCurrentText(self.defaults["thickness_mode"])

        f.addRow("thickness", self.s_th)
        f.addRow("thickness_mode", self.cb_mode)

        g2_f = QGroupBox("thickness_factor")
        f2 = QFormLayout(g2_f)
        self.s_tfa = FloatSlider(0.0, 1.0, 0.01, self.defaults["thickness_factor"][0])
        self.s_tfb = FloatSlider(0.0, 1.0, 0.01, self.defaults["thickness_factor"][1])
        self.s_tfc = FloatSlider(0.0, 1.0, 0.01, self.defaults["thickness_factor"][2])
        f2.addRow("a", self.s_tfa); f2.addRow("b", self.s_tfb); f2.addRow("c", self.s_tfc)

        g2_f2 = QGroupBox("thickness_factor2")
        f3 = QFormLayout(g2_f2)
        self.s_tf2a = FloatSlider(0.0, 1.0, 0.01, self.defaults["thickness_factor2"][0])
        self.s_tf2b = FloatSlider(0.0, 1.0, 0.01, self.defaults["thickness_factor2"][1])
        self.s_tf2c = FloatSlider(0.0, 1.0, 0.01, self.defaults["thickness_factor2"][2])
        f3.addRow("a2", self.s_tf2a); f3.addRow("b2", self.s_tf2b); f3.addRow("c2", self.s_tf2c)

        for g in (g2_core, g2_f, g2_f2):
            lay.addWidget(g)
        lay.addStretch(1)

        # connections
        for w in (self.s_th, self.s_tfa, self.s_tfb, self.s_tfc, self.s_tf2a, self.s_tf2b, self.s_tf2c):
            w.valueChanged.connect(self._redraw)
        self.cb_mode.currentIndexChanged.connect(self._redraw)

        return tab

    # Build Params (also supplies safe defaults for removed "extras")
    def _params(self) -> SimpleNamespace:
        min_y_frac = [self.s_my0.value(), self.s_my1.value(), self.s_my2.value(), self.s_my3.value()]
        n_curves = int(round(self.s_nc.value()))
        n_periods_req = int(math.ceil(n_curves/2))
        period_factors = _pad_to([self.s_pf0.value(), self.s_pf1.value(), self.s_pf2.value()], n_periods_req, 1.0)

        return SimpleNamespace(
            amplitude0=self.s_amp.value(),
            desired_radius=self.s_rad.value(),
            n_curves=n_curves,
            period_factors=period_factors,
            offset_factor_x=self.s_ofx.value(),
            mid_factor_x=self.s_mfx.value(),
            mid_factor_y=self.s_mfy.value(),
            min_y_positions=min_y_frac,       # fractions; converted in fallback
            curve_weight=self.s_wt.value(),
            weights=self.defaults["weights"],

            thickness=self.s_th.value(),
            thickness_mode=self.cb_mode.currentText(),
            thickness_factor=[self.s_tfa.value(), self.s_tfb.value(), self.s_tfc.value()],
            thickness_factor2=[self.s_tf2a.value(), self.s_tf2b.value(), self.s_tf2c.value()],

            # Safe defaults for removed extras:
            cap_thickness=self.defaults["cap_thickness"],
            center_offset=self.defaults["center_offset"],
            bending_enabled=self.defaults["bending_enabled"],
            angular_section=self.defaults["angular_section"],

            export_filename=self.defaults["export_filename"],
            export_folder=self.defaults["export_folder"],
        )

    def _redraw(self):
        p = self._params()

        # Compute 1D
        try:
            if build_one_d is not None:
                curves1d = build_one_d(p)
            else:
                raise RuntimeError("High-level 1D builder unavailable")
        except Exception:
            curves1d = _compute_curves_fallback(p)

        _draw_1d(self.ax1, curves1d)

        # Compute 2D
        self.ax2.clear()
        try:
            if build_two_d is None:
                raise RuntimeError("High-level 2D builder unavailable")
            xsec2d = build_two_d(curves1d, p)
            _draw_2d(self.ax2, xsec2d)
        except Exception as e:
            self.ax2.text(0.5, 0.5, f"2D error:\n{e}", color="red", ha="center", va="center")
            self.ax2.set_axis_off()

        self.canvas.draw_idle()

    # ---------- actions ----------
    def _save_image(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Figure as PNG", "curves.png", "PNG Image (*.png)")
        if not fn: return
        self.fig.savefig(fn, dpi=200, bbox_inches="tight")

    def _export_csv(self):
        # Build two rows like your example (0° and 180°), plus all relevant columns
        p = self._params()
        rows = []
        for ang in (0, 180):
            row = {
                "angular_section": ang,
                "amplitude0": float(p.amplitude0),
                "desired_radius": float(p.desired_radius),
                "n_curves": int(p.n_curves),
                "period_factors": json.dumps(list(p.period_factors)),
                "offset_factor_x": float(p.offset_factor_x),
                "mid_factor_x": float(p.mid_factor_x),
                "mid_factor_y": float(p.mid_factor_y),
                "curve_weight": float(p.curve_weight),
                "min_y_positions": json.dumps(list(p.min_y_positions)),
                "thickness_factor": json.dumps(list(p.thickness_factor)),
                "thickness_factor2": json.dumps(list(p.thickness_factor2)),
                "thickness": float(p.thickness),
                "thickness_mode": str(p.thickness_mode),
                "center_offset": float(getattr(p, "center_offset", baseline.revolve_offset)),
            }
            rows.append(row)

        # Choose columns & order (keeps your example columns first)
        fieldnames = [
            "angular_section",
            "amplitude0",
            "desired_radius",
            "period_factors",
            "offset_factor_x",
            "mid_factor_x",
            "curve_weight",
            "min_y_positions",
            "thickness_factor",
            "thickness",
            "thickness_mode",
            # extras for completeness:
            "mid_factor_y",
            "thickness_factor2",
            "n_curves",
            "center_offset",
        ]

        fn, _ = QFileDialog.getSaveFileName(self, "Export Config CSV", "config_export.csv", "CSV (*.csv)")
        if not fn: return
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    app = QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
