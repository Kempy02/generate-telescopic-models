# core/param_builder.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import ast

from core.types import Params
from core.config import BaselineGeometryConfig, optionsConfig

def _parse_list(val: Any, default: List[float]) -> List[float]:
    """Accept list or string like '[1, 1, 1]'; fall back to default."""
    if val is None:
        return list(default)
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        s = val.strip()
        try:
            out = ast.literal_eval(s)
            if isinstance(out, (list, tuple)):
                return list(out)
        except Exception:
            pass
    return list(default)

def _is_truthy_path(x: Any) -> bool:
    if x is None:
        return False
    s = str(x).strip().strip('"').strip("'")
    return s not in ("", "None", "none", "null")

def build_params_from_row(
    row: Dict[str, Any],
    baseline: BaselineGeometryConfig,
    *,
    bending_configs_dir: str = "datasets/bending_configs/",
) -> Tuple[Params, Optional[str]]:
    """
    Build a Params object directly from a CSV row.
    Any fields not present in the row are filled from simple defaults and config.
    Returns (params, bending_csv_path or None).
    """

    # --- bending decision & path ---
    bend_spec = row.get("bending_config")
    bending_enabled = _is_truthy_path(bend_spec)
    bending_csv_path = None
    if bending_enabled:
        fname = str(bend_spec).strip().strip('"').strip("'")
        bending_csv_path = bending_configs_dir + fname

    # --- core design fields (come from CSV, else sensible defaults) ---
    amplitude0       = float(row.get("amplitude0", 20.0))
    desired_radius   = float(row.get("desired_radius", 25.0))
    period_factors   = _parse_list(row.get("period_factors"),   [1.0, 1.0, 1.0])
    offset_factor_x  = float(row.get("offset_factor_x", 0.0))
    mid_factor_x     = float(row.get("mid_factor_x", 0.0))
    mid_factor_y     = float(row.get("mid_factor_y", 0.0))
    min_y_positions  = _parse_list(row.get("min_y_positions"),  [0.0, 0.0, 0.0])
    curve_weight     = float(row.get("curve_weight", 5.0))
    # If weights not provided, keep your previous pattern that references curve_weight
    weights          = _parse_list(row.get("weights"), [1.0, curve_weight, 1.0, curve_weight, 1.0])
    thickness        = float(row.get("thickness", 1.0))
    thickness_factor = _parse_list(row.get("thickness_factor"), [1.0, 1.0, 1.0])

    # --- meta / export ---
    export_filename  = str(row.get("Prototype ID") or "model")
    export_folder    = str(row.get("export_folder") or "prototype_models")

    # --- motion / angles (defaults come from config conventions) ---
    revolve_angle    = float(row.get("revolve_angle", 360.0))
    angular_section  = float(row.get("angular_section", 360.0))

    # --- center offset depends on bending-or-not ---
    center_offset    = baseline.loft_offset if bending_enabled else baseline.revolve_offset

    params = Params(
        amplitude0=amplitude0,
        desired_radius=desired_radius,
        period_factors=period_factors,
        offset_factor_x=offset_factor_x,
        mid_factor_x=mid_factor_x,
        mid_factor_y=mid_factor_y,
        min_y_positions=min_y_positions,
        curve_weight=curve_weight,
        weights=weights,
        thickness=thickness,
        thickness_factor=thickness_factor,
        center_offset=center_offset,
        export_filename=export_filename,
        export_folder=export_folder,
        bending_enabled=bending_enabled,
        angular_section=angular_section,
        revolve_angle=revolve_angle,
    )

    return params, bending_csv_path
