# core/param_builder.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
from core.types import Params
from core.config import BaselineGeometryConfig
from io_modules.read_csv import read_param_rows_csv

def _pick_angle0(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the row with angular_section == 0. If missing, take the smallest angle."""
    rows2 = [dict(r) for r in rows if "angular_section" in r]
    if not rows2:
        raise ValueError("Config CSV must include at least one row with angular_section.")
    for r in rows2:
        r["angular_section"] = float(r["angular_section"])
    rows2.sort(key=lambda r: r["angular_section"])
    if rows2[0]["angular_section"] != 0.0:
        # allow “closest to 0” as base if 0 not present
        pass
    return rows2[0]

def build_params_from_config_csv(
    top_row: Dict[str, Any],
    baseline: BaselineGeometryConfig,
) -> Tuple[Params, str, bool]:
    """
    Reads the per-model config CSV and builds a Params for the base (angle 0) row.
    Returns: (params, config_csv_path, use_linear_fast)
    """
    config_csv = str(top_row.get("config_csv") or "").strip().strip('"').strip("'")
    if not config_csv:
        raise ValueError("`config_csv` is required in the top-level CSV.")

    use_linear_fast = bool(top_row.get("use_linear_fast", False))

    rows = read_param_rows_csv(config_csv)
    base = _pick_angle0(rows)

    # Build Params from angle-0 row + required meta from the top CSV
    export_filename = str(top_row.get("Prototype ID") or "").strip()
    export_folder  = str(top_row.get("export_folder") or "prototype_models").strip()

    # Fills in only the fields your builders actually use.
    params = Params(
        amplitude0       = float(base.get("amplitude0", 20.0)),
        desired_radius   = float(base.get("desired_radius", 25.0)),
        period_factors   = list(base.get("period_factors", [1.0,1.0,1.0])),
        offset_factor_x  = float(base.get("offset_factor_x", 0.0)),
        mid_factor_x     = float(base.get("mid_factor_x", 0.0)),
        mid_factor_y     = float(base.get("mid_factor_y", 0.0)),
        min_y_positions  = list(base.get("min_y_positions", [0.0,0.0,0.0,0.0])),
        curve_weight     = float(base.get("curve_weight", 5.0)),
        weights          = list(base.get("weights", [1.0, 5.0, 1.0, 5.0, 1.0])),
        thickness        = float(base.get("thickness", 1.0)),
        thickness_factor = base.get("thickness_factor", [1.0,1.0,1.0]),
        thickness_factor2= base.get("thickness_factor2", [1.0,1.0,1.0]),
        thickness_mode   = str(base.get("thickness_mode", "variable")),
        n_curves         = int(base.get("n_curves", baseline.n_curves)),
        cap_thickness    = float(base.get("cap_thickness", baseline.cap_thickness)),
        center_offset    = baseline.revolve_offset,  # may switch below
        export_filename  = export_filename,
        export_folder    = export_folder,
        bending_enabled  = (not use_linear_fast),
        angular_section  = float(base.get("angular_section", 0.0)),
    )

    # center offset: revolve when linear_fast, else loft/bend
    params.center_offset = baseline.revolve_offset if use_linear_fast else baseline.loft_offset
    return params, config_csv, use_linear_fast

