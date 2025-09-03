# builders/one_d_build.py

# import struct types
from core.types import Params, Curves1D

# import necessary functions
from builders.build_modules.oneD_helpers import (
    validate_parameters,
    compute_x_increments_and_y_positions,
    generate_cap_curve,
    generate_curves
)

# -----------------------------
# 1a) Hard-coded or derived "globals" from config file
#  - Should change to struct at some point
# -----------------------------

from core.config import (optionsConfig,
                            BaselineGeometryConfig,
                            NURBSConfig,
                            CurveSettings
                            )

options = optionsConfig()
baseline = BaselineGeometryConfig()
nurbs = NURBSConfig()
curves = CurveSettings()

def generate_1D_curves(params: Params) -> Curves1D:

    """
    Generate 1D curves based on the provided parameters.

    Args:
        params (Params): Parameters for generating the curves.

    Returns:
        Curves1D: An object containing the generated curves.
    """
    
    # ---------------------------------
    # 1) Import user-supplied parameters from params
    # ---------------------------------

    amplitude0       = params.amplitude0
    desired_radius   = params.desired_radius
    period_factors   = params.period_factors
    offset_factor_x  = params.offset_factor_x
    mid_factor_x     = params.mid_factor_x
    mid_factor_y     = params.mid_factor_y
    min_y_positions  = params.min_y_positions
    curve_weight     = params.curve_weight
    thickness        = params.thickness
    thickness_factor = params.thickness_factor
    export_filename  = params.export_filename
    export_folder    = params.export_folder
    center_offset    = params.center_offset

    # ---------------------------------
    # 1c) Derived parameters
    # --------------------------------

    weights  = [1, curve_weight, 1, curve_weight, 1]
    offset_factor_x0 = offset_factor_x

    # -----------------------------
    # 2) Validate & compute
    # -----------------------------
    period_values, min_y_positions, n_periods, n_descending_curves = validate_parameters(
        period_factors     = period_factors,
        min_y_positions    = min_y_positions,
        desired_radius     = desired_radius,
        inside_tolerance   = baseline.inside_tolerance,
        n_curves           = baseline.n_curves
    )

    x_increments, y_positions = compute_x_increments_and_y_positions(
        baseline.n_curves,
        amplitude0,
        min_y_positions,
        period_values,
        baseline.start_y
    )
    
    # -----------------------------
    # 3) CAP + main curves
    # -----------------------------
    (all_control_points, all_curve_points,
        control_points_idx, control_points_idx_names,
        curve_points_idx, curve_points_idx_names,
        end_x0, end_y0
    ) = generate_cap_curve( 
        baseline.start_x, baseline.start_y,
        baseline.cap_height, baseline.cap_length,
        nurbs.weights0, nurbs.degree0
    )

    (all_control_points, all_curve_points,
        control_points_idx, control_points_idx_names,
        curve_points_idx, curve_points_idx_names
    ) = generate_curves(
        baseline.n_curves,
        x_increments, y_positions,
        all_control_points, all_curve_points,
        control_points_idx, control_points_idx_names,
        curve_points_idx, curve_points_idx_names,
        end_x0, end_y0,
        offset_factor_x0, curves.offset_factor_y0,
        mid_factor_x, mid_factor_y,
        curves.true_mid, curves.rel_mid,
        thickness, baseline.inside_tolerance,
        nurbs.degree, nurbs.order, nurbs.knot_c,
        curves.resolution,
        weights, center_offset
    )

    # # -----------------------------
    # # 4) Process the generated curves ready for 2d construction
    # # -----------------------------

    # oneD_cross_section_points = process_outer_points(all_curve_points)

    # vt_control_points = generate_vt_control_points(all_control_points, control_points_idx)

    return Curves1D(
        control_points=all_control_points,
        curve_points=all_curve_points,
        cp_idx=control_points_idx
    )