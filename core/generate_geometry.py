# core/generate_geometry.py

import copy
import numpy as np
import cadquery as cq

from pytictoc import TicToc
t = TicToc()

from core.types import Params, BuildReport, BuildReportBend
from builders.one_d_build import generate_1D_curves
from builders.two_d_build import generate_2D_cross_sections
from builders.three_d_build import generate_3D_model

from io_modules.progress_bar import start_progress_bar, stop_progress_bar

from builders.build_modules.interpolate_bend import interpolate_bending_config

# -----------------------------
# Configuration Parameters
# -----------------------------

from core.config import (
    optionsConfig,
    BaselineGeometryConfig,
    NURBSConfig,
    CurveSettings,
    BendSettings
                            )

options = optionsConfig()
baseline = BaselineGeometryConfig()
nurbs = NURBSConfig()
curves = CurveSettings()
bend = BendSettings()

def generate_geometry(params: Params):

    print("Preparing build")

    Curve_1D = generate_1D_curves(params)

    # Generate the 2D cross-section
    Xsec_2D = generate_2D_cross_sections(Curve_1D, params)

    # Build the 3D model
    Model_3D = generate_3D_model(Xsec_2D, params, None)

    # return Model_3D
    return BuildReport(
        params=params,
        curves1d=Curve_1D,
        xsections2d=Xsec_2D,
        model3d=Model_3D
    )

def _apply_overrides(dst_params: Params, overrides: dict) -> None:
    """
    Sets attributes only if they exist on dst_params.
    Useful keys in bending CSV: amplitude0, desired_radius, period_factors,
    offset_factor_x, mid_factor_x, curve_weight, thickness_factor, thickness, etc.
    """
    for k, v in overrides.items():
        if k == "angular_section":
            continue
        if v is None:
            continue
        if hasattr(dst_params, k):
            setattr(dst_params, k, v)

def generate_geometry_bend(params: Params, bending_csv_path: str, testing_mode: bool) -> BuildReportBend:
    """
    Bending mode:
      - Use the incoming 'params' as angle 0°.
      - Read CSV of per-angle overrides; each row defines a cross-section at that angle.
      - Generate a 1D curve + 2D cross-section for each angle (0° + CSV rows).
      - Store lists in BuildReportBend; your 3D builder (already implemented)
        will loft these sequentially.
    """

    print("Processing input configurations")

    # 1) set up lists for storing generated geometries
    curves1d_list = []
    xsections2d_list = []
    xsection_params = []

    # 2) Interpolate bending factors for each row
    rows = interpolate_bending_config(
        input_csv_path=bending_csv_path,
        base_params=params,
        step_deg=getattr(params, "bending_step_deg", bend.angle_intervals)
    )

    # # print build data from interpolation
    # for row in rows:
    #     print(row)

    # inferred total angular section
    total_angular_section = rows[-1]["angular_section"] if rows else 360.0

    # 3) For each row, clone 'params', apply overrides, build 1D + 2D

    print("Preparing build")
    t.tic()

    for i, row in enumerate(rows, start=0):
        params_i = copy.deepcopy(params)
        _apply_overrides(params_i, row)

        curve_i = generate_1D_curves(params_i)
        xsec_i  = generate_2D_cross_sections(curve_i, params_i)

        curves1d_list.append(curve_i)
        xsections2d_list.append(xsec_i)

        xsection_params.append(params_i)

    prep_time = t.tocvalue()
    print(f"Build preparation time: {prep_time:.2f} seconds")

    # 4) 3D builder consumes the list of 2d cross-sections and generates the 3D model

    # comp time estimate
    ct_estimate =  prep_time * (4.0 * np.random.uniform(0.96, 1.04))

    print("Beginning Build")
    print(f"Estimated build time: {ct_estimate:.2f} seconds")

    # Start the timer
    t.tic()

    if testing_mode:
        setattr(params, "bending_enabled", False)  # for testing - produce only plots
        model_3d = generate_3D_model(xsections2d_list[0], params, None)
    else:
        start_progress_bar(ct_estimate)
        try:
            model_3d = generate_3D_model(xsections2d_list, params, total_angular_section)
        finally:
            stop_progress_bar()

    # Stop the timer
    t.toc("Total build time:")

    return BuildReportBend(
        params=params,
        curves1d_list=curves1d_list,
        xsections2d_list=xsections2d_list,
        model3d=model_3d
    )