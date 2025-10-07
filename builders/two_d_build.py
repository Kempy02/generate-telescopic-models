from typing import List, Dict, Any
from core.types import Params, Curves1D, CrossSections2D

from builders.build_modules.twoD_helpers import (
    process_outer_points,
    generate_vt_control_points,
    apply_thickness,
    handle_thickness_factor,
    create_2d_cross_section_points,
)

from builders.build_modules.resample_points import resample_cross_section_points

from core.config import (
    BaselineGeometryConfig,
    ResampleSettings,
    optionsConfig
)

baseline = BaselineGeometryConfig()
resample = ResampleSettings()
options = optionsConfig()

def _build_thickness_factors(params: Params) -> List[List[float]]:

    thickness_factors: List[List[float]] = []
    thickness_factor_cap = [baseline.cap_thickness, baseline.cap_thickness]
    thickness_factors.append(thickness_factor_cap)

    for i in range(int(params.n_curves)):
        tf = handle_thickness_factor(params.thickness_factor)
        if i < int(params.n_curves) - 1:
            thickness_factors.append(tf)
        else:
            tf[-1] = baseline.cap_thickness
        thickness_factors.append(tf)


    # Multiply each by baseline thickness
    thickness_factors = [[v * params.thickness for v in sub] for sub in thickness_factors]
    # Reverse lists to match existing ordering
    thickness_factors = [t[::-1] for t in thickness_factors[::-1]]
    return thickness_factors


def generate_2D_cross_sections(curves: Curves1D, params: Params) -> CrossSections2D:

    # process 1d curve - arrange into suitable format and extract relevant data
    oneD_cross_section_points = process_outer_points(curves.curve_points)
    vt_control_points = generate_vt_control_points(curves.control_points, curves.cp_idx)

    # Build thickness factors
    thickness_factors = _build_thickness_factors(params)

    # set cap_thickness option (based on thickness_mode)
    if params.thickness_mode == "sbend":
        options.constant_cap_thickness = True
    else:
        options.constant_cap_thickness = options.constant_cap_thickness

    # Apply thickness to the outer points
    point_thicknesses, cap_thickness = apply_thickness(
        outer_points=oneD_cross_section_points,
        mode=params.thickness_mode,                         
        vt_control_points=vt_control_points,
        all_thicknesses=thickness_factors,
        thickness_factor1=params.thickness_factor,
        thickness_factor2=params.thickness_factor2,
        baseline_thickness=params.thickness
    )
    # Create the 2d cross-section
    twoD_cross_section = create_2d_cross_section_points(
        oneD_points=oneD_cross_section_points,
        point_thicknesses=point_thicknesses,
        cap_thickness=baseline.cap_thickness if options.constant_cap_thickness else cap_thickness
    )

    total = resample.resolution
    twoD_cross_section_resampled, arc_length = resample_cross_section_points(twoD_cross_section, total)

    return CrossSections2D(
        twoD_cross_section=twoD_cross_section_resampled,
        thickness_factors=thickness_factors,
        arc_length=arc_length,
        thickness_value=params.thickness,
        cap_thickness=cap_thickness
    )