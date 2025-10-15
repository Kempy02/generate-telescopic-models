# builders/three_d_build

import cadquery as cq

from core.types import (
    Params, 
    CrossSections2D, 
    Model3D
)
from builders.build_modules.threeD_helpers import (
    create_3d_model,
    create_3d_model_bending,
)
from core.config import (
    BaselineGeometryConfig,
    optionsConfig
)
baseline = BaselineGeometryConfig()
options = optionsConfig()

def generate_3D_model(xsec: CrossSections2D | list[CrossSections2D], params: Params, angular_section: float | None) -> Model3D:
    """
    If bending is enabled, expect xsec to be a list of CrossSections2D,
    otherwise expect a single CrossSections2D.
    """
    if getattr(params, "bending_enabled", False):
        if not isinstance(xsec, list):
            raise ValueError("bending_enabled=True but xsec is not a list")

        # Extract 2D sections and thickness factors as lists
        cross_sections = [xs.twoD_cross_section for xs in xsec]
        thickness_factors_list = [xs.thickness_factors for xs in xsec] 

        # Extract cap thicknesses
        cap_thicknesses = [xs.cap_thickness for xs in xsec]

        model = create_3d_model_bending(
            cross_sections,
            loft_offset=params.center_offset,
            angular_section=angular_section,
            params=params,
            keying_enabled=options.keying_feature,
            cap_thickness=cap_thicknesses[0] if not options.constant_cap_thickness else baseline.cap_thickness
        )
    else:
        model = create_3d_model(
            xsec.twoD_cross_section,
            revolve_offset=params.center_offset,
            params=params,
            keying_enabled=options.keying_feature,
            cap_thickness=xsec.cap_thickness if not options.constant_cap_thickness else baseline.cap_thickness
        )

    return Model3D(threeD_model=model)