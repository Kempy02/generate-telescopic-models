# geometry/base_params.py

from dataclasses import dataclass, field
from typing import List, Union

@dataclass
class GeometryParams:
    def __init__(self,
                amplitude0: float = 20.0,
                desired_radius: float = 25.0,
                period_factors: List[float] = None,
                offset_factor_x: float = 0.0,
                mid_factor_x: float = 0.0,
                mid_factor_y: float = 0.0,
                min_y_positions: List[float] = None, 
                curve_weight: float = 5.0,
                weights: List[float] = None,
                thickness: float = 1.0, 
                thickness_factor: List[float] = None,
                center_offset: float = 0.0,
                bending_enabled: bool = False,
                angular_section: float = 360,
                revolve_angle: float = 360,
                export_filename: str = "final_model",
                export_folder: str = "test_models"):

        self.amplitude0 = amplitude0
        self.desired_radius = desired_radius
        self.period_factors = period_factors if period_factors is not None else [1, 1, 1]
        self.offset_factor_x = offset_factor_x
        self.mid_factor_x = mid_factor_x
        self.mid_factor_y = mid_factor_y
        self.min_y_positions = min_y_positions if min_y_positions is not None else [0, 0, 0]
        self.curve_weight = curve_weight
        self.weights = weights if weights is not None else [1.0, curve_weight, 1.0, curve_weight, 1.0]
        self.thickness = thickness
        self.thickness_factor = thickness_factor if thickness_factor is not None else [1, 1, 1]
        self.center_offset = center_offset
        self.bending_enabled = bending_enabled
        self.angular_section = angular_section
        self.revolve_angle = revolve_angle
        self.export_filename = export_filename
        self.export_folder = export_folder

# @dataclass
# class GeometryParams:
#     amplitude0: float = 20.0,
#     desired_radius: float = 25.0,
#     period_factors: List[float] = None,
#     offset_factor_x: float = 0.0,
#     mid_factor_x: float = 0.0,
#     mid_factor_y: float = 0.0,
#     min_y_positions: List[float] = None, 
#     curve_weight: float = 5.0,
#     weights: List[float] = None,
#     thickness: float = 1.0, 
#     thickness_factor: List[float] = None,
#     center_offset: float = 0.0,
#     bending_enabled: bool = False,
#     angular_section: float = 360,
#     revolve_angle: float = 360,
#     export_filename: str = "final_model",
#     export_folder: str = "test_models"