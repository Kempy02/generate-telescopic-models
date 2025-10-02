# core.config.py

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class optionsConfig:
    plot_curve_flag: bool = True
    export_base_exploded_flag: bool = True
    export_bases_flag: bool = True
    use_base: int = 2                     # 0 = auto, 1 = min, 2 = mid, 3 = max
    export_exploded_system: bool = False
    export_crossSection_flag: bool = False
    export_model_flag: bool = True
    plot_prototypes_flag: bool = True
    test_2d_mode = False              # If True, only generate 2D cross-section (no 3D models - useful if failing to generate 3D)
    calculate_area_flag: bool = False
    keying_feature: bool = False        # If True, add keying features to the sections
    engrave_text: bool = False          # If True, engrave text on the sections
    ruled_flag: bool = True             # Set True unless failing - If True, use ruled loft (straighter lines between sections), else smooth loft
    constant_cap_thickness: bool = False  # If True, keep cap thickness constant, else scale with thickness value
    thickness_mode: str = "variable"    # "constant", "linear", "variable", "collapsed", "sbend"
    default_thickness_factors: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

@dataclass
class BaselineGeometryConfig:
    start_x: float = 0
    start_y: float = 0
    cap_height: float = 1
    cap_length: float = 10
    upper_cap_length: float = 2.5
    cap_thickness: float = 1.0
    cap_pts_ratio: float = 0.95
    inside_tolerance: float = 5
    n_curves: int = 5
    revolve_offset: float = 1.0
    loft_offset: float = 0.0
    keying_offset: float = 2.0

@dataclass
class NURBSConfig:
    # Setup the degrees/order/weights
    degree: int = 3
    order: int = None
    knot_c: int = 1
    degree0: int = 2
    cp1_weight0: int = 5
    weights0: list[int] | None = None
    def __post_init__(self):
        if self.order is None:
            self.order = self.degree + 1
        if self.weights0 is None:
            self.weights0 = [1, self.cp1_weight0, 1]

@dataclass
class CurveSettings:
    # Resolution for the curve points
    resolution: int = 250
    # Offset factors for the curves
    offset_factor_y0: float = 1   # fixed at 1
    true_mid: float = 1
    rel_mid: float = 1 - true_mid

@dataclass
class ResampleSettings:
    resolution: int = 1000           # 500~1500
    edge_fraction: float = 0.01      # 0.8% to bridges
    vertex_threshold: float = 100    # 0.1 mm to detect vertices

@dataclass
class BendSettings:
    total_angular_section: float = 360  # degrees
    angle_intervals: float = 10.0  # degrees between each workplane

@dataclass
class BaseBuildSettings:
    foundation_radius: float = 48.0
    base_tolerance: float = 0.2  # 0.2 mm
    baseline_geo_radius: float = 35.25
    base_extension: float = 15.0  # Extra length for the base beyond the main body
    foundation_height: float = 5.0
    slide_length: float = 3.0
    f_screw_tolerance: float = 5.0
    no_screws: int = 6
    squeeze_tolerance: float = 0.8  # 0.4 mm

