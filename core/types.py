# core/types.py
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

Point = Tuple[float, float]
Point3 = Tuple[float, float, float]

@dataclass
class Params:
    amplitude0: float
    desired_radius: float
    period_factors: List[float]
    offset_factor_x: float
    mid_factor_x: float
    mid_factor_y: float
    min_y_positions: List[float]
    curve_weight: float
    weights: List[float]
    thickness: float
    thickness_factor: Any
    center_offset: float
    export_filename: str
    export_folder: str
    bending_enabled: bool
    angular_section: float
    revolve_angle: float

@dataclass
class Curves1D:
    control_points: List[Point]
    curve_points: List[Point]
    cp_idx: List[int]

@dataclass
class CrossSections2D:
    # one or many cross-sections; if many, ordered along centerline param s
    twoD_cross_section: List[Point]
    thickness_factors: List[List[float]]
    arc_length: float

@dataclass
class BendingSections:
    no_sections: int
    angular_sections: List[float]
    sectional_cross_sections: List[List[Point]]

@dataclass
class Model3D:
    threeD_model: Any

@dataclass
class BaseComponents:
    base_exploded: Any
    foundation: Any
    seal: Any

@dataclass
class RunOptions:
    export_model: bool = True
    export_cross_sections: bool = False
    export_bases: bool = False
    export_exploded_system: bool = False
    plot_1d: bool = False
    plot_2d: bool = False
    plot_3d: bool = False
    directory: Optional[str] = None
    ch_export_type: str = None
    overwrite: bool = True

@dataclass
class BuildReport:
    params: Params
    curves1d: Curves1D
    xsections2d: CrossSections2D
    model3d: Model3D
    bases: Optional[Dict[str, Any]] = None
    timings: Dict[str, float] = None

@dataclass
class BuildReportBend:
    params: "Params"
    curves1d_list: List["Curves1D"]
    xsections2d_list: List["CrossSections2D"]
    model3d: "Model3D"
    bases: Optional[Dict[str, Any]] = None
    timings: Dict[str, float] = None
