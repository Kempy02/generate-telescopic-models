# builders.build_modules.general_helpers

from shapely.geometry import Point

# helper functions to find/calculate min/max values
def find_min_x(curve_points):
    min_x = min(pt[0] for pt in curve_points)
    return min_x

def find_min_y(curve_points):
    min_y = min(pt[1] for pt in curve_points)
    return min_y

def find_max_x(curve_points):
    max_x = max(pt[0] for pt in curve_points)
    return max_x

def find_max_y(curve_points):
    max_y = max(pt[1] for pt in curve_points)
    return max_y

def find_min_value(curve_points: list[Point]) -> float:
    min_hypot = min((pt.x**2 + pt.y**2)**0.5 for pt in curve_points)
    return min_hypot

def find_max_value(curve_points: list[Point]) -> float:
    max_hypot = max((pt.x**2 + pt.y**2)**0.5 for pt in curve_points)
    return max_hypot

# helper function for base generation
def create_circle_of_radius(radius: float) -> list[tuple[float, float]]:
        center_point = Point(0, 0)
        circle = center_point.buffer(radius)
        return list(circle.exterior.coords)