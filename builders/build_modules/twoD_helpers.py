# builders/build_modules/twoD_helpers.py

import math
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from typing import List, Tuple

import builders.build_modules.general_helpers as helpers
import builders.build_modules.thickness_distributions as thickness_distributions

from core.config import (
    BaselineGeometryConfig,
    optionsConfig
)
baseline = BaselineGeometryConfig()
options = optionsConfig()

def process_outer_points(all_curve_points):
    """
    Stack all curve arrays, reverse them, shift so that max_x => x=0, etc.
    Returns (outer_points, outer_min_x, outer_min_y, outer_max_x, outer_max_y).
    """
    all_points = np.vstack(all_curve_points)
    outer_points = np.array(all_points)

    # Find max x
    # max_x = max(p[0] for p in outer_points)
    max_x = helpers.find_max_x(outer_points)

    # Reverse order
    outer_points = outer_points[::-1]

    # Shift so max_x => x=0
    shifted = []
    for (x,y) in outer_points:
        shifted.append((x - max_x, y))

    outer_points = shifted

    return outer_points

def generate_vt_control_points(all_control_points, control_points_idx):
    """
    Based on the original variable-thickness logic, keep 'even' indices, invert them, shift them.
    """
    import numpy as np

    vt_cp_idx = [np.where(np.arange(len(idx)) % 2 == 1, 0, idx)
                 for idx in control_points_idx]
    vt_control_points = []

    for i in range(len(all_control_points)):
        cp_array = all_control_points[i]
        vt_cp_idx_array = vt_cp_idx[i]
        filtered = []
        for j in range(len(cp_array)):
            if vt_cp_idx_array[j] != 0:
                filtered.append(cp_array[j])
        vt_control_points.append(filtered)

    # Reverse each sublist and shift so max_x => 0
    vt_control_points = [cp[::-1] for cp in vt_control_points[::-1]]
    max_x = max(pt[0] for cp in vt_control_points for pt in cp)

    out = []
    for cp in vt_control_points:
        out_cp = [(x - max_x, y) for (x,y) in cp]
        out.append(out_cp)

    return out

def apply_thickness(
    outer_points,
    mode,
    vt_control_points=None,
    all_thicknesses=None,
    constant_value=1.0,
    linear_start=1.0,
    linear_end=3.0,
    # new delayed args:
    delayed_start=0.10,
    delayed_ramp=0.10,
    delayed_end=0.10,
    delayed_apply_sections=None,
    delayed_baseline=(0.5, 1.0, 0.5),
):
    """
    Returns a thickness array for each point in outer_points, 
    according to the chosen mode: "constant", "linear", or "variable".
    """
    if mode == "constant":
        return thickness_distributions.constant_thd(outer_points, constant_value)
    elif mode == "linear":
        return thickness_distributions.linear_thd(outer_points, linear_start, linear_end)
    elif mode == "variable":
        if vt_control_points is None or all_thicknesses is None:
            raise ValueError("For 'variable' mode, provide vt_control_points and all_thicknesses.")
        return thickness_distributions.variable_thd(outer_points, vt_control_points, all_thicknesses)
    if mode == "delayed":
        return thickness_distributions.variable_thd_delayed(
            outer_points=outer_points,
            vt_control_points=vt_control_points,
            all_thicknesses=all_thicknesses,
            apply_sections=delayed_apply_sections,
            baseline_values=delayed_baseline,
            start_frac=delayed_start,
            ramp_frac=delayed_ramp,
            end_frac=delayed_end
        )
    else:
        raise ValueError(f"Unknown thickness mode: {mode}")

def generate_2d_profile(outer_points, thicknesses, cap_thickness):
    """
    Automatic Inner Profile Generation:
    1) For each (x_i, y_i), create a small circle with radius=thicknesses[i].
    2) Union + take convex_hull of pairs => 'capsules'.
    3) Union all capsules => final shape => output exterior boundary coords.
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union

    # Find the maximum x-coordinate among the outer points
    max_x = helpers.find_max_x(outer_points)

    point_buffers = []
    for (x,y), r in zip(outer_points, thicknesses):
        if x < baseline.cap_length and y < baseline.cap_height:
            circle = Point(x, y).buffer(baseline.cap_thickness, resolution=32)
        elif x > (max_x - baseline.upper_cap_length):
            circle = Point(x, y).buffer(cap_thickness, resolution=32)
        else:
            circle = Point(x, y).buffer(r, resolution=32)
        point_buffers.append(circle)

    capsules = []
    for i in range(len(point_buffers) - 1):
        unioned = point_buffers[i].union(point_buffers[i+1])
        cap = unioned.convex_hull
        capsules.append(cap)

    if len(capsules) == 0:
        if len(point_buffers) == 1:
            final_shape = point_buffers[0]
        else:
            return []
    else:
        final_shape = unary_union(capsules)

    if final_shape is None:
        return []

    if final_shape.geom_type == "Polygon":
        return list(final_shape.exterior.coords)
    elif final_shape.geom_type == "MultiPolygon":
        biggest = max(final_shape.geoms, key=lambda g: g.area)
        return list(biggest.exterior.coords)
    else:
        return []

def handle_thickness_factor(thickness_factor):

    thickness_factor_handled = []

    for i in range(len(thickness_factor)):
        
        x = thickness_factor[i]

        if i == 1:
            if x >= 1.00:
                # print(f"Warning: {x} is greater than 1.0. Setting to 0.99.")
                x = 0.9
            elif x <= 0.00:
                # print(f"Warning: {x} is less than 0.0. Setting to 0.01.")
                x = 0.01
            else:
                # print(f"Warning: {x} is not a problem")
                x = x
        else:
            x = x

        thickness_factor_handled.append(x)
    
    # thickness_factor_handled = [x for x in thickness_factor]

    return thickness_factor_handled

def remove_consecutive_duplicates(points):
    """
    Removes consecutive duplicate points from a list of (x,y).
    """
    if not points:
        return []
    cleaned_points = [points[0]]
    for point in points[1:]:
        if point != cleaned_points[-1]:
            cleaned_points.append(point)
    return cleaned_points


def filter_points_by_threshold(points, x_threshold=None, y_threshold=None,
                               x_operator='lt', y_operator='lt'):
    """
    Filters out points based on threshold conditions for x and y.
    x_operator='lt' => remove points with x > x_threshold, etc.
    """
    filtered = []
    for x, y in points:
        if x_threshold is not None:
            if x_operator == 'lt' and x > x_threshold:
                continue
            elif x_operator == 'gt' and x < x_threshold:
                continue
        if y_threshold is not None:
            if y_operator == 'lt' and y > y_threshold:
                continue
            elif y_operator == 'gt' and y < y_threshold:
                continue
        filtered.append((x, y))
    return filtered

def shift_and_close(coords):
    """
    Shift the coords so that the first point => x=0, preserve y,
    optionally ensure the ring is 'closed' if needed.
    """
    if not coords:
        return []

    first_x, first_y = coords[0]
    shifted = [(x - first_x, y) for (x,y) in coords]
    return shifted

def reorient_coords(coords_list, start_from_max_x=True, reverse_order=False):
    """
    Reorient a closed ring so that it starts from the max-Y point, optionally reverse it.
    """
    if not coords_list:
        return []

    if coords_list[-1] == coords_list[0]:
        coords_list = coords_list[:-1]

    if start_from_max_x:
        i_max = max(range(len(coords_list)), key=lambda i: coords_list[i][0])
        coords_list = coords_list[i_max:] + coords_list[:i_max]

    if reverse_order:
        coords_list.reverse()

    if coords_list[0] != coords_list[-1]:
        coords_list.append(coords_list[0])

    return coords_list

def create_2d_cross_section_points(oneD_points, point_thicknesses, cap_thickness):

    max_x = helpers.find_max_x(oneD_points)
    min_x = helpers.find_min_x(oneD_points)

    # generate the inner profile using the outer points and thickness variations
    twoD_points = generate_2d_profile(oneD_points, point_thicknesses, cap_thickness)

    # filter inner points to include only those with x less than outer_max_x
    twoD_points_filtered = filter_points_by_threshold(twoD_points, x_threshold=max_x, x_operator='lt')
    # further filter inner points to include only those with x greater than outer_min_x
    twoD_points_filtered = filter_points_by_threshold(twoD_points_filtered, x_threshold=min_x, x_operator='gt')

    # reorient the coordinates to start from max y and maintain the original order
    twoD_points_filtered = reorient_coords(twoD_points_filtered, start_from_max_x=True, reverse_order=False)
    # shift the points to close the polygon and generate the final cross-section points
    cross_section_points = shift_and_close(twoD_points_filtered)

    return cross_section_points
