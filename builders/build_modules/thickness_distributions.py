# builders/build_modules/thickness_distributions.py

import math
import numpy as np

def constant_thd(outer_points, thickness_value=1.0):
    """All points => the same thickness."""
    return [thickness_value]*len(outer_points)

def linear_thd(outer_points, start_thickness=1.0, end_thickness=2.0):
    """
    Interpolate from start_thickness to end_thickness
    across the entire set of outer_points.
    """
    n_points = len(outer_points)
    if n_points <= 1:
        return [start_thickness]*(n_points or 1)

    thicknesses = []
    for i in range(n_points):
        t = i/(n_points-1)
        val = start_thickness + t*(end_thickness - start_thickness)
        thicknesses.append(val)
    return thicknesses

def variable_thd(outer_points, vt_control_points, all_thicknesses):
    """
    Piecewise interpolation approach from your original code.
    'all_thicknesses' is a list-of-lists with thickness values for each segment.
    """
    # Flatten vt_control_points
    vt_cp_array = np.vstack(vt_control_points)
    cp_in_curve_idx = []

    # For each control point, find the closest outer point
    for (x_cp, y_cp) in vt_cp_array:
        distances = [math.hypot(x_cp - x_o, y_cp - y_o) for (x_o,y_o) in outer_points]
        idx = distances.index(min(distances))
        cp_in_curve_idx.append(idx)

    # The first should map to 0, the last to len(outer_points)-1
    if cp_in_curve_idx[0] != 0 or cp_in_curve_idx[-1] != len(outer_points)-1:
        raise ValueError("First control point must map to 0 and last must map to the final outer_point index.")

    # Segment them
    seg_indices = []
    start_idx = 0
    for segment in vt_control_points:
        seg_len = len(segment)
        seg = cp_in_curve_idx[start_idx : start_idx + seg_len]
        # remove duplicates while preserving order
        unique = []
        for x in seg:
            if x not in unique:
                unique.append(x)
        seg_indices.append(unique)
        start_idx += seg_len

    point_thicknesses = []
    for i, seg_idx in enumerate(seg_indices):
        thickness_vals = all_thicknesses[i]  
        for j in range(len(seg_idx)-1):
            start_t = thickness_vals[j]
            end_t   = thickness_vals[j+1]

            idx_start = seg_idx[j]
            idx_end   = seg_idx[j+1]
            
            n = idx_end - idx_start

            for k in range(n):
                frac = k/n
                interp = start_t + frac*(end_t - start_t)
                point_thicknesses.append(interp)

            
        # Add the last thickness for the segment
        # Add the final thickness of that segment
        point_thicknesses.append(thickness_vals[-1])

    return point_thicknesses