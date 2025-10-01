# builders/one_d.py

import numpy as np
from geomdl import NURBS


def validate_parameters(
    period_factors,
    min_y_positions,
    desired_radius,
    inside_tolerance,
    n_curves
):
    """
    Check lengths of min_y_positions, period_factors vs n_curves,
    then normalize them to produce period_values that sum to (desired_radius - inside_tolerance/2).
    Returns the computed period_values, plus derived counters like n_periods, n_descending_curves.
    """
    # Decide how many ascending/descending curves
    n_periods           = int(np.ceil(n_curves / 2))
    n_descending_curves = int(np.floor(n_curves / 2))
    max_radius = desired_radius - inside_tolerance/2

    # 1) Validate min_y_positions
    if len(min_y_positions) < n_descending_curves + 1:
        raise ValueError(f"Length of min_y_positions must be at least {n_descending_curves + 1}")
    elif len(min_y_positions) > n_descending_curves + 1:
        min_y_positions = min_y_positions[:n_descending_curves + 1]
        # print(f"Warning: min_y_positions had extra entries. Trimmed to {len(min_y_positions)}.")

    # 2) Validate & trim period_factors
    if len(period_factors) < n_periods:
        raise ValueError(f"Length of period_factors must be at least {n_periods}")
    elif len(period_factors) > n_periods:
        period_factors = period_factors[:n_periods]
        # print(f"Warning: period_factors had extra entries. Trimmed to {len(period_factors)}.")

    sum_f = sum(period_factors)
    if sum_f <= 0:
        raise ValueError("Sum of period_factors must be > 0.")

    # Determine usage based on even or odd n_curves
    if n_curves % 2 == 0:
        usage = sum_f
    else:
        usage = sum_f - (period_factors[-1] / 2.0)

    # Scale so the usage matches max_radius
    scale = max_radius / usage
    period_values = [f * scale for f in period_factors]

    # print(f"Normalized period_values: {period_values}")
    # print(f"  => Sum usage: {usage * scale:.2f} mm (should match {max_radius} mm)")

    return period_values, min_y_positions, n_periods, n_descending_curves

def compute_x_increments_and_y_positions(
    n_curves,
    amplitude0,
    min_y_positions,
    period_values,
    start_y=0
):
    """
    Compute x-increments and y-positions for each ascending/descending curve segment.
    Returns (x_increments, y_positions).
    """
    x_increments = [0] * n_curves
    y_positions_array  = [0] * (n_curves + 1)
    y_positions_array[0] = start_y

    p = 0  # period index
    for i in range(n_curves):
        if i % 2 == 0:  # Ascending
            amplitude        = amplitude0 - min_y_positions[p]
            y_positions_array[i+1] = y_positions_array[i] + amplitude
            x_increments[i]  = period_values[p] / 2
        else:            # Descending
            y_positions_array[i+1] = min_y_positions[p+1]
            x_increments[i]  = period_values[p] / 2
            p += 1

    return x_increments, y_positions_array

def generate_cap_curve(start_x, start_y,
                       cap_height, cap_length,
                       weights0, degree0):
    """
    Generate the initial 'cap' curve at the base of the actuator.
    """
    all_control_points = []
    all_curve_points   = []
    control_points_idx = []
    control_points_idx_names = []
    curve_points_idx   = []
    curve_points_idx_names   = []

    cp0_x0 = start_x
    cp0_y0 = start_y
    cp1_x0 = cap_length
    cp1_y0 = start_y
    cp2_x0 = cap_length
    cp2_y0 = cap_height

    control_points0 = [
        [cp0_x0, cp0_y0],
        [cp1_x0, cp1_y0],
        [cp2_x0, cp2_y0],
    ]
    all_control_points.append(control_points0)
    curve0_cp_idx = np.arange(1, len(control_points0)+1)
    control_points_idx.append(curve0_cp_idx)
    control_points_idx_names.append(['curve0'])
    curve_weight0 = weights0

    order0 = degree0 + 1
    knot_c0 = 1
    no_control_points0 = len(control_points0)
    knot_vector_length0 = no_control_points0 + order0
    n0 = no_control_points0 - 1
    internal_knots0 = n0 - order0 + 1

    if (knot_vector_length0 - 2*order0) >= 0:
        knot_vector0 = (
            [0]*order0 +
            [(i*knot_c0) for i in range(1, internal_knots0+1)] +
            [internal_knots0+1]*order0
        )
    else:
        raise ValueError("Number of cap control points must be >= degree.")

    curve = NURBS.Curve()
    curve.degree = degree0
    curve.ctrlpts = [[pt[0], pt[1]] for pt in control_points0]
    curve.weights = curve_weight0
    curve.knotvector = knot_vector0
    curve.delta = 0.01
    curve.evaluate()
    curve_points0 = np.array(curve.evalpts)
    all_curve_points.append(curve_points0[:-1])
    curve0_idx = np.arange(1, len(curve_points0)+1)
    curve_points_idx.append(curve0_idx)
    curve_points_idx_names.append(['curve0']*len(curve_points0))

    end_x0 = curve_points0[-1][0]
    end_y0 = curve_points0[-1][1]

    return (all_control_points, all_curve_points,
            control_points_idx, control_points_idx_names,
            curve_points_idx, curve_points_idx_names,
            end_x0, end_y0)


def generate_curves(
    n_curves,
    x_increments, y_positions,
    all_control_points, all_curve_points,
    control_points_idx, control_points_idx_names,
    curve_points_idx, curve_points_idx_names,
    end_x0, end_y0,
    offset_factor_x0, offset_factor_y0,
    mid_factor_x, mid_factor_y,
    true_mid, rel_mid,
    thickness, inside_tolerance,
    degree, order, knot_c,
    resolution,
    weights, center_offset
):
    """
    Generate the main sequential NURBS curves after the cap.
    """
    import numpy as np
    from geomdl import NURBS

    start_x = end_x0

    for i in range(n_curves):
        # Y start/end
        y_start = end_y0 + y_positions[i]
        y_end   = end_y0 + y_positions[i+1]
        polarity = np.sign(y_end - y_start)
        dx = x_increments[i]

        if i < 1:
            x_end = start_x + dx/2
        else:
            x_end = start_x + dx

        # Midpoint
        mid_x = (start_x + x_end)/2
        mid_y = (y_start + y_end)/2

        # Offsets
        offset_factor_x = offset_factor_x0
        offset_factor_y = offset_factor_y0 * polarity

        if i < 1:
            cp0_x, cp0_y = start_x, y_start
            cp2_x        = cp0_x
        else:
            cp0_x, cp0_y = start_x, y_start
            cp2_x        = mid_x + ((x_end - mid_x)*mid_factor_x)*polarity
        cp2_y            = mid_y + ((x_end - mid_x)*mid_factor_y)*polarity

        if i < 1:
            cp1_x, cp1_y = cp0_x, cp0_y
        else:
            cp1_y = (mid_y*true_mid + cp2_y*rel_mid
                     - abs(y_start - mid_y)*offset_factor_y)
            cp1_x = cp2_x + (x_end - mid_x)*offset_factor_x

        cp3_x = cp2_x - (x_end - mid_x)*offset_factor_x
        cp3_y = (mid_y*true_mid + cp2_y*rel_mid
                 + abs(y_start - mid_y)*offset_factor_y)

        if i < n_curves - 1:
            cp4_x = x_end
        else:
            cp4_x = x_end - center_offset + inside_tolerance
        cp4_y   = y_end

        control_points = [
            [cp0_x, cp0_y],
            [cp1_x, cp1_y],
            [cp2_x, cp2_y],
            [cp3_x, cp3_y],
            [cp4_x, cp4_y]
        ]
        all_control_points.append(control_points)

        curve_cp_idx = np.arange(1, len(control_points)+1)
        control_points_idx.append(curve_cp_idx)
        cp_idx_name = [f"curve{i+1}"] * len(control_points)
        control_points_idx_names.append(cp_idx_name)

        no_control_points = len(control_points)
        knot_vector_length= no_control_points + order
        n                = no_control_points - 1
        internal_knots   = n - order + 1

        if (knot_vector_length - 2*order) >= 0:
            knot_vector = (
                [0]*order +
                [(j*knot_c) for j in range(1, internal_knots+1)] +
                [internal_knots+1]*order
            )
        else:
            raise ValueError("Number of control points must be >= degree.")

        curve = NURBS.Curve()
        curve.degree = degree
        curve.ctrlpts = [[pt[0], pt[1]] for pt in control_points]
        curve.weights = weights
        curve.knotvector = knot_vector
        curve.delta = 1/resolution
        curve.evaluate()
        curve_points = np.array(curve.evalpts)

        if i < n_curves - 1:
            all_curve_points.append(curve_points[:-1])
        else:
            all_curve_points.append(curve_points)

        curve_idx = np.arange(1, len(curve_points)+1)
        if i < n_curves - 1:
            curve_points_idx.append(curve_idx[:-1])
        else:
            curve_points_idx.append(curve_idx)

        start_x = x_end

    return (all_control_points, all_curve_points,
            control_points_idx, control_points_idx_names,
            curve_points_idx, curve_points_idx_names)