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

    # cap thickness is the first thickness value
    cap_thickness = float(point_thicknesses[0]) if point_thicknesses else 1.0

    return point_thicknesses, cap_thickness


def _resample_list(vals, m):
    m = int(m)
    if m <= 0: return []
    vals = [float(v) for v in vals]
    if len(vals) == m: return vals
    if len(vals) == 1: return [vals[0]] * m
    x_old = np.linspace(0.0, 1.0, num=len(vals))
    x_new = np.linspace(0.0, 1.0, num=m)
    return np.interp(x_new, x_old, vals).tolist()

def _map_cp_indices(outer_points, vt_control_points):
    """Map each VT control point to nearest outer index, per segment; drop consecutive duplicates.
       Anchor first CP to 0 and last CP to len(outer)-1 for coverage."""
    vt_cp_array = np.vstack(vt_control_points)  # (total_cp, 2)
    outer = np.asarray(outer_points, dtype=float)
    # nearest index for each CP
    cp_to_idx = []
    for xcp, ycp in vt_cp_array:
        d2 = np.sum((outer - (xcp, ycp))**2, axis=1)
        cp_to_idx.append(int(np.argmin(d2)))

    seg_indices, start = [], 0
    for seg in vt_control_points:
        seg_len = len(seg)
        raw = cp_to_idx[start:start+seg_len]
        uniq = []
        for u in raw:
            if not uniq or u != uniq[-1]:  # keep monotone indices
                uniq.append(u)
        seg_indices.append(uniq)
        start += seg_len

    if seg_indices:
        seg_indices[0][0] = 0
        seg_indices[-1][-1] = len(outer_points) - 1
    return seg_indices

def variable_thd_collapsed(
    outer_points,
    vt_control_points,
    all_thicknesses,
    baseline_thickness,
    *,
    apply_sections=None,               # e.g. [0,2,4] ; None => apply to all segments
    baseline_values=(1.0, 1.0, 1.0),   # used on non-selected segments
    start_frac=0.10,
    ramp_frac=0.10,
    end_frac=None                      # None => mirror start_frac
):
    """
    For each CP-to-CP span [i0..i1]:
      - hold v0 for 'start_frac' of the integer span,
      - ramp v0->v1 for 'ramp_frac',
      - hold v1 for 'end_frac' (defaults to start_frac).

    All fractions are applied in *indices* so the effect is visible even on short spans.
    Returns a list of len(outer_points).
    """
    n_outer = len(outer_points)
    if n_outer == 0:
        return []
    
    # handle baseline values (convert from factor to absolute thickness values)
    # convert baseline_values (factors) to absolute thickness using baseline_thickness [params.thickness]
    baseline_values = tuple(float(v) * baseline_thickness for v in baseline_values)

    # normalize fractions
    start_frac = float(np.clip(start_frac, 0.0, 1.0))
    ramp_frac  = float(max(0.0, ramp_frac))
    if end_frac is None:
        end_frac = start_frac
    end_frac = float(np.clip(end_frac, 0.0, 1.0))

    seg_indices = _map_cp_indices(outer_points, vt_control_points)

    # which segments get the windowed profile
    if apply_sections is None:
        apply_set = set(range(len(seg_indices)))
    else:
        apply_set = set(int(s) for s in apply_sections)

    out = np.empty(n_outer, dtype=float)
    filled_to = 0  # how many entries already written

    for seg_id, idxs in enumerate(seg_indices):
        if len(idxs) < 2:
            continue

        # pick controls for this segment
        if seg_id in apply_set:
            ctrl = list(all_thicknesses[seg_id])
            # anchor first value to second to avoid jump
            if seg_id == 0:
                ctrl[0] = ctrl[1]
                # cap thickness is the first thickness value in the first ctrl segment
                cap_thickness = float(ctrl[0])
            windowed = True
        else:
            # baseline [a,m,b] expanded to this segment's control count
            ctrl = _resample_list(list(baseline_values), len(idxs))
            windowed = False

        # print(f"  seg {seg_id} idxs={idxs} ctrl={ctrl} windowed={windowed}")

        if len(ctrl) != len(idxs):
            ctrl = _resample_list(ctrl, len(idxs))

        for j in range(len(idxs) - 1):
            i0, i1 = idxs[j], idxs[j+1]
            if i1 <= i0:
                continue
            span = i1 - i0  # number of edges; points are (span+1)

            v0, v1 = float(ctrl[j]), float(ctrl[j+1])

            if windowed:
                # compute integer windows
                hold0 = int(round(start_frac * (span)))
                hold1 = int(round(end_frac   * (span)))
                max_ramp_span = max(0, span - hold0 - hold1)
                ramp_len = int(round(ramp_frac * span))
                ramp_len = max(1, min(ramp_len, max_ramp_span)) if max_ramp_span > 0 else 0

                k_start = i0 + hold0
                k_rend  = k_start + ramp_len  # exclusive

                # 1) hold v0
                # if seg_id > 0:
                out[i0:k_start] = v0

                # 2) ramp v0 -> v1
                if ramp_len > 0:
                    if ramp_len == 1:
                        out[k_start:k_rend] = v1  # single step jump
                    else:
                        t = np.linspace(0.0, 1.0, num=ramp_len, endpoint=False)
                        out[k_start:k_rend] = v0 + t * (v1 - v0)

                # 3) hold v1
                out[k_rend:i1] = v1

            else:
                # simple linear
                t = np.linspace(0.0, 1.0, num=span, endpoint=False)
                out[i0:i1] = v0 + t * (v1 - v0)

        # include last control value at the final index
        out[idxs[-1]] = float(ctrl[-1])
        filled_to = max(filled_to, idxs[-1] + 1)

    # fill any gaps (can occur if segs don’t cover everything)
    if filled_to < n_outer:
        out[filled_to:] = out[filled_to - 1] if filled_to > 0 else 1.0

    return out.tolist(), cap_thickness

def variable_thd_sbend(
    outer_points,
    vt_control_points,
    baseline_thickness,
    profile1,                 # e.g. [0.5, 1.0, 0.5] or any length
    sections1=None,           # e.g. [0,2,4]
    profile2=None,            # optional second profile (same semantics)
    sections2=None,           # e.g. [1,3]
    default_profile=(1.0,1.0,1.0)
):
    """
    Assign thickness *profiles* to segments:
      - segments in `sections1` use `profile1`
      - segments in `sections2` use `profile2` (takes precedence if overlap)
      - all other segments use `default_profile`

    Each profile is resampled to the number of control points for that segment.
    Continuity is enforced: the first value of a segment is set to the previous
    segment's last value.

    Returns: list of per-outer-point thickness values (length ~= len(outer_points)).
    """
    seg_indices = _map_cp_indices(outer_points, vt_control_points)
    n_outer = len(outer_points)

    s1 = set(sections1 or [])
    s2 = set(sections2 or [])

    # convert s1 and s2 (factors) to absolute thickness using baseline_thickness [params.thickness]
    profile1 = [float(v) * baseline_thickness for v in profile1]
    if profile2 is not None:
        profile2 = [float(v) * baseline_thickness for v in profile2]
    default_profile = [float(v) * baseline_thickness for v in default_profile]

    point_thicknesses = []
    prev_last = None  # to enforce continuity

    for seg_id, idxs in enumerate(seg_indices):
        if len(idxs) == 0:
            continue

        # choose the source profile by section
        if seg_id in s2 and profile2 is not None:
            src = list(profile2)
        elif seg_id in s1:
            src = list(profile1)
        else:
            src = list(default_profile)

        # resample to the segment's control count
        src_vals = _resample_list(src, len(idxs))

        # enforce continuity: first value must equal previous segment's last value
        if prev_last is not None and prev_last != src_vals[0]:
            # average them
            avg_seg_val = (prev_last + src_vals[0]) / 2
            src_vals[0] = avg_seg_val
            # adjust the rest of the profile to preserve shape
            point_thicknesses[-1] = avg_seg_val  # fix last of previous segment

        # interpolate linearly between consecutive control values over the
        # outer-point spans defined by idxs
        for j in range(len(idxs) - 1):
            i0, i1 = idxs[j], idxs[j + 1]
            span = max(1, i1 - i0)
            v0, v1 = float(src_vals[j]), float(src_vals[j + 1])
            for k in range(span):
                t = k / float(span)  # [0,1)
                point_thicknesses.append(v0 + t * (v1 - v0))

        # include the last control value at the end of segment
        last_val = float(src_vals[-1])
        point_thicknesses.append(last_val)
        prev_last = last_val

    # tidy length to match outer_points
    if len(point_thicknesses) > n_outer:
        point_thicknesses = point_thicknesses[:n_outer]
    elif len(point_thicknesses) < n_outer:
        fill = point_thicknesses[-1] if point_thicknesses else 1.0
        point_thicknesses += [fill] * (n_outer - len(point_thicknesses))

    # cap thickness is the first thickness value
    cap_thickness = float(point_thicknesses[0]) if point_thicknesses else baseline_thickness

    return point_thicknesses, cap_thickness