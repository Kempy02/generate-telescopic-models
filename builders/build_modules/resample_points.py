# builders/build_modules/twoD_helpers.py (append these)

from typing import List, Tuple, Optional, Dict, Set
import math
import numpy as np

import builders.build_modules.general_helpers as helpers

def _ensure_open_ring(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return ring without duplicated end point."""
    if not pts:
        return pts
    if pts[0] == pts[-1]:
        return pts[:-1]
    return pts

def _argmin_idx_by_x(pts: List[Tuple[float, float]]) -> int:
    """Return the index of the point with the minimum x-coordinate."""
    return min(range(len(pts)), key=lambda i: pts[i][0])

def _argmax_idx_by_x(pts: List[Tuple[float, float]]) -> int:
    """Return the index of the point with the maximum x-coordinate."""
    return max(range(len(pts)), key=lambda i: pts[i][0])

def _walk_forward(pts: List[Tuple[float, float]], start: int, end: int) -> List[Tuple[float, float]]:
    """
    Walk forward along the ring indices (wrapping) from start to end (inclusive).
    """
    n = len(pts)
    out = [pts[start]]
    i = start
    while i != end:
        i = (i + 1) % n
        out.append(pts[i])
    return out

def _segment_lengths(poly: List[Tuple[float, float]]) -> List[float]:
    """Compute lengths of each segment in a polyline."""
    L = []
    for i in range(len(poly) - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i+1]
        L.append(math.hypot(x1 - x0, y1 - y0))
    return L

def _resample_polyline_even(poly: List[Tuple[float, float]], count: int, include_start=True, include_end=True) -> List[Tuple[float, float]]:
    """
    Resample a polyline to 'count' points evenly by arc length.
    If include_start=False or include_end=False, the corresponding endpoints are dropped.
    """
    if count <= 0 or len(poly) < 2:
        return []

    # Build cumulative length
    segL = _segment_lengths(poly)
    totalL = sum(segL)
    if totalL == 0:
        # Degenerate: all points same.
        return [poly[0]] * count

    # count = int(np.ceil((count) / totalL) * int(totalL))

    # print(f"Resampling polyline of length {totalL} to {count} points")

    # Determine the parameter positions we want
    # We space in [0, totalL] inclusive; then optionally drop ends
    target = [i * (totalL / (count-1)) for i in range(count)]
    # If excluding start/end, trim now while keeping "count" total
    if not include_start:
        target = target[1:]
    if not include_end:
        target = target[:-1]
    if len(target) == 0:
        return []

    # Walk segments to interpolate
    res = []
    cum = 0.0
    idx = 0
    for s in target:
        # advance until the segment covering 's' is found
        while idx < len(segL) and cum + segL[idx] < s:
            cum += segL[idx]
            idx += 1
        if idx >= len(segL):
            # numeric edge: clamp to last
            res.append(poly[-1])
            continue
        # interpolate within segment idx
        x0, y0 = poly[idx]
        x1, y1 = poly[idx + 1]
        seg = segL[idx]
        t = 0.0 if seg == 0 else (s - cum) / seg
        res.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return res

def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return dx * dx + dy * dy

def resample_cross_section_points(
    cross_section_points: List[Tuple[float, float]],
    total_samples: int = 400
) -> List[Tuple[float, float]]:
    """
    Resample a closed cross-section (polygon boundary) into four logical segments:
      1) 'Outer' arc: from min_x extremum to max_x extremum (forward along ring)
      3) 'Inner' arc: from max_x back to min_x (forward along ring)

    - Points are placed at evenly spaced arc-length intervals within each segment.
    - Segment budgets (n for arcs) are derived from total_samples

    Returns a CLOSED ring (first point repeated at the end).
    """

    # inc_res = arc_length / total_samples

    pts = _ensure_open_ring(cross_section_points)
    # pts = cross_section_points
    if len(pts) < 4:
        # nothing to do
        return cross_section_points
    
    # snap pts
    pts = snap_x_to_extremes(pts)

    # Identify extrema by X to split sides
    i_min = _argmin_idx_by_x(pts)
    i_max = _argmax_idx_by_x(pts)

    # Build the two long arcs (forward order)
    outer_arc = _walk_forward(pts, i_min, i_max)            # min_x -> max_x
    # print(f"Outer arc: {outer_arc}\n")
    inner_arc = _walk_forward(pts, i_max, i_min)            # max_x -> min_x
    # print(f"Inner arc: {inner_arc}\n")

    # if the first 2 points of outer-arc share an x-coordinate, remove the point with lower y-coordinate
    if outer_arc[0][0] == outer_arc[1][0]:
        outer_arc = [p for p in outer_arc if p[1] > outer_arc[0][1]]
    # if the first 2 points of inner-arc share an x-coordinate, remove the point with higher y-coordinate
    if inner_arc[0][0] == inner_arc[1][0]:
        inner_arc = [p for p in inner_arc if p[1] < inner_arc[0][1]]

    n = max(2,total_samples // 2)

    # Resample each segment
    # Include starts/ends to keep continuity but avoid duplicates on concatenation:
    # - outer_arc: include start, exclude end (since next segment begins at that end)
    # - min_edge:  include start, and include end so we land exactly back at i_min
    outer_rs = _resample_polyline_even(outer_arc, n, include_start=True,  include_end=True)
    # print(f"Resampled outer arc points: {outer_rs} \n")
    inner_rs = _resample_polyline_even(inner_arc, n, include_start=True,  include_end=True)
    # print(f"Resampled inner arc points: {inner_rs} \n")

    # Stitch into one closed ring
    ring = outer_rs + inner_rs
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    # Calculate arc length [1d]
    arc_length = sum(_segment_lengths(outer_rs))

    return ring, arc_length

def snap_x_to_extremes(ring, thresh=0.5):
        """
        Snap x coordinates in 'ring' that are close to min/max x in 'pts' to the exact extreme.
        """
        min_x = min(p[0] for p in ring)
        max_x = max(p[0] for p in ring)

        snapped = []
        for x, y in ring:
            dmin = abs(x - min_x)
            dmax = abs(x - max_x)
            if dmin <= thresh or dmax <= thresh:
                # snap to the nearest extreme
                x = min_x if dmin <= dmax else max_x
            snapped.append((x, y))
        return snapped