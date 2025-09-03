import copy
import numpy as np
import cadquery as cq

from pytictoc import TicToc
t = TicToc()

from core.types import Params, BuildReportBend
from io_modules.read_csv import read_bending_factors_csv

def interpolate_bending_config(bending_csv_path: str, params: Params):

    rows = read_bending_factors_csv(bending_csv_path)

    for row in rows:
        print(row)

    # Interpolate bending factors for each row
    interpolated_rows = []
    for i in range(len(rows) - 1):
        start = rows[i]
        end = rows[i + 1]
        for j in range(1, 5):  # Interpolate between 1° and 4° increments
            interpolated_row = {
                "angular_section": start["angular_section"] + j,
                "amplitude0": np.interp(j, [1, 4], [start["amplitude0"], end["amplitude0"]]),
                "desired_radius": np.interp(j, [1, 4], [start["desired_radius"], end["desired_radius"]]),
                "period_factors": np.interp(j, [1, 4], [start["period_factors"], end["period_factors"]]),
                "offset_factor_x": np.interp(j, [1, 4], [start["offset_factor_x"], end["offset_factor_x"]]),
                "mid_factor_x": np.interp(j, [1, 4], [start["mid_factor_x"], end["mid_factor_x"]]),
                "curve_weight": np.interp(j, [1, 4], [start["curve_weight"], end["curve_weight"]]),
                "thickness_factor": np.interp(j, [1, 4], [start["thickness_factor"], end["thickness_factor"]]),
                "thickness": np.interp(j, [1, 4], [start["thickness"], end["thickness"]]),
            }
            interpolated_rows.append(interpolated_row)

    print(interpolated_rows)

    return interpolated_rows
