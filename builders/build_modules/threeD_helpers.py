import math
import numpy as np
import cadquery as cq

from pytictoc import TicToc
t = TicToc()

import builders.build_modules.general_helpers as helpers
from core.types import Params, Model3D

from core.config import BaselineGeometryConfig
baseline = BaselineGeometryConfig()
# -----------------------------------------

def create_3d_cap(thickness_factors, revolve_offset, y_translate, x_translate):
    # top cap
    thicknesses = np.concatenate(thickness_factors)
    cap_thickness = thicknesses[-1] if len(thicknesses) > 0 else 1
    cap = (
        cq.Workplane("XZ")
        .circle(revolve_offset)
        .extrude(cap_thickness, both=True)
        .translate((x_translate, y_translate-cap_thickness, 0))
    )

    return cap

def create_3d_model(cross_section_points, thickness_factors, params: Params, revolve_offset=1.0, revolve_angle=360):
    """
    Revolve the 2D cross_section around the Y axis to form the actuator body,
    then add a top 'cap'.
    """

    profile = (cq.Workplane("XY")
               .polyline(cross_section_points)
               .close()
               .edges()
               .revolve(
                    angleDegrees=revolve_angle,
                    axisStart=(revolve_offset, 0, 0),
                    axisEnd=(revolve_offset, 1, 0)
                )
                .translate((-revolve_offset, 0, 0))
    )

    y_translate = helpers.find_max_y(cross_section_points)

    cap = create_3d_cap(thickness_factors, revolve_offset, y_translate, 0)
    final = profile + cap

    # create keying feature at base
    x_translate = abs(helpers.find_max_x(cross_section_points) - helpers.find_min_x(cross_section_points)) + revolve_offset
    final = (
        final
        .transformed(offset=cq.Vector(-x_translate, 0, 0), rotate=cq.Vector(0,270,0))
        .workplane()
        .rect(params.thickness*2, params.thickness*2)
        .extrude(baseline.keying_offset, both=True, combine="a")
    )

    return final

# -----------------------------------------
# 3D Model Generation - Bending
# -----------------------------------------

def create_3d_model_bending(
        cross_sections: list[list[tuple[float, float]]],
        thickness_factors: list[list[float]],
        params: Params,
        loft_offset: float = 0.0,
        angular_section: float | None = None,
    ):
    """
    Build a bending model from a list of 2D cross-sections and matching thickness factors.
    cross_sections[0] is the base (0°), then subsequent entries for each increment.
    """

    # handle increments and angular_section
    increments = len(cross_sections) - 1

    # create array to store workplanes
    workplanes = []

    # define initial workplane [cross_section0]
    initial_pts = cross_sections[0]

    # create initial workplane
    workplane0 = (cq.Workplane("XY")
               .polyline(initial_pts)
               .close()
    )

    # append initial workplane to the list
    workplanes.append(workplane0)

    # calculate the increment angle
    angle = calculate_angle(increments, angular_section)

    x_prev = calculate_x_length(initial_pts)

    profile = None
    for i in range(0,(increments)):

        # (1) DERIVE THE X AND Z OFFSETS

        # extract curve points
        curve_points = cross_sections[i+1]

        # Get the current rectangle dimensions
        x_len = calculate_x_length(curve_points)
        y_len = helpers.find_max_y(curve_points) - helpers.find_min_y(curve_points)
        # print(f"Increment {i}: x_length = {x}, y_length = {y}")

        # Get the current workplane
        workplane_now = workplanes[i]

        # Calculate the z and x offsets
        x_variance = calculate_x_inc_variance(x_len, x_prev)

        z_offset = calculate_minimum_z_offset(angle, x_len, centre_offset=loft_offset)
        x_offset = calculate_minimum_x_offset(angle, x_len, centre_offset=loft_offset)

        # (2) CREATE THE WORKPLANES AND LOFT

        workplane_new = workplane_now.transformed(offset=cq.Vector(x_offset, 0, z_offset),rotate=cq.Vector(0, angle, 0)).polyline(curve_points).close()

        workplanes.append(workplane_new)

        # print(f"completed increment {i+1}")

    # Get the maximum y value for the cap
    last_curve_points = cross_sections[-1]
    y_max = helpers.find_max_y(last_curve_points)
    last_thickness_factors = thickness_factors[-1]

    profile = workplanes[-1].loft(ruled=True,combine="a")
    
    # create keying feature at base
    x_translate = abs(helpers.find_max_x(initial_pts) - helpers.find_min_x(initial_pts)) + loft_offset
    profile = (
        profile
        .transformed(offset=cq.Vector(-x_translate, 0, 0), rotate=cq.Vector(0,270,0))
        .workplane()
        .rect(params.thickness*2, params.thickness*2)
        .extrude(baseline.keying_offset, both=True, combine="s")
    )

    if loft_offset > 0:
        cap = create_3d_cap(last_thickness_factors, loft_offset, y_max, loft_offset)
        final = profile + cap
    else:
        final = profile

    return final
# -----------------------------------------
# Helper functions for bending calculations
# -----------------------------------------

# calculate minimum offset for each increment
def calculate_minimum_z_offset(inc_angle_deg, x_length, centre_offset):
    inc_angle_rad = math.radians(inc_angle_deg)
    # z_min_offset = ((x_length + centre_offset) / 2) * math.sin(inc_angle_rad)
    z_min_offset = (centre_offset) * math.sin(inc_angle_rad)
    return z_min_offset

# calculate minimum offset for each increment
def calculate_z_offset(inc_angle_deg, x_variance, centre_offset):
    inc_angle_rad = math.radians(inc_angle_deg)
    # z_min_offset = ((x_length + centre_offset) / 2) * math.sin(inc_angle_rad)
    z_offset = (centre_offset - x_variance) * math.sin(inc_angle_rad)
    return z_offset

# calculate minimum offset for each increment
def calculate_minimum_x_offset(inc_angle_deg, x_length, centre_offset):
    inc_angle_rad = math.radians(inc_angle_deg)
    # x_min_offset = ((x_length + centre_offset) / 2) * (1 - math.cos(inc_angle_rad))
    x_min_offset = (centre_offset) * (1 - math.cos(inc_angle_rad))
    return x_min_offset

# calculate minimum offset for each increment
def calculate_x_offset(inc_angle_deg, x_variance, centre_offset):
    inc_angle_rad = math.radians(inc_angle_deg)
    # x_min_offset = ((x_length + centre_offset) / 2) * (1 - math.cos(inc_angle_rad))
    x_offset = (centre_offset - x_variance) * (1 - math.cos(inc_angle_rad))
    return x_offset

# calculate the angle for each increment 
def calculate_angle(increments, angular_section):
    angle = angular_section / increments
    return angle

def calculate_x_inc_variance(x_length, x_length_prev):
    x_variance = x_length - x_length_prev
    return x_variance

def calculate_x_length(cross_section_points):
    x_length = helpers.find_max_x(cross_section_points) - helpers.find_min_x(cross_section_points)
    return x_length

