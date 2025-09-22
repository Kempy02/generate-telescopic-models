# builders/build_modules/threeD_helpers.py

import math
import numpy as np
import cadquery as cq

from pytictoc import TicToc
t = TicToc()

import builders.build_modules.general_helpers as helpers
from core.types import Params, Model3D

from core.config import BaselineGeometryConfig, optionsConfig, BendSettings
baseline = BaselineGeometryConfig()
options = optionsConfig()
bend = BendSettings()
# -----------------------------------------

def create_3d_cap(revolve_offset, y_translate, x_translate, cap_thickness):
    # top cap
    cap = (
        cq.Workplane("XZ")
        .circle(revolve_offset)
        .extrude(cap_thickness, both=True)
        .translate((x_translate, y_translate-cap_thickness, 0))
    )

    return cap

def create_3d_model(cross_section_points, params: Params, revolve_offset=1.0, keying_enabled=False):
    """
    Revolve the 2D cross_section around the Y axis to form the actuator body,
    then add a top 'cap'.
    """

    max_x = helpers.find_max_x(cross_section_points)
    points_max_x = [pt for pt in cross_section_points if math.isclose(pt[0], max_x, rel_tol=1e-9, abs_tol=1e-12)]
    y_translate = helpers.find_max_y(points_max_x)
    x_translate = (max_x - helpers.find_min_x(cross_section_points)) + revolve_offset

    profile = (cq.Workplane("XY")
               .polyline(cross_section_points)
               .close()
               .edges()
               .revolve(
                    angleDegrees=bend.total_angular_section,
                    axisStart=(revolve_offset, 0, 0),
                    axisEnd=(revolve_offset, 1, 0)
                )
                .translate((-revolve_offset, 0, 0))
    )

    # create text engraving
    if options.engrave_text:
        profile = (
            profile.faces(">Y")
            .transformed(offset=(0, baseline.cap_thickness*2, (x_translate-5)), rotate=(270, 0, 0))
            .text("T1", fontsize=4, distance=-2.0, cut=True, combine=True, kind='bold')
            # .un-transform
            .transformed(offset=(0, 0, -(x_translate-5)), rotate=(-90, 0, 0))
        )

    # create keying feature at base
    if keying_enabled:
        profile = (
            profile
            .transformed(offset=cq.Vector(-x_translate, 0, 0), rotate=cq.Vector(0,270,0))
            .workplane()
            .rect(params.thickness*2, params.thickness*2)
            .extrude(baseline.keying_offset, both=True, combine="a")
            # un-transform
            .transformed(offset=cq.Vector(x_translate, 0, 0), rotate=cq.Vector(0,90,0))
        )
    else:
        profile = profile

    cap_thickness = baseline.cap_thickness if options.constant_cap_thickness else params.thickness
    cap = create_3d_cap(revolve_offset, y_translate, 0, cap_thickness)
    final = profile + cap

    return final

# -----------------------------------------
# 3D Model Generation - Bending
# -----------------------------------------

def create_3d_model_bending(
        cross_sections: list[list[tuple[float, float]]],
        params: Params,
        loft_offset: float = 0.0,
        angular_section: float | None = None,
        keying_enabled: bool = False,
    ):
    """
    Build a bending model from a list of 2D cross-sections and matching thickness factors.
    cross_sections[0] is the base (0°), then subsequent entries for each angular increment.
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
        workplane_now = workplanes[i].faces().workplane()

        # Calculate the z and x offsets
        x_variance = calculate_x_inc_variance(x_len, x_prev)

        z_offset = calculate_minimum_z_offset(angle, x_len, centre_offset=loft_offset)
        x_offset = calculate_minimum_x_offset(angle, x_len, centre_offset=loft_offset)

        # (2) CREATE THE WORKPLANES AND LOFT

        workplane_new = workplane_now.transformed(offset=cq.Vector(x_offset, 0, z_offset),rotate=cq.Vector(0, angle, 0)).polyline(curve_points).close()

        workplanes.append(workplane_new)

        # print(f"completed increment at angle {(i+1) * angle}°")
        
    #  LOFT THE WORKPLANES TO FROM THE 3D PROFILE
    profile = workplanes[-1].loft(combine=True, ruled=options.ruled_flag)

    # Get the maximum y value for the cap
    last_curve_points = cross_sections[-1]
    y_max = helpers.find_max_y(last_curve_points)
    
    # create keying feature at base
    if keying_enabled:
        x_translate = abs(helpers.find_max_x(initial_pts) - helpers.find_min_x(initial_pts)) + loft_offset
        profile = (
            profile
            .transformed(offset=cq.Vector(-x_translate, 0, 0), rotate=cq.Vector(0,270,0))
            .workplane()
            .rect(params.thickness*2, params.thickness*2)
            .extrude(baseline.keying_offset, both=True, combine="a")
        )
    else:
        profile = profile

    if loft_offset > 0:
        cap_thickness = baseline.cap_thickness if options.constant_cap_thickness else params.thickness
        cap = create_3d_cap(loft_offset, y_max, loft_offset, cap_thickness)
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
    z_min_offset = (centre_offset) * math.sin(inc_angle_rad)
    return z_min_offset

# # calculate offset for each increment
# def calculate_z_offset(inc_angle_deg, x_variance, centre_offset):
#     inc_angle_rad = math.radians(inc_angle_deg)
#     z_offset = (centre_offset - x_variance) * math.sin(inc_angle_rad)
#     return z_offset

# calculate minimum offset for each increment
def calculate_minimum_x_offset(inc_angle_deg, x_length, centre_offset):
    inc_angle_rad = math.radians(inc_angle_deg)
    x_min_offset = (centre_offset) * (1 - math.cos(inc_angle_rad))
    return x_min_offset

# # calculate offset for each increment
# def calculate_x_offset(inc_angle_deg, x_variance, centre_offset):
#     inc_angle_rad = math.radians(inc_angle_deg)
#     x_offset = (centre_offset - x_variance) * (1 - math.cos(inc_angle_rad))
#     return x_offset

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

