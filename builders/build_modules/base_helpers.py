# builders/build_modules/base_creation.py

import cadquery as cq
import numpy as np
from shapely.geometry import Point

from core.types import BaseComponents, Params, BuildReport, BuildReportBend

import builders.build_modules.general_helpers as helpers

from core.config import BaselineGeometryConfig, optionsConfig, BendSettings, BaseBuildSettings
baseline = BaselineGeometryConfig()
options = optionsConfig()
bend = BendSettings()
base_build = BaseBuildSettings()

def create_base(params: Params, xsection2D) -> BaseComponents:
    """
    Creates a base with holes, clamp geometry, etc.
    """

    exploded_factor = 10  # Factor to control the explosion of the base in Z direction (for exploded view of base rig)

    screw_diameter  = 3.0
    screw_head_diameter = 5.5
    valve_diameter  = 11.2 # 1/4" BSPT Tapping Drill Size

    base_plate_height  = 2
    base_internal_height= baseline.cap_height - params.thickness 
    height_factor      = 0.8
    base_input_radius  = valve_diameter/2
    wall_thickness     = 3

    screw_radius       = screw_diameter/2
    screw_head_radius  = screw_head_diameter/2
    screw_tolerance    = screw_head_radius * 2
    outer_screw_tolerance = screw_head_radius * 2 
    # desired_diameter   = 2*48.0 - 2*(screw_tolerance/2 + outer_screw_tolerance)
    # desired_side_length= desired_diameter/np.sqrt(2)

    # foundation screw slot
    foundation_screw_slot = (screw_radius * 2) + (base_build.slide_length * 2) # x mm slide length on either side of screw

    clamp_side_length = 30
    clamp_depth       = 30

    # Calculate base radius from 2D cross-section

    # if xsection2D is a list
    if isinstance(xsection2D, list):

        # validate that enclosed shape can be created
        if ((len(xsection2D)-1) * bend.angle_intervals) % 360 != 0:
            raise ValueError("Base creation requires complete revolution of model. \n If testing, turn off base creation")

        base_radius_outline = []
        base_internal_radius_outline = []
        seal_internal_outline = []
        seal_clamp_outline = []
        # twoD_cross_section_points = [x.twoD_cross_section for x in xsection2D]
        for i in range(len(xsection2D)):
            twoD_cross_section_points = xsection2D[i].twoD_cross_section
            # calculate min and max x
            min_x = helpers.find_min_x(twoD_cross_section_points)
            max_x = helpers.find_max_x(twoD_cross_section_points)

            base_rad_point = abs(max_x - min_x) + baseline.loft_offset + params.thickness + base_build.base_extension
            base_internal_rad_point = (abs(max_x - min_x) + baseline.loft_offset) - (baseline.cap_length + params.thickness) + base_build.base_tolerance
            seal_internal_rad_point = (abs(max_x - min_x) + baseline.loft_offset) - (baseline.cap_length) + (params.thickness*2 + base_build.base_tolerance)

            start_point = Point(0, 0)

            # create base_rad_outline points
            base_rad_outline_pt_x = start_point.x + base_rad_point * np.cos(np.radians(bend.angle_intervals*i))
            base_rad_outline_pt_y = start_point.y + base_rad_point * np.sin(np.radians(bend.angle_intervals*i))
            base_rad_outline_pt = Point(base_rad_outline_pt_x, base_rad_outline_pt_y)
            base_radius_outline.append(base_rad_outline_pt)
            # create base_internal_rad_outline points
            base_internal_rad_outline_pt_x = start_point.x + base_internal_rad_point * np.cos(np.radians(bend.angle_intervals*i))
            base_internal_rad_outline_pt_y = start_point.y + base_internal_rad_point * np.sin(np.radians(bend.angle_intervals*i))
            base_internal_rad_outline_pt = Point(base_internal_rad_outline_pt_x, base_internal_rad_outline_pt_y)
            base_internal_radius_outline.append(base_internal_rad_outline_pt)
            # create seal_internal_outline points
            seal_internal_outline_pt_x = start_point.x + (seal_internal_rad_point) * np.cos(np.radians(bend.angle_intervals*i))
            seal_internal_outline_pt_y = start_point.y + (seal_internal_rad_point) * np.sin(np.radians(bend.angle_intervals*i))
            seal_internal_outline_pt = Point(seal_internal_outline_pt_x, seal_internal_outline_pt_y)
            seal_internal_outline.append(seal_internal_outline_pt)
            # create seal_clamp_outline_points
            seal_clamp_radius_pt_x = seal_internal_outline_pt_x + (baseline.cap_length - params.thickness) * np.cos(np.radians(bend.angle_intervals*i))
            seal_clamp_radius_pt_y = seal_internal_outline_pt_y + (baseline.cap_length - params.thickness) * np.sin(np.radians(bend.angle_intervals*i))
            seal_clamp_radius_pt = Point(seal_clamp_radius_pt_x, seal_clamp_radius_pt_y)
            seal_clamp_outline.append(seal_clamp_radius_pt)

        # print first value of base_internal_radius_outline
        print("Base Internal Radius Outline Point 0: ", base_internal_radius_outline[0].x)

        base_radius = helpers.find_max_value(base_radius_outline)
        base_internal_radius = helpers.find_min_value(base_internal_radius_outline)

        base_radius_outline = list((pt.x, pt.y) for pt in base_radius_outline)
        base_internal_radius_outline = list((pt.x, pt.y) for pt in base_internal_radius_outline)
        seal_internal_radius_outline = list((pt.x, pt.y) for pt in seal_internal_outline)
        seal_clamp_outline = list((pt.x, pt.y) for pt in seal_clamp_outline)

    else:
        twoD_cross_section_points = xsection2D.twoD_cross_section
            # calculate min and max x
        min_x = helpers.find_min_x(twoD_cross_section_points)
        max_x = helpers.find_max_x(twoD_cross_section_points)

        base_radius = abs(max_x - min_x) + baseline.revolve_offset + params.thickness + base_build.base_extension
        base_internal_radius = (abs(max_x - min_x) + baseline.revolve_offset) - (baseline.cap_length + params.thickness) + base_build.base_tolerance
        seal_internal_radius = (abs(max_x - min_x) + baseline.revolve_offset) - (baseline.cap_length) + (params.thickness*2 + base_build.base_tolerance)

        # create base_radius outline points
        base_radius_outline = helpers.create_circle_of_radius(base_radius)
        # create base_radius_internal outline points
        base_internal_radius_outline = helpers.create_circle_of_radius(base_internal_radius)
        # create seal_internal_radius circle points
        seal_internal_radius_outline = helpers.create_circle_of_radius(seal_internal_radius)
        # create the seal clamp outline points
        seal_clamp_outline = helpers.create_circle_of_radius(seal_internal_radius + (baseline.cap_length - params.thickness))

    # Calculate screw placement geometry
    # Clamp and Seal
    base_diameter   = base_radius * 2
    # Foundation (Constant)
    baseline_diameter = 2 * (base_build.baseline_geo_radius + base_build.base_extension + params.thickness)
    desired_diameters = [baseline_diameter-base_build.slide_length*2, baseline_diameter, baseline_diameter+base_build.slide_length*2]

    # Determine which base size to use
    if base_diameter <= baseline_diameter - base_build.slide_length/2:
        desired_diameter = desired_diameters[0]
        print("Base 1 [min] used")
    elif base_diameter >= baseline_diameter + base_build.slide_length/2:
        desired_diameter = desired_diameters[2]
        print("Base 3 [max] used")
    else:
        desired_diameter = desired_diameters[1]
        print("Base 2 [std] used")

    # distance from center to screw holes
    screw_hole_diameter = desired_diameter - 2*(screw_tolerance/2 + outer_screw_tolerance)

    # calculate displacement from outer edge to inner edge of the seal
    # First point (0 degrees) will always be at faces(">X")
    print("Seal Clamp Outline Point 0: ", seal_clamp_outline[0][0])
    first_displacement = desired_diameter/2 - seal_clamp_outline[0][0]

    jimstron_clamp_plate_width = 8
    jimstron_clamp_max_distance = 56
    clamp_screw_req_distance = screw_tolerance + jimstron_clamp_plate_width
    # if clamp_side_length > (desired_diameter/2 - clamp_screw_req_distance):
    #     max_clamp_side_length = desired_diameter/2 - clamp_screw_req_distance
    #     raise ValueError(f"Clamp side length blocks screws; max feasible = {max_clamp_side_length}")
    # elif clamp_side_length > (jimstron_clamp_max_distance - 2 * screw_tolerance):
    #     max_clamp_side_length = jimstron_clamp_max_distance - 2 * screw_tolerance
    #     raise ValueError(f"Clamp side length too large for jimstrun; max feasible = {max_clamp_side_length}")

    jimstron_max_depth = 80
    pneu_head_height = 25
    pneu_head_tolerance = 10
    clamp_max_depth = jimstron_max_depth - (pneu_head_height + pneu_head_tolerance)
    if clamp_depth > clamp_max_depth:
        raise ValueError(f"Clamp depth is too large; max feasible = {clamp_max_depth}")

    # SEAL
    base1 = (
        cq.Workplane("XZ")
        .circle(desired_diameter/2)
        # .polyline(base_radius_outline)
        # .close()
        .extrude(base_plate_height, both=True)
        # Create the seal clamp
        .faces("<Y")
        .workplane()
        .polyline(seal_clamp_outline)
        .close()
        .extrude(-(params.thickness * 2 - base_build.squeeze_tolerance), combine='cut')
        # Cut screw holes
        .faces(">Y")
        .polarArray(radius=screw_hole_diameter/2, startAngle=0, angle=360, count=base_build.no_screws, rotate=True)
        .circle(screw_radius*1.15)
        .cutThruAll()
        # Hollow the plate
        .faces(">Y")
        .workplane()
        .polyline(seal_internal_radius_outline)
        .close()
        .extrude(-base_plate_height*2, combine='cut')
        # add key feature to seal
        .transformed(offset=cq.Vector(seal_clamp_outline[0][0], 0, -(base_plate_height*2 - (base_plate_height-params.thickness-base_build.squeeze_tolerance))), rotate=cq.Vector(0,270,0))
        .workplane()
        .rect(params.thickness*2, params.thickness*2)
        .extrude(baseline.keying_offset, both=True, combine="s")
        # explode base for visualisation
        .translate((0, params.thickness*2*exploded_factor, 0))
    )

    # add key feature to seal
    # base1 = base1.faces(">X").workplane().transformed(offset=cq.Vector(0, 0, -first_displacement)).faces().workplane().rect(params.thickness*2, params.thickness*2).extrude(baseline.keying_offset, combine=True)
    # base1 = base1.faces(">X").workplane().rect(params.thickness*2, params.thickness*2).extrude(baseline.keying_offset, combine="s")
    # base1 = base1.transformed(offset=cq.Vector(seal_clamp_outline[0][0], 0, 0)).faces().workplane().rect(params.thickness*2, params.thickness*2).extrude(baseline.keying_offset, combine="s")


    # Foundation (Constant)
    base2 = (
        cq.Workplane("XZ")
        .circle(desired_diameter/2)
        .extrude(base_plate_height + 2)
        .translate((0, -params.thickness*exploded_factor, 0))
        # Screw holes
        .faces(">Y")
        .polarArray(radius=(screw_hole_diameter/2), startAngle=0, angle=360, count=base_build.no_screws, rotate=True)
        # .slot2D(foundation_screw_slot, screw_radius*2, 0)
        .circle(screw_radius*1.15)
        .cutThruAll()
        # The clamp
        .faces("<Y")
        .workplane()
        .rect(clamp_side_length, clamp_side_length)
        .extrude(clamp_depth, combine=True)
        # The input hole
        .faces("<Y")
        .workplane()
        .circle(base_input_radius)
        .cutThruAll()
        # .rotateAboutCenter((0, 1, 0), (180/base_build.no_screws))
    )

    # add fillets to foundation base
    base2 = base2.faces("<<Y[-2]").edges().fillet(5)

    base = base1 + base2

    return BaseComponents(
        base_exploded=base, 
        foundation=base2, 
        seal=base1
    )


def create_fem_models(final, base_temp):
    """
    Example of combining final with a 'temp' base for FE analyses, etc.
    """
    fem_model = final + base_temp
    return fem_model
