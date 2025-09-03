# io/exporting.py
import cadquery as cq
import os

import matplotlib.pyplot as plt

# import modules.geometry_utils as geometry_utils
import builders.build_modules.general_helpers as helpers

def create_2d_crossSection(cross_section_points):
    """
    Create a 2D cross‐section from the combined inner/outer points.
    """
    twoD_cross_section = (
        cq.Workplane("XY")
        .polyline(cross_section_points)
        .close()
    )
    return twoD_cross_section

def create_3d_crossSection(cross_section_points, thickness_factors, revolve_offset):
    """
    Revolve the 2D cross_section 180 degrees around the Y axis to form half an actuator body,
    then add a top 'cap'.
    """
    import numpy as np

    # calculate y-dimension of cross_section
    y_length = helpers.find_max_y(cross_section_points) - helpers.find_min_y(cross_section_points)

    profile = (
        cq.Workplane("XY")
        .polyline(cross_section_points)
        .close()
        .edges()
        .revolve(
            angleDegrees=180,
            axisStart=(revolve_offset,0,0),
            axisEnd=(revolve_offset,1,0)
        )
        .translate((-revolve_offset,0,0))
    )
    thicknesses = np.concatenate(thickness_factors)
    cap_thickness = thicknesses[-1] if len(thicknesses) else 1
    cap = (
        cq.Workplane("XZ")
        .circle(revolve_offset)
        .extrude(cap_thickness, both=True)
        .translate((0, y_length, 0))
    )
    threeD_cross_section = profile + cap
    return threeD_cross_section

def export(model, title, export_type='stl', directory=None, folder='Models', overwrite=False):
    """
    Export a CadQuery model to a file, with optional directory and overwrite check.
    """

    # import os

    supported_types = {
        'stl': '.stl',
        'STEP': '.STEP',
        'dxf': '.dxf'
    }
    
    # export_type = export_type.lower()
    if export_type not in supported_types:
        print(f"Error: Unsupported export type '{export_type}'")
        return
    extension = supported_types[export_type]
    if title.lower().endswith(extension):
        title = title[:-len(extension)]
    filename = f"{title}{extension}"
    filename = os.path.join(folder, filename)

    if directory is None:
        try:
            directory = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # import os
            directory = os.getcwd()

    if directory and not os.path.isdir(directory):
        try:
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        except Exception as e:
            print(f"Error creating directory '{directory}': {e}")
            return
    filepath = os.path.join(directory, filename) if directory else filename

    if os.path.exists(filepath) and not overwrite:
        response = input(f"File '{filepath}' exists. Overwrite? (y/n): ")#.lower()
        if response != 'y':
            print("Export canceled.")
            return

    try:
        model.export(filepath)
        print(f"Model exported as '{filepath}'")
    except AttributeError:
        print("Error: The provided model does not have an 'export' method.")
    except IOError as e:
        print(f"IOError writing '{filepath}': {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def export_plot(fig, title, export_type='png', directory=None, folder='Model_plots', overwrite=False):
    """
    Export a Matplotlib Figure img to a file, with optional directory and overwrite check.
    """

    supported_types = {
        'png': '.png'
    }
    
    export_type = export_type.lower()
    if export_type not in supported_types:
        print(f"Error: Unsupported export type '{export_type}'")
        return
    extension = supported_types[export_type]
    if title.lower().endswith(extension):
        title = title[:-len(extension)]
    filename = f"{title}{extension}"
    filename = os.path.join(folder, filename)

    if directory is None:
        try:
            directory = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # import os
            directory = os.getcwd()

    if directory and not os.path.isdir(directory):
        try:
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        except Exception as e:
            print(f"Error creating directory '{directory}': {e}")
            return
    filepath = os.path.join(directory, filename) if directory else filename

    if os.path.exists(filepath) and not overwrite:
        response = input(f"File '{filepath}' exists. Overwrite? (y/n): ").lower()
        if response != 'y':
            print("Export canceled.")
            return

    try:
        fig.savefig(filepath)
        print(f"Plot exported as '{filepath}'")
    except AttributeError:
        print("Error: The provided model does not have an 'export' method.")
    except IOError as e:
        print(f"IOError writing '{filepath}': {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def combine(final_obj, base_obj):
    return final_obj + base_obj

def export_model(final_obj):
    final_obj.export("final_model.stl")
    print("Model exported as 'final_model.stl'")
