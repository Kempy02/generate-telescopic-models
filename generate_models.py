# main.py
from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional

from core.types import Params, RunOptions, BuildReport
from core.generate_geometry import generate_geometry, generate_geometry_bend

from io_modules.exporting import export, export_plot
from io_modules.plotting import (
    plot_twoD_xsection_by_model,
    plot_grouped_curves_by_model,
    plot_twoD_xsection_by_model_3d
)
from io_modules.read_csv import read_param_rows_csv

from io_modules.write_output_csv import write_design_metrics_csv 

from core.config import (
    optionsConfig,
    BaselineGeometryConfig,
    NURBSConfig,
    CurveSettings
)
from core.param_builder import build_params_from_row 

options = optionsConfig()
baseline = BaselineGeometryConfig()

CSV_PATH = "datasets/test.csv"
BENDING_CONFIGS_DIR = "datasets/bending_configs/"

def generate_prototypes(
    csv_path: str,
    run: RunOptions,
    default_model_export_folder: str = "prototype_models",
    base_export_folder: str = "prototype_bases",
    plots_export_folder: str = "prototype_plots",
    oneD_plots_filename: str = "prototypes_1D",
    twoD_plots_filename: str = "prototypes_2D",
    threeD_plots_filename: str = "prototypes_3D"
) -> Dict[str, List[Dict[str, Any]]]:

    rows = read_param_rows_csv(csv_path)
    plots_data: Dict[str, List[Dict[str, Any]]] = {}
    bucket = "_all"
    arc_length_by_id: Dict[str, Optional[float]] = {}

    for i, row in enumerate(rows, start=1):
        # Build Params directly from CSV row (no GeometryParams default)
        params, bending_csv_path = build_params_from_row(
            row, baseline, bending_configs_dir=BENDING_CONFIGS_DIR
        )

        # Ensure export meta (CSV should already have these)
        proto_id = params.export_filename
        if not proto_id:
            proto_id = f"row_{i}"
            params.export_filename = proto_id
        if not params.export_folder:
            params.export_folder = default_model_export_folder

        print(f"[{i}/{len(rows)}] Building {proto_id} "
              f"{'(bending)' if params.bending_enabled else '(revolve)'}")

        # ---- Build
        if params.bending_enabled:
            report = generate_geometry_bend(params, bending_csv_path, testing_mode=False)
        else:
            report = generate_geometry(params)

        # # ---- Save arc length metric
        if params.bending_enabled:
            arc_length_by_angle = {}
            for idx, xsec in enumerate(report.xsections2d_list):
                arc_length_by_angle[f"{proto_id}_ang{idx}"] = xsec.arc_length
                # Save max arc length for this prototype (ignore None values)
            lengths = [v for v in arc_length_by_angle.values()]
            arc_length_by_id[proto_id] = max(lengths)
        else:
            arc_length_by_id[proto_id] = report.xsections2d.arc_length

        # ---- Export model
        if run.export_model:
            print(f"Exporting 3D Model as {run.ch_export_type}...")
            export(
                report.model3d.threeD_model,
                title=params.export_filename,
                overwrite=run.overwrite,
                directory=run.directory,
                export_type=run.ch_export_type,
                folder=params.export_folder,
            )

        # ---- Optional: export base components
        if getattr(run, "export_bases", False):
            from builders.build_modules.base_helpers import create_base
            print("Generating Base Components...")
            base_components = create_base(
                params=params,
                xsection2D=report.xsections2d_list if params.bending_enabled else report.xsections2d
            )
            export_components = ("foundation", "seal", "clamp", "base_exploded") if options.export_base_exploded_flag else ("foundation", "seal", "clamp")
            for comp_name in export_components:
                export(
                    getattr(base_components, comp_name),
                    title=f"{params.export_filename}_{comp_name}",
                    overwrite=run.overwrite,
                    directory=run.directory,
                    export_type="stl",
                    folder=base_export_folder,
                )

        # ---- Collect for grouped plotting
        if params.bending_enabled:
            for idx, xsec in enumerate(report.xsections2d_list):
                plots_data.setdefault(bucket, []).append({
                    "proto_id": f"{proto_id}_ang{idx}",
                    "model_id": proto_id,
                    "section_idx": idx,
                    "value": None,
                    "curve_points": report.curves1d_list[idx].curve_points,
                    "control_points": report.curves1d_list[idx].control_points,
                    "cross_section_points": xsec.twoD_cross_section,
                    "angular_section": params.angular_section
                })
        else:
            plots_data.setdefault(bucket, []).append({
                "proto_id": proto_id,
                "model_id": proto_id,
                "section_idx": 0,
                "value": None,
                "curve_points": report.curves1d.curve_points,
                "control_points": report.curves1d.control_points,
                "cross_section_points": report.xsections2d.twoD_cross_section,
            })

    # ---- Write output metrics to CSV
    write_design_metrics_csv(csv_path, arc_length_by_id)

    # ---- Plot once at the end
    if plots_data and getattr(run, "plot_1d", False):
        fig1 = plot_grouped_curves_by_model(plots_data, cols=3, share_axes=True)
        export_plot(fig1, title=oneD_plots_filename, export_type="png",
                    directory=run.directory, folder=plots_export_folder, overwrite=True)

    if plots_data and getattr(run, "plot_2d", False):
        if any(e.get("section_idx", 0) > 0 for e in plots_data.get(bucket, [])):
            fig2 = plot_twoD_xsection_by_model(plots_data, plot_points=True, markersize=1, cols=3, share_axes=True)
        else:
            fig2 = plot_twoD_xsection_by_model(plots_data)
        export_plot(fig2, title=twoD_plots_filename, export_type="png",
                    directory=run.directory, folder=plots_export_folder, overwrite=True)

    if plots_data and getattr(run, "plot_3d", False):
        fig3d = plot_twoD_xsection_by_model_3d(plots_data, resolution=6, projection="ortho", elev=25, azim=-20, plot_points=True)
        export_plot(fig3d, title=threeD_plots_filename, export_type="png",
                    directory=run.directory, folder=plots_export_folder, overwrite=True)

    return plots_data

def main():
    run = RunOptions(
        export_model=True,
        export_bases=options.export_bases_flag,
        plot_1d=True,
        plot_2d=True,
        plot_3d=True,
        overwrite=True,
        ch_export_type="stl",
        directory=".",
    )
    generate_prototypes(CSV_PATH, run=run)

if __name__ == "__main__":
    main()
