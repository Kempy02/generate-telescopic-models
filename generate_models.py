# main.py
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from core.types import Params, RunOptions, BuildReport

from builders.build_modules.base_helpers import create_base

from core.generate_geometry import generate_geometry, generate_geometry_bend
from io_modules.exporting import export, export_plot
from io_modules.plotting import plot_grouped_curves, plot_thickness_and_factors, plot_twoD_xsection, plot_twoD_xsection_by_model, plot_grouped_curves_by_model, plot_twoD_xsection_by_model_3d
from io_modules.read_csv import read_param_rows_csv  # <-- NEW reader

from core.config import (
    optionsConfig,
    BaselineGeometryConfig,
    NURBSConfig,
    CurveSettings
)
options = optionsConfig()
baseline = BaselineGeometryConfig()

from base_params import GeometryParams

CSV_PATH = "datasets/test.csv"
BENDING_CONFIGS_DIR = "datasets/bending_configs/"

def _apply_row_to_params(params: Params, row: Dict[str, Any]) -> None:
    """
    Set attributes on params for every CSV column that matches a field
    on Params/GeometryParams. Skips control columns like Prototype ID.
    """
    skip_cols = {"Prototype ID", "export_folder"}
    for key, val in row.items():
        if key in skip_cols:
            continue
        if val is None:
            continue
        # Only set attributes that exist on params to avoid typos breaking runs
        if hasattr(params, key):
            setattr(params, key, val)


def generate_prototypes(
    csv_path: str,
    run: RunOptions,
    base_params: Optional[Params] = None,
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

    for i, row in enumerate(rows, start=1):
        proto_id = str(row.get("Prototype ID") or f"row_{i}")
        export_folder = str(row.get("export_folder") or default_model_export_folder)

        params: Params = copy.deepcopy(base_params) if base_params is not None else GeometryParams()
        _apply_row_to_params(params, row)
        setattr(params, "export_filename", proto_id)
        setattr(params, "export_folder", export_folder)

        # --- NEW: per-row bending decision ---
        bending_spec = row.get("bending_config")
        # normalize: treat '', 'None', None as no-bend
        def _is_truthy_path(x) -> bool:
            if x is None: return False
            s = str(x).strip().strip('"').strip("'")
            return s not in ("", "None", "none", "null")
        bending_enabled_now = _is_truthy_path(bending_spec)
        bending_csv_filename = str(bending_spec).strip().strip('"').strip("'")
        bending_csv_path = (BENDING_CONFIGS_DIR + bending_csv_filename) if bending_enabled_now else None

        # Reflect into params for downstream checks
        setattr(params, "bending_enabled", bool(bending_enabled_now))

        print(f"[{i}/{len(rows)}] Building {proto_id} "
              f"{'(bending)' if bending_enabled_now else '(revolve)'}")

        # Build
        if bending_enabled_now:
            setattr(params, "center_offset", baseline.loft_offset)
            report = generate_geometry_bend(params, bending_csv_path, testing_mode=False)
        else:
            setattr(params, "center_offset", baseline.revolve_offset)
            report = generate_geometry(params)

        # Export model
        if run.export_model:
            export(
                report.model3d.threeD_model,
                title=params.export_filename,
                overwrite=run.overwrite,
                directory=run.directory,
                export_type="stl",
                folder=params.export_folder,
            )

        # Optional: export base components
        if getattr(run, "export_bases", False):
            print("Generating Base Components...")
            base_components = create_base(params=params, xsection2D=report.xsections2d_list if bending_enabled_now else report.xsections2d)
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

        if bending_enabled_now:
            for idx, xsec in enumerate(report.xsections2d_list):
                plots_data.setdefault(bucket, []).append({
                    "proto_id": f"{proto_id}_ang{idx}",
                    "model_id": proto_id,        # <--- add
                    "section_idx": idx,          # <--- add
                    "value": None,
                    "curve_points": report.curves1d_list[idx].curve_points,
                    "control_points": report.curves1d_list[idx].control_points,
                    "cross_section_points": xsec.twoD_cross_section,
                    "angular_section": params.angular_section
                })
        else:
            plots_data.setdefault(bucket, []).append({
                "proto_id": proto_id,
                "model_id": proto_id,            # <--- add
                "section_idx": 0,                # <--- add
                "value": None,
                "curve_points": report.curves1d.curve_points,
                "control_points": report.curves1d.control_points,
                "cross_section_points": report.xsections2d.twoD_cross_section,
                # "angular_section": params.angular_section
            })

    # Plot once at the end
    if plots_data and getattr(run, "plot_1d", False):
        fig1 = plot_grouped_curves_by_model(
            plots_data,
            cols=3,        # adjust
            share_axes=True
        )
        export_plot(
            fig1,
            title=oneD_plots_filename,
            export_type="png",
            directory=run.directory,
            folder=plots_export_folder,
            overwrite=True,
        )

    if plots_data and getattr(run, "plot_2d", False):
        if bending_enabled_now:
            fig2 = plot_twoD_xsection_by_model(
                plot_data=plots_data,
                plot_points=True,
                markersize=1,
                cols=3,          # adjust
                share_axes=True  # useful for visual comparison across models
            )
        else:
            fig2 = plot_thickness_and_factors(
                plots_data
            )
        export_plot(
            fig2,
            title=twoD_plots_filename,
            export_type="png",
            directory=run.directory,
            folder=plots_export_folder,
            overwrite=True,
        )

    if plots_data and getattr(run, "plot_3d", False):
        fig3d = plot_twoD_xsection_by_model_3d(
            plots_data,
            resolution=6,
            projection="ortho",
            elev=25,
            azim=-20,
            plot_points=True
        )
        export_plot(
            fig3d,
            title=threeD_plots_filename,
            export_type="png",
            directory=run.directory,
            folder=plots_export_folder,
            overwrite=True,
        )

    return plots_data
 
# ---------- main ----------

def main():
    run = RunOptions(
        export_model=True,
        export_bases=options.export_bases_flag,
        plot_1d=True,
        plot_2d=True,
        plot_3d=True,
        overwrite=True,
        directory=".",
        # export_bases=True
    )

    base_params = GeometryParams()  # serves as baseline for each row
    generate_prototypes(
        CSV_PATH, 
        run=run, 
        base_params=base_params
        )

if __name__ == "__main__":
    main()
