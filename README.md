# Generate Telescopic Models

Parametric, CSV‑driven generation of telescopic actuator CAD models using CadQuery. The pipeline builds 1D curve cross-sections, derives 2D cross‑sections with configurable thickness behavior, and finally constructs 3D models by revolve (linear/fast mode) or lofting through bending sections (bending mode). Batch runs, plots, and metrics are produced from simple CSVs.

## Highlights

- CSV‑driven: change geometry without code edits; one row = one model
- Two modes:
	- Linear/fast: revolve a single 2D section into a 3D model
	- Bending: interpolate per‑angle “keyframes” and loft across many 2D sections
- Exports: 3D models (STL/STEP), optional base components and exploded systems
- Plots: grouped 1D curves, 2D cross‑sections, and a 3D stacked view
- Metrics: arc length and thickness summaries saved back to CSV

## Quick start

### 1) Environment
Use the provided Conda environment file `generate_geometry.min.yml`.

Summary of requirements:
- Python 3.10
- CadQuery 2.5.x (conda-forge, strict channel priority)
- numpy, pandas, shapely, matplotlib, ipython, ffmpeg
- pip packages: geomdl==5.3.1, pytictoc

Install and activate:

```zsh
conda env create -f generate_geometry.min.yml 
conda activate generate-geometry
```

### 2) Run the sample

```zsh
python generate_models.py
```

This reads the input csv [CSV_PATH], builds the specified model(s), exports CAD models, and writes plots and metrics.

Outputs appear in:

- 3D models: `prototype_models/`
- Base parts: `prototype_bases/` (optional)
- Plots: `prototype_plots/`
- Metrics CSV: `datasets/test_with_metrics.csv`

## How it works

### Overview

1. Top‑level CSV lists models to build and points to per‑model “config” CSVs.
2. For each model, we build a base `Params` from the 0° row (or smallest angle if 0° missing).
3. If bending is enabled, we interpolate the keyframes to regular angle steps (e.g., every 10°) and generate a 2D cross‑section per angle; otherwise, we generate a single section.
4. We generate a 3D model by either revolving (linear) or lofting (bending).
5. We export models/plots and append per‑model metrics to a CSV.

### Entry points

- `generate_models.py`
	- Main runner. Reads input csv [CSV_PATH] by default and orchestrates the build/export/plot flow.
- `core/generate_geometry.py`
	- `generate_geometry(params)`: linear/fast (revolve)
	- `generate_geometry_bend(params, bending_csv_path, testing_mode)`: bending/lofting

### Builders

- `builders/one_d_build.py` → 1D curves (nurbs 1D curve generation, control point manipulation)
- `builders/two_d_build.py` → 2D cross‑section (thickness application, resampling, arc length)
- `builders/three_d_build.py` → 3D model (revolve or loft across angular sections)

### Configuration

`core/config.py` contains the main knobs:

- `optionsConfig`
	- `export_bases_flag`: export base components (Foundation, Seal, Clamp, Exploded)
	- `export_exploded_system`: export model + base as an exploded system
	- `export_model_flag`: export the prototype model (should be True for normal runs)
	- `plot_prototypes_flag`: enable plotting
	- `test_2d_mode`: generate cross‑sections only (speeds up debugging)
	- `thickness_mode`: "constant" | "linear" | "variable" | "collapsed" | "sbend"
	- `ruled_flag`: use ruled loft for straighter transitions between sections
- `BaselineGeometryConfig`: general baseline geometry settings (cap sizes, offsets)
- `NURBSConfig`, `CurveSettings`: curve generation settings
- `BendSettings`:
	- `total_angular_section`: 360° for complete model or 180° for cross-section slice
	- `angle_intervals`: step size (e.g., 10°) used for interpolation

## CSV schemas

There are two CSV layers: a top‑level models list and a per‑model “bending config” (keyframe) CSV.

### 1) Top‑level models list

File: `datasets/test.csv` (example)

Columns:

- `Prototype ID` (string) – model name and file stem
- `export_folder` (string) – destination subfolder (default `prototype_models`)
- `config_csv` (string) – path to the per‑model keyframe CSV
- `use_linear_fast` (True/False) – True: revolve a single section; False: bending/loft

Example:

```csv
Prototype ID,export_folder,config_csv,use_linear_fast
baseline1,prototype_models,datasets/configs/baseline1.csv,False
```

### 2) Cross-Section Config CSV (keyframe)

File: e.g., `datasets/configs/baseline.csv`

Required:

- `angular_section` (float) – angle in degrees for this keyframe

Typical fields (subset):

- `amplitude0` (float)
- `desired_radius` (float)
- `period_factors` (list) e.g., `[0.9, 1.0, 1.1]`
- `offset_factor_x`, `mid_factor_x`, `mid_factor_y` (floats)
- `min_y_positions` (list) e.g., `[0.0, 0.35, 0.35]` (scaled by `amplitude0`)
- `curve_weight` (float) → curve weights derived: `[1, curve_weight, 1, curve_weight, 1]`
- `thickness` (float), `thickness_factor` (list), `thickness_factor2` (list)
- `thickness_mode` (string) – see options
- `n_curves` (int, optional) – if present in any keyframe, it must be constant across all keyframes

Lists in CSV:

- Use bracketed list syntax, e.g. `[0.9, 1.0, 1.1]`. The parser accepts Python/JSON‑like list strings.

Interpolation rules (`builders/build_modules/interpolate_bend.py`):

- Ensures a 0° row (copies the smallest angle row if necessary)
- Optionally ensures a final row at `total_angular_section` by copying 0° values
- Linearly interpolates numeric values to regular steps of `angle_intervals`
- Interpolates lists element‑wise; non‑numeric fields fall back to step‑wise pick (A before midpoint, B after)
- If `n_curves` is present, it must be a single integer across all keyframes

The interpolated CSV is written next to your source (e.g., `baseline1_10deg_interp.csv`) and then used for building.

## Parameters used in the build

Built `Params` (from `core/types.py`) contains the fields your builders consume:

- amplitude0, desired_radius, period_factors
- offset_factor_x, mid_factor_x, mid_factor_y
- min_y_positions
- curve_weight, weights (derived), n_curves
- thickness, thickness_factor, thickness_factor2, thickness_mode
- cap_thickness (from baseline), center_offset
- export_filename, export_folder
- bending_enabled, angular_section (for base row)

## Outputs

- Models (default STL): `prototype_models/<PrototypeID>.stl`
- Base components (optional): `prototype_bases/`
	- `Foundation_<base>.stl`, `Seal_<PrototypeID>.stl`, `Clamp_Cutout_<base>.stl`, `Base_Exploded_<PrototypeID>.stl`
- Exploded system (optional): `prototype_models/<PrototypeID>_exploded_system.stl`
- Plots:
	- `prototype_plots/prototypes_1D.png`
	- `prototype_plots/prototypes_2D.png`
	- `prototype_plots/prototypes_3D.png`
- Metrics CSV: a `*_with_metrics.csv` sibling to your top‑level models CSV (e.g., `datasets/test_with_metrics.csv`)
	- arc_length_min/max (+ ratio)
	- thickness_min/max (+ ratio)

## Tuning and flags

Key runtime flags live in `core/config.py` (class `optionsConfig`). Notable ones:

- `export_model_flag`: keep True to write CAD (turn off for quick debugging)
- `export_bases_flag`, `export_exploded_system`: base components/exploded system exports
- `plot_prototypes_flag`: enable 1D/2D/3D plot generation
- `test_2d_mode`: only compute cross‑sections (no 3D build). Helpful if 3D fails.
- `ruled_flag`: toggles a straighter “ruled” loft vs smoother transitions
- `constant_cap_thickness`: force the end cap thickness to be constant

`BendSettings` controls interpolation and plotting cadence:

- `total_angular_section`: bend coverage, usually 360° (testing may use 180° for visualisation)
- `angle_intervals`: step size in degrees for interpolation (e.g., 10)

## Repository structure (brief)

- `generate_models.py` – main runner
- `core/` – configs, types, geometry orchestration
- `builders/` – 1D, 2D, 3D build steps and helper modules
- `io_modules/` – CSV IO, exporting, plotting, progress bar, metrics writer
- `datasets/` – sample model lists, keyframe configs, and output metrics
- `prototype_models/`, `prototype_bases/`, `prototype_plots/` – default output locations

## Troubleshooting

- 3D build failures:
	- Try `optionsConfig.test_2d_mode=True` to generate only 2D cross‑sections for debugging. Visually inspect the validity of 2D cross-sections.
    - Try reducing the `total_angular_section` to identify at what stage (of somplete revolution) the geometry generation is failing. 
    - Try vary `angle_intervals` in BendSettings [Config]. Sometimes high resolution (low `angle_intervals` values) can cause OCP failure.
	- Ensure your keyframes interpolate cleanly (lists consistent length; numeric fields numeric).
- Overwrite prompts:
	- Exports prompt if a file exists and `overwrite` is False. The default run passes `overwrite=True`.
- Constant `n_curves`:
	- If you include `n_curves` in keyframes, it must be an integer and the consistent across all keyframes.

## Extending

To add a new parameter:

1. Add a field to `core/types.Params` (if it’s used beyond CSV passthrough).
2. Pass it through in `core/param_builder.py` from your keyframe CSV.
3. Consume it in the relevant builder(s): `one_d_build.py` / `two_d_build.py` / `three_d_build.py`.
4. Add the parameter to your keyframe CSV rows. If it’s numeric or a same‑length list, it will be interpolated. Otherwise, it will be step‑wise.

## Revised Detailed Workflow Overview

1) Environment
- Create and activate the conda environment from `generate_geometry.min.yml` (see Quick start). This installs CadQuery 2.5.x and required Python packages.

2) Prepare input CSVs
- Top-level models list (e.g., `datasets/test.csv`) with columns: `Prototype ID, export_folder, config_csv, use_linear_fast` — one row per model to build.
- Per-model keyframe CSVs (e.g., under `datasets/configs/`) that include `angular_section` and geometry fields (`amplitude0`, `desired_radius`, `period_factors`, thickness fields, etc.).

3) Run the generator
- Execute `python generate_models.py`. The script reads the models list and processes each row.

4) Build per model
- Base parameters: `core/param_builder.build_params_from_config_csv` reads a model’s keyframe CSV and constructs a base `Params` from the 0° (or smallest angle) row.
- Linear/fast mode (`use_linear_fast=True`): `core.generate_geometry.generate_geometry(params)`
	- 1D curves (`builders/one_d_build.py`) → 2D cross-section (`builders/two_d_build.py`) → revolve to 3D (`builders/three_d_build.py`).
- Bending mode (`use_linear_fast=False`): `core.generate_geometry.generate_geometry_bend(params, config_csv_path, testing_mode=False)`
	- Keyframes interpolated via `builders/build_modules/interpolate_bend.interpolate_bending_config_from_config` using `BendSettings.angle_intervals`.
	- Generate 1D/2D per angular section; loft a 3D model through sections (`builders/three_d_build.py`).

5) Export artifacts
- Prototype model: `io_modules/exporting.export(...)` to `prototype_models/` (export type/overwrite controlled in `generate_models.py` and `core/config.py`).
- Optional base components: `builders/build_modules/base_helpers.create_base` exported to `prototype_bases/` (controlled by `optionsConfig.export_bases_flag` and `optionsConfig.export_exploded_system`).
- Plots: grouped 1D/2D/3D summaries via `io_modules.plotting` to `prototype_plots/` when plotting flags are enabled.

6) Save metrics
- Arc length (min/max) and thickness (min/max) metrics per model appended to a sibling CSV (e.g., `datasets/test_with_metrics.csv`) via `io_modules.write_output_csv.write_design_metrics_csv`.

7) Review
- Inspect models in `prototype_models/`, bases in `prototype_bases/`, plots in `prototype_plots/`, and metrics CSVs in `datasets/`.


