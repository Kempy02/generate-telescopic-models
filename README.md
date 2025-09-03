## Workflow Overview

### 1. Define Baseline Design parameters:

modules/config.py defines the parameters of the baseline design, from which, relevant parameters will be overwridden to generate new designs

### 2. Populate datasets/prototypes1.csv with four columns:

factor_varied; var_name; proto_id; value

e.g. “Amplitude”; amplitude0; AMP1; 1.2

One row = one design.

### 3. GeneratePrototypes.py: 
reads each row, copies a fresh GeometryParams() object, overrides the given attribute, and calls generateGeometry().

### 4. Telesctopic Models + Base components:
are exported via modules/exporting.export().

### 5. Optional plots:
are grouped and saved via modules/plotting.

## Key Features:

### CSV‑driven batch generation (e.g. prototypes1.csv):

One row = one prototype; automatically loops through as many variants as you like.

### Param‑object overrides:

Baseline parameters are defined in 'module/config.py'.
Once 'GeneratatePrototypes.py' is run, every row/design will override any relevant attribute in GeometryParams (if varied). This decouples design configurations from the main logic of the generation script.

## Outputs

### Models/Prototypes:
One CAD file (set either .STEP or .stl) is output for each design

### Base Components:
Four CAD files (set either .STEP or .stl) are output for each design; this includes three base components and an exploded base view for validation

### Cross-sectional Plots
Relevant cross-sectional plots are grouped into 1D and 2D cross-sectional plots and output for visualisation and validation of design configurations

### Output Destination
Output destinations are set in  'GeneratePrototypes.py'.

## Optional Flags

Several flags in 'modules/config.py' can be set depending on desired outputs:

### create_base_flag & export_bases_flag: 
Set these flags true if you'd like design specific base rig CAD models output for each design
### export_crossSection_flag: 
Set true if you'd like a cross-sectional model output for each design; useful for simple validation of output

### export_model_flag: 
should always be set true to output prototype CAD models; can be turned off if troubleshooting or validating other components

### plot_prototypes_flag:
Recommended to set true, for validation and saving of cross-sectional design characteristics

### calculate_area_flag:
Useful for understanding the impact of explicit design characteristics on internal volume - cross-sectional area

## Adding New Parameters

### 1. Edit base_params.py: 
add an attribute to GeometryParams.

### 2. Update modules/generateGeometry.py:
this is where you will add the logic to handle the new paramater

### 3. Add a row in the CSV: 
set var_name = the new attribute name, and set value

No further code changes required, the override mechanism should pick it up automatically.

## Recommended Additions

### Save and Export parameteric settings for all designs:

The current implementation saves and exports plots for each design but not exact parametric settings. This was sufficient for OFAT design exploration but if multiple paramaters are varied for each design, it is recommended to save and export these configurations. 

To implement this, simply export the updated GeometryParams class for each design/iteration, in any desired format (e.g. .csv)