import csv
import ast

def read_prototypes_csv(csv_filename):
    """
    Reads a CSV containing columns:
      Factor Varied, Variable Name, Prototype ID, Value
    Returns separate lists: factor_varied_list, var_name_list, proto_id_list, value_list
    """
    factor_varied_list = []
    var_name_list = []
    proto_id_list = []
    value_list = []

    with open(csv_filename, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            factor_varied_list.append(row['Factor Varied'])
            var_name_list.append(row['Variable Name'])
            proto_id_list.append(row['Prototype ID'])
            
            val_str = row['Value']
            # If it's something like "[1.5, 1, 0.5]", parse it as a Python list:
            try:
                val_parsed = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                # If it's just a number or string, you could parse float or keep as-is
                # Example: parse as float if numeric
                try:
                    val_parsed = float(val_str)
                except ValueError:
                    val_parsed = val_str
            value_list.append(val_parsed)

    return factor_varied_list, var_name_list, proto_id_list, value_list

def read_bending_factors_csv(csv_filename):
    """
    Reads a CSV containing columns:
      Angular section, Factor Varied, Variable Name, Factor
    Returns separate lists:
      angular_list, factor_varied_list, var_name_list, factor_list
    """
    angular_list = []
    factor_varied_list = []
    var_name_list = []
    factor_list = []

    with open(csv_filename, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # angular section: try int -> float -> keep as string
            ang = row.get('Angular section', '')
            try:
                ang_parsed = int(ang)
            except ValueError:
                try:
                    ang_parsed = float(ang)
                except ValueError:
                    ang_parsed = ang
            angular_list.append(ang_parsed)

            factor_varied_list.append(row.get('Factor Varied'))
            var_name_list.append(row.get('Variable Name'))

            # Factor: allow list-like strings, numeric, or raw string
            factor_str = row.get('Factor', '')
            try:
                factor_parsed = ast.literal_eval(factor_str)
            except (ValueError, SyntaxError):
                try:
                    factor_parsed = float(factor_str)
                except ValueError:
                    factor_parsed = factor_str
            factor_list.append(factor_parsed)

    return angular_list, factor_varied_list, var_name_list, factor_list

import csv, ast
from typing import Any, Dict, List

def _parse_cell(v: str) -> Any:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    low = s.lower()
    if low == "none":
        return None
    if low in ("true", "false"):
        return low == "true"
    # try literal (numbers, lists, tuples, dicts, etc.)
    try:
        return ast.literal_eval(s)
    except Exception:
        return s  # fallback: raw string

def read_param_rows_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Reads a 'full-params-per-row' CSV.
    Returns a list of dictionaries (column_name -> parsed_value).
    """
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip(): _parse_cell(v) for k, v in raw_row.items()}
            rows.append(row)
    return rows

def read_bending_factors_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Expected headers (at minimum):
      angular_section, amplitude0, desired_radius, period_factors,
      offset_factor_x, mid_factor_x, curve_weight, thickness_factor, thickness_factor2, thickness, thickness_mode, n_curves
    Returns a list of dicts, parsed and sorted by angular_section ascending.
    """
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k.strip(): _parse_cell(v) for k, v in raw.items()}
            # ensure angular_section is numeric
            if "angular_section" not in row or row["angular_section"] is None:
                raise ValueError("bending CSV row missing 'angular_section'")
            row["angular_section"] = float(row["angular_section"])
            rows.append(row)
    rows.sort(key=lambda r: r["angular_section"])
    return rows