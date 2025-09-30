# io_modules/write_output_csv.py

import csv
from typing import Dict, Optional

def write_design_metrics_csv(
    input_csv_path: str,
    arc_length_min_by_id: Dict[str, float],
    arc_length_max_by_id: Dict[str, float],
    thickness_min_by_id: Dict[str, float],
    thickness_max_by_id: Dict[str, float],
    out_csv_path: Optional[str] = None,
    id_col: str = "Prototype ID",
    metric_col1: str = "arc_length_min",
    metric_col2: str = "arc_length_max",
    metric_col3: str = "arc_length_ratio",
    metric_col4: str = "thickness_min",
    metric_col5: str = "thickness_max",
    metric_col6: str = "thickness_ratio"
) -> str:
    """
    Read the input design CSV and write a new CSV with an extra column `metric_col`.
    Values are pulled from `arc_length_min_by_id[id]` and `arc_length_max_by_id[id]` where `id` is the value in `id_col`.
    Missing metrics are left blank.
    """
    # Decide output path
    if out_csv_path is None:
        out_csv_path = (
            input_csv_path[:-4] + "_with_metrics.csv"
            if input_csv_path.lower().endswith(".csv")
            else input_csv_path + "_with_metrics.csv"
        )

    # Read original rows
    with open(input_csv_path, "r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # Ensure new column exists (append at end)
    if metric_col1 not in fieldnames:
        fieldnames.append(metric_col1)

    if metric_col2 not in fieldnames:
        fieldnames.append(metric_col2)

    if metric_col3 not in fieldnames:
        fieldnames.append(metric_col3)

    if metric_col4 not in fieldnames:
        fieldnames.append(metric_col4)

    if metric_col5 not in fieldnames:
        fieldnames.append(metric_col5)

    if metric_col6 not in fieldnames:
        fieldnames.append(metric_col6)

    # Fill metric
    for row in rows:
        pid = row.get(id_col, "")
        val = arc_length_min_by_id.get(pid)
        row[metric_col1] = "" if val is None else f"{float(val):.6f}"
        val = arc_length_max_by_id.get(pid)
        row[metric_col2] = "" if val is None else f"{float(val):.6f}"
        if row[metric_col1] and row[metric_col2]:
            try:
                ratio = max((float(row[metric_col1]) / float(row[metric_col2])), 
                            (float(row[metric_col2]) / float(row[metric_col1])))
                row[metric_col3] = f"{ratio:.6f}"
            except ZeroDivisionError:
                row[metric_col3] = ""
        else:
            row[metric_col3] = ""
        val = thickness_min_by_id.get(pid)
        row[metric_col4] = "" if val is None else f"{float(val):.6f}"
        val = thickness_max_by_id.get(pid)
        row[metric_col5] = "" if val is None else f"{float(val):.6f}"
        if row[metric_col4] and row[metric_col5]:
            try:
                ratio = max((float(row[metric_col4]) / float(row[metric_col5])), 
                            (float(row[metric_col5]) / float(row[metric_col4])))
                row[metric_col6] = f"{ratio:.6f}"
            except ZeroDivisionError:
                row[metric_col6] = ""
        else:
            row[metric_col6] = ""

    # Write new CSV
    with open(out_csv_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[metrics] wrote: {out_csv_path}")
    return out_csv_path
