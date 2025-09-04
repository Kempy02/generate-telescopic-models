# io_modules/write_output_csv.py

import csv
from typing import Dict, Optional

def write_design_metrics_csv(
    input_csv_path: str,
    arc_length_by_id: Dict[str, float],
    out_csv_path: Optional[str] = None,
    id_col: str = "Prototype ID",
    metric_col: str = "arc_length",
) -> str:
    """
    Read the input design CSV and write a new CSV with an extra column `metric_col`.
    Values are pulled from `arc_length_by_id[id]` where `id` is the value in `id_col`.
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
    if metric_col not in fieldnames:
        fieldnames.append(metric_col)

    # Fill metric
    for row in rows:
        pid = row.get(id_col, "")
        val = arc_length_by_id.get(pid)
        row[metric_col] = "" if val is None else f"{float(val):.6f}"

    # Write new CSV
    with open(out_csv_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[metrics] wrote: {out_csv_path}")
    return out_csv_path
