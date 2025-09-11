# builders/build_modules/interpolate_bend.py
import csv, os, json
from typing import List, Dict, Any
from io_modules.read_csv import read_param_rows_csv

def interpolate_bending_config_from_config(input_csv_path: str,
                                           step_deg: float = 5.0,
                                           close_cycle: bool = True,
                                           out_csv_path: str | None = None) -> List[Dict[str, Any]]:
    """
    Read a per-model config CSV that contains one or more keyframes with 'angular_section'.
    - Ensures there is a 0° keyframe (uses the smallest angle if missing).
    - If close_cycle=True, ensures a 360° keyframe by copying the 0° row if missing.
    - Linearly interpolates between consecutive keyframes to multiples of 'step_deg'.
    - Writes an interpolated CSV and returns the parsed rows (as dicts).
    """
    rows = read_param_rows_csv(input_csv_path)
    if not rows:
        raise ValueError(f"No rows in {input_csv_path}")

    # ensure float angles + collect fields
    for r in rows:
        r["angular_section"] = float(r["angular_section"])
    rows.sort(key=lambda r: r["angular_section"])
    fields = sorted({k for r in rows for k in r.keys()} - {"angular_section"})

    # ensure 0°
    if rows[0]["angular_section"] != 0.0:
        zero = dict(rows[0]); zero["angular_section"] = 0.0
        rows = [zero] + rows

    # ensure 360° (optional)
    if close_cycle and rows[-1]["angular_section"] != 360.0:
        end = dict(rows[0])  # copy 0° values
        end["angular_section"] = 360.0
        rows.append(end)

    # interpolate
    dense: List[Dict[str, Any]] = []
    for i in range(len(rows)-1):
        A, B = rows[i], rows[i+1]
        a, b = A["angular_section"], B["angular_section"]
        span = b - a
        if span <= 0: 
            continue
        n = max(1, int(round(span / float(step_deg))))
        step = span / n
        for j in range(n+1):
            ang = a + j*step
            if dense and abs(ang - dense[-1]["angular_section"]) < 1e-9:
                continue
            t = 0.0 if n == 0 else j/float(n)
            out = {"angular_section": round(ang, 6)}
            for k in fields:
                va, vb = A.get(k, None), B.get(k, None)
                if va is None and vb is None:
                    out[k] = None
                else:
                    if va is None: va = vb
                    if vb is None: vb = va
                    if isinstance(va, list) and isinstance(vb, list):
                        m = min(len(va), len(vb))
                        out[k] = [va[idx]*(1-t) + vb[idx]*t for idx in range(m)]
                    else:
                        try:
                            out[k] = float(va)*(1-t) + float(vb)*t
                        except Exception:
                            out[k] = va if t < 0.5 else vb
            dense.append(out)

    if not dense and rows:
        dense = [rows[0]]

    # write CSV
    if out_csv_path is None:
        root, _ = os.path.splitext(input_csv_path)
        out_csv_path = f"{root}_{int(step_deg)}deg_interp.csv"

    cols = ["angular_section"] + fields
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in dense:
            row = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
            w.writerow({k: row.get(k) for k in cols})

    # return parsed dicts for the builder
    return read_param_rows_csv(out_csv_path)

