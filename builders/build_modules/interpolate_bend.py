# builders/build_modules/interpolate_bend.py
import csv, os, json
from typing import List, Dict, Any
from io_modules.read_csv import read_param_rows_csv

from core.config import BendSettings
bend = BendSettings()

def _coerce_int_like(val, *, name: str) -> int:
    """Coerce val to an exact integer, allowing e.g. '3' or 3.0 (but not 3.2)."""
    if val is None:
        raise ValueError(f"'{name}' is None but required to be an integer.")
    # try numeric first
    try:
        f = float(val)
        i = int(round(f))
        if abs(f - i) <= 1e-9:
            return i
        raise ValueError
    except ValueError:
        # try pure int string
        try:
            return int(str(val).strip())
        except Exception:
            raise ValueError(f"'{name}' must be an integer (got {val!r}).")

def interpolate_bending_config_from_config(input_csv_path: str,
                                           step_deg: float = 5.0,
                                           close_cycle: bool = True,
                                           out_csv_path: str | None = None) -> List[Dict[str, Any]]:
    """
    Read a per-model config CSV that contains one or more keyframes with 'angular_section'.
    - Ensures there is a 0° keyframe (uses the smallest angle if missing).
    - If close_cycle=True, ensures a 360° keyframe by copying the 0° row if missing.
    - Linearly interpolates between consecutive keyframes to multiples of 'step_deg'.
    - Enforces constant integer 'n_curves' if present (no interpolation).
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

    # ---- Enforce constant integer n_curves if present in ANY row ----
    constant_n_curves = None
    if any("n_curves" in r for r in rows):
        values = []
        for r in rows:
            if "n_curves" in r and r["n_curves"] not in (None, ""):
                values.append(_coerce_int_like(r["n_curves"], name="n_curves"))
        if not values:
            # column exists but all missing → require explicit value
            raise ValueError("Column 'n_curves' is present but contains no usable values.")
        uniq = sorted(set(values))
        if len(uniq) != 1:
            # show which angles have which values to help debug
            by_val = {}
            for r in rows:
                v = r.get("n_curves", None)
                if v in (None, ""): continue
                iv = _coerce_int_like(v, name="n_curves")
                by_val.setdefault(iv, []).append(r["angular_section"])
            msg_parts = [f"{val}: angles {sorted(angs)}" for val, angs in by_val.items()]
            raise ValueError("Inconsistent 'n_curves' across keyframes → " + "; ".join(msg_parts))
        constant_n_curves = uniq[0]
        # ensure it's listed in fields so it is written out
        if "n_curves" not in fields:
            fields.append("n_curves")
    # ------------------------------------------------------------------

    # ensure 0°
    if rows[0]["angular_section"] != 0.0:
        zero = dict(rows[0]); zero["angular_section"] = 0.0
        # stamp constant n_curves if we have it
        if constant_n_curves is not None:
            zero["n_curves"] = constant_n_curves
        rows = [zero] + rows

    # ensure 360° (optional)
    if close_cycle and rows[-1]["angular_section"] != bend.total_angular_section:
        end = dict(rows[0])  # copy 0° values
        end["angular_section"] = bend.total_angular_section
        if constant_n_curves is not None:
            end["n_curves"] = constant_n_curves
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

            # always set constant n_curves if defined
            if constant_n_curves is not None:
                out["n_curves"] = int(constant_n_curves)

            for k in fields:
                if k in ("angular_section", "n_curves"):
                    continue  # already handled

                va, vb = A.get(k, None), B.get(k, None)
                if va is None and vb is None:
                    out[k] = None
                    continue

                if va is None: va = vb
                if vb is None: vb = va

                # lists → elementwise linear interpolation
                if isinstance(va, list) and isinstance(vb, list):
                    m = min(len(va), len(vb))
                    out[k] = [va[idx]*(1-t) + vb[idx]*t for idx in range(m)]
                else:
                    # numeric → linear; fallback to step-wise pick if non-numeric
                    try:
                        out[k] = float(va)*(1-t) + float(vb)*t
                    except Exception:
                        out[k] = va if t < 0.5 else vb

            dense.append(out)

    if not dense and rows:
        # degenerate case: only one row
        only = dict(rows[0])
        if constant_n_curves is not None:
            only["n_curves"] = int(constant_n_curves)
        dense = [only]

    # write CSV
    if out_csv_path is None:
        root, _ = os.path.splitext(input_csv_path)
        out_csv_path = f"{root}_{int(step_deg)}deg_interp.csv"

    cols = ["angular_section"] + sorted(set(fields + (["n_curves"] if constant_n_curves is not None else [])))
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in dense:
            row = {}
            for k in cols:
                v = r.get(k)
                if isinstance(v, list):
                    row[k] = json.dumps(v)
                elif k == "n_curves" and v is not None:
                    row[k] = int(v)  # ensure integer in the file
                else:
                    row[k] = v
            w.writerow(row)

    # return parsed dicts for the builder
    return read_param_rows_csv(out_csv_path)
