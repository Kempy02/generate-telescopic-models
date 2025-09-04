# builders/build_modules/interpolate_bend.py

import csv, os, json
from io_modules.read_csv import read_param_rows_csv  # assumes it returns proper types

def interpolate_bending_config(input_csv_path, base_params, step_deg=5, out_csv_path=None):
    """
    Read input bending config CSV, add base(0°) from base_params, interpolate to step_deg, write a full CSV.
    Assumptions:
      - read_param_rows_csv returns dicts with numbers/lists already parsed.
      - Lists on both sides have same length.
      - If one side is None, we copy the other side's value.
    """
    # 1) Read sparse rows
    rows = read_param_rows_csv(input_csv_path)  # list of dicts
    if not rows:
        raise ValueError("No rows found in bending CSV.")

    # 2) Work out which fields to include (everything except the angle)
    fieldnames = sorted({k for r in rows for k in r.keys()} - {"angular_section"})

    # 3) Make a base row (0°) from base_params, and if final row is equal to 355°, a target row (360°) from base_params
    base_row = {"angular_section": 0.0}

    if base_params.angular_section == 360.0:
        for k in fieldnames:
            base_row[k] = getattr(base_params, k, None)

        target_row = None
        # if final row is equal to 360°, raise error and correct
        if float(rows[-1]["angular_section"]) == 360.0:
            for k in fieldnames:
                if getattr(base_params, k, None) != getattr(rows[-1], k, None):
                    print("ERROR: Final row (360°) does not match base parameters.\n    Proceeding: Setting final row to base parameters.")
                    break
            target_row = {"angular_section": 360.0}
            for k in fieldnames:
                target_row[k] = getattr(base_params, k, None)
        # else set final row
        # elif float(rows[-1]["angular_section"]) == 355.0:
        else:
            target_row = {"angular_section": 360.0}
            for k in fieldnames:
                target_row[k] = getattr(base_params, k, None)
    else:
        target_row = {"angular_section": base_params.angular_section}
        for k in fieldnames:
            target_row[k] = getattr(base_params, k, None)


    # 4) Merge base, target + CSV rows; ensure angles are floats; sort by angle
    keyframes = [base_row, target_row] + rows
    keyframes = [dict(r) for r in keyframes if "angular_section" in r]
    for r in keyframes:
        r["angular_section"] = float(r["angular_section"])
    keyframes.sort(key=lambda r: r["angular_section"])

    # 5) Interpolate between consecutive pairs at fixed step
    dense = []
    for i in range(len(keyframes) - 1):
        A = keyframes[i]
        B = keyframes[i + 1]
        a = float(A["angular_section"])
        b = float(B["angular_section"])
        span = b - a
        if span <= 0:
            continue

        # number of intervals to reach 'b' from 'a'
        n = max(1, int(round(span / float(step_deg))))
        step = span / n

        # j = 0..n produces angles a, a+step, ..., b
        for j in range(n + 1):
            ang = a + j * step
            # avoid duplicates where segments meet
            if dense and abs(ang - dense[-1]["angular_section"]) < 1e-9:
                continue

            t = 0.0 if n == 0 else (j / float(n))  # interpolation factor
            out = {"angular_section": round(ang, 6)}

            for k in fieldnames:
                va = A.get(k, None)
                vb = B.get(k, None)

                if va is None and vb is None:
                    out[k] = None
                else:
                    if va is None: va = vb
                    if vb is None: vb = va

                    # element-wise for lists, numeric lerp otherwise
                    if isinstance(va, list) and isinstance(vb, list):
                        m = min(len(va), len(vb))
                        out[k] = [va[idx] * (1.0 - t) + vb[idx] * t for idx in range(m)]
                    else:
                        try:
                            out[k] = float(va) * (1.0 - t) + float(vb) * t
                        except Exception:
                            # non-numeric -> just pick one side
                            out[k] = va if t < 0.5 else vb

            dense.append(out)

    # Edge case: only one keyframe
    if not dense and keyframes:
        dense = [keyframes[0]]

    # 6) Write CSV
    if out_csv_path is None:
        root, _ = os.path.splitext(input_csv_path)
        out_csv_path = f"{root}_{int(step_deg)}deg_interp.csv"

    cols = ["angular_section"] + fieldnames
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in dense:
            # lists -> JSON strings so they round-trip cleanly
            row = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
            w.writerow({k: row.get(k) for k in cols})

    print(f"[interpolate_bending_config] wrote {len(dense)} rows → {out_csv_path}")

    rows = read_param_rows_csv(out_csv_path)

    return rows

