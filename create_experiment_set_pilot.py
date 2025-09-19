# create_experiment_set_pilot.py
# -----------------------------------
# Pilot run config generator:
# - For bending prototypes: emit two angular sections:
#       0°   -> every parameter at its MIN
#       180° -> every parameter at its MAX
# - For linear prototypes: emit one section at 0° with MIN values.
#
# Notes on vectors with constraints:
# - If PRESERVE_VECTOR_CONSTRAINTS = True, setting all elements to min/max
#   will renormalize (e.g., to mean=1 or sum=3), so both angles may end up
#   identical for those vectors. This is expected and keeps downstream
#   assumptions intact.
# -----------------------------------

import os, json
import numpy as np
import pandas as pd

# ---------------------------
# Settings
# ---------------------------
SEED = 42
N_PROTOTYPES = 10
P_BENDING = 1.0                     # pilot: make all bending by default (set to 0.5 if you want a mix)
EXPORT_FOLDER = "prototype_models"
MASTER_CSV_PATH = "datasets/experiment_pilot_dataset.csv"
CONFIG_DIR = "datasets/configs"
os.makedirs(CONFIG_DIR, exist_ok=True)

# Pilot angles are fixed to [0, 180]
ANGLE_MIN = 0.0
ANGLE_MAX = 180.0

# ---------------------------
# Parameter bounds
# ---------------------------
PARAM_RANGES = {
    "amplitude0":      (20, 20),   # fixed
    "desired_radius":  (20, 25),
    "offset_factor_x": (-0.7, 0.7),
    "mid_factor_x":    (-0.5, 0.5),
    "curve_weight":    (5.0, 10.0),
    "thickness":       (0.5, 1.5),
}

VEC_META = {
    # mode "sum3": elements renormalized to sum to 3
    "period_factors":   dict(lo=0.5, hi=1.5, m=3, mode="sum3"),
    # mode "mirror": normalize to mean 1, then make last = first (your prior behavior)
    "thickness_factor": dict(lo=0.5, hi=1.5, m=3, mode="mirror"),
}

# ---------------------------
# Rounding / formatting controls (CSV readability)
# ---------------------------
ROUND_MODE = "dp"                # "dp" (decimal places) or "sig" (significant figs)
SCALAR_DP = 2                    # used if ROUND_MODE == "dp"
VECTOR_DP = 2                    # used if ROUND_MODE == "dp"
SIG_FIGS_SCALAR = 3              # used if ROUND_MODE == "sig"
SIG_FIGS_VECTOR = 3              # used if ROUND_MODE == "sig"

ANGLE_AS_INT = True              # write angles as ints (e.g., 0, 180)
PRESERVE_VECTOR_CONSTRAINTS = True  # keep sum=3 or mean=1 exactly after rounding

rng = np.random.default_rng(SEED)

# ---------------------------
# Rounding / serialization
# ---------------------------
def _round_sig(x: float, sig: int) -> float:
    if x == 0 or not np.isfinite(x):
        return float(x)
    p = int(np.floor(np.log10(abs(x))))
    return float(round(x, sig - 1 - p))

def _round_scalar(x: float) -> float:
    if ROUND_MODE == "dp":
        return float(round(x, SCALAR_DP))
    else:
        return _round_sig(x, SIG_FIGS_SCALAR)

def _round_vector(arr: np.ndarray) -> np.ndarray:
    if ROUND_MODE == "dp":
        r = np.round(arr.astype(float), VECTOR_DP)
    else:
        r = np.array([_round_sig(float(v), SIG_FIGS_VECTOR) for v in arr], dtype=float)
    return r

def _format_angle(a: float):
    return int(round(a)) if ANGLE_AS_INT else float(a)

def _serialize_vector(vec: np.ndarray) -> str:
    v = _round_vector(vec)
    return json.dumps([float(x) for x in v])

def _serialize_vector_with_constraint(vec: np.ndarray, mode: str) -> str:
    v = _round_vector(vec)
    if PRESERVE_VECTOR_CONSTRAINTS:
        if mode == "sum3":
            s = float(np.sum(v))
            v = (v * (3.0 / s)) if s != 0 else v
            v = _round_vector(v)
        elif mode == "mean1":
            m = float(np.mean(v))
            v = (v / m) if m != 0 else v
            v = _round_vector(v)
        elif mode == "mirror":
            # normalize to mean 1, then enforce symmetry last=first
            m = float(np.mean(v))
            v = (v / m) if m != 0 else v
            v = _round_vector(v)
            lst = list(v)
            if len(lst) >= 1:
                first_val = lst[0]
                lst[-1] = first_val
            v = _round_vector(np.array(lst, dtype=float))
    return json.dumps([float(x) for x in v])

# ---------------------------
# Helpers to create extreme parameter sets
# ---------------------------
def make_extreme_params(which: str) -> dict:
    """
    which: "min" or "max"
    Returns a dict of scalar and vector params at extreme bounds.
    """
    assert which in ("min", "max")
    params = {}

    # Scalars at bounds
    for name, (lo, hi) in PARAM_RANGES.items():
        params[name] = float(lo if which == "min" else hi)

    # Vectors at bounds (then optionally renormalize per mode)
    for k, meta in VEC_META.items():
        lo, hi, m, mode = meta["lo"], meta["hi"], meta["m"], meta["mode"]
        val = lo if which == "min" else hi
        vec = np.full(m, float(val), dtype=float)
        params[k] = vec
    return params

# ---------------------------
# Per-prototype config writer (pilot extremes only)
# ---------------------------
def write_config_csv_pilot(prototype_id: str, is_linear: bool) -> str:
    path = os.path.join(CONFIG_DIR, f"{prototype_id}.csv")
    cols = [
        "angular_section",
        "amplitude0",
        "desired_radius",
        "period_factors",
        "offset_factor_x",
        "mid_factor_x",
        "curve_weight",
        "thickness_factor",
        "thickness",
    ]

    rows = []

    # 0° row (mins)
    p_min = make_extreme_params("min")
    rows.append(dict(
        angular_section=_format_angle(ANGLE_MIN),
        amplitude0=_round_scalar(p_min["amplitude0"]),
        desired_radius=_round_scalar(p_min["desired_radius"]),
        period_factors=_serialize_vector_with_constraint(
            np.array(p_min["period_factors"], dtype=float),
            VEC_META["period_factors"]["mode"]
        ),
        offset_factor_x=_round_scalar(p_min["offset_factor_x"]),
        mid_factor_x=_round_scalar(p_min["mid_factor_x"]),
        curve_weight=_round_scalar(p_min["curve_weight"]),
        thickness_factor=_serialize_vector_with_constraint(
            np.array(p_min["thickness_factor"], dtype=float),
            VEC_META["thickness_factor"]["mode"]
        ),
        thickness=_round_scalar(p_min["thickness"]),
    ))

    # 180° row (maxes) only for bending
    if not is_linear:
        p_max = make_extreme_params("max")
        rows.append(dict(
            angular_section=_format_angle(ANGLE_MAX),
            amplitude0=_round_scalar(p_max["amplitude0"]),
            desired_radius=_round_scalar(p_max["desired_radius"]),
            period_factors=_serialize_vector_with_constraint(
                np.array(p_max["period_factors"], dtype=float),
                VEC_META["period_factors"]["mode"]
            ),
            offset_factor_x=_round_scalar(p_max["offset_factor_x"]),
            mid_factor_x=_round_scalar(p_max["mid_factor_x"]),
            curve_weight=_round_scalar(p_max["curve_weight"]),
            thickness_factor=_serialize_vector_with_constraint(
                np.array(p_max["thickness_factor"], dtype=float),
                VEC_META["thickness_factor"]["mode"]
            ),
            thickness=_round_scalar(p_max["thickness"]),
        ))

    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path

# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(os.path.dirname(MASTER_CSV_PATH), exist_ok=True)
    master_rows = []

    for i in range(N_PROTOTYPES):
        is_bending = (rng.random() < P_BENDING)
        proto_id = f"{'bending' if is_bending else 'linear'}{i+1}"

        config_path = write_config_csv_pilot(proto_id, is_linear=not is_bending)

        master_rows.append({
            "Prototype ID": proto_id,
            "export_folder": EXPORT_FOLDER,
            "config_csv": config_path.replace("\\", "/"),
            "use_linear_fast": (not is_bending)
        })

    pd.DataFrame(
        master_rows,
        columns=["Prototype ID", "export_folder", "config_csv", "use_linear_fast"]
    ).to_csv(MASTER_CSV_PATH, index=False)

    print(f"[OK] Wrote master → {MASTER_CSV_PATH}")
    print(f"[OK] Wrote {len(master_rows)} per-prototype configs to → {CONFIG_DIR}")

if __name__ == "__main__":
    main()
