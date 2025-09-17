# create_experiment_set.py

import os, json
import numpy as np
import pandas as pd
from scipy.stats import qmc

# ---------------------------
# Settings
# ---------------------------
SEED = 42
N_PROTOTYPES = 10
P_BENDING = 0.5                     # fraction of prototypes that are bending
EXPORT_FOLDER = "prototype_models"
MASTER_CSV_PATH = "datasets/experiment0_dataset.csv"
CONFIG_DIR = "datasets/configs"
os.makedirs(CONFIG_DIR, exist_ok=True)

# >>> USER-DEFINED angle sets <<<
#   - Use one list to apply the same angles to all bending prototypes, e.g. [[0, 90, 180, 270]]
#   - Or provide several sets; one will be chosen per prototype (random/cycle)
ANGLE_SETS = [
    [0, 180]
]
ANGLE_SET_MODE = "cycle"   # "random" or "cycle"

# ---------------------------
# Parameter bounds
# ---------------------------
PARAM_RANGES = {
    "amplitude0":      (10, 20),   # fixed
    "desired_radius":  (22, 25),
    "offset_factor_x": (-0.7, 0.7),
    "mid_factor_x":    (-0.5, 0.5),
    "curve_weight":    (1.0, 10.0),
    "thickness":       (0.5, 1.5),
}

VEC_META = {
    # mode "sum3": elements renormalized to sum to 3
    "period_factors":   dict(lo=0.5, hi=1.5, m=3, mode="sum3"),
    # mode "mean1": elements renormalized to mean 1
    "thickness_factor": dict(lo=0.5, hi=1.5, m=3, mode="mirror"),
}

# Per-angle step bounds (small deltas to avoid infeasible jumps)
STEP_DELTA = {
    "amplitude0":      0.0,
    "desired_radius":  5.0,
    "offset_factor_x": 0.5,
    "mid_factor_x":    0.5,
    "curve_weight":    5.0,
    "thickness":       0.5,
}
VEC_STEP_DELTA = {
    "period_factors":   0.5,
    "thickness_factor": 0.5,
}

rng = np.random.default_rng(SEED)

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
        # elementwise significant-figures rounding
        r = np.array([_round_sig(float(v), SIG_FIGS_VECTOR) for v in arr], dtype=float)
    return r

def _format_angle(a: float):
    return int(round(a)) if ANGLE_AS_INT else float(a)

def _serialize_vector(vec: np.ndarray) -> str:
    # round first
    v = _round_vector(vec)
    return json.dumps([float(x) for x in v])

def _serialize_vector_with_constraint(vec: np.ndarray, mode: str) -> str:
    # round, then (optionally) re-enforce constraint so it remains exactly true.
    v = _round_vector(vec)
    if PRESERVE_VECTOR_CONSTRAINTS:
        if mode == "sum3":
            s = float(np.sum(v))
            v = (v * (3.0 / s)) if s != 0 else v
            # final cosmetic touch: round again for nice printout after rescale
            v = _round_vector(v)
        elif mode == "mean1":
            m = float(np.mean(v))
            v = (v / m) if m != 0 else v
            v = _round_vector(v)
        elif mode == "mirror":
            m = float(np.mean(v))
            v = (v / m) if m != 0 else v
            v = _round_vector(v)
            # make the vector symmetric by replacing the last element with the first
            lst = list(v)
            if len(lst) >= 1:
                first_val = lst[0]
                lst.pop()          # remove last element
                lst.append(first_val)
            v = _round_vector(np.array(lst, dtype=float))

    return json.dumps([float(x) for x in v])

# ---------------------------
# Helpers
# ---------------------------
def clip(x, lo, hi): return float(np.clip(x, lo, hi))
def sample_scalar(lo, hi, u): return lo + u * (hi - lo)

def apply_scalar_step(prev, lo, hi, delta_max):
    return clip(prev + rng.uniform(-delta_max, delta_max), lo, hi)

def sample_vec_sum3(m, lo, hi):
    v = rng.dirichlet(np.ones(m)) * 3.0
    v = np.clip(v, lo, hi)
    v *= (3.0 / v.sum())
    return v

def sample_vec_mean1(m, lo, hi):
    v = rng.uniform(lo, hi, size=m)
    v /= np.mean(v)
    return v

def apply_vec_step(prev, lo, hi, delta_max, mode):
    v = prev + rng.uniform(-delta_max, delta_max, size=prev.shape)
    v = np.clip(v, lo, hi)
    if mode == "sum3":
        v *= (3.0 / v.sum())
    elif mode == "mean1":
        v /= np.mean(v)
    return v

def choose_angles(i):
    """Pick an angle set for prototype i according to ANGLE_SET_MODE."""
    assert len(ANGLE_SETS) >= 1, "ANGLE_SETS must contain at least one list of angles."
    if ANGLE_SET_MODE == "cycle":
        chosen = ANGLE_SETS[i % len(ANGLE_SETS)]
    else:  # "random"
        chosen = ANGLE_SETS[rng.integers(0, len(ANGLE_SETS))]
    # Clean: ensure unique, sorted, ensure 0 present
    s = sorted(set(float(a) for a in chosen))
    if 0.0 not in s:
        s = [0.0] + s
    return s

# ---------------------------
# Base design sampling (LHS for scalars; structured for vectors)
# ---------------------------
scalar_names = list(PARAM_RANGES.keys())
sampler = qmc.LatinHypercube(d=len(scalar_names), seed=SEED)
U = sampler.random(n=N_PROTOTYPES)

def sample_base_params(u_row):
    params = {}
    for j, name in enumerate(scalar_names):
        lo, hi = PARAM_RANGES[name]
        params[name] = sample_scalar(lo, hi, u_row[j])
    for k, meta in VEC_META.items():
        lo, hi, m, mode = meta["lo"], meta["hi"], meta["m"], meta["mode"]
        params[k] = sample_vec_sum3(m, lo, hi) if mode == "sum3" else sample_vec_mean1(m, lo, hi)
    return params

# ---------------------------
# Write per-prototype config CSV
# (We do NOT add 360° here; your pipeline will add it to match 0°.)
# ---------------------------
def write_config_csv(prototype_id, angles, base_params, is_linear):
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
    cur = dict(base_params)
    cur["period_factors"]   = np.array(cur["period_factors"], dtype=float)
    cur["thickness_factor"] = np.array(cur["thickness_factor"], dtype=float)

    def serialize_row(angle_value: float):
        # Scalars rounded for readability
        return dict(
            angular_section=_format_angle(float(angle_value)),
            amplitude0=_round_scalar(cur["amplitude0"]),
            desired_radius=_round_scalar(cur["desired_radius"]),
            period_factors=_serialize_vector_with_constraint(
                cur["period_factors"], VEC_META["period_factors"]["mode"]
            ),
            offset_factor_x=_round_scalar(cur["offset_factor_x"]),
            mid_factor_x=_round_scalar(cur["mid_factor_x"]),
            curve_weight=_round_scalar(cur["curve_weight"]),
            thickness_factor=_serialize_vector_with_constraint(
                cur["thickness_factor"], VEC_META["thickness_factor"]["mode"]
            ),
            thickness=_round_scalar(cur["thickness"]),
        )

    # Always include the first angle (should be 0)
    rows.append(serialize_row(angles[0]))

    if not is_linear:
        for a in angles[1:]:
            # scalars
            for name, (lo, hi) in PARAM_RANGES.items():
                if name in STEP_DELTA:
                    cur[name] = apply_scalar_step(cur[name], lo, hi, STEP_DELTA[name])
            # vectors
            for k, meta in VEC_META.items():
                lo, hi, m, mode = meta["lo"], meta["hi"], meta["m"], meta["mode"]
                cur[k] = apply_vec_step(cur[k], lo, hi, VEC_STEP_DELTA[k], mode)

            rows.append(serialize_row(a))

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
        base = sample_base_params(U[i])

        if is_bending:
            angles = choose_angles(i)
            config_path = write_config_csv(proto_id, angles, base, is_linear=False)
            use_linear_fast = False
        else:
            config_path = write_config_csv(proto_id, [0], base, is_linear=True)
            use_linear_fast = True

        master_rows.append({
            "Prototype ID": proto_id,
            "export_folder": EXPORT_FOLDER,
            "config_csv": config_path.replace("\\", "/"),
            "use_linear_fast": use_linear_fast
        })

    pd.DataFrame(master_rows, columns=["Prototype ID", "export_folder", "config_csv", "use_linear_fast"])\
      .to_csv(MASTER_CSV_PATH, index=False)

    print(f"[OK] Wrote master → {MASTER_CSV_PATH}")
    print(f"[OK] Wrote {len(master_rows)} per-prototype configs to → {CONFIG_DIR}")

if __name__ == "__main__":
    main()
