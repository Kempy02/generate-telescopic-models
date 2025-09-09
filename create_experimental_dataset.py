import numpy as np
import pandas as pd
from scipy.stats import qmc

# -----------------
# Define Bounds
# -----------------
param_ranges = {
    "amplitude0":      (20, 20),
    "desired_radius":  (22, 28),
    "offset_factor_x": (-0.7, 0.7),
    "mid_factor_x":    (-0.5, 0.5),
    "curve_weight":    (1.0, 10.0),
    "thickness":       (0.5, 1.5),
}

vec_params = {
    "period_factors":   (0.5, 1.5, 3),   # (low, high, length)
    "thickness_factor": (0.5, 1.5, 3),
}

# -----------------
# 2. Latin Hypercube Sampling
# -----------------
N = 15  # number of prototypes
dim = len(param_ranges) + sum(v[2] for v in vec_params.values())

sampler = qmc.LatinHypercube(d=dim, seed=42)
U = sampler.random(n=N)

# -----------------
# 3. Scale each dimension into parameter ranges
# -----------------
rows = []
col_names = list(param_ranges.keys()) + [
    f"{k}_{i}" for k, (lo, hi, m) in vec_params.items() for i in range(m)
]

for i in range(N):
    row = {}
    offset = 0

    # scalars
    for j, (name, (lo, hi)) in enumerate(param_ranges.items()):
        # take the single uniform sample and scale manually to avoid passing a 1D array to qmc.scale
        u_val = U[i, j+offset]
        row[name] = lo + u_val * (hi - lo)

    offset = len(param_ranges)

    # vectors
    for k, (lo, hi, m) in vec_params.items():
        vals = lo + U[i, offset:offset+m] * (hi - lo)
        row[k] = (vals / np.mean(vals)).tolist()  # normalize mean ≈ 1
        offset += m

    # metadata columns
    row["Prototype ID"] = f"P{i+1}"
    row["export_folder"] = "prototype_models"
    row["bending_config"] = "bending5.csv" if i % 2 == 0 else None
    rows.append(row)

# -----------------
# 4. Write to CSV
# -----------------
df = pd.DataFrame(rows)
# Flatten lists into JSON strings so your reader can handle them
for k in vec_params.keys():
    df[k] = df[k].apply(lambda x: str(x))

cols_order = ["Prototype ID", "export_folder"] + list(param_ranges.keys()) + list(vec_params.keys()) + ["bending_config"]
df = df[cols_order]

df.to_csv("datasets/lhs_designs.csv", index=False)
print("Wrote LHS design → datasets/lhs_designs.csv")