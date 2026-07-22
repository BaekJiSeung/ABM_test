# %% ================== Plot handwash-specific beta_ABM -> beta_SM mappings ==================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# settings
# -----------------------------
data_type = "A"
init_env = 9
tau0 = 140

handwash_values = [0.8, 0.9, 0.95, 0.99]

smfit_dir = "sm_fit"
fig_dir = os.path.join(smfit_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

mapping_csv = os.path.join(
    smfit_dir,
    f"theta_pairs_subset_cumGaussian_{data_type}{init_env}{tau0}_handwashALL.csv"
)

print("reading:", mapping_csv)

df = pd.read_csv(mapping_csv)

print(df.head())
print(df.columns)
print(df.shape)
# %%
