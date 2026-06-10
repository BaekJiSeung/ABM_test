# %%
import numpy as np
import matplotlib.pyplot as plt

# β_ABM
beta_abm = np.array([
    0.010,0.015,0.020,0.025,0.030,0.035,
    0.040,0.045,0.050,0.055,0.060,0.065,0.070
])

# 각 initial condition별 θ_hat
data = {
    "init=40": [1.070234,1.640468,2.155518,2.523411,2.854515,3.038462,
                3.240803,3.498328,3.700669,4.068562,4.270903,4.510033,5.172241],

    "init=70": [1.167224,1.819398,2.220736,2.538462,2.73913,3.056856,
                3.29097,3.441472,3.658863,3.826087,4.394649,4.712375,5.063545],

    "init=100": [1.419732,1.934783,2.32107,2.707358,3.001672,3.204013,
                 3.351171,3.62709,3.792642,4.031773,4.436455,4.951505,5.337793],

    "init=140": [1.316583,1.984925,2.424623,2.723618,3.005025,3.21608,
                 3.479899,3.726131,3.884422,4.165829,4.467337,4.768844,5.452261]
}

plt.figure(figsize=(10,6))

for key, val in data.items():
    plt.plot(beta_abm, val, marker='o', label=key)

plt.xlabel("β_ABM", fontsize=14)
plt.ylabel("θ_hat (SM parameter)", fontsize=14)
plt.title("Mapping: β_ABM → θ_hat under different initial conditions", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm

rows = []

beta_abm = np.array([
    0.010,0.015,0.020,0.025,0.030,0.035,
    0.040,0.045,0.050,0.055,0.060,0.065,0.070
])

init_vals = [0.40,0.70,1,1.40]
data_vals = list(data.values())

rows = []

for i, init in enumerate(init_vals):
    for j, beta in enumerate(beta_abm):
        rows.append({
            "beta": beta,
            "beta2": beta**2,
            "beta3": beta**3,
            "beta_init": beta * init,
            "beta2_init": (beta**2) * init,
            "beta3_init": (beta**3) * init,
            "theta": data_vals[i][j]
        })

df = pd.DataFrame(rows)

X = df[["beta","beta2","beta3","beta_init","beta2_init","beta3_init"]]
y = df["theta"]

model = sm.OLS(y, X).fit()
print(model.summary())
# %%
