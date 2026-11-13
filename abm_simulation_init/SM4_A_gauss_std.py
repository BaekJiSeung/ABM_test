
# %% 1. Imports and data

import os
import ast
import numpy as np
import pandas as pd
from scipy.optimize import brentq

monthly_PI = pd.Series({
    "2017-01": 0, "2017-02": 0, "2017-03": 1,
    "2017-04": 0, "2017-05": 1, "2017-06": 0,
    "2017-07": 0, "2017-08": 0, "2017-09": 2,
    "2017-10": 0, "2017-11": 2, "2017-12": 0,
    "2018-01": 0, "2018-02": 0, "2018-03": 0,
    "2018-04": 0, "2018-05": 1, "2018-06": 0,
    "2018-07": 0
})

PI_dates = [
    "2017-03-01", "2017-05-30",
    "2017-09-17", "2017-09-29",
    "2017-11-14", "2017-11-20",
    "2018-05-25"
]


def _make_AI_from_dates(pi_dates, days):
    A = np.zeros(len(days))
    idx_map = {d: i for i, d in enumerate(days)}

    for d in pi_dates:
        idx = idx_map.get(pd.to_datetime(d))
        if idx is not None:
            A[idx] += 1.0

    return A


# %% 2. Surrogate Model

def simulate_theta(
    beta, init_env, tau0,
    monthly_PI=monthly_PI,
    pi_dates=PI_dates,
    return_diagnostics=False
):
    C_total = 30
    N_H, N_E = 19, 30

    mu_S = 1 / 7
    mu_HAI = 1 / 14
    mu_I = 1 / 7

    p_wash = 0.90
    cleaning_eff = 0.90  # Independent of handwashing probability
    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    iso_factor = 0.75
    sigma = 1 / 7

    start = pd.Period(
        monthly_PI.index.min(), freq="M"
    ).to_timestamp(how="start")

    end = pd.Period(
        monthly_PI.index.max(), freq="M"
    ).to_timestamp(how="end")

    days = pd.date_range(start, end, freq="D")
    T = len(days)

    A_I_day = _make_AI_from_dates(pi_dates, days)

    P_S_sh = np.zeros(T)
    P_HAI_sh = np.zeros(T)
    P_HAI_iso = np.zeros(T)
    P_I = np.zeros(T)
    H_C = np.zeros(T)
    Env_C = np.zeros(T)
    NewHAI_day = np.zeros(T)

    P_S_sh[0] = C_total - 1
    P_I[0] = 1
    Env_C[0] = init_env

    admission_records = []

    for t in range(T):

        # Environmental deep cleaning (기존 로직 유지)
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            Env_C[t] = (1 - cleaning_eff) * Env_C[t]

        PS_sh = P_S_sh[t]
        PH_sh = P_HAI_sh[t]
        PH_iso = P_HAI_iso[t]
        PI = P_I[t]
        HC = H_C[t]
        EC = Env_C[t]

        # Imported admissions: 수정
        inc = A_I_day[t]

        if inc > 0:
            total_P = PS_sh + PH_sh + PH_iso + PI
            stay_free = max(0.0, C_total - total_P)

            # 일반 환자를 Imported 환자로 대체
            taken = min(PS_sh, inc)

            PS_sh -= taken
            PI += taken

            admission_records.append({
                "date": days[t],
                "requested": float(inc),
                "total_patients": float(total_P),
                "free_beds": float(stay_free),
                "admitted": float(taken),
                "not_admitted": float(inc - taken)
            })

        for _ in range(contacts_per_day):

            B_tot = max(
                PS_sh + PH_sh + PH_iso + PI, 1e-9
            )

            lam_HP_sh = beta * (HC / N_H)

            lam_PH = beta * (
                (PH_sh + iso_factor * PH_iso + PI)
                / B_tot
            )

            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            # New HAI
            hai_sh = lam_HP_sh * PS_sh * dt

            # Isolation: Imported 환자 격리 추가
            move_HA = sigma * PH_sh * dt
            move_I = sigma * PI * dt

            # Discharges
            outS_sh = mu_S * PS_sh * dt
            outH_sh = mu_HAI * PH_sh * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI = mu_I * PI * dt

            leaving = (
                outS_sh + outH_sh + outH_iso + outI
            )

            total_P = PS_sh + PH_sh + PH_iso + PI

            AS_tot = max(
                0.0, C_total - (total_P - leaving)
            )

            # Patient compartments
            PS_sh += AS_tot - outS_sh - hai_sh
            PH_sh += hai_sh - outH_sh - move_HA

            # HAI + Imported -> Isolation
            PH_iso += move_HA + move_I - outH_iso
            PI += -outI - move_I

            PS_sh = np.clip(PS_sh, 0, C_total)
            PH_sh = np.clip(PH_sh, 0, C_total)
            PH_iso = np.clip(PH_iso, 0, C_total)
            PI = np.clip(PI, 0, C_total)

            # HCW: 기존 로직 유지
            new_H = (
                (lam_PH + lam_EH)
                * (N_H - HC) * dt
            )

            HC = (HC + new_H) * (1 - p_wash)
            HC = np.clip(HC, 0, N_H)

            # Environment
            EC += lam_HE * (N_E - EC) * dt
            EC = np.clip(EC, 0, N_E)

            NewHAI_day[t] += hai_sh

        if t < T - 1:
            P_S_sh[t + 1] = PS_sh
            P_HAI_sh[t + 1] = PH_sh
            P_HAI_iso[t + 1] = PH_iso
            P_I[t + 1] = PI
            H_C[t + 1] = HC
            Env_C[t + 1] = EC

    df = pd.DataFrame({
        "date": days,
        "NewHAI": NewHAI_day
    })

    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["NewHAI"]
        .sum()
        .reset_index()
        .rename(columns={
            "date": "month",
            "NewHAI": "NewHAI_month"
        })
    )

    monthly["cum_NewHAI"] = (
        monthly["NewHAI_month"].cumsum()
    )

    comp_df = pd.DataFrame({
        "date": days,
        "P_S_sh": P_S_sh,
        "P_HAI_sh": P_HAI_sh,
        "P_HAI_iso": P_HAI_iso,
        "P_I": P_I,
        "H_S": N_H - H_C,
        "H_C": H_C,
        "Env_S": N_E - Env_C,
        "Env_C": Env_C
    }).set_index("date")

    if return_diagnostics:
        return (
            days, NewHAI_day, monthly,
            comp_df, pd.DataFrame(admission_records)
        )

    return days, NewHAI_day, monthly, comp_df


# %% 3. Imported admission check

_, _, _, _, diag_df = simulate_theta(
    beta=3.387,
    init_env=9,
    tau0=140,
    return_diagnostics=True
)

print("\n=== Imported Admission Check ===")
print(diag_df.to_string(index=False))

requested = diag_df["requested"].sum()
admitted = diag_df["admitted"].sum()
missing = diag_df["not_admitted"].sum()

print("\n=== Summary ===")
print("Requested:", requested)
print("Admitted:", admitted)
print("Not admitted:", missing)


# %% 4. Step 4 settings

abm_csv = (
    "../result/"
    "interv_prob_transmission_summary_A9140_0.01-0.07.csv"
)

init_env = 9
tau0 = 140

theta_min, theta_max = 1.0, 6.0

beta_list = [
    0.01, 0.015, 0.02, 0.025,
    0.03, 0.035, 0.04, 0.045,
    0.05, 0.055, 0.06, 0.065, 0.07
]

df_abm = pd.read_csv(abm_csv)

df_abm["mean_vec"] = df_abm["mean"].apply(
    lambda s: np.array(ast.literal_eval(s), dtype=float)
)

df_abm["std_vec"] = df_abm["std"].apply(
    lambda s: np.array(ast.literal_eval(s), dtype=float)
)

n_months = len(df_abm["mean_vec"].iloc[0])

months = pd.period_range(
    "2017-01", periods=n_months, freq="M"
).to_timestamp()


# %% 5. Gaussian MLE functions

def model_monthly_and_cum(
    theta, init_env=init_env, tau0=tau0
):
    _, _, monthly_df, _ = simulate_theta(
        theta, init_env, tau0
    )

    mdf = monthly_df.copy()
    mdf["month"] = pd.to_datetime(
        mdf["month"].astype(str)
    )
    mdf = mdf.set_index("month")

    monthly = np.array([
        mdf["NewHAI_month"].get(m, 0.0)
        for m in months
    ])

    return monthly, np.cumsum(monthly)


def make_cum_std_from_monthly_std(y_std):
    y_std = np.asarray(y_std, dtype=float)

    cum_std = np.sqrt(
        np.cumsum(y_std ** 2)
    )

    return np.maximum(cum_std, 1e-6)


def negloglik_theta_cum_gaussian(
    theta, cum_obs, cum_std, init_env, tau0
):
    _, cum_model = model_monthly_and_cum(
        theta, init_env, tau0
    )

    y = np.asarray(cum_obs, dtype=float)
    mu = np.asarray(cum_model, dtype=float)
    sd = np.asarray(cum_std, dtype=float)

    m = min(len(y), len(mu), len(sd))

    y = y[:m]
    mu = mu[:m]
    sd = np.maximum(sd[:m], 1e-6)

    resid = y - mu

    nll = 0.5 * np.sum(
        np.log(2 * np.pi * sd ** 2)
        + (resid ** 2) / (sd ** 2)
    )

    return float(nll)


def ci95_profile_theta_gaussian(
    cum_obs, cum_std, init_env, tau0,
    theta_hat, nll_hat,
    bounds=(0.5, 6.0),
    grid_n=200
):
    thr = nll_hat + 1.92

    a, b = bounds
    grid = np.linspace(a, b, grid_n)

    vals = np.array([
        negloglik_theta_cum_gaussian(
            th, cum_obs, cum_std, init_env, tau0
        )
        for th in grid
    ])

    g = vals - thr
    i_hat = np.searchsorted(grid, theta_hat)

    left = a
    for i in range(i_hat, 0, -1):
        if g[i - 1] > 0 and g[i] <= 0:
            left = brentq(
                lambda x: negloglik_theta_cum_gaussian(
                    x, cum_obs, cum_std, init_env, tau0
                ) - thr,
                grid[i - 1], grid[i]
            )
            break

    right = b
    for i in range(i_hat, len(grid) - 1):
        if g[i] <= 0 and g[i + 1] > 0:
            right = brentq(
                lambda x: negloglik_theta_cum_gaussian(
                    x, cum_obs, cum_std, init_env, tau0
                ) - thr,
                grid[i], grid[i + 1]
            )
            break

    return float(left), float(right)


def fit_theta_cum_gaussian_for_one(
    beta_abm, y_mean, y_std
):
    cum_obs = np.cumsum(y_mean)
    cum_std = make_cum_std_from_monthly_std(y_std)

    theta_grid = np.linspace(
        theta_min, theta_max, 300
    )

    vals = np.array([
        negloglik_theta_cum_gaussian(
            th, cum_obs, cum_std, init_env, tau0
        )
        for th in theta_grid
    ])

    idx = vals.argmin()

    theta_hat = float(theta_grid[idx])
    nll_min = float(vals[idx])

    _, cum_model_hat = model_monthly_and_cum(
        theta_hat, init_env, tau0
    )

    m = min(len(cum_obs), len(cum_model_hat))
    resid = cum_obs[:m] - cum_model_hat[:m]

    sigma_hat = float(
        np.sqrt(np.mean(resid ** 2))
    )

    weighted_rmse = float(
        np.sqrt(
            np.mean((resid / cum_std[:m]) ** 2)
        )
    )

    theta_low, theta_high = (
        ci95_profile_theta_gaussian(
            cum_obs, cum_std,
            init_env, tau0,
            theta_hat, nll_min,
            bounds=(theta_min, theta_max),
            grid_n=200
        )
    )

    return (
        theta_hat, theta_low, theta_high,
        nll_min, sigma_hat, weighted_rmse
    )


# %% 6. Run mapping

results = []

for b in beta_list:

    sub = df_abm.loc[
        np.isclose(df_abm["beta"], b)
    ]

    if sub.empty:
        print(f"[WARNING] beta={b} not found")
        continue

    row = sub.iloc[0]

    (
        theta_hat, theta_low, theta_high,
        nll_min, sigma_hat, weighted_rmse
    ) = fit_theta_cum_gaussian_for_one(
        float(row["beta"]),
        row["mean_vec"],
        row["std_vec"]
    )

    print(
        f"beta_ABM={b:.3f}, "
        f"beta_SM={theta_hat:.4f}, "
        f"CI=[{theta_low:.4f}, {theta_high:.4f}], "
        f"NLL={nll_min:.2f}"
    )

    results.append({
        "beta_abm": b,
        "theta_hat": theta_hat,
        "theta_low": theta_low,
        "theta_high": theta_high,
        "sigma_hat": sigma_hat,
        "weighted_rmse": weighted_rmse,
        "neg_loglik_cum_min": nll_min,
        "init_env": init_env,
        "tau0": tau0
    })


# %% 7. Save temporary mapping

df_res = pd.DataFrame(results)

out_dir = os.path.join("sm_fit", "sm_fit_final")
os.makedirs(out_dir, exist_ok=True)

out_csv = os.path.join(
    out_dir,
    "theta_pairs_subset_cumGaussian_A9140_imported_TEST.csv"
)

df_res.to_csv(
    out_csv,
    index=False,
    encoding="utf-8"
)

print("\n저장 완료:", out_csv)
print(df_res.head())
