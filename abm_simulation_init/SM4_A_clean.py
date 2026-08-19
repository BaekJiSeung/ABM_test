# %% ================== cleaningDay raw files -> summary files + ALL ==================

import os
import ast
import numpy as np
import pandas as pd


# -----------------------------
# settings
# -----------------------------
data_type = "A"
init_env = 9
first_clean_day = 20

scenario_tag = f"{data_type}{init_env}_firstclean{first_clean_day}"

variable_name = "prob_transmission"

beta_tag1 = 0.02
beta_tag2 = 0.06

cleaning_values = [60, 90, 180, 360]

days_per_month = 30

if data_type == "A":
    n_months = 19
elif data_type == "B":
    n_months = 36
else:
    raise ValueError("data_type must be A or B")


try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

result_dir = os.path.join(base_dir, "..", "result")


# -----------------------------
# helper functions
# -----------------------------
def parse_series(x):
    if pd.isna(x):
        return None

    if isinstance(x, list):
        return x

    if isinstance(x, np.ndarray):
        return x.tolist()

    return ast.literal_eval(x)


def daily_to_monthly(daily_series, days_per_month=30, n_months=None):
    arr = np.array(daily_series, dtype=float)

    if n_months is not None:
        needed = days_per_month * n_months
        arr = arr[:needed]

    m = len(arr) // days_per_month
    arr = arr[:m * days_per_month]

    monthly = arr.reshape(m, days_per_month).sum(axis=1)

    return monthly.tolist()


def summarize_monthly_runs(monthly_runs):
    monthly_arr = np.array(monthly_runs, dtype=float)

    mean_ = monthly_arr.mean(axis=0).tolist()
    std_ = monthly_arr.std(axis=0, ddof=0).tolist()
    max_ = monthly_arr.max(axis=0).tolist()
    median_ = np.median(monthly_arr, axis=0).tolist()

    nonzero_mean_ = []
    disease_free_ = []

    for j in range(monthly_arr.shape[1]):
        col = monthly_arr[:, j]
        nz = col[col > 0]

        if len(nz) == 0:
            nonzero_mean_.append(0.0)
        else:
            nonzero_mean_.append(float(nz.mean()))

        disease_free_.append(float((col == 0).mean() * 100.0))

    return {
        "mean": mean_,
        "std": std_,
        "n": int(monthly_arr.shape[0]),
        "max": max_,
        "median": median_,
        "nonzero_mean": nonzero_mean_,
        "disease_free(%)": disease_free_
    }


# -----------------------------
# convert each cleaningDay raw file
# -----------------------------
all_summary_rows = []

for clean_day in cleaning_values:

    raw_csv = os.path.join(
        result_dir,
        f"interv_{variable_name}_"
        f"{scenario_tag}_{beta_tag1}-{beta_tag2}_cleaning{clean_day}.csv"
    )

    print("\n" + "=" * 80)
    print("reading raw:", raw_csv)

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(raw_csv)

    raw_df = pd.read_csv(raw_csv)

    print("raw shape:", raw_df.shape)
    print("raw columns:", list(raw_df.columns))

    rows = []

    for beta in raw_df.columns:

        series_list = (
            raw_df[beta]
            .dropna()
            .apply(parse_series)
            .tolist()
        )

        monthly_runs = [
            daily_to_monthly(
                daily_series=s,
                days_per_month=days_per_month,
                n_months=n_months
            )
            for s in series_list
        ]

        summary = summarize_monthly_runs(monthly_runs)

        row = {
            "cleaningDay": clean_day,
            "first_clean_day": first_clean_day,
            "tau_offset_days": clean_day - first_clean_day,
            "beta": float(beta),
            "mean": summary["mean"],
            "std": summary["std"],
            "n": summary["n"],
            "max": summary["max"],
            "median": summary["median"],
            "nonzero_mean": summary["nonzero_mean"],
            "disease_free(%)": summary["disease_free(%)"]
        }

        rows.append(row)
        all_summary_rows.append(row)

    summary_df = (
        pd.DataFrame(rows)
        .sort_values("beta")
        .reset_index(drop=True)
    )

    out_csv = os.path.join(
        result_dir,
        f"interv_{variable_name}_summary_"
        f"{scenario_tag}_{beta_tag1}-{beta_tag2}_cleaning{clean_day}.csv"
    )

    summary_df.to_csv(out_csv, index=False, encoding="utf-8")

    print("summary shape:", summary_df.shape)
    print(summary_df[["cleaningDay", "beta", "first_clean_day", "tau_offset_days", "n"]])
    print("saved summary:", out_csv)


# -----------------------------
# save ALL summary
# -----------------------------
summary_all_df = (
    pd.DataFrame(all_summary_rows)
    .sort_values(["cleaningDay", "beta"])
    .reset_index(drop=True)
)

out_all_csv = os.path.join(
    result_dir,
    f"interv_{variable_name}_summary_"
    f"{scenario_tag}_{beta_tag1}-{beta_tag2}_cleaningALL.csv"
)

summary_all_df.to_csv(out_all_csv, index=False, encoding="utf-8")

print("\n" + "=" * 80)
print("saved ALL summary:", out_all_csv)
print("ALL summary shape:", summary_all_df.shape)
print(summary_all_df[["cleaningDay", "beta", "first_clean_day", "tau_offset_days", "n"]])
print("=" * 80)

print("\nDONE cleaningDay raw -> summary + ALL")

# %%

















# %% ================== SM4_A_cleaning_mapping.py ==================
# cleaningDay-specific beta_ABM -> beta_SM mapping reconstruction

import os
import ast
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from joblib import Parallel, delayed


# ============================================================
# Period A imported colonized patient input
# ============================================================

monthly_PI = pd.Series({
    "2017-01": 0,
    "2017-02": 0,
    "2017-03": 1,
    "2017-04": 0,
    "2017-05": 1,
    "2017-06": 0,
    "2017-07": 0,
    "2017-08": 0,
    "2017-09": 2,
    "2017-10": 0,
    "2017-11": 2,
    "2017-12": 0,
    "2018-01": 0,
    "2018-02": 0,
    "2018-03": 0,
    "2018-04": 0,
    "2018-05": 1,
    "2018-06": 0,
    "2018-07": 0
})

PI_dates = [
    "2017-03-01",
    "2017-05-30",
    "2017-09-17",
    "2017-09-29",
    "2017-11-14",
    "2017-11-20",
    "2018-05-25",
]


def _make_AI_from_dates(pi_dates, days):
    T = len(days)
    A = np.zeros(T)

    idx_map = {d: i for i, d in enumerate(days)}

    for d in pi_dates:
        ts = pd.to_datetime(d)
        idx = idx_map.get(ts, None)

        if idx is not None:
            A[idx] += 1.0

    return A


# ============================================================
# Surrogate model
# ============================================================

def simulate_theta(
    beta,
    init_env,
    clean_day,
    tau0,
    p_wash=0.9,
    monthly_PI=monthly_PI,
    pi_dates=PI_dates
):
    """
    Difference-equation surrogate model for cleaningDay-specific mapping.

    beta      : surrogate model transmission parameter, beta_SM
    init_env  : initial environmental contamination level
    clean_day : deep cleaning period
    tau0      : cleaning phase offset
    p_wash    : HCW handwashing rate, fixed in cleaning intervention
    """

    C_total = 30
    C_iso = 30
    C_sh = C_total

    N_H = 19
    N_E = 30

    mu_S = 1 / 7
    mu_HAI = 1 / 14
    mu_I = 1 / 7

    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = clean_day
    cleaning_eff = 0.90

    iso_factor = 0.75

    # 현재 ABM cleaning intervention baseline에 맞춰 14로 둠.
    # 만약 기존 handwash surrogate와 완전히 맞추려면 7.0으로 바꾸면 됨.
    isol_time = 7.0
    sigma = 1.0 / isol_time

    start = pd.Period(monthly_PI.index.min(), freq="M").to_timestamp(how="start")
    end = pd.Period(monthly_PI.index.max(), freq="M").to_timestamp(how="end")

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

    for t in range(T):

        # deep cleaning
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            Env_C[t] = (1 - cleaning_eff) * Env_C[t]

        PS_sh = P_S_sh[t]
        PH_sh = P_HAI_sh[t]
        PH_iso = P_HAI_iso[t]
        PI = P_I[t]
        HC = H_C[t]
        EC = Env_C[t]

        inc = A_I_day[t]

        if inc > 0:
            total_P = PS_sh + PH_sh + PH_iso + PI
            stay_free = max(0.0, C_total - total_P)
            inc_eff = min(inc, stay_free)

            taken = min(PS_sh, inc_eff)

            PS_sh -= taken
            PI += taken

        for _ in range(contacts_per_day):

            B_tot = max(PS_sh + PH_sh + PH_iso + PI, 1e-9)

            lam_HP_sh = beta * (HC / N_H)
            lam_PH = beta * ((PH_sh + iso_factor * PH_iso + PI) / B_tot)
            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            hai_sh = lam_HP_sh * PS_sh * dt
            move_HA = sigma * PH_sh * dt

            outS_sh = mu_S * PS_sh * dt
            outH_sh = mu_HAI * PH_sh * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI = mu_I * PI * dt

            leaving = outS_sh + outH_sh + outH_iso + outI
            total_P = PS_sh + PH_sh + PH_iso + PI

            AS_tot = max(0.0, C_total - (total_P - leaving))
            AS_sh = AS_tot

            PS_sh += AS_sh - outS_sh - hai_sh
            PH_sh += hai_sh - outH_sh - move_HA
            PH_iso += move_HA - outH_iso
            PI += -outI

            PS_sh = np.clip(PS_sh, 0, C_sh)
            PH_sh = np.clip(PH_sh, 0, C_sh)
            PH_iso = np.clip(PH_iso, 0, C_iso)
            PI = np.clip(PI, 0, C_total)

            # handwashing fixed
            new_H = (lam_PH + lam_EH) * (N_H - HC) * dt
            HC = (HC + new_H) * (1 - p_wash)
            HC = np.clip(HC, 0, N_H)

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
        df
        .groupby(df["date"].dt.to_period("M"))["NewHAI"]
        .sum()
        .reset_index()
        .rename(columns={
            "date": "month",
            "NewHAI": "NewHAI_month"
        })
    )

    monthly["cum_NewHAI"] = monthly["NewHAI_month"].cumsum()

    H_S = N_H - H_C
    Env_S = N_E - Env_C

    comp_df = pd.DataFrame({
        "date": days,
        "P_S_sh": P_S_sh,
        "P_HAI_sh": P_HAI_sh,
        "P_HAI_iso": P_HAI_iso,
        "P_I": P_I,
        "H_S": H_S,
        "H_C": H_C,
        "Env_S": Env_S,
        "Env_C": Env_C,
    }).set_index("date")

    return days, NewHAI_day, monthly, comp_df


print("Cell 1 done")


# ============================================================
# Step 4 settings
# ============================================================

data_type = "A"

init_env = 9
first_clean_day = 20

p_wash_fixed = 0.9

cleaning_values = [60, 90, 180, 360]

variable_name = "prob_transmission"

beta_tag1 = 0.02
beta_tag2 = 0.06

# beta_SM search range
# cleaningDay에서는 handwash0.99보다 낮은 beta_SM도 나올 수 있어서 넓게 잡음
theta_min = 0.1
theta_max = 12.0

theta_grid_n = 400
ci_grid_n = 400

n_jobs = 8

if data_type == "A":
    start_month = "2017-01"
    n_months_expected = 19
elif data_type == "B":
    start_month = "2021-01"
    n_months_expected = 36
else:
    raise ValueError("data_type must be 'A' or 'B'")

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

result_dir = os.path.join(base_dir, "..", "result")
smfit_dir = os.path.join(base_dir, "sm_fit")
os.makedirs(smfit_dir, exist_ok=True)


def parse_vec(s):
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=float)

    return np.array(ast.literal_eval(s), dtype=float)


def model_monthly_and_cum(
    theta,
    clean_day,
    tau0,
    months,
    init_env=init_env,
    p_wash=p_wash_fixed
):
    days, daily_inc, monthly_df, comp_df = simulate_theta(
        beta=theta,
        init_env=init_env,
        clean_day=clean_day,
        tau0=tau0,
        p_wash=p_wash
    )

    mdf = monthly_df.copy()
    mdf["month"] = pd.to_datetime(mdf["month"].astype(str))
    mdf = mdf.set_index("month")

    monthly = np.array([
        mdf["NewHAI_month"].get(m, 0.0)
        for m in months
    ])

    cum = np.cumsum(monthly)

    return monthly, cum


def make_cum_std_from_monthly_std(y_std):
    y_std = np.asarray(y_std, dtype=float)
    cum_std = np.sqrt(np.cumsum(y_std ** 2))
    cum_std = np.maximum(cum_std, 1e-6)

    return cum_std


def negloglik_theta_cum_gaussian(
    theta,
    cum_obs,
    cum_std,
    months,
    init_env,
    clean_day,
    tau0,
    p_wash
):
    _, cum_model = model_monthly_and_cum(
        theta=theta,
        clean_day=clean_day,
        tau0=tau0,
        months=months,
        init_env=init_env,
        p_wash=p_wash
    )

    y = np.asarray(cum_obs, dtype=float)
    mu = np.asarray(cum_model, dtype=float)
    sd = np.asarray(cum_std, dtype=float)

    m = min(len(y), len(mu), len(sd))

    y = y[:m]
    mu = mu[:m]
    sd = sd[:m]

    sd = np.maximum(sd, 1e-6)

    resid = y - mu

    nll = 0.5 * np.sum(
        np.log(2 * np.pi * sd ** 2) + (resid ** 2) / (sd ** 2)
    )

    return float(nll)


def ci95_profile_theta_gaussian(
    cum_obs,
    cum_std,
    months,
    init_env,
    clean_day,
    tau0,
    p_wash,
    theta_hat,
    nll_hat,
    bounds=(0.1, 12.0),
    grid_n=400
):
    thr = nll_hat + 1.92

    a, b = bounds
    grid = np.linspace(a, b, grid_n)

    vals = np.array([
        negloglik_theta_cum_gaussian(
            theta=th,
            cum_obs=cum_obs,
            cum_std=cum_std,
            months=months,
            init_env=init_env,
            clean_day=clean_day,
            tau0=tau0,
            p_wash=p_wash
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
                    theta=x,
                    cum_obs=cum_obs,
                    cum_std=cum_std,
                    months=months,
                    init_env=init_env,
                    clean_day=clean_day,
                    tau0=tau0,
                    p_wash=p_wash
                ) - thr,
                grid[i - 1],
                grid[i]
            )
            break

    right = b

    for i in range(i_hat, len(grid) - 1):
        if g[i] <= 0 and g[i + 1] > 0:
            right = brentq(
                lambda x: negloglik_theta_cum_gaussian(
                    theta=x,
                    cum_obs=cum_obs,
                    cum_std=cum_std,
                    months=months,
                    init_env=init_env,
                    clean_day=clean_day,
                    tau0=tau0,
                    p_wash=p_wash
                ) - thr,
                grid[i],
                grid[i + 1]
            )
            break

    return float(left), float(right)


def fit_theta_cum_gaussian_for_one(
    beta_abm,
    y_mean,
    y_std,
    clean_day,
    tau0,
    months,
    p_wash=p_wash_fixed
):
    cum_obs = np.cumsum(y_mean)
    cum_std = make_cum_std_from_monthly_std(y_std)

    theta_grid = np.linspace(theta_min, theta_max, theta_grid_n)

    vals = np.array([
        negloglik_theta_cum_gaussian(
            theta=th,
            cum_obs=cum_obs,
            cum_std=cum_std,
            months=months,
            init_env=init_env,
            clean_day=clean_day,
            tau0=tau0,
            p_wash=p_wash
        )
        for th in theta_grid
    ])

    idx = vals.argmin()

    theta_hat = float(theta_grid[idx])
    nll_min = float(vals[idx])

    _, cum_model_hat = model_monthly_and_cum(
        theta=theta_hat,
        clean_day=clean_day,
        tau0=tau0,
        months=months,
        init_env=init_env,
        p_wash=p_wash
    )

    m = min(len(cum_obs), len(cum_model_hat))

    resid = cum_obs[:m] - cum_model_hat[:m]

    sigma_hat = float(np.sqrt(np.mean(resid ** 2)))
    weighted_rmse = float(np.sqrt(np.mean((resid / cum_std[:m]) ** 2)))

    theta_low, theta_high = ci95_profile_theta_gaussian(
        cum_obs=cum_obs,
        cum_std=cum_std,
        months=months,
        init_env=init_env,
        clean_day=clean_day,
        tau0=tau0,
        p_wash=p_wash,
        theta_hat=theta_hat,
        nll_hat=nll_min,
        bounds=(theta_min, theta_max),
        grid_n=ci_grid_n
    )

    return theta_hat, theta_low, theta_high, nll_min, sigma_hat, weighted_rmse


def fit_one_row(row_dict, clean_day, tau0, months):
    beta_abm = float(row_dict["beta"])
    y_mean = row_dict["mean_vec"]
    y_std = row_dict["std_vec"]

    theta_hat, theta_low, theta_high, nll_min, sigma_hat, weighted_rmse = (
        fit_theta_cum_gaussian_for_one(
            beta_abm=beta_abm,
            y_mean=y_mean,
            y_std=y_std,
            clean_day=clean_day,
            tau0=tau0,
            months=months,
            p_wash=p_wash_fixed
        )
    )

    print(
        f"done: cleaningDay={clean_day}, beta_ABM={beta_abm:0.3f}, "
        f"theta_hat={theta_hat:.4f}, NLL={nll_min:.2f}"
    )

    return {
        "cleaningDay": clean_day,
        "first_clean_day": first_clean_day,
        "tau_offset_days": tau0,
        "p_wash": p_wash_fixed,
        "beta_abm": beta_abm,
        "theta_hat": theta_hat,
        "theta_low": theta_low,
        "theta_high": theta_high,
        "sigma_hat": sigma_hat,
        "weighted_rmse": weighted_rmse,
        "neg_loglik_cum_min": nll_min,
        "init_env": init_env,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "theta_grid_n": theta_grid_n,
        "ci_grid_n": ci_grid_n,
        "cleaning_eff": 0.90,
    }


print("Cell 2 done")


# ============================================================
# Run Step 4 mapping reconstruction for each cleaningDay
# ============================================================

all_results = []

scenario_tag = f"{data_type}{init_env}_firstclean{first_clean_day}"

for clean_day in cleaning_values:

    tau0 = clean_day - first_clean_day

    abm_csv = os.path.join(
        result_dir,
        f"interv_{variable_name}_summary_"
        f"{scenario_tag}_{beta_tag1}-{beta_tag2}_cleaning{clean_day}.csv"
    )

    print("\n" + "=" * 80)
    print(f"START Step4 for cleaningDay = {clean_day}")
    print(f"tau_offset_days = {tau0}")
    print("ABM summary csv:", abm_csv)
    print("=" * 80)

    if not os.path.exists(abm_csv):
        print("[ERROR] file not found:", abm_csv)
        continue

    df_abm = pd.read_csv(abm_csv)

    print("df_abm shape:", df_abm.shape)
    print("df_abm columns:", list(df_abm.columns))

    df_abm["mean_vec"] = df_abm["mean"].apply(parse_vec)
    df_abm["std_vec"] = df_abm["std"].apply(parse_vec)

    n_months = len(df_abm["mean_vec"].iloc[0])

    if n_months != n_months_expected:
        print(
            f"[WARNING] expected n_months={n_months_expected}, "
            f"but CSV has n_months={n_months}"
        )

    months = pd.period_range(
        start_month,
        periods=n_months,
        freq="M"
    ).to_timestamp()

    row_dicts = (
        df_abm
        .sort_values("beta")
        .to_dict(orient="records")
    )

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_one_row)(row_dict, clean_day, tau0, months)
        for row_dict in row_dicts
    )

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("beta_abm").reset_index(drop=True)

    out_csv = os.path.join(
        smfit_dir,
        f"theta_pairs_subset_cumGaussian_"
        f"{scenario_tag}_cleaning{clean_day}.csv"
    )

    df_res.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n저장 완료 →", out_csv)
    print(df_res.head())

    all_results.extend(results)


# ============================================================
# Save ALL mapping
# ============================================================

df_all = pd.DataFrame(all_results)

if not df_all.empty:
    df_all = df_all.sort_values(["cleaningDay", "beta_abm"]).reset_index(drop=True)

    out_all_csv = os.path.join(
        smfit_dir,
        f"theta_pairs_subset_cumGaussian_"
        f"{scenario_tag}_cleaningALL.csv"
    )

    df_all.to_csv(out_all_csv, index=False, encoding="utf-8")

    print("\n전체 저장 완료 →", out_all_csv)
    print(df_all.head())
    print("done.")
else:
    print("[ERROR] no results generated.")
# %% 이제 플롯 abm-sm 비교(0.02-0.06)











# %% ==============================
# Step 4 validation: ABM vs SM cumulative curves
# for each cleaningDay and each beta_ABM
# ==============================

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# settings
# -----------------------------
data_type = "A"
init_env = 9
first_clean_day = 20

p_wash_fixed = 0.9

cleaning_values = [60, 90, 180, 360]

beta_values = np.round(np.arange(0.02, 0.0601, 0.005), 5)

beta_tag1 = 0.02
beta_tag2 = 0.06

start_month = "2017-01"

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

result_dir = os.path.join(base_dir, "..", "result")
smfit_dir = os.path.join(base_dir, "sm_fit")
fig_dir = os.path.join(smfit_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

scenario_tag = f"{data_type}{init_env}_firstclean{first_clean_day}"


# -----------------------------
# helper functions
# -----------------------------
def parse_vec(x):
    if isinstance(x, list):
        return np.array(x, dtype=float)

    if isinstance(x, np.ndarray):
        return x.astype(float)

    return np.array(ast.literal_eval(x), dtype=float)


def get_month_axis(n_months, start_month="2017-01"):
    return pd.period_range(
        start_month,
        periods=n_months,
        freq="M"
    ).to_timestamp()


def load_abm_summary_cleaning(clean_day):
    csv_path = os.path.join(
        result_dir,
        f"interv_prob_transmission_summary_"
        f"{scenario_tag}_{beta_tag1}-{beta_tag2}_cleaning{clean_day}.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    df["mean_vec"] = df["mean"].apply(parse_vec)
    df["std_vec"] = df["std"].apply(parse_vec)

    df = df.sort_values("beta").reset_index(drop=True)

    return df


def load_mapping_cleaning(clean_day):
    # 개별 mapping file 우선 사용
    csv_path = os.path.join(
        smfit_dir,
        f"theta_pairs_subset_cumGaussian_"
        f"{scenario_tag}_cleaning{clean_day}.csv"
    )

    # 없으면 ALL에서 filtering
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        all_path = os.path.join(
            smfit_dir,
            f"theta_pairs_subset_cumGaussian_"
            f"{scenario_tag}_cleaningALL.csv"
        )

        if not os.path.exists(all_path):
            raise FileNotFoundError(
                f"mapping file not found:\n{csv_path}\n{all_path}"
            )

        df_all = pd.read_csv(all_path)

        df = df_all.loc[
            np.isclose(df_all["cleaningDay"].astype(float), clean_day)
        ].copy()

        if df.empty:
            raise ValueError(
                f"cleaningDay={clean_day} not found in mapping ALL file.\n"
                f"available = {sorted(df_all['cleaningDay'].unique())}"
            )

    df = df.sort_values("beta_abm").reset_index(drop=True)

    return df


def get_beta_sm_from_mapping(beta_abm, df_map):
    x = df_map["beta_abm"].to_numpy(dtype=float)
    y = df_map["theta_hat"].to_numpy(dtype=float)

    if beta_abm < x.min() or beta_abm > x.max():
        print("[WARNING] beta_abm outside mapping range")
        print("mapping range:", x.min(), "~", x.max())
        print("target:", beta_abm)

    beta_sm = float(np.interp(beta_abm, x, y))

    return beta_sm


def get_surrogate_monthly_cleaning(beta_sm, clean_day, tau0, n_months):
    """
    Requires cleaningDay-version simulate_theta() already defined.
    simulate_theta(beta, init_env, clean_day, tau0, p_wash)
    """

    days, daily_inc, monthly_df, comp_df = simulate_theta(
        beta=beta_sm,
        init_env=init_env,
        clean_day=clean_day,
        tau0=tau0,
        p_wash=p_wash_fixed
    )

    mdf = monthly_df.copy()
    mdf["month"] = pd.to_datetime(mdf["month"].astype(str))
    mdf = mdf.set_index("month")

    months = get_month_axis(n_months, start_month=start_month)

    sm_monthly = np.array([
        mdf["NewHAI_month"].get(m, 0.0)
        for m in months
    ], dtype=float)

    return sm_monthly


# -----------------------------
# plot cumulative curves
# -----------------------------
for clean_day in cleaning_values:

    tau0 = clean_day - first_clean_day

    print("\n" + "=" * 80)
    print(f"Plot cumulative curves: cleaningDay = {clean_day}")
    print(f"tau_offset_days = {tau0}")
    print("=" * 80)

    df_abm = load_abm_summary_cleaning(clean_day)
    df_map = load_mapping_cleaning(clean_day)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()

    for ax, beta_abm in zip(axes, beta_values):

        # --- ABM summary row ---
        row_abm = df_abm.loc[
            np.isclose(df_abm["beta"].astype(float), beta_abm)
        ]

        if row_abm.empty:
            ax.set_title(f"beta={beta_abm:.3f}\n(no ABM row)")
            ax.axis("off")
            continue

        row_abm = row_abm.iloc[0]

        abm_monthly = row_abm["mean_vec"]
        n_months = len(abm_monthly)

        months = get_month_axis(n_months, start_month=start_month)

        abm_cum = np.cumsum(abm_monthly)

        # --- beta_ABM -> beta_SM mapping ---
        beta_sm = get_beta_sm_from_mapping(
            beta_abm=beta_abm,
            df_map=df_map
        )

        # --- surrogate run ---
        sm_monthly = get_surrogate_monthly_cleaning(
            beta_sm=beta_sm,
            clean_day=clean_day,
            tau0=tau0,
            n_months=n_months
        )

        sm_cum = np.cumsum(sm_monthly)

        # --- RMSE ---
        cum_rmse = float(np.sqrt(np.mean((abm_cum - sm_cum) ** 2)))

        # --- plot ---
        ax.plot(
            months,
            abm_cum,
            "o-",
            linewidth=2,
            markersize=4,
            label="ABM cumulative"
        )

        ax.plot(
            months,
            sm_cum,
            "s--",
            linewidth=2,
            markersize=4,
            label="SM cumulative"
        )

        ax.set_title(
            f"$\\beta_{{ABM}}$={beta_abm:.3f}\n"
            f"$\\beta_{{SM}}$={beta_sm:.3f}, RMSE={cum_rmse:.2f}",
            fontsize=10
        )

        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    # 남는 subplot 제거
    for j in range(len(beta_values), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)

    fig.suptitle(
        f"Step 4 validation: ABM vs SM cumulative curves "
        f"(cleaningDay={clean_day}, first clean day={first_clean_day})",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = os.path.join(
        fig_dir,
        f"step4_validation_cumulative_curves_"
        f"{scenario_tag}_cleaning{clean_day}.png"
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    print("saved:", out_png)


print("\nDONE cleaningDay Step 4 cumulative validation plots")
# %% abm과비교0.03847그리고매핑




# %% ================== cleaningDay mapping + ABM vs mapped SM (beta_ABM=0.03847) ==================

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# settings
# --------------------------------------------------
data_type = "A"
init_env = 9
first_clean_day = 20
cleaning_values = [60, 90, 180, 360]

beta_abm_target = 0.03847

days_per_month = 30
if data_type == "A":
    n_months = 19
elif data_type == "B":
    n_months = 36
else:
    raise ValueError("data_type must be 'A' or 'B'")

smfit_dir = "sm_fit"
fig_dir = os.path.join(smfit_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

result_dir = "../result"

mapping_csv = os.path.join(
    smfit_dir,
    "theta_pairs_subset_cumGaussian_A9_firstclean20_cleaningALL.csv"
)

summary_all_csv = os.path.join(
    result_dir,
    "interv_prob_transmission_summary_A9_firstclean20_0.02-0.06_cleaningALL.csv"
)


# --------------------------------------------------
# helper functions
# --------------------------------------------------
def parse_vec(x):
    if isinstance(x, list):
        return np.array(x, dtype=float)
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if pd.isna(x):
        return np.array([], dtype=float)
    return np.array(ast.literal_eval(x), dtype=float)


# --------------------------------------------------
# check required function
# --------------------------------------------------
if "model_monthly_and_cum" not in globals():
    raise NameError(
        "model_monthly_and_cum is not defined.\n"
        "Run your cleaning SM4 cell first, then run this cell."
    )


# --------------------------------------------------
# read files
# --------------------------------------------------
print("reading mapping:", mapping_csv)
df_map = pd.read_csv(mapping_csv)
print(df_map.head())
print(df_map.columns)

print("\nreading summary:", summary_all_csv)
df_sum = pd.read_csv(summary_all_csv)
print(df_sum.head())
print(df_sum.columns)


# --------------------------------------------------
# Part 1. Plot cleaning-specific beta_ABM -> beta_SM mappings
#        and compute mapped beta_SM at beta_ABM = 0.03847
# --------------------------------------------------
mapped_rows = []

plt.figure(figsize=(8, 6))

for clean_day in cleaning_values:

    sub = (
        df_map[df_map["cleaningDay"].astype(float) == float(clean_day)]
        .sort_values("beta_abm")
        .reset_index(drop=True)
    )

    if sub.empty:
        print(f"[WARNING] no mapping rows for cleaningDay={clean_day}")
        continue

    x = sub["beta_abm"].to_numpy(float)
    y = sub["theta_hat"].to_numpy(float)

    # CI columns optional
    has_ci = ("theta_low" in sub.columns) and ("theta_high" in sub.columns)
    if has_ci:
        yl = sub["theta_low"].to_numpy(float)
        yh = sub["theta_high"].to_numpy(float)

    # interpolation for beta_ABM target
    beta_sm_target = float(np.interp(beta_abm_target, x, y))

    mapped_rows.append({
        "cleaningDay": clean_day,
        "first_clean_day": first_clean_day,
        "tau_offset_days": clean_day - first_clean_day,
        "beta_abm_target": beta_abm_target,
        "beta_sm_mapped": beta_sm_target
    })

    # plot mapping
    plt.plot(
        x, y,
        "o-",
        linewidth=2,
        markersize=5,
        label=f"cleaningDay={clean_day}"
    )

    if has_ci:
        plt.fill_between(x, yl, yh, alpha=0.12)

    plt.scatter(
        [beta_abm_target],
        [beta_sm_target],
        s=100,
        marker="*",
        zorder=5
    )

plt.axvline(
    beta_abm_target,
    linestyle="--",
    linewidth=1.5,
    label=r"$\beta_{\mathrm{ABM}}=0.03847$"
)

plt.xlabel(r"$\beta_{\mathrm{ABM}}$", fontsize=14)
plt.ylabel(r"$\beta_{\mathrm{SM}}$", fontsize=14)
plt.title(
    r"Cleaning-specific mapping: $\beta_{\mathrm{ABM}} \rightarrow \beta_{\mathrm{SM}}$",
    fontsize=15
)
plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()

mapping_png = os.path.join(
    fig_dir,
    "mapping_cleaning_betaABM_to_betaSM_betaABM0p03847.png"
)
plt.savefig(mapping_png, dpi=300, bbox_inches="tight")
plt.show()

print("saved:", mapping_png)

df_mapped = pd.DataFrame(mapped_rows)

mapped_csv = os.path.join(
    smfit_dir,
    "mapped_betaSM_betaABM0p03847_A9_firstclean20_cleaning.csv"
)
df_mapped.to_csv(mapped_csv, index=False, encoding="utf-8")

print("saved:", mapped_csv)
print(df_mapped.to_string(index=False))

# --------------------------------------------------
# Part 2. ABM vs mapped SM comparison
# using ACTUAL ABM beta_ABM = 0.03847 raw file
# --------------------------------------------------

actual_raw_csv = os.path.join(
    result_dir,
    "interv_prob_transmission_LONG_A9_firstclean20_0.03847-0.03847_cleaningALL.csv"
)

print("\nreading actual ABM raw:", actual_raw_csv)
df_actual_raw = pd.read_csv(actual_raw_csv)

print(df_actual_raw.head())
print(df_actual_raw.columns)


def daily_to_monthly_local(daily_series, days_per_month=30, n_months=19):
    arr = np.array(daily_series, dtype=float)

    needed = days_per_month * n_months
    arr = arr[:needed]

    m = len(arr) // days_per_month
    arr = arr[:m * days_per_month]

    return arr.reshape(m, days_per_month).sum(axis=1)


def summarize_actual_abm_cleaning(df_raw, clean_day):
    sub = df_raw[df_raw["cleaningDay"].astype(float) == float(clean_day)].copy()

    if sub.empty:
        raise ValueError(f"actual raw file에 cleaningDay={clean_day} 없음")

    monthly_runs = []

    for s in sub["HCW_related_infecs"].dropna():
        daily = parse_vec(s)
        monthly = daily_to_monthly_local(
            daily,
            days_per_month=days_per_month,
            n_months=n_months
        )
        monthly_runs.append(monthly)

    monthly_arr = np.array(monthly_runs, dtype=float)

    mean_ = monthly_arr.mean(axis=0)
    std_ = monthly_arr.std(axis=0, ddof=0)

    return mean_, std_, len(monthly_runs)


compare_rows = []

fig_m, axes_m = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes_m = axes_m.ravel()

fig_c, axes_c = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes_c = axes_c.ravel()

for i, clean_day in enumerate(cleaning_values):

    tau0 = clean_day - first_clean_day

    # actual ABM beta=0.03847
    abm_mean, abm_std, n_runs = summarize_actual_abm_cleaning(
        df_actual_raw,
        clean_day
    )

    beta_used = beta_abm_target

    # mapped beta_SM
    row_map = df_mapped[df_mapped["cleaningDay"] == clean_day]

    if row_map.empty:
        print(f"[WARNING] no mapped beta_SM for cleaningDay={clean_day}")
        continue

    beta_sm = float(row_map.iloc[0]["beta_sm_mapped"])

    months = pd.period_range(
        start="2017-01",
        periods=n_months,
        freq="M"
    ).to_timestamp()

    # run SM
    sm_monthly, sm_cum = model_monthly_and_cum(
        theta=beta_sm,
        clean_day=clean_day,
        tau0=tau0,
        months=months,
        init_env=init_env,
        p_wash=p_wash_fixed
    )

    sm_monthly = np.array(sm_monthly, dtype=float)
    sm_cum = np.array(sm_cum, dtype=float)

    abm_cum = np.cumsum(abm_mean)
    abm_cum_std = np.sqrt(np.cumsum(abm_std ** 2))

    m = min(
        len(abm_mean),
        len(sm_monthly),
        len(abm_std),
        len(abm_cum),
        len(sm_cum)
    )

    xx = np.arange(1, m + 1)

    abm_mean = abm_mean[:m]
    abm_std = abm_std[:m]
    abm_cum = abm_cum[:m]
    abm_cum_std = abm_cum_std[:m]
    sm_monthly = sm_monthly[:m]
    sm_cum = sm_cum[:m]

    rmse_monthly = float(np.sqrt(np.mean((abm_mean - sm_monthly) ** 2)))
    rmse_cum = float(np.sqrt(np.mean((abm_cum - sm_cum) ** 2)))

    compare_rows.append({
        "cleaningDay": clean_day,
        "first_clean_day": first_clean_day,
        "tau_offset_days": tau0,
        "beta_abm_actual": beta_abm_target,
        "beta_sm_mapped": beta_sm,
        "n_runs": n_runs,
        "abm_total": float(abm_cum[-1]),
        "sm_total": float(sm_cum[-1]),
        "rmse_monthly": rmse_monthly,
        "rmse_cumulative": rmse_cum
    })

    # ---------------- monthly plot ----------------
    axm = axes_m[i]

    axm.plot(
        xx,
        abm_mean,
        "o-",
        linewidth=2,
        label=rf"Actual ABM mean ($\beta_{{ABM}}$={beta_used:.5f})"
    )

    axm.fill_between(
        xx,
        abm_mean - abm_std,
        abm_mean + abm_std,
        alpha=0.2,
        label="ABM ±1 SD"
    )

    axm.plot(
        xx,
        sm_monthly,
        "s--",
        linewidth=2,
        label=rf"Mapped SM ($\beta_{{SM}}$={beta_sm:.3f})"
    )

    axm.set_title(
        f"cleaningDay={clean_day}, tau0={tau0}, n={n_runs}",
        fontsize=12
    )
    axm.set_xlabel("Month")
    axm.set_ylabel("Monthly HAI")
    axm.grid(alpha=0.3)
    axm.legend(fontsize=8)

    # ---------------- cumulative plot ----------------
    axc = axes_c[i]

    axc.plot(
        xx,
        abm_cum,
        "o-",
        linewidth=2,
        label=rf"Actual ABM cumulative ($\beta_{{ABM}}$={beta_used:.5f})"
    )

    axc.fill_between(
        xx,
        abm_cum - abm_cum_std,
        abm_cum + abm_cum_std,
        alpha=0.2,
        label="ABM cumulative ± approx SD"
    )

    axc.plot(
        xx,
        sm_cum,
        "s--",
        linewidth=2,
        label=rf"Mapped SM ($\beta_{{SM}}$={beta_sm:.3f})"
    )

    axc.set_title(
        f"cleaningDay={clean_day}, tau0={tau0}, n={n_runs}",
        fontsize=12
    )
    axc.set_xlabel("Month")
    axc.set_ylabel("Cumulative HAI")
    axc.grid(alpha=0.3)
    axc.legend(fontsize=8)


fig_m.suptitle(
    r"Actual ABM vs mapped SM monthly trajectories "
    r"at $\beta_{\mathrm{ABM}}=0.03847$",
    fontsize=14
)
fig_m.tight_layout(rect=[0, 0, 1, 0.96])

monthly_png = os.path.join(
    fig_dir,
    "actual_cleaning_ABM_vs_mappedSM_monthly_betaABM0p03847.png"
)
fig_m.savefig(monthly_png, dpi=300, bbox_inches="tight")
plt.show()

print("saved:", monthly_png)


fig_c.suptitle(
    r"Actual ABM vs mapped SM cumulative trajectories "
    r"at $\beta_{\mathrm{ABM}}=0.03847$",
    fontsize=14
)
fig_c.tight_layout(rect=[0, 0, 1, 0.96])

cum_png = os.path.join(
    fig_dir,
    "actual_cleaning_ABM_vs_mappedSM_cumulative_betaABM0p03847.png"
)
fig_c.savefig(cum_png, dpi=300, bbox_inches="tight")
plt.show()

print("saved:", cum_png)


# --------------------------------------------------
# save comparison table
# --------------------------------------------------
df_compare = pd.DataFrame(compare_rows)

compare_csv = os.path.join(
    smfit_dir,
    "actual_cleaning_ABM_vs_mappedSM_betaABM0p03847.csv"
)

df_compare.to_csv(compare_csv, index=False, encoding="utf-8")

print("saved:", compare_csv)
print(df_compare.to_string(index=False))
# %%



# %% ================== Plot cleaningDay-specific beta_ABM -> beta_SM mappings ==================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# settings
# -----------------------------
data_type = "A"
init_env = 9
first_clean_day = 20

smfit_dir = "sm_fit"
fig_dir = os.path.join(smfit_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

mapping_csv = os.path.join(
    smfit_dir,
    "theta_pairs_subset_cumGaussian_A9_firstclean20_cleaningALL.csv"
)

print("reading:", mapping_csv)

df = pd.read_csv(mapping_csv)

print(df.head())
print(df.columns.tolist())
print(df.shape)


# -----------------------------
# column names
# -----------------------------
if "cleaningDay" in df.columns:
    clean_col = "cleaningDay"
elif "cleaning_day" in df.columns:
    clean_col = "cleaning_day"
elif "clean_day" in df.columns:
    clean_col = "clean_day"
else:
    raise ValueError("cleaningDay column not found.")

if "beta_abm" not in df.columns:
    raise ValueError("beta_abm column not found.")

if "theta_hat" in df.columns:
    theta_col = "theta_hat"
elif "beta_sm_hat" in df.columns:
    theta_col = "beta_sm_hat"
else:
    raise ValueError("theta_hat or beta_sm_hat column not found.")

if "theta_low" in df.columns:
    low_col = "theta_low"
elif "beta_sm_low" in df.columns:
    low_col = "beta_sm_low"
else:
    raise ValueError("theta_low or beta_sm_low column not found.")

if "theta_high" in df.columns:
    high_col = "theta_high"
elif "beta_sm_high" in df.columns:
    high_col = "beta_sm_high"
else:
    raise ValueError("theta_high or beta_sm_high column not found.")


# -----------------------------
# cleaningDay values
# -----------------------------
clean_values = sorted(df[clean_col].dropna().unique())
print("cleaning values:", clean_values)


# -----------------------------
# plot 1: overlay mapping curves
# -----------------------------
plt.figure(figsize=(9, 6))

for clean_day in clean_values:

    sub = (
        df[df[clean_col] == clean_day]
        .sort_values("beta_abm")
        .reset_index(drop=True)
    )

    x = sub["beta_abm"].to_numpy(float)
    y = sub[theta_col].to_numpy(float)
    yl = sub[low_col].to_numpy(float)
    yh = sub[high_col].to_numpy(float)

    tau0 = float(clean_day) - first_clean_day

    label_main = f"cleaningDay={clean_day}, tau0={tau0:.0f}"

    plt.plot(
        x, y,
        "o-",
        linewidth=2,
        markersize=5,
        label=label_main
    )

    plt.fill_between(
        x, yl, yh,
        alpha=0.15
    )

plt.xlabel(r"$\beta_{\mathrm{ABM}}$", fontsize=15)
plt.ylabel(r"$\beta_{\mathrm{SM}}$", fontsize=15)
plt.title("Cleaning-specific mapping: ABM $\\beta$ to SM $\\beta$", fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

out_png1 = os.path.join(
    fig_dir,
    "step4_mapping_A9_firstclean20_cleaningALL_overlay.png"
)

plt.savefig(out_png1, dpi=200)
plt.show()

print("saved ->", out_png1)


# -----------------------------
# plot 2: same plot but CI endpoints visible
# -----------------------------
plt.figure(figsize=(9, 6))

for clean_day in clean_values:

    sub = (
        df[df[clean_col] == clean_day]
        .sort_values("beta_abm")
        .reset_index(drop=True)
    )

    x = sub["beta_abm"].to_numpy(float)
    y = sub[theta_col].to_numpy(float)
    yl = sub[low_col].to_numpy(float)
    yh = sub[high_col].to_numpy(float)

    yerr = np.vstack([y - yl, yh - y])

    tau0 = float(clean_day) - first_clean_day
    label_main = f"cleaningDay={clean_day}, tau0={tau0:.0f}"

    plt.errorbar(
        x, y,
        yerr=yerr,
        fmt="o-",
        capsize=3,
        linewidth=1.8,
        markersize=5,
        label=label_main
    )

plt.xlabel(r"$\beta_{\mathrm{ABM}}$", fontsize=15)
plt.ylabel(r"$\beta_{\mathrm{SM}}$", fontsize=15)
plt.title("Cleaning-specific mapping with 95% CI", fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

out_png2 = os.path.join(
    fig_dir,
    "step4_mapping_A9_firstclean20_cleaningALL_errorbar.png"
)

plt.savefig(out_png2, dpi=200)
plt.show()

print("saved ->", out_png2)


# -----------------------------
# print table preview
# -----------------------------
show_cols = [clean_col, "beta_abm", theta_col, low_col, high_col]
print(df[show_cols].sort_values([clean_col, "beta_abm"]).to_string(index=False))
# %%
