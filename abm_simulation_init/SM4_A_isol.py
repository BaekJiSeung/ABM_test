# %% ================== isolationTime raw files -> summary files + ALL ==================

import os
import ast
import numpy as np
import pandas as pd


# -----------------------------
# settings
# -----------------------------
data_type = "A"
init_env = 9

# isolation intervention에서는 cleaningDay와 tau_offset_days는 baseline으로 고정
cleaningDay = 180
tau_offset_days = 140

scenario_tag = f"{data_type}{init_env}{tau_offset_days}"

variable_name = "prob_transmission"

beta_tag1 = 0.02
beta_tag2 = 0.06

isoltime_values = [6, 14, 20, 28]

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
# convert each isolationTime raw file
# -----------------------------
all_summary_rows = []

for isol_time in isoltime_values:

    raw_csv = os.path.join(
        result_dir,
        f"interv_{variable_name}_"
        f"{scenario_tag}_{beta_tag1}-{beta_tag2}_isoltime{isol_time}.csv"
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
            "isolation_time": isol_time,
            "cleaningDay": cleaningDay,
            "tau_offset_days": tau_offset_days,
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
        f"{scenario_tag}_{beta_tag1}-{beta_tag2}_isoltime{isol_time}.csv"
    )

    summary_df.to_csv(out_csv, index=False, encoding="utf-8")

    print("summary shape:", summary_df.shape)
    print(summary_df[["isolation_time", "beta", "cleaningDay", "tau_offset_days", "n"]])
    print("saved summary:", out_csv)


# -----------------------------
# save ALL summary
# -----------------------------
summary_all_df = (
    pd.DataFrame(all_summary_rows)
    .sort_values(["isolation_time", "beta"])
    .reset_index(drop=True)
)

out_all_csv = os.path.join(
    result_dir,
    f"interv_{variable_name}_summary_"
    f"{scenario_tag}_{beta_tag1}-{beta_tag2}_isoltimeALL.csv"
)

summary_all_df.to_csv(out_all_csv, index=False, encoding="utf-8")

print("\n" + "=" * 80)
print("saved ALL summary:", out_all_csv)
print("ALL summary shape:", summary_all_df.shape)
print(summary_all_df[["isolation_time", "beta", "cleaningDay", "tau_offset_days", "n"]])
print("=" * 80)

print("\nDONE isolationTime raw -> summary + ALL")
# %% 매핑 4단계







# %% ================== SM4_A_isoltime_mapping.py ==================
# isolationTime-specific beta_ABM -> beta_SM mapping reconstruction

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
    tau0,
    isol_time_abm,
    p_wash=0.9,
    cleaningDay=180,
    monthly_PI=monthly_PI,
    pi_dates=PI_dates
):
    """
    Difference-equation surrogate model for isolationTime-specific mapping.

    beta          : surrogate model transmission parameter, beta_SM
    init_env      : initial environmental contamination level
    tau0          : cleaning phase offset
    isol_time_abm : ABM isolation_time parameter
                    In ABM, isolation delay is randomly sampled up to this value.
                    Therefore, surrogate uses mean isolation time = isol_time_abm / 2.
    p_wash        : HCW handwashing rate, fixed
    cleaningDay   : deep cleaning period, fixed
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

    deep_clean_period = cleaningDay
    cleaning_eff = 0.90
    iso_factor = 0.75

    # 중요:
    # ABM isolation_time은 0~isolation_time 사이 random draw의 upper bound.
    # surrogate는 평균 이동시간을 쓰므로 isolation_time / 2 사용.
    isol_time_mean = isol_time_abm / 2.0
    sigma = 1.0 / isol_time_mean

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

        # environmental deep cleaning fixed
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
tau0 = 140
cleaningDay_fixed = 180
p_wash_fixed = 0.9

isoltime_values = [6, 14, 20, 28]

variable_name = "prob_transmission"

beta_tag1 = 0.02
beta_tag2 = 0.06

# beta_SM search range
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

scenario_tag = f"{data_type}{init_env}{tau0}"


# ============================================================
# Helper functions
# ============================================================

def parse_vec(s):
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=float)

    return np.array(ast.literal_eval(s), dtype=float)


def model_monthly_and_cum(
    theta,
    isol_time_abm,
    months,
    init_env=init_env,
    tau0=tau0,
    p_wash=p_wash_fixed,
    cleaningDay=cleaningDay_fixed
):
    days, daily_inc, monthly_df, comp_df = simulate_theta(
        beta=theta,
        init_env=init_env,
        tau0=tau0,
        isol_time_abm=isol_time_abm,
        p_wash=p_wash,
        cleaningDay=cleaningDay
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
    tau0,
    isol_time_abm,
    p_wash,
    cleaningDay
):
    _, cum_model = model_monthly_and_cum(
        theta=theta,
        isol_time_abm=isol_time_abm,
        months=months,
        init_env=init_env,
        tau0=tau0,
        p_wash=p_wash,
        cleaningDay=cleaningDay
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
    tau0,
    isol_time_abm,
    p_wash,
    cleaningDay,
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
            tau0=tau0,
            isol_time_abm=isol_time_abm,
            p_wash=p_wash,
            cleaningDay=cleaningDay
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
                    tau0=tau0,
                    isol_time_abm=isol_time_abm,
                    p_wash=p_wash,
                    cleaningDay=cleaningDay
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
                    tau0=tau0,
                    isol_time_abm=isol_time_abm,
                    p_wash=p_wash,
                    cleaningDay=cleaningDay
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
    isol_time_abm,
    months,
    p_wash=p_wash_fixed,
    cleaningDay=cleaningDay_fixed
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
            tau0=tau0,
            isol_time_abm=isol_time_abm,
            p_wash=p_wash,
            cleaningDay=cleaningDay
        )
        for th in theta_grid
    ])

    idx = vals.argmin()

    theta_hat = float(theta_grid[idx])
    nll_min = float(vals[idx])

    _, cum_model_hat = model_monthly_and_cum(
        theta=theta_hat,
        isol_time_abm=isol_time_abm,
        months=months,
        init_env=init_env,
        tau0=tau0,
        p_wash=p_wash,
        cleaningDay=cleaningDay
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
        tau0=tau0,
        isol_time_abm=isol_time_abm,
        p_wash=p_wash,
        cleaningDay=cleaningDay,
        theta_hat=theta_hat,
        nll_hat=nll_min,
        bounds=(theta_min, theta_max),
        grid_n=ci_grid_n
    )

    return theta_hat, theta_low, theta_high, nll_min, sigma_hat, weighted_rmse


def fit_one_row(row_dict, isol_time_abm, months):
    beta_abm = float(row_dict["beta"])
    y_mean = row_dict["mean_vec"]
    y_std = row_dict["std_vec"]

    theta_hat, theta_low, theta_high, nll_min, sigma_hat, weighted_rmse = (
        fit_theta_cum_gaussian_for_one(
            beta_abm=beta_abm,
            y_mean=y_mean,
            y_std=y_std,
            isol_time_abm=isol_time_abm,
            months=months,
            p_wash=p_wash_fixed,
            cleaningDay=cleaningDay_fixed
        )
    )

    print(
        f"done: isolation_time={isol_time_abm}, beta_ABM={beta_abm:0.3f}, "
        f"theta_hat={theta_hat:.4f}, NLL={nll_min:.2f}"
    )

    return {
        "isolation_time": isol_time_abm,
        "surrogate_isol_time_mean": isol_time_abm / 2.0,
        "beta_abm": beta_abm,
        "theta_hat": theta_hat,
        "theta_low": theta_low,
        "theta_high": theta_high,
        "sigma_hat": sigma_hat,
        "weighted_rmse": weighted_rmse,
        "neg_loglik_cum_min": nll_min,
        "init_env": init_env,
        "tau0": tau0,
        "cleaningDay": cleaningDay_fixed,
        "p_wash": p_wash_fixed,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "theta_grid_n": theta_grid_n,
        "ci_grid_n": ci_grid_n,
        "cleaning_eff": 0.90,
    }


print("Cell 2 done")


# ============================================================
# Run Step 4 mapping reconstruction for each isolation_time
# ============================================================

all_results = []

for isol_time in isoltime_values:

    abm_csv = os.path.join(
        result_dir,
        f"interv_{variable_name}_summary_"
        f"{scenario_tag}_{beta_tag1}-{beta_tag2}_isoltime{isol_time}.csv"
    )

    print("\n" + "=" * 80)
    print(f"START Step4 for isolation_time = {isol_time}")
    print(f"surrogate mean isolation time = {isol_time / 2.0}")
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
        delayed(fit_one_row)(row_dict, isol_time, months)
        for row_dict in row_dicts
    )

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("beta_abm").reset_index(drop=True)

    out_csv = os.path.join(
        smfit_dir,
        f"theta_pairs_subset_cumGaussian_"
        f"{scenario_tag}_isoltime{isol_time}.csv"
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
    df_all = df_all.sort_values(["isolation_time", "beta_abm"]).reset_index(drop=True)

    out_all_csv = os.path.join(
        smfit_dir,
        f"theta_pairs_subset_cumGaussian_"
        f"{scenario_tag}_isoltimeALL.csv"
    )

    df_all.to_csv(out_all_csv, index=False, encoding="utf-8")

    print("\n전체 저장 완료 →", out_all_csv)
    print(df_all.head())
    print("done.")
else:
    print("[ERROR] no results generated.")

# %%플롯하는거





# %% ================== Plot isolation-specific beta_ABM -> beta_SM mappings ==================

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

smfit_dir = "sm_fit"
fig_dir = os.path.join(smfit_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# 파일명 후보들
candidate_files = [
    os.path.join(smfit_dir, f"theta_pairs_subset_cumGaussian_{data_type}{init_env}{tau0}_isolALL.csv"),
    os.path.join(smfit_dir, f"theta_pairs_subset_cumGaussian_{data_type}{init_env}{tau0}_isoltimeALL.csv"),
    os.path.join(smfit_dir, f"theta_pairs_subset_cumGaussian_{data_type}{init_env}{tau0}_isolationALL.csv"),
]

mapping_csv = None
for f in candidate_files:
    if os.path.exists(f):
        mapping_csv = f
        break

if mapping_csv is None:
    raise FileNotFoundError(
        "isolation mapping csv 파일을 찾지 못했습니다.\n"
        "예상 파일명 예시:\n"
        + "\n".join(candidate_files)
    )

print("reading:", mapping_csv)

df = pd.read_csv(mapping_csv)

print(df.head())
print(df.columns.tolist())
print(df.shape)


# -----------------------------
# column names
# -----------------------------
if "isolationTime" in df.columns:
    isol_col = "isolationTime"
elif "isoltime" in df.columns:
    isol_col = "isoltime"
elif "isolation_time" in df.columns:
    isol_col = "isolation_time"
else:
    raise ValueError("isolation time column not found.")

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
# isolation values
# -----------------------------
isol_values = sorted(df[isol_col].dropna().unique())
print("isolation values:", isol_values)


# -----------------------------
# helper: ABM isolation time -> SM mean isolation time
# ABM에서 0 ~ T 사이로 랜덤하게 잡으면 평균은 T/2
# -----------------------------
def abm_to_sm_isol(t_abm):
    return t_abm / 2.0


# -----------------------------
# plot 1: overlay mapping curves
# -----------------------------
plt.figure(figsize=(9, 6))

for isol in isol_values:
    sub = (
        df[df[isol_col] == isol]
        .sort_values("beta_abm")
        .reset_index(drop=True)
    )

    x = sub["beta_abm"].to_numpy(float)
    y = sub[theta_col].to_numpy(float)
    yl = sub[low_col].to_numpy(float)
    yh = sub[high_col].to_numpy(float)

    sm_isol = abm_to_sm_isol(float(isol))

    label_main = f"ABM isol={isol}, SM isol={sm_isol:.1f}"

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
plt.title("Isolation-specific mapping: ABM $\\beta$ to SM $\\beta$", fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

out_png1 = os.path.join(
    fig_dir,
    f"step4_mapping_{data_type}{init_env}{tau0}_isolALL_overlay.png"
)

plt.savefig(out_png1, dpi=200)
plt.show()

print("saved ->", out_png1)


# -----------------------------
# plot 2: same plot but CI endpoints visible
# -----------------------------
plt.figure(figsize=(9, 6))

for isol in isol_values:
    sub = (
        df[df[isol_col] == isol]
        .sort_values("beta_abm")
        .reset_index(drop=True)
    )

    x = sub["beta_abm"].to_numpy(float)
    y = sub[theta_col].to_numpy(float)
    yl = sub[low_col].to_numpy(float)
    yh = sub[high_col].to_numpy(float)

    yerr = np.vstack([y - yl, yh - y])

    sm_isol = abm_to_sm_isol(float(isol))
    label_main = f"ABM isol={isol}, SM isol={sm_isol:.1f}"

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
plt.title("Isolation-specific mapping with 95% CI", fontsize=16)
plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

out_png2 = os.path.join(
    fig_dir,
    f"step4_mapping_{data_type}{init_env}{tau0}_isolALL_errorbar.png"
)

plt.savefig(out_png2, dpi=200)
plt.show()

print("saved ->", out_png2)


# -----------------------------
# print table preview
# -----------------------------
show_cols = [isol_col, "beta_abm", theta_col, low_col, high_col]
print(df[show_cols].sort_values([isol_col, "beta_abm"]).to_string(index=False))


# %%







#
# %%
# %% ==============================
# Step 4 validation: ABM vs SM cumulative curves
# for each isolation_time and each beta_ABM
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
tau0 = 140

cleaningDay_fixed = 180
p_wash_fixed = 0.9

isoltime_values = [6, 14, 20, 28]

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

scenario_tag = f"{data_type}{init_env}{tau0}"


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
    months = pd.period_range(
        start_month,
        periods=n_months,
        freq="M"
    ).to_timestamp()

    return months


def load_abm_summary_isoltime(isol_time):
    csv_path = os.path.join(
        result_dir,
        f"interv_prob_transmission_summary_"
        f"{scenario_tag}_{beta_tag1}-{beta_tag2}_isoltime{isol_time}.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    df["mean_vec"] = df["mean"].apply(parse_vec)

    if "std" in df.columns:
        df["std_vec"] = df["std"].apply(parse_vec)

    df = df.sort_values("beta").reset_index(drop=True)

    return df


def load_mapping_isoltime(isol_time):
    # 개별 mapping file 우선 사용
    csv_path = os.path.join(
        smfit_dir,
        f"theta_pairs_subset_cumGaussian_"
        f"{scenario_tag}_isoltime{isol_time}.csv"
    )

    # 개별 파일이 없으면 ALL에서 filtering
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

    else:
        all_path = os.path.join(
            smfit_dir,
            f"theta_pairs_subset_cumGaussian_"
            f"{scenario_tag}_isoltimeALL.csv"
        )

        if not os.path.exists(all_path):
            raise FileNotFoundError(
                f"mapping file not found:\n{csv_path}\n{all_path}"
            )

        df_all = pd.read_csv(all_path)

        if "isolation_time" in df_all.columns:
            isol_col = "isolation_time"
        elif "isoltime" in df_all.columns:
            isol_col = "isoltime"
        elif "isolationTime" in df_all.columns:
            isol_col = "isolationTime"
        else:
            raise ValueError(
                "isolation time column not found in mapping ALL file."
            )

        df = df_all.loc[
            np.isclose(df_all[isol_col].astype(float), isol_time)
        ].copy()

        if df.empty:
            raise ValueError(
                f"isolation_time={isol_time} not found in mapping ALL file.\n"
                f"available = {sorted(df_all[isol_col].unique())}"
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


def get_surrogate_monthly_isoltime(beta_sm, isol_time_abm, n_months):
    """
    Requires model_monthly_and_cum() from SM4_A_isol.py.

    Important:
    - isol_time_abm is the ABM isolation_time parameter.
    - Inside model_monthly_and_cum() / simulate_theta(),
      surrogate mean isolation time should be isol_time_abm / 2.
    """

    months = get_month_axis(
        n_months=n_months,
        start_month=start_month
    )

    sm_monthly, sm_cum = model_monthly_and_cum(
        theta=beta_sm,
        isol_time_abm=isol_time_abm,
        months=months,
        init_env=init_env,
        tau0=tau0
    )

    sm_monthly = np.asarray(sm_monthly, dtype=float)
    sm_cum = np.asarray(sm_cum, dtype=float)

    return sm_monthly, sm_cum


# -----------------------------
# plot cumulative curves
# -----------------------------
for isol_time in isoltime_values:

    sm_isol_mean = isol_time / 2.0

    print("\n" + "=" * 80)
    print(f"Plot cumulative curves: ABM isolation_time = {isol_time}")
    print(f"Surrogate mean isolation time = {sm_isol_mean}")
    print("=" * 80)

    df_abm = load_abm_summary_isoltime(isol_time)
    df_map = load_mapping_isoltime(isol_time)

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

        months = get_month_axis(
            n_months=n_months,
            start_month=start_month
        )

        abm_cum = np.cumsum(abm_monthly)

        # --- beta_ABM -> beta_SM mapping ---
        beta_sm = get_beta_sm_from_mapping(
            beta_abm=beta_abm,
            df_map=df_map
        )

        # --- surrogate run ---
        sm_monthly, sm_cum = get_surrogate_monthly_isoltime(
            beta_sm=beta_sm,
            isol_time_abm=isol_time,
            n_months=n_months
        )

        # --- length matching ---
        m = min(len(abm_cum), len(sm_cum), len(months))

        abm_cum_plot = abm_cum[:m]
        sm_cum_plot = sm_cum[:m]
        months_plot = months[:m]

        # --- RMSE ---
        cum_rmse = float(
            np.sqrt(np.mean((abm_cum_plot - sm_cum_plot) ** 2))
        )

        # --- plot ---
        ax.plot(
            months_plot,
            abm_cum_plot,
            "o-",
            linewidth=2,
            markersize=4,
            label="ABM cumulative"
        )

        ax.plot(
            months_plot,
            sm_cum_plot,
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
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=12
    )

    fig.suptitle(
        f"Step 4 validation: ABM vs SM cumulative curves "
        f"(ABM isolation_time={isol_time}, SM mean={sm_isol_mean:.1f})",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = os.path.join(
        fig_dir,
        f"step4_validation_cumulative_curves_"
        f"{scenario_tag}_isoltime{isol_time}.png"
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

    print("saved:", out_png)


print("\nDONE isolationTime Step 4 cumulative validation plots")
# %% abm 과비교







 

# %% ================== isoltime mapping + actual ABM vs mapped SM at beta_ABM=0.03847 ==================

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
tau0 = 140

isoltime_values = [6, 14, 20, 28]

beta_abm_target = 0.03847
start_month = "2017-01"

days_per_month = 30

if data_type == "A":
    n_months = 19
elif data_type == "B":
    n_months = 36
else:
    raise ValueError("data_type must be 'A' or 'B'")


try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

result_dir = os.path.join(base_dir, "..", "result")
smfit_dir = os.path.join(base_dir, "sm_fit")
fig_dir = os.path.join(smfit_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

scenario_tag = f"{data_type}{init_env}{tau0}"


# --------------------------------------------------
# files
# --------------------------------------------------
mapping_csv = os.path.join(
    smfit_dir,
    f"theta_pairs_subset_cumGaussian_{scenario_tag}_isoltimeALL.csv"
)

actual_raw_csv = os.path.join(
    result_dir,
    f"interv_prob_transmission_LONG_{scenario_tag}_0.03847-0.03847_isoltimeALL.csv"
)


# --------------------------------------------------
# required function check
# --------------------------------------------------
if "model_monthly_and_cum" not in globals():
    raise NameError(
        "model_monthly_and_cum is not defined.\n"
        "Run your isolation SM4 mapping code first."
    )


# --------------------------------------------------
# read files
# --------------------------------------------------
print("reading mapping:", mapping_csv)
df_map = pd.read_csv(mapping_csv)
print(df_map.head())
print(df_map.columns)

print("\nreading actual ABM raw:", actual_raw_csv)
df_actual = pd.read_csv(actual_raw_csv)
print(df_actual.head())
print(df_actual.columns)


# --------------------------------------------------
# helper functions
# --------------------------------------------------
def parse_vec_local(x):
    if isinstance(x, list):
        return np.array(x, dtype=float)

    if isinstance(x, np.ndarray):
        return x.astype(float)

    return np.array(ast.literal_eval(x), dtype=float)


def get_months(n_months):
    return pd.period_range(
        start_month,
        periods=n_months,
        freq="M"
    ).to_timestamp()


def find_isol_col(df):
    for c in ["isolation_time", "isoltime", "isolationTime"]:
        if c in df.columns:
            return c

    raise ValueError(f"isolation time column not found: {list(df.columns)}")


def daily_to_monthly_local(daily_series, days_per_month=30, n_months=19):
    arr = np.array(daily_series, dtype=float)

    needed = days_per_month * n_months
    arr = arr[:needed]

    m = len(arr) // days_per_month
    arr = arr[:m * days_per_month]

    return arr.reshape(m, days_per_month).sum(axis=1)


def summarize_actual_abm_isol(df_raw, isol_time):
    """
    actual ABM raw file에서 특정 isolation_time의 50회 run을 monthly mean/std로 요약.
    """

    isol_col = find_isol_col(df_raw)

    sub = df_raw[
        df_raw[isol_col].astype(float) == float(isol_time)
    ].copy()

    if sub.empty:
        raise ValueError(
            f"actual raw file에 isolation_time={isol_time} 없음.\n"
            f"available = {sorted(df_raw[isol_col].unique())}"
        )

    monthly_runs = []

    for s in sub["HCW_related_infecs"].dropna():
        daily = parse_vec_local(s)

        monthly = daily_to_monthly_local(
            daily_series=daily,
            days_per_month=days_per_month,
            n_months=n_months
        )

        monthly_runs.append(monthly)

    monthly_arr = np.array(monthly_runs, dtype=float)

    abm_mean = monthly_arr.mean(axis=0)
    abm_std = monthly_arr.std(axis=0, ddof=0)

    return abm_mean, abm_std, len(monthly_runs)


def get_mapping_sub(df_map, isol_time):
    isol_col = find_isol_col(df_map)

    sub = df_map[
        df_map[isol_col].astype(float) == float(isol_time)
    ].copy()

    if sub.empty:
        raise ValueError(
            f"mapping file에 isolation_time={isol_time} 없음.\n"
            f"available = {sorted(df_map[isol_col].unique())}"
        )

    sub = sub.sort_values("beta_abm").reset_index(drop=True)

    return sub


# mapping file isolation column
map_isol_col = find_isol_col(df_map)


# --------------------------------------------------
# Part 1. Mapping plot + beta_SM interpolation
# --------------------------------------------------
mapped_rows = []

plt.figure(figsize=(8, 6))

for isol_time in isoltime_values:

    sub = get_mapping_sub(df_map, isol_time)

    x = sub["beta_abm"].to_numpy(float)
    y = sub["theta_hat"].to_numpy(float)

    beta_sm = float(np.interp(beta_abm_target, x, y))

    mapped_rows.append({
        "isolation_time_abm": isol_time,
        "isolation_time_sm_mean": isol_time / 2.0,
        "beta_abm_target": beta_abm_target,
        "beta_sm_mapped": beta_sm
    })

    plt.plot(
        x,
        y,
        "o-",
        linewidth=2,
        markersize=5,
        label=f"ABM isol={isol_time}, SM mean={isol_time / 2.0:.1f}"
    )

    if "theta_low" in sub.columns and "theta_high" in sub.columns:
        plt.fill_between(
            x,
            sub["theta_low"].to_numpy(float),
            sub["theta_high"].to_numpy(float),
            alpha=0.12
        )

    plt.scatter(
        [beta_abm_target],
        [beta_sm],
        s=110,
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
    r"Isolation-specific mapping: $\beta_{\mathrm{ABM}} \rightarrow \beta_{\mathrm{SM}}$",
    fontsize=15
)
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()

out_mapping_png = os.path.join(
    fig_dir,
    f"mapping_isoltime_betaABM_to_betaSM_betaABM0p03847_{scenario_tag}.png"
)

plt.savefig(out_mapping_png, dpi=300, bbox_inches="tight")
plt.show()

print("saved:", out_mapping_png)


df_mapped = pd.DataFrame(mapped_rows)

out_mapped_csv = os.path.join(
    smfit_dir,
    f"mapped_betaSM_betaABM0p03847_{scenario_tag}_isoltime.csv"
)

df_mapped.to_csv(out_mapped_csv, index=False, encoding="utf-8")

print("saved:", out_mapped_csv)
print(df_mapped.to_string(index=False))


# --------------------------------------------------
# Part 2. Actual ABM vs mapped SM comparison
# --------------------------------------------------
compare_rows = []

fig_m, axes_m = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes_m = axes_m.ravel()

fig_c, axes_c = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes_c = axes_c.ravel()

for i, isol_time in enumerate(isoltime_values):

    # actual ABM at beta_ABM = 0.03847
    abm_mean, abm_std, n_runs = summarize_actual_abm_isol(
        df_actual,
        isol_time
    )

    row_map = df_mapped[
        df_mapped["isolation_time_abm"] == isol_time
    ]

    if row_map.empty:
        print(f"[WARNING] no mapped beta_SM for isol_time={isol_time}")
        continue

    beta_sm = float(row_map.iloc[0]["beta_sm_mapped"])

    months = get_months(len(abm_mean))

    # SM run
    # 중요:
    # isol_time_abm을 그대로 넣는다.
    # 네 isol surrogate 함수 내부에서 isol_time_abm / 2.0을 SM mean isolation time으로 사용해야 함.
    sm_monthly, sm_cum = model_monthly_and_cum(
        theta=beta_sm,
        isol_time_abm=isol_time,
        months=months,
        init_env=init_env,
        tau0=tau0
    )

    sm_monthly = np.array(sm_monthly, dtype=float)
    sm_cum = np.array(sm_cum, dtype=float)

    abm_cum = np.cumsum(abm_mean)
    abm_cum_std = np.sqrt(np.cumsum(abm_std ** 2))

    m = min(
        len(abm_mean),
        len(abm_std),
        len(abm_cum),
        len(abm_cum_std),
        len(sm_monthly),
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
        "isolation_time_abm": isol_time,
        "isolation_time_sm_mean": isol_time / 2.0,
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
        markersize=4,
        label=rf"Actual ABM mean ($\beta_{{ABM}}$={beta_abm_target:.5f})"
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
        markersize=4,
        label=rf"Mapped SM ($\beta_{{SM}}$={beta_sm:.3f})"
    )

    axm.set_title(
        f"ABM isol={isol_time}, SM mean={isol_time / 2.0:.1f}, n={n_runs}",
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
        markersize=4,
        label=rf"Actual ABM cumulative ($\beta_{{ABM}}$={beta_abm_target:.5f})"
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
        markersize=4,
        label=rf"Mapped SM ($\beta_{{SM}}$={beta_sm:.3f})"
    )

    axc.set_title(
        f"ABM isol={isol_time}, SM mean={isol_time / 2.0:.1f}, n={n_runs}",
        fontsize=12
    )

    axc.set_xlabel("Month")
    axc.set_ylabel("Cumulative HAI")
    axc.grid(alpha=0.3)
    axc.legend(fontsize=8)


# --------------------------------------------------
# save monthly plot
# --------------------------------------------------
fig_m.suptitle(
    r"Actual isolation ABM vs mapped SM monthly trajectories "
    r"at $\beta_{\mathrm{ABM}}=0.03847$",
    fontsize=14
)

fig_m.tight_layout(rect=[0, 0, 1, 0.96])

out_monthly = os.path.join(
    fig_dir,
    f"actual_isoltime_ABM_vs_mappedSM_monthly_betaABM0p03847_{scenario_tag}.png"
)

fig_m.savefig(out_monthly, dpi=300, bbox_inches="tight")
plt.show()

print("saved:", out_monthly)


# --------------------------------------------------
# save cumulative plot
# --------------------------------------------------
fig_c.suptitle(
    r"Actual isolation ABM vs mapped SM cumulative trajectories "
    r"at $\beta_{\mathrm{ABM}}=0.03847$",
    fontsize=14
)

fig_c.tight_layout(rect=[0, 0, 1, 0.96])

out_cum = os.path.join(
    fig_dir,
    f"actual_isoltime_ABM_vs_mappedSM_cumulative_betaABM0p03847_{scenario_tag}.png"
)

fig_c.savefig(out_cum, dpi=300, bbox_inches="tight")
plt.show()

print("saved:", out_cum)


# --------------------------------------------------
# Part 3. save comparison table
# --------------------------------------------------
df_compare = pd.DataFrame(compare_rows)

out_compare_csv = os.path.join(
    smfit_dir,
    f"actual_isoltime_ABM_vs_mappedSM_betaABM0p03847_{scenario_tag}.csv"
)

df_compare.to_csv(out_compare_csv, index=False, encoding="utf-8")

print("saved:", out_compare_csv)
print(df_compare.to_string(index=False))