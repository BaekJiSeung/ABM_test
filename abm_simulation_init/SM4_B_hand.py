# %% ================== 01_make_summary_handwash_B260.py ==================
# LONG raw ABM output -> handwash-specific summary CSV
# 서머리로바꿔서 저장
# Input:
#   ../result/interv_prob_transmission_LONG_B260_0.02-0.06_handwashALL.csv
#
# Output:
#   ../result/interv_prob_transmission_summary_B260_0.02-0.06_handwash0p8.csv
#   ../result/interv_prob_transmission_summary_B260_0.02-0.06_handwash0p9.csv
#   ../result/interv_prob_transmission_summary_B260_0.02-0.06_handwash0p95.csv
#   ../result/interv_prob_transmission_summary_B260_0.02-0.06_handwash0p99.csv
#   ../result/interv_prob_transmission_summary_B260_0.02-0.06_handwashALL.csv

import os
import ast
import numpy as np
import pandas as pd


# --------------------------------------------------
# Settings
# --------------------------------------------------
data_type = "B"
init_env = 2
tau0 = 60

variable_name = "prob_transmission"

beta_values = np.round(np.arange(0.02, 0.0601, 0.005), 5)
handwash_values = [0.8, 0.9, 0.95, 0.99]

num_iter = 50
n_months_expected = 36

beta_tag1 = f"{beta_values[0]:.2f}"
beta_tag2 = f"{beta_values[-1]:.2f}"

result_dir = "../result"

long_csv = os.path.join(
    result_dir,
    f"interv_{variable_name}_LONG_"
    f"{data_type}{init_env}{tau0}_"
    f"{beta_tag1}-{beta_tag2}_handwashALL.csv"
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def wash_to_tag(wash):
    return str(wash).replace(".", "p")


def parse_vec(x):
    """
    Convert stored daily HAI trajectory string to numpy array.
    """
    if isinstance(x, list):
        return np.array(x, dtype=float)

    if isinstance(x, np.ndarray):
        return x.astype(float)

    return np.array(ast.literal_eval(x), dtype=float)


def daily_to_monthly_30days(daily_vec, n_months):
    """
    Convert daily ABM output to monthly counts using 30-day months.
    B period has 36 months, so the expected length is 30 * 36 = 1080.
    """
    daily_vec = np.asarray(daily_vec, dtype=float)

    expected_days = 30 * n_months

    if len(daily_vec) < expected_days:
        raise ValueError(
            f"daily vector too short: len={len(daily_vec)}, "
            f"expected at least {expected_days}"
        )

    if len(daily_vec) > expected_days:
        daily_vec = daily_vec[:expected_days]

    monthly_vec = daily_vec.reshape(n_months, 30).sum(axis=1)

    return monthly_vec


# --------------------------------------------------
# Load LONG raw file
# --------------------------------------------------
print("=" * 80)
print("START make handwash summary - Period B")
print("reading:", long_csv)
print("=" * 80)

if not os.path.exists(long_csv):
    raise FileNotFoundError(long_csv)

df_long = pd.read_csv(long_csv)

print("\nLONG shape:", df_long.shape)
print("columns:", list(df_long.columns))
print(df_long.head())


# --------------------------------------------------
# Basic checks
# --------------------------------------------------
required_cols = [
    "prob_transmission",
    "hcw_wash_rate",
    "HCW_related_infecs",
    "my_iteration"
]

missing_cols = [c for c in required_cols if c not in df_long.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df_long["prob_transmission"] = df_long["prob_transmission"].astype(float)
df_long["hcw_wash_rate"] = df_long["hcw_wash_rate"].astype(float)

check_counts = (
    df_long
    .groupby(["hcw_wash_rate", "prob_transmission"])
    .size()
    .reset_index(name="n")
)

print("\ncheck counts:")
print(check_counts)
print("min n:", check_counts["n"].min())
print("max n:", check_counts["n"].max())

if check_counts["n"].min() != num_iter or check_counts["n"].max() != num_iter:
    print("[WARNING] Some handwash-beta settings do not have num_iter runs.")
else:
    print("[OK] Every handwash-beta setting has num_iter runs.")


# --------------------------------------------------
# Make summary for each handwash value
# --------------------------------------------------
all_summary_rows = []

for wash in handwash_values:

    wash_tag = wash_to_tag(wash)

    print("\n" + "=" * 80)
    print("Summarizing handwash =", wash)
    print("=" * 80)

    df_wash = df_long.loc[
        np.isclose(df_long["hcw_wash_rate"], wash)
    ].copy()

    if df_wash.empty:
        print("[WARNING] No rows for handwash =", wash)
        continue

    summary_rows = []

    for beta in beta_values:

        sub = df_wash.loc[
            np.isclose(df_wash["prob_transmission"], beta)
        ].copy()

        print(f"handwash={wash}, beta={beta:.3f}, n={len(sub)}")

        if len(sub) == 0:
            print("[WARNING] Empty beta group")
            continue

        monthly_mat = []

        for _, row in sub.iterrows():
            daily_vec = parse_vec(row["HCW_related_infecs"])

            monthly_vec = daily_to_monthly_30days(
                daily_vec,
                n_months=n_months_expected
            )

            monthly_mat.append(monthly_vec)

        monthly_mat = np.vstack(monthly_mat)

        mean_vec = monthly_mat.mean(axis=0)
        std_vec = monthly_mat.std(axis=0, ddof=1)

        row_out = {
            "beta": float(beta),
            "mean": mean_vec.tolist(),
            "std": std_vec.tolist(),
            "n_iter": int(monthly_mat.shape[0]),
            "handwash": float(wash),
            "data_type": data_type,
            "init_env": init_env,
            "tau0": tau0
        }

        summary_rows.append(row_out)
        all_summary_rows.append(row_out.copy())

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values("beta").reset_index(drop=True)

    out_csv = os.path.join(
        result_dir,
        f"interv_{variable_name}_summary_"
        f"{data_type}{init_env}{tau0}_"
        f"{beta_tag1}-{beta_tag2}_handwash{wash_tag}.csv"
    )

    df_summary.to_csv(out_csv, index=False, encoding="utf-8")

    print("\nsaved ->", out_csv)
    print("summary shape:", df_summary.shape)
    print(df_summary.head())


# --------------------------------------------------
# Save ALL summary
# --------------------------------------------------
df_all_summary = pd.DataFrame(all_summary_rows)

df_all_summary = (
    df_all_summary
    .sort_values(["handwash", "beta"])
    .reset_index(drop=True)
)

out_all_csv = os.path.join(
    result_dir,
    f"interv_{variable_name}_summary_"
    f"{data_type}{init_env}{tau0}_"
    f"{beta_tag1}-{beta_tag2}_handwashALL.csv"
)

df_all_summary.to_csv(out_all_csv, index=False, encoding="utf-8")

print("\n전체 summary 저장 완료 ->", out_all_csv)
print("ALL summary shape:", df_all_summary.shape)
print(df_all_summary.head())

print("\nDONE make handwash summary - Period B")
# %% 매핑과 발리데이션 






# %% ================== 02_fit_mapping_handwash_B260.py ==================
# Summary CSV -> handwash-specific beta_ABM -> beta_SM mapping
#
# Input:
#   ../result/interv_prob_transmission_summary_B260_0.02-0.06_handwash0p8.csv
#   ../result/interv_prob_transmission_summary_B260_0.02-0.06_handwash0p9.csv
#   ../result/interv_prob_transmission_summary_B260_0.02-0.06_handwash0p95.csv
#   ../result/interv_prob_transmission_summary_B260_0.02-0.06_handwash0p99.csv
#
# Output:
#   sm_fit/theta_pairs_subset_cumGaussian_B260_handwash0p8.csv
#   sm_fit/theta_pairs_subset_cumGaussian_B260_handwash0p9.csv
#   sm_fit/theta_pairs_subset_cumGaussian_B260_handwash0p95.csv
#   sm_fit/theta_pairs_subset_cumGaussian_B260_handwash0p99.csv
#   sm_fit/theta_pairs_subset_cumGaussian_B260_handwashALL.csv

import os
import ast
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from joblib import Parallel, delayed


# ==================================================
# 0. B period observed data and imported input
# ==================================================

# ---- Observed monthly HAI data: 2021-01 ~ 2023-12 ----
y_month = np.array(
    [1, 2, 2, 2, 1, 0, 0, 3, 2, 2, 2, 0,
     3, 0, 1, 2, 0, 1, 4, 5, 4, 2, 4, 1,
     0, 0, 1, 0, 1, 1, 0, 3, 1, 0, 3, 0],
    dtype=float
)

y_cum = np.cumsum(y_month)


# ---- Monthly imported colonized patient counts ----
monthly_PI = pd.Series({
    "2021-01": 2, "2021-02": 2, "2021-03": 3, "2021-04": 1,
    "2021-05": 0, "2021-06": 3, "2021-07": 3, "2021-08": 5,
    "2021-09": 2, "2021-10": 5, "2021-11": 0, "2021-12": 2,

    "2022-01": 4, "2022-02": 1, "2022-03": 1, "2022-04": 3,
    "2022-05": 5, "2022-06": 2, "2022-07": 2, "2022-08": 5,
    "2022-09": 1, "2022-10": 4, "2022-11": 7, "2022-12": 2,

    "2023-01": 3, "2023-02": 2, "2023-03": 1, "2023-04": 0,
    "2023-05": 5, "2023-06": 3, "2023-07": 1, "2023-08": 3,
    "2023-09": 4, "2023-10": 7, "2023-11": 7, "2023-12": 3
})


# ---- Imported colonized patient admission dates ----
PI_dates = [
    "2021-01-08", "2021-01-29",
    "2021-02-17", "2021-02-19",
    "2021-03-02", "2021-03-11", "2021-03-28",
    "2021-04-04",
    "2021-06-16", "2021-06-28", "2021-06-30",
    "2021-07-06", "2021-07-07", "2021-07-20",
    "2021-08-10", "2021-08-14", "2021-08-20", "2021-08-21", "2021-08-23",
    "2021-09-15", "2021-09-18",
    "2021-10-04", "2021-10-20", "2021-10-27", "2021-10-28", "2021-10-29",
    "2021-12-27", "2021-12-27",

    "2022-01-04", "2022-01-05", "2022-01-07", "2022-01-14",
    "2022-02-18",
    "2022-03-12",
    "2022-04-15", "2022-04-15", "2022-04-17",
    "2022-05-06", "2022-05-14", "2022-05-16", "2022-05-21", "2022-05-24",
    "2022-06-27", "2022-06-30",
    "2022-07-03", "2022-07-20",
    "2022-08-10", "2022-08-15", "2022-08-16", "2022-08-18", "2022-08-25",
    "2022-09-20",
    "2022-10-02", "2022-10-11", "2022-10-15", "2022-10-21",
    "2022-11-08", "2022-11-13", "2022-11-15", "2022-11-17",
    "2022-11-20", "2022-11-21", "2022-11-23",
    "2022-12-21", "2022-12-29",

    "2023-01-16", "2023-01-26", "2023-01-30",
    "2023-02-03", "2023-02-13",
    "2023-03-29",
    "2023-05-10", "2023-05-10", "2023-05-10", "2023-05-15", "2023-05-23",
    "2023-06-07", "2023-06-12", "2023-06-15",
    "2023-07-01",
    "2023-08-04", "2023-08-10", "2023-08-16",
    "2023-09-10", "2023-09-17", "2023-09-24", "2023-09-28",
    "2023-10-05", "2023-10-06", "2023-10-08", "2023-10-10",
    "2023-10-14", "2023-10-20", "2023-10-31",
    "2023-11-08", "2023-11-08", "2023-11-13", "2023-11-17",
    "2023-11-19", "2023-11-23", "2023-11-28",
    "2023-12-04", "2023-12-09", "2023-12-09"
]


# ==================================================
# 1. Helper for imported patient dates
# ==================================================
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


# ==================================================
# 2. B surrogate model
# ==================================================
def simulate_theta(beta, init_env, tau0, p_wash,
                   monthly_PI=monthly_PI,
                   pi_dates=PI_dates):
    """
    Period B difference-equation surrogate model.

    beta     : surrogate model transmission parameter, beta_SM
    init_env : initial environmental contamination
    tau0     : cleaning phase offset
    p_wash   : HCW handwashing rate
    """

    # ---- Parameters ----
    C_total = 30
    C_iso = 30
    C_sh = C_total

    N_H = 19
    N_E = 30

    # Period B hospital stay assumption
    mu_S = 1 / 14
    mu_HAI = 1 / 21
    mu_I = 1 / 14

    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    cleaning_eff = 0.90
    iso_factor = 0.75

    isol_time = 7.0
    sigma = 1.0 / isol_time

    # ---- Time axis ----
    start = pd.Period(monthly_PI.index.min(), freq="M").to_timestamp(how="start")
    end = pd.Period(monthly_PI.index.max(), freq="M").to_timestamp(how="end")

    days = pd.date_range(start, end, freq="D")
    T = len(days)

    # ---- Imported colonized patient input ----
    A_I_day = _make_AI_from_dates(pi_dates, days)

    # ---- State variables ----
    P_S_sh = np.zeros(T)
    P_HAI_sh = np.zeros(T)
    P_HAI_iso = np.zeros(T)
    P_I = np.zeros(T)
    H_C = np.zeros(T)
    Env_C = np.zeros(T)
    NewHAI_day = np.zeros(T)

    # ---- Initial condition ----
    P_S_sh[0] = C_total - 1
    P_I[0] = 1
    Env_C[0] = init_env

    for t in range(T):

        # environmental deep cleaning
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            xx = Env_C[t]
            Env_C[t] = (1 - cleaning_eff) * xx

        PS_sh = P_S_sh[t]
        PH_sh = P_HAI_sh[t]
        PH_iso = P_HAI_iso[t]
        PI = P_I[t]
        HC = H_C[t]
        EC = Env_C[t]

        # imported colonized patient admission
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

            # HCW -> susceptible patient
            lam_HP_sh = beta * (HC / N_H)

            # patient -> HCW
            lam_PH = beta * ((PH_sh + iso_factor * PH_iso + PI) / B_tot)

            # environment <-> HCW
            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            # new HAI
            hai_sh = lam_HP_sh * PS_sh * dt

            # shared HAI -> isolated HAI
            move_HA = sigma * PH_sh * dt

            # discharge
            outS_sh = mu_S * PS_sh * dt
            outH_sh = mu_HAI * PH_sh * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI = mu_I * PI * dt

            leaving = outS_sh + outH_sh + outH_iso + outI
            total_P = PS_sh + PH_sh + PH_iso + PI

            AS_tot = max(0.0, C_total - (total_P - leaving))
            AS_sh = AS_tot

            # update patients
            PS_sh += AS_sh - outS_sh - hai_sh
            PH_sh += hai_sh - outH_sh - move_HA
            PH_iso += move_HA - outH_iso
            PI += -outI

            # clip patient states
            PS_sh = np.clip(PS_sh, 0, C_sh)
            PH_sh = np.clip(PH_sh, 0, C_sh)
            PH_iso = np.clip(PH_iso, 0, C_iso)
            PI = np.clip(PI, 0, C_total)

            # update HCW contamination
            new_H = (lam_PH + lam_EH) * (N_H - HC) * dt
            HC = (HC + new_H) * (1 - p_wash)
            HC = np.clip(HC, 0, N_H)

            # update environmental contamination
            EC += lam_HE * (N_E - EC) * dt
            EC = np.clip(EC, 0, N_E)

            # incidence
            NewHAI_day[t] += hai_sh

        if t < T - 1:
            P_S_sh[t + 1] = PS_sh
            P_HAI_sh[t + 1] = PH_sh
            P_HAI_iso[t + 1] = PH_iso
            P_I[t + 1] = PI
            H_C[t + 1] = HC
            Env_C[t + 1] = EC

    # ---- Monthly incidence ----
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

    # ---- Compartment dataframe ----
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


# ==================================================
# 3. Mapping fitting settings
# ==================================================
data_type = "B"
init_env = 2
tau0 = 60

# B handwash 전체 mapping용 theta range
# 0.9 baseline 근처 theta가 3~4대였으므로 기존 6.5~11.5는 부적절
theta_min = 0.5
theta_max = 12.0

theta_grid_n = 800
ci_grid_n = 800

variable_name = "prob_transmission"

beta_values = np.round(np.arange(0.02, 0.0601, 0.005), 5)
beta_tag1 = f"{beta_values[0]:.2f}"
beta_tag2 = f"{beta_values[-1]:.2f}"

handwash_values = [0.8, 0.9, 0.95, 0.99]

start_month = "2021-01"
n_months_expected = 36

# ABM 돌리는 중이면 2~4 추천
n_jobs = 4

result_dir = "../result"
smfit_dir = "sm_fit"

os.makedirs(smfit_dir, exist_ok=True)


# ==================================================
# 4. Fitting helper functions
# ==================================================
def wash_to_tag(wash):
    return str(wash).replace(".", "p")


def parse_vec(s):
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=float)
    return np.array(ast.literal_eval(s), dtype=float)


def get_month_axis(n_months, start_month="2021-01"):
    return pd.period_range(
        start_month,
        periods=n_months,
        freq="M"
    ).to_timestamp()


def model_monthly_and_cum(theta, p_wash, months, init_env=init_env, tau0=tau0):
    days, daily_inc, monthly_df, comp_df = simulate_theta(
        beta=theta,
        init_env=init_env,
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


def negloglik_theta_cum_gaussian(theta, cum_obs, cum_std,
                                 months, init_env, tau0, p_wash):

    _, cum_model = model_monthly_and_cum(
        theta,
        p_wash=p_wash,
        months=months,
        init_env=init_env,
        tau0=tau0
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


def ci95_profile_theta_gaussian(cum_obs, cum_std,
                                months, init_env, tau0, p_wash,
                                theta_hat, nll_hat,
                                bounds=(0.5, 12.0),
                                grid_n=800):

    thr = nll_hat + 1.92

    a, b = bounds
    grid = np.linspace(a, b, grid_n)

    vals = np.array([
        negloglik_theta_cum_gaussian(
            th,
            cum_obs,
            cum_std,
            months,
            init_env,
            tau0,
            p_wash
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
                    x,
                    cum_obs,
                    cum_std,
                    months,
                    init_env,
                    tau0,
                    p_wash
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
                    x,
                    cum_obs,
                    cum_std,
                    months,
                    init_env,
                    tau0,
                    p_wash
                ) - thr,
                grid[i],
                grid[i + 1]
            )
            break

    return float(left), float(right)


def fit_theta_cum_gaussian_for_one(beta_abm, y_mean, y_std, p_wash, months):
    cum_obs = np.cumsum(y_mean)
    cum_std = make_cum_std_from_monthly_std(y_std)

    theta_grid = np.linspace(theta_min, theta_max, theta_grid_n)

    vals = np.array([
        negloglik_theta_cum_gaussian(
            th,
            cum_obs,
            cum_std,
            months,
            init_env,
            tau0,
            p_wash
        )
        for th in theta_grid
    ])

    idx = vals.argmin()

    theta_hat = float(theta_grid[idx])
    nll_min = float(vals[idx])

    _, cum_model_hat = model_monthly_and_cum(
        theta_hat,
        p_wash=p_wash,
        months=months,
        init_env=init_env,
        tau0=tau0
    )

    m = min(len(cum_obs), len(cum_model_hat))

    resid = cum_obs[:m] - cum_model_hat[:m]

    sigma_hat = float(np.sqrt(np.mean(resid ** 2)))
    weighted_rmse = float(np.sqrt(np.mean((resid / cum_std[:m]) ** 2)))

    theta_low, theta_high = ci95_profile_theta_gaussian(
        cum_obs,
        cum_std,
        months,
        init_env,
        tau0,
        p_wash,
        theta_hat,
        nll_min,
        bounds=(theta_min, theta_max),
        grid_n=ci_grid_n
    )

    return theta_hat, theta_low, theta_high, nll_min, sigma_hat, weighted_rmse


def fit_one_row(row_dict, wash, months):
    beta_abm = float(row_dict["beta"])
    y_mean = row_dict["mean_vec"]
    y_std = row_dict["std_vec"]

    theta_hat, theta_low, theta_high, nll_min, sigma_hat, weighted_rmse = (
        fit_theta_cum_gaussian_for_one(
            beta_abm,
            y_mean,
            y_std,
            p_wash=wash,
            months=months
        )
    )

    print(
        f"done: handwash={wash}, beta_ABM={beta_abm:0.3f}, "
        f"theta_hat={theta_hat:.4f}, NLL={nll_min:.2f}"
    )

    return {
        "handwash": float(wash),
        "beta_abm": float(beta_abm),
        "theta_hat": theta_hat,
        "theta_low": theta_low,
        "theta_high": theta_high,
        "sigma_hat": sigma_hat,
        "weighted_rmse": weighted_rmse,
        "neg_loglik_cum_min": nll_min,
        "init_env": init_env,
        "tau0": tau0,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "theta_grid_n": theta_grid_n,
        "ci_grid_n": ci_grid_n,
        "cleaning_eff": 0.90,
        "data_type": data_type
    }


def load_abm_summary(wash):
    wash_tag = wash_to_tag(wash)

    csv_path = os.path.join(
        result_dir,
        f"interv_{variable_name}_summary_"
        f"{data_type}{init_env}{tau0}_"
        f"{beta_tag1}-{beta_tag2}_handwash{wash_tag}.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    df["mean_vec"] = df["mean"].apply(parse_vec)
    df["std_vec"] = df["std"].apply(parse_vec)

    df = df.sort_values("beta").reset_index(drop=True)

    return df


# ==================================================
# 5. Fit mapping and save
# ==================================================
print("=" * 80)
print("START handwash-specific beta_ABM -> beta_SM mapping fitting - Period B")
print("data_type:", data_type)
print("init_env:", init_env)
print("tau0:", tau0)
print("theta range:", theta_min, "~", theta_max)
print("theta_grid_n:", theta_grid_n)
print("ci_grid_n:", ci_grid_n)
print("beta_values:", beta_values)
print("handwash_values:", handwash_values)
print("n_jobs:", n_jobs)
print("=" * 80)

all_results = []

for wash in handwash_values:

    print("\n" + "=" * 80)
    print(f"START fitting for handwash = {wash}")
    print("=" * 80)

    df_abm = load_abm_summary(wash)

    print("df_abm shape:", df_abm.shape)
    print("df_abm columns:", list(df_abm.columns))

    n_months = len(df_abm["mean_vec"].iloc[0])

    if n_months != n_months_expected:
        print(
            f"[WARNING] expected n_months={n_months_expected}, "
            f"but CSV has n_months={n_months}"
        )

    months = get_month_axis(n_months, start_month=start_month)

    row_dicts = (
        df_abm
        .sort_values("beta")
        .to_dict(orient="records")
    )

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_one_row)(row_dict, wash, months)
        for row_dict in row_dicts
    )

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("beta_abm").reset_index(drop=True)

    wash_tag = wash_to_tag(wash)

    out_csv = os.path.join(
        smfit_dir,
        f"theta_pairs_subset_cumGaussian_"
        f"{data_type}{init_env}{tau0}_handwash{wash_tag}.csv"
    )

    df_res.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n저장 완료 ->", out_csv)
    print(df_res.head())

    # boundary check
    n_low = np.sum(np.isclose(df_res["theta_hat"], theta_min))
    n_high = np.sum(np.isclose(df_res["theta_hat"], theta_max))

    if n_low > 0 or n_high > 0:
        print("[WARNING] Some theta_hat values are on the boundary.")
        print("n at theta_min:", n_low)
        print("n at theta_max:", n_high)

    all_results.extend(results)


df_all = pd.DataFrame(all_results)

df_all = (
    df_all
    .sort_values(["handwash", "beta_abm"])
    .reset_index(drop=True)
)

out_all_csv = os.path.join(
    smfit_dir,
    f"theta_pairs_subset_cumGaussian_"
    f"{data_type}{init_env}{tau0}_handwashALL.csv"
)

df_all.to_csv(out_all_csv, index=False, encoding="utf-8")

print("\n전체 저장 완료 ->", out_all_csv)
print("df_all shape:", df_all.shape)
print(df_all.head())

print("\nBoundary check by handwash:")
print(
    df_all
    .assign(
        at_lower=np.isclose(df_all["theta_hat"], theta_min),
        at_upper=np.isclose(df_all["theta_hat"], theta_max)
    )
    .groupby("handwash")[["at_lower", "at_upper"]]
    .sum()
)

print("\nDONE handwash-specific mapping fitting - Period B")






# %%
