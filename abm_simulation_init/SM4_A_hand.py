# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
from scipy.optimize import brentq
from joblib import Parallel, delayed


# Period A: 2017 Jan. – 2018 Jul.
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


def simulate_theta(beta, init_env, tau0, p_wash,
                   monthly_PI=monthly_PI, pi_dates=PI_dates):
    """
    Difference-equation surrogate model.

    beta     : surrogate model transmission parameter, beta_SM
    init_env : initial environmental contamination
    tau0     : cleaning phase offset
    p_wash   : HCW handwashing rate

    중요:
    - p_wash는 HCW handwashing에만 적용
    - environmental deep cleaning effect는 cleaning_eff = 0.90으로 고정
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

    deep_clean_period = 180
    cleaning_eff = 0.90
    iso_factor = 0.75

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

        # 환경청소는 p_wash와 무관하게 0.90 고정
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            xx = Env_C[t]
            Env_C[t] = (1 - cleaning_eff) * xx

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

            # p_wash는 HCW contamination 제거에만 적용
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

# ================== Step4 settings ==================

data_type = "A"

init_env = 9
tau0 = 140

theta_min = 6.5
theta_max = 11.5

theta_grid_n = 400
ci_grid_n = 400

variable_name = "prob_transmission"

beta_tag1 = 0.02
beta_tag2 = 0.06

handwash_values = [0.99]

n_jobs = 8

if data_type == "A":
    start_month = "2017-01"
    n_months_expected = 19
elif data_type == "B":
    start_month = "2021-01"
    n_months_expected = 36
else:
    raise ValueError("data_type must be 'A' or 'B'")

os.makedirs("sm_fit", exist_ok=True)


def wash_to_tag(wash):
    return str(wash).replace(".", "p")


def parse_vec(s):
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=float)
    return np.array(ast.literal_eval(s), dtype=float)


def model_monthly_and_cum(theta, p_wash, months, init_env=init_env, tau0=tau0):
    days, daily_inc, monthly_df, comp_df = simulate_theta(
        theta,
        init_env,
        tau0,
        p_wash
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
                                bounds=(6.5, 10.5),
                                grid_n=400):

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
        "handwash": wash,
        "beta_abm": beta_abm,
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
    }


print("Cell 2 done")
all_results = []

for wash in handwash_values:

    wash_tag = wash_to_tag(wash)

    abm_csv = (
        f"../result/interv_{variable_name}_summary_"
        f"{data_type}{init_env}{tau0}_{beta_tag1}-{beta_tag2}_handwash{wash_tag}.csv"
    )

    print("\n" + "=" * 80)
    print(f"START Step4 for handwash = {wash}")
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

    # pandas row를 그대로 넘기면 joblib에서 꼬일 수 있어서 dict로 변환
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

    out_csv = (
        f"sm_fit/theta_pairs_subset_cumGaussian_"
        f"{data_type}{init_env}{tau0}_handwash{wash_tag}.csv"
    )

    df_res.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n저장 완료 →", out_csv)
    print(df_res.head())

    all_results.extend(results)


df_all = pd.DataFrame(all_results)
df_all = df_all.sort_values(["handwash", "beta_abm"]).reset_index(drop=True)

out_all_csv = (
    f"sm_fit/theta_pairs_subset_cumGaussian_"
    f"{data_type}{init_env}{tau0}_handwashALL.csv"
)

df_all.to_csv(out_all_csv, index=False, encoding="utf-8")

print("\n전체 저장 완료 →", out_all_csv)
print(df_all.head())
print("done.")
# %%











# %% ==============================
# Step 4 validation for handwash-specific mappings
# beta_ABM = 0.039 example
# ABM mean trajectory vs surrogate trajectory
# ==============================

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# --------------------------------
# basic settings
# --------------------------------
data_type = "A"
init_env = 9
tau0 = 140

beta_target = 0.03847 # 여기를 베타 abm넣기
handwash_values = [0.8, 0.9, 0.95, 0.99]

beta_tag1 = 0.02
beta_tag2 = 0.06

start_month = "2017-01"

result_dir = "../result"
smfit_dir = "sm_fit"
fig_dir = os.path.join(smfit_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)


# --------------------------------
# helper functions
# --------------------------------
def wash_to_tag(w):
    return str(w).replace(".", "p")


def parse_vec(x):
    if isinstance(x, list):
        return np.array(x, dtype=float)
    if isinstance(x, np.ndarray):
        return x.astype(float)
    return np.array(ast.literal_eval(x), dtype=float)


def load_abm_summary(wash):
    wash_tag = wash_to_tag(wash)

    csv_path = os.path.join(
        result_dir,
        f"interv_prob_transmission_summary_{data_type}{init_env}{tau0}_{beta_tag1}-{beta_tag2}_handwash{wash_tag}.csv"
    )

    df = pd.read_csv(csv_path)
    df["mean_vec"] = df["mean"].apply(parse_vec)
    df["std_vec"] = df["std"].apply(parse_vec)
    df = df.sort_values("beta").reset_index(drop=True)

    return df


def load_mapping(wash):
    wash_tag = wash_to_tag(wash)

    csv_path = os.path.join(
        smfit_dir,
        f"theta_pairs_subset_cumGaussian_{data_type}{init_env}{tau0}_handwash{wash_tag}.csv"
    )

    df = pd.read_csv(csv_path)
    df = df.sort_values("beta_abm").reset_index(drop=True)

    return df


def infer_beta_sm_from_mapping(beta_target, df_map):
    """
    beta_ABM -> beta_SM interpolation
    """
    x = df_map["beta_abm"].to_numpy(float)
    y = df_map["theta_hat"].to_numpy(float)

    f = PchipInterpolator(x, y, extrapolate=False)
    beta_sm = float(f(beta_target))

    return beta_sm


def interpolate_abm_monthly(df_abm, beta_target):
    """
    ABM summary CSV contains values only on beta grid.
    For beta_target=0.039, interpolate monthly mean and std across beta.
    """
    beta_grid = df_abm["beta"].to_numpy(float)

    mean_mat = np.vstack(df_abm["mean_vec"].to_numpy())   # shape = (n_beta, n_month)
    std_mat  = np.vstack(df_abm["std_vec"].to_numpy())    # shape = (n_beta, n_month)

    n_months = mean_mat.shape[1]

    mean_interp = np.array([
        np.interp(beta_target, beta_grid, mean_mat[:, j])
        for j in range(n_months)
    ])

    std_interp = np.array([
        np.interp(beta_target, beta_grid, std_mat[:, j])
        for j in range(n_months)
    ])

    return mean_interp, std_interp


def get_month_axis(n_months, start_month="2017-01"):
    months = pd.period_range(start_month, periods=n_months, freq="M").to_timestamp()
    return months


def get_surrogate_monthly(beta_sm, wash, n_months):
    """
    run surrogate model with inferred beta_sm and p_wash=wash
    """
    days, daily_inc, monthly_df, comp_df = simulate_theta(
        beta=beta_sm,
        init_env=init_env,
        tau0=tau0,
        p_wash=wash
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


# --------------------------------
# single summary table
# --------------------------------
summary_rows = []


# --------------------------------
# 1) monthly comparison plots
# --------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, wash in zip(axes, handwash_values):

    df_abm = load_abm_summary(wash)
    df_map = load_mapping(wash)

    abm_mean, abm_std = interpolate_abm_monthly(df_abm, beta_target)
    n_months = len(abm_mean)

    beta_sm = infer_beta_sm_from_mapping(beta_target, df_map)
    sm_monthly = get_surrogate_monthly(beta_sm, wash, n_months)

    months = get_month_axis(n_months, start_month=start_month)

    ax.plot(months, abm_mean, "o-", linewidth=2, label="ABM mean")
    ax.fill_between(months, abm_mean - abm_std, abm_mean + abm_std, alpha=0.2, label="ABM ±1 SD")
    ax.plot(months, sm_monthly, "s--", linewidth=2, label="Surrogate")

    monthly_rmse = float(np.sqrt(np.mean((abm_mean - sm_monthly) ** 2)))

    ax.set_title(
        f"p_wash={wash}, beta_ABM={beta_target:.3f}\n"
        f"beta_SM={beta_sm:.4f}, RMSE={monthly_rmse:.4f}"
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly HAI")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    summary_rows.append({
        "p_wash": wash,
        "beta_abm_target": beta_target,
        "beta_sm_hat": beta_sm,
        "monthly_rmse": monthly_rmse
    })

plt.suptitle("Step 4 validation: ABM mean vs surrogate trajectory (monthly)", fontsize=16)
plt.tight_layout()

out_png_monthly = os.path.join(
    fig_dir,
    f"step4_validation_monthly_beta{str(beta_target).replace('.', 'p')}.png"
)
plt.savefig(out_png_monthly, dpi=300, bbox_inches="tight")
plt.show()

print("saved ->", out_png_monthly)


# --------------------------------
# 2) cumulative comparison plots
# --------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, wash in zip(axes, handwash_values):

    df_abm = load_abm_summary(wash)
    df_map = load_mapping(wash)

    abm_mean, abm_std = interpolate_abm_monthly(df_abm, beta_target)
    n_months = len(abm_mean)

    beta_sm = infer_beta_sm_from_mapping(beta_target, df_map)
    sm_monthly = get_surrogate_monthly(beta_sm, wash, n_months)

    abm_cum = np.cumsum(abm_mean)
    sm_cum = np.cumsum(sm_monthly)

    # monthly std를 이용한 approximate cumulative std
    abm_cum_std = np.sqrt(np.cumsum(abm_std ** 2))

    months = get_month_axis(n_months, start_month=start_month)

    ax.plot(months, abm_cum, "o-", linewidth=2, label="ABM cumulative mean")
    ax.fill_between(
        months,
        abm_cum - abm_cum_std,
        abm_cum + abm_cum_std,
        alpha=0.2,
        label="ABM cumulative ± approx SD"
    )
    ax.plot(months, sm_cum, "s--", linewidth=2, label="Surrogate cumulative")

    cumulative_rmse = float(np.sqrt(np.mean((abm_cum - sm_cum) ** 2)))

    ax.set_title(
        f"p_wash={wash}, beta_ABM={beta_target:.3f}\n"
        f"beta_SM={beta_sm:.4f}, RMSE={cumulative_rmse:.4f}"
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative HAI")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # summary_rows에 cumulative RMSE도 추가
    for row in summary_rows:
        if np.isclose(row["p_wash"], wash):
            row["cumulative_rmse"] = cumulative_rmse
            break

plt.suptitle("Step 4 validation: ABM mean vs surrogate trajectory (cumulative)", fontsize=16)
plt.tight_layout()

out_png_cum = os.path.join(
    fig_dir,
    f"step4_validation_cumulative_beta{str(beta_target).replace('.', 'p')}.png"
)
plt.savefig(out_png_cum, dpi=300, bbox_inches="tight")
plt.show()

print("saved ->", out_png_cum)


# --------------------------------
# 3) summary table save
# --------------------------------
df_summary = pd.DataFrame(summary_rows)

out_csv = os.path.join(
    smfit_dir,
    f"step4_validation_summary_beta{str(beta_target).replace('.', 'p')}.csv"
)

df_summary.to_csv(out_csv, index=False, encoding="utf-8")
print("saved ->", out_csv)
print(df_summary.round(6).to_string(index=False))
















# %%
# %% ================== 2) Actual ABM intervention vs SM surrogate comparison ==================
# actual ABM intervention file:
# interv_hcw_wash_rate_summary_A9140_0.7-0.99
#
# beta column in this file = hcw_wash_rate p
# Compare p = [0.8, 0.9, 0.95, 0.99]
#
# Required:
# - simulate_theta() must already be defined
# - theta_pairs_subset_cumGaussian_A9140_handwashALL.csv must already exist

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

# SMoRe ParS로 추정된 beta_ABM
beta_abm_target = 0.03847

# 비교할 handwash p 값
target_p_values = [0.8, 0.9, 0.95, 0.99]

start_month = "2017-01"

result_dir = "../result"
smfit_dir = "sm_fit"
fig_dir = os.path.join(smfit_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)


# -----------------------------
# actual ABM intervention summary file
# -----------------------------
actual_abm_csv = os.path.join(
    result_dir,
    "interv_hcw_wash_rate_summary_A9140_0.7-0.99.csv"
)

# 확장자 없이 저장되어 있으면 자동으로 이 파일명 사용
if not os.path.exists(actual_abm_csv):
    actual_abm_csv_no_ext = os.path.join(
        result_dir,
        "interv_hcw_wash_rate_summary_A9140_0.7-0.99"
    )

    if os.path.exists(actual_abm_csv_no_ext):
        actual_abm_csv = actual_abm_csv_no_ext
    else:
        raise FileNotFoundError(
            "actual ABM intervention summary file not found:\n"
            f"{actual_abm_csv}\n"
            f"{actual_abm_csv_no_ext}"
        )


# -----------------------------
# SMoRe ParS handwash-specific mapping file
# -----------------------------
mapping_csv = os.path.join(
    smfit_dir,
    f"theta_pairs_subset_cumGaussian_{data_type}{init_env}{tau0}_handwashALL.csv"
)

if not os.path.exists(mapping_csv):
    raise FileNotFoundError(mapping_csv)


# -----------------------------
# helper functions
# -----------------------------
def parse_vec(x):
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


def get_beta_sm_from_mapping(df_map, beta_abm_target, p_target):
    """
    handwash-specific beta_ABM -> beta_SM mapping에서
    beta_ABM target에 대응되는 beta_SM 계산.
    """

    sub = df_map.loc[
        np.isclose(df_map["handwash"].astype(float), p_target)
    ].copy()

    if sub.empty:
        raise ValueError(
            f"mapping에서 handwash={p_target} 없음.\n"
            f"available handwash = {sorted(df_map['handwash'].unique())}"
        )

    sub = sub.sort_values("beta_abm")

    x = sub["beta_abm"].to_numpy(dtype=float)
    y = sub["theta_hat"].to_numpy(dtype=float)

    if beta_abm_target < x.min() or beta_abm_target > x.max():
        print("[WARNING] beta_abm_target outside mapping range")
        print("mapping range:", x.min(), "~", x.max())
        print("target:", beta_abm_target)

    beta_sm = np.interp(beta_abm_target, x, y)

    if "theta_low" in sub.columns and "theta_high" in sub.columns:
        beta_sm_low = np.interp(
            beta_abm_target,
            x,
            sub["theta_low"].to_numpy(dtype=float)
        )

        beta_sm_high = np.interp(
            beta_abm_target,
            x,
            sub["theta_high"].to_numpy(dtype=float)
        )
    else:
        beta_sm_low = np.nan
        beta_sm_high = np.nan

    return float(beta_sm), float(beta_sm_low), float(beta_sm_high)


def get_actual_abm_by_p(df_actual, p_target):
    """
    actual ABM intervention summary에서 p_target에 해당하는 monthly mean/std를 가져옴.

    주의:
    이 파일의 beta column은 transmission beta가 아니라 hcw_wash_rate p.
    """

    sub = df_actual.loc[
        np.isclose(df_actual["p_wash"].astype(float), p_target)
    ].copy()

    if sub.empty:
        raise ValueError(
            f"actual ABM file에서 p={p_target} 없음.\n"
            f"available p = {df_actual['p_wash'].tolist()}"
        )

    row = sub.iloc[0]

    mean_vec = parse_vec(row["mean"])
    std_vec = parse_vec(row["std"])

    return mean_vec, std_vec


def run_sm_surrogate(beta_sm, p_wash, n_months):
    """
    beta_SM과 p_wash를 surrogate model에 넣고 monthly HAI 반환.
    simulate_theta는 위에서 이미 정의되어 있어야 함.
    """

    days, daily_inc, monthly_df, comp_df = simulate_theta(
        beta=beta_sm,
        init_env=init_env,
        tau0=tau0,
        p_wash=p_wash
    )

    months = get_months(n_months)

    mdf = monthly_df.copy()
    mdf["month"] = pd.to_datetime(mdf["month"].astype(str))
    mdf = mdf.set_index("month")

    sm_monthly = np.array([
        mdf["NewHAI_month"].get(m, 0.0)
        for m in months
    ], dtype=float)

    return sm_monthly


# -----------------------------
# load files
# -----------------------------
print("reading actual ABM intervention summary:")
print(actual_abm_csv)

df_actual = pd.read_csv(actual_abm_csv)

print("actual columns:")
print(list(df_actual.columns))
print(df_actual.head())


print("\nreading mapping:")
print(mapping_csv)

df_map = pd.read_csv(mapping_csv)

print("mapping columns:")
print(list(df_map.columns))
print(df_map.head())


# -----------------------------
# preprocess actual ABM file
# -----------------------------
# 여기서 beta column은 handwashing p로 사용
df_actual = df_actual.rename(columns={"beta": "p_wash"})

df_actual["p_wash"] = df_actual["p_wash"].astype(float)

print("\nactual p values in file:")
print(df_actual["p_wash"].tolist())

print("\ntarget p values:")
print(target_p_values)


# -----------------------------
# compare actual ABM intervention vs SM surrogate
# -----------------------------
comparison_rows = []
comparison_store = {}

for p in target_p_values:

    print("\n" + "=" * 80)
    print(f"Compare actual ABM vs SM for p_wash = {p}")
    print("=" * 80)

    # 1. actual ABM intervention result
    abm_monthly, abm_std = get_actual_abm_by_p(
        df_actual=df_actual,
        p_target=p
    )

    n_months = len(abm_monthly)

    # 2. beta_ABM -> beta_SM using p-specific mapping
    beta_sm, beta_sm_low, beta_sm_high = get_beta_sm_from_mapping(
        df_map=df_map,
        beta_abm_target=beta_abm_target,
        p_target=p
    )

    # 3. run surrogate
    sm_monthly = run_sm_surrogate(
        beta_sm=beta_sm,
        p_wash=p,
        n_months=n_months
    )

    # 4. cumulative
    abm_cum = np.cumsum(abm_monthly)
    sm_cum = np.cumsum(sm_monthly)

    abm_cum_std = np.sqrt(np.cumsum(abm_std ** 2))

    # 5. metrics
    monthly_rmse = float(np.sqrt(np.mean((abm_monthly - sm_monthly) ** 2)))
    cumulative_rmse = float(np.sqrt(np.mean((abm_cum - sm_cum) ** 2)))

    abm_total = float(abm_cum[-1])
    sm_total = float(sm_cum[-1])

    total_diff = sm_total - abm_total
    total_rel_diff = float(total_diff / abm_total * 100) if abm_total != 0 else np.nan

    comparison_rows.append({
        "p_wash": p,
        "beta_abm_target": beta_abm_target,
        "beta_sm": beta_sm,
        "beta_sm_low": beta_sm_low,
        "beta_sm_high": beta_sm_high,
        "actual_abm_total_hai": abm_total,
        "sm_total_hai": sm_total,
        "total_difference": total_diff,
        "total_relative_difference_percent": total_rel_diff,
        "monthly_rmse": monthly_rmse,
        "cumulative_rmse": cumulative_rmse,
        "init_env": init_env,
        "tau0": tau0
    })

    comparison_store[p] = {
        "p_wash": p,
        "beta_sm": beta_sm,
        "beta_sm_low": beta_sm_low,
        "beta_sm_high": beta_sm_high,
        "abm_monthly": abm_monthly,
        "abm_std": abm_std,
        "abm_cum": abm_cum,
        "abm_cum_std": abm_cum_std,
        "sm_monthly": sm_monthly,
        "sm_cum": sm_cum,
        "n_months": n_months
    }

    print(
        f"p={p}, beta_ABM={beta_abm_target}, beta_SM={beta_sm:.4f}, "
        f"ABM total={abm_total:.3f}, SM total={sm_total:.3f}, "
        f"cum_RMSE={cumulative_rmse:.3f}"
    )


df_compare = pd.DataFrame(comparison_rows)
df_compare = df_compare.sort_values("p_wash").reset_index(drop=True)

out_compare_csv = os.path.join(
    smfit_dir,
    f"actual_intervention_vs_SM_betaABM{str(beta_abm_target).replace('.', 'p')}_"
    f"{data_type}{init_env}{tau0}_handwash_selected.csv"
)

df_compare.to_csv(out_compare_csv, index=False, encoding="utf-8")

print("\nsaved comparison csv:")
print(out_compare_csv)
print(df_compare.round(4).to_string(index=False))
# %%
# %% ================== Plot actual ABM intervention vs SM: monthly ==================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, p in zip(axes, sorted(comparison_store.keys())):

    item = comparison_store[p]

    months = get_months(item["n_months"])

    ax.plot(
        months,
        item["abm_monthly"],
        marker="o",
        linewidth=2,
        label="Actual ABM intervention mean"
    )

    ax.fill_between(
        months,
        item["abm_monthly"] - item["abm_std"],
        item["abm_monthly"] + item["abm_std"],
        alpha=0.2,
        label="ABM ± 1 SD"
    )

    ax.plot(
        months,
        item["sm_monthly"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="SM surrogate"
    )

    ax.set_title(
        f"p_wash={p}, β_ABM={beta_abm_target:.3f}\n"
        f"β_SM={item['beta_sm']:.3f}"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly HAI")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(
    "Actual ABM intervention vs SM surrogate: monthly HAI",
    fontsize=16
)

plt.tight_layout()

out_fig = os.path.join(
    fig_dir,
    f"actual_intervention_vs_SM_monthly_betaABM{str(beta_abm_target).replace('.', 'p')}_"
    f"{data_type}{init_env}{tau0}_handwash_selected.png"
)

plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.show()

print("saved:", out_fig)
# %%
# %% ================== Plot actual ABM intervention vs SM: cumulative ==================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, p in zip(axes, sorted(comparison_store.keys())):

    item = comparison_store[p]

    months = get_months(item["n_months"])

    ax.plot(
        months,
        item["abm_cum"],
        marker="o",
        linewidth=2,
        label="Actual ABM intervention cumulative"
    )

    ax.fill_between(
        months,
        item["abm_cum"] - item["abm_cum_std"],
        item["abm_cum"] + item["abm_cum_std"],
        alpha=0.2,
        label="ABM cumulative ± approx SD"
    )

    ax.plot(
        months,
        item["sm_cum"],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="SM surrogate cumulative"
    )

    ax.set_title(
        f"p_wash={p}, β_ABM={beta_abm_target:.3f}\n"
        f"β_SM={item['beta_sm']:.3f}"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative HAI")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(
    "Actual ABM intervention vs SM surrogate: cumulative HAI",
    fontsize=16
)

plt.tight_layout()

out_fig = os.path.join(
    fig_dir,
    f"actual_intervention_vs_SM_cumulative_betaABM{str(beta_abm_target).replace('.', 'p')}_"
    f"{data_type}{init_env}{tau0}_handwash_selected.png"
)

plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.show()

print("saved:", out_fig)
# %%
