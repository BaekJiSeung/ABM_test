
# %%
import os
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. Period A data
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
    "2018-05-25"
]


def _make_AI_from_dates(pi_dates, days):

    T = len(days)
    A = np.zeros(T)

    idx_map = {
        d: i for i, d in enumerate(days)
    }

    for d in pi_dates:
        ts = pd.to_datetime(d)
        idx = idx_map.get(ts, None)

        if idx is not None:
            A[idx] += 1.0

    return A


# ============================================================
# 2. Surrogate Model
# ============================================================

def simulate_theta(
    beta,
    init_env,
    tau0,
    p_wash,
    monthly_PI=monthly_PI,
    pi_dates=PI_dates
):

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

    start = pd.Period(
        monthly_PI.index.min(),
        freq="M"
    ).to_timestamp(how="start")

    end = pd.Period(
        monthly_PI.index.max(),
        freq="M"
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

    for t in range(T):

        if t > 0 and (t + tau0) % deep_clean_period == 0:
            Env_C[t] *= (1 - cleaning_eff)

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

            B_tot = max(
                PS_sh + PH_sh + PH_iso + PI,
                1e-9
            )

            lam_HP_sh = beta * (HC / N_H)

            lam_PH = beta * (
                (PH_sh + iso_factor * PH_iso + PI)
                / B_tot
            )

            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            hai_sh = lam_HP_sh * PS_sh * dt
            move_HA = sigma * PH_sh * dt

            outS_sh = mu_S * PS_sh * dt
            outH_sh = mu_HAI * PH_sh * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI = mu_I * PI * dt

            leaving = (
                outS_sh + outH_sh + outH_iso + outI
            )

            total_P = (
                PS_sh + PH_sh + PH_iso + PI
            )

            AS_tot = max(
                0.0,
                C_total - (total_P - leaving)
            )

            AS_sh = AS_tot

            PS_sh += AS_sh - outS_sh - hai_sh
            PH_sh += hai_sh - outH_sh - move_HA
            PH_iso += move_HA - outH_iso
            PI += -outI

            PS_sh = np.clip(PS_sh, 0, C_sh)
            PH_sh = np.clip(PH_sh, 0, C_sh)
            PH_iso = np.clip(PH_iso, 0, C_iso)
            PI = np.clip(PI, 0, C_total)

            new_H = (
                (lam_PH + lam_EH)
                * (N_H - HC)
                * dt
            )

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
        df.groupby(
            df["date"].dt.to_period("M")
        )["NewHAI"]
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
        "Env_C": Env_C
    }).set_index("date")

    return days, NewHAI_day, monthly, comp_df


# ============================================================
# 3. Settings
# ============================================================

data_type = "A"

init_env = 9
tau0 = 140

beta_abm_target = 0.03847

handwash_values = [
    0.80,
    0.90,
    0.95,
    0.99
]

start_month = "2017-01"

result_dir = "../result"
smfit_dir = "sm_fit"

output_dir = os.path.join(
    smfit_dir,
    "export_period_A_handwashing"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 4. Helper Functions
# ============================================================

def parse_vec(x):

    if isinstance(x, np.ndarray):
        return x.astype(float)

    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float)

    return np.asarray(
        ast.literal_eval(str(x)),
        dtype=float
    )


def get_months(n_months):

    return pd.period_range(
        start_month,
        periods=n_months,
        freq="M"
    ).to_timestamp()


def get_beta_sm_from_mapping(
    df_map,
    beta_abm_target,
    p_target
):

    sub = df_map.loc[
        np.isclose(
            df_map["handwash"].astype(float),
            p_target
        )
    ].copy()

    if sub.empty:
        raise ValueError(
            f"No mapping for handwash={p_target}"
        )

    sub = sub.sort_values("beta_abm")

    x = sub["beta_abm"].to_numpy(float)
    y = sub["theta_hat"].to_numpy(float)

    if len(x) < 2:
        raise ValueError(
            f"Insufficient mapping points for {p_target}"
        )

    if not (x.min() <= beta_abm_target <= x.max()):
        raise ValueError(
            f"beta_ABM outside mapping range for {p_target}"
        )

    beta_sm = np.interp(
        beta_abm_target,
        x,
        y
    )

    beta_sm_low = np.nan
    beta_sm_high = np.nan

    if "theta_low" in sub.columns:
        beta_sm_low = np.interp(
            beta_abm_target,
            x,
            sub["theta_low"].to_numpy(float)
        )

    if "theta_high" in sub.columns:
        beta_sm_high = np.interp(
            beta_abm_target,
            x,
            sub["theta_high"].to_numpy(float)
        )

    return (
        float(beta_sm),
        float(beta_sm_low),
        float(beta_sm_high)
    )


def get_actual_abm_by_p(df_actual, p_target):

    sub = df_actual.loc[
        np.isclose(
            df_actual["p_wash"].astype(float),
            p_target
        )
    ]

    if len(sub) != 1:
        raise ValueError(
            f"Expected one ABM row for p={p_target}, "
            f"found {len(sub)}"
        )

    row = sub.iloc[0]

    return (
        parse_vec(row["mean"]),
        parse_vec(row["std"])
    )


def run_sm_surrogate(beta_sm, p_wash, n_months):

    days, daily_inc, monthly_df, comp_df = (
        simulate_theta(
            beta=beta_sm,
            init_env=init_env,
            tau0=tau0,
            p_wash=p_wash
        )
    )

    months = get_months(n_months)

    mdf = monthly_df.copy()

    mdf["month"] = pd.to_datetime(
        mdf["month"].astype(str)
    )

    mdf = mdf.set_index("month")

    sm_monthly = np.array([
        mdf["NewHAI_month"].get(m, 0.0)
        for m in months
    ], dtype=float)

    return sm_monthly


# ============================================================
# 5. Load ABM Intervention Results
# ============================================================

actual_abm_csv = os.path.join(
    result_dir,
    "interv_hcw_wash_rate_summary_A9140_0.7-0.99.csv"
)

if not os.path.exists(actual_abm_csv):

    no_ext = os.path.splitext(actual_abm_csv)[0]

    if os.path.exists(no_ext):
        actual_abm_csv = no_ext
    else:
        raise FileNotFoundError(actual_abm_csv)


df_actual = pd.read_csv(actual_abm_csv)

if "p_wash" not in df_actual.columns:
    df_actual = df_actual.rename(
        columns={"beta": "p_wash"}
    )

df_actual["p_wash"] = (
    df_actual["p_wash"].astype(float)
)


# ============================================================
# 6. Load Parameter Mapping
# ============================================================

mapping_csv = os.path.join(
    smfit_dir,
    "theta_pairs_subset_cumGaussian_A9140_handwashALL.csv"
)

if not os.path.exists(mapping_csv):
    raise FileNotFoundError(mapping_csv)

df_map = pd.read_csv(mapping_csv)


# ============================================================
# 7. Actual ABM vs Surrogate Model
# ============================================================

comparison_rows = []
timeseries_rows = []
comparison_store = {}

for p in handwash_values:

    print("=" * 60)
    print(f"Handwashing = {p}")
    print("=" * 60)

    # Actual ABM
    abm_monthly, abm_std = get_actual_abm_by_p(
        df_actual,
        p
    )

    n_months = len(abm_monthly)

    if n_months != 19:
        raise ValueError(
            f"Expected 19 months, got {n_months}"
        )

    months = get_months(n_months)

    # Parameter Mapping
    beta_sm, beta_sm_low, beta_sm_high = (
        get_beta_sm_from_mapping(
            df_map,
            beta_abm_target,
            p
        )
    )

    # Surrogate Model
    sm_monthly = run_sm_surrogate(
        beta_sm,
        p,
        n_months
    )

    # Cumulative HAI
    abm_cum = np.cumsum(abm_monthly)
    sm_cum = np.cumsum(sm_monthly)

    # Approximate cumulative SD
    abm_cum_std = np.sqrt(
        np.cumsum(abm_std ** 2)
    )

    # RMSE
    monthly_rmse = float(
        np.sqrt(
            np.mean(
                (abm_monthly - sm_monthly) ** 2
            )
        )
    )

    cumulative_rmse = float(
        np.sqrt(
            np.mean(
                (abm_cum - sm_cum) ** 2
            )
        )
    )

    abm_total = float(abm_cum[-1])
    sm_total = float(sm_cum[-1])

    # --------------------------------
    # Summary Results
    # --------------------------------

    comparison_rows.append({
        "Period": data_type,
        "Handwashing_Rate": p,
        "Beta_ABM": beta_abm_target,
        "Beta_SM": beta_sm,
        "Beta_SM_Low": beta_sm_low,
        "Beta_SM_High": beta_sm_high,
        "ABM_Total_HAI": abm_total,
        "SM_Total_HAI": sm_total,
        "Total_Difference": sm_total - abm_total,
        "Monthly_RMSE": monthly_rmse,
        "Cumulative_RMSE": cumulative_rmse
    })

    # --------------------------------
    # Monthly Time Series
    # --------------------------------

    for i, month in enumerate(months):

        timeseries_rows.append({
            "Period": data_type,
            "Month": month.strftime("%Y-%m"),
            "Handwashing_Rate": p,
            "Beta_ABM": beta_abm_target,
            "Beta_SM": beta_sm,
            "ABM_Monthly_Mean": abm_monthly[i],
            "ABM_Monthly_SD": abm_std[i],
            "SM_Monthly": sm_monthly[i],
            "ABM_Cumulative": abm_cum[i],
            "ABM_Cumulative_SD_Approx": abm_cum_std[i],
            "SM_Cumulative": sm_cum[i]
        })

    # Store for plotting
    comparison_store[p] = {
        "months": months,
        "abm_monthly": abm_monthly,
        "abm_std": abm_std,
        "sm_monthly": sm_monthly,
        "abm_cum": abm_cum,
        "abm_cum_std": abm_cum_std,
        "sm_cum": sm_cum,
        "beta_sm": beta_sm
    }

    print(
        f"beta_SM={beta_sm:.5f}, "
        f"ABM total={abm_total:.3f}, "
        f"SM total={sm_total:.3f}, "
        f"Cumulative RMSE={cumulative_rmse:.4f}"
    )


# ============================================================
# 8. Save CSV Files
# ============================================================

df_summary = pd.DataFrame(comparison_rows)
df_timeseries = pd.DataFrame(timeseries_rows)

summary_csv = os.path.join(
    output_dir,
    "period_A_handwashing_summary.csv"
)

timeseries_csv = os.path.join(
    output_dir,
    "period_A_handwashing_timeseries.csv"
)

df_summary.to_csv(
    summary_csv,
    index=False,
    encoding="utf-8-sig"
)

df_timeseries.to_csv(
    timeseries_csv,
    index=False,
    encoding="utf-8-sig"
)

print("Saved:", summary_csv)
print("Saved:", timeseries_csv)


# ============================================================
# 9. Monthly HAI Figure
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 9)
)

axes = axes.flatten()

for ax, p in zip(axes, handwash_values):

    item = comparison_store[p]

    months = item["months"]
    abm = item["abm_monthly"]
    std = item["abm_std"]
    sm = item["sm_monthly"]

    ax.plot(
        months, abm,
        "o-",
        label="ABM Mean"
    )

    ax.fill_between(
        months,
        abm - std,
        abm + std,
        alpha=0.2,
        label="ABM ± 1 SD"
    )

    ax.plot(
        months, sm,
        "s--",
        label="Surrogate"
    )

    ax.set_title(f"Handwashing = {p}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly HAI")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "period_A_handwashing_monthly.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. Cumulative HAI Figure
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 9)
)

axes = axes.flatten()

for ax, p in zip(axes, handwash_values):

    item = comparison_store[p]

    months = item["months"]
    abm = item["abm_cum"]
    std = item["abm_cum_std"]
    sm = item["sm_cum"]

    ax.plot(
        months, abm,
        "o-",
        label="ABM Cumulative"
    )

    ax.fill_between(
        months,
        abm - std,
        abm + std,
        alpha=0.2,
        label="ABM Cumulative ± Approx. SD"
    )

    ax.plot(
        months, sm,
        "s--",
        label="Surrogate Cumulative"
    )

    ax.set_title(f"Handwashing = {p}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative HAI")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "period_A_handwashing_cumulative.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. Save Run Settings
# ============================================================

settings = {
    "Period": "A",
    "Start_Month": start_month,
    "Number_of_Months": 19,
    "Initial_Environment": init_env,
    "Cleaning_Offset": tau0,
    "Beta_ABM_Target": beta_abm_target,
    "Handwashing_Values": handwash_values,
    "ABM_Source": actual_abm_csv,
    "Mapping_Source": mapping_csv,
    "Cumulative_SD_Note":
        "Approximate SD assuming zero covariance "
        "between monthly incidence values."
}

settings_path = os.path.join(
    output_dir,
    "run_settings.json"
)

with open(
    settings_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        settings,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nAll results saved to:", output_dir)
print(df_summary.round(5))

# %%
