
# %%
# ============================================================
# Period A - Handwashing
# Actual ABM intervention vs Surrogate Model
#
# Original comparison + monthly/cumulative CSV export
# Mapping fitting is NOT repeated.
# ABM interpolation is NOT used.
# ============================================================

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    "2018-05-25",
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
# 2. Original Surrogate Model
# ============================================================

def simulate_theta(
    beta,
    init_env,
    tau0,
    p_wash,
    monthly_PI=monthly_PI,
    pi_dates=PI_dates
):
    """
    Difference-equation surrogate model.

    beta     : beta_SM
    init_env : initial environmental contamination
    tau0     : cleaning phase offset
    p_wash   : HCW handwashing rate

    p_wash affects HCW contamination only.
    Environmental cleaning efficiency is fixed at 0.90.
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

        # Environmental deep cleaning
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            xx = Env_C[t]
            Env_C[t] = (1 - cleaning_eff) * xx

        PS_sh = P_S_sh[t]
        PH_sh = P_HAI_sh[t]
        PH_iso = P_HAI_iso[t]
        PI = P_I[t]

        HC = H_C[t]
        EC = Env_C[t]

        # Imported colonized patients
        inc = A_I_day[t]

        if inc > 0:

            total_P = PS_sh + PH_sh + PH_iso + PI
            stay_free = max(0.0, C_total - total_P)
            inc_eff = min(inc, stay_free)

            taken = min(PS_sh, inc_eff)

            PS_sh -= taken
            PI += taken

        # Within-day contacts
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
                outS_sh
                + outH_sh
                + outH_iso
                + outI
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

            # HCW handwashing
            new_H = (
                (lam_PH + lam_EH)
                * (N_H - HC)
                * dt
            )

            HC = (HC + new_H) * (1 - p_wash)
            HC = np.clip(HC, 0, N_H)

            # Environmental contamination
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
        "Env_C": Env_C,
    }).set_index("date")

    return days, NewHAI_day, monthly, comp_df


# ============================================================
# 3. Comparison settings
# ============================================================

data_type = "A"

init_env = 9
tau0 = 140

beta_abm_target = 0.03847

target_p_values = [
    0.8,
    0.9,
    0.95,
    0.99
]

start_month = "2017-01"

result_dir = "../result"
smfit_dir = "sm_fit"

fig_dir = os.path.join(
    smfit_dir,
    "figures"
)

os.makedirs(fig_dir, exist_ok=True)


# ============================================================
# 4. Input files
# ============================================================

# Actual ABM intervention results
actual_abm_csv = os.path.join(
    result_dir,
    "interv_hcw_wash_rate_summary_A9140_0.7-0.99.csv"
)

if not os.path.exists(actual_abm_csv):

    actual_abm_csv_no_ext = os.path.join(
        result_dir,
        "interv_hcw_wash_rate_summary_A9140_0.7-0.99"
    )

    if os.path.exists(actual_abm_csv_no_ext):
        actual_abm_csv = actual_abm_csv_no_ext
    else:
        raise FileNotFoundError(
            "Actual ABM intervention file not found:\n"
            f"{actual_abm_csv}\n"
            f"{actual_abm_csv_no_ext}"
        )


# Existing beta_ABM -> beta_SM mapping
mapping_csv = os.path.join(
    smfit_dir,
    f"theta_pairs_subset_cumGaussian_"
    f"{data_type}{init_env}{tau0}_handwashALL.csv"
)

if not os.path.exists(mapping_csv):
    raise FileNotFoundError(mapping_csv)


# ============================================================
# 5. Helper functions
# ============================================================

def parse_vec(x):

    if isinstance(x, list):
        return np.array(x, dtype=float)

    if isinstance(x, np.ndarray):
        return x.astype(float)

    return np.array(
        ast.literal_eval(x),
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
    """
    Handwashing-specific beta_ABM -> beta_SM mapping.
    """

    sub = df_map.loc[
        np.isclose(
            df_map["handwash"].astype(float),
            p_target
        )
    ].copy()

    if sub.empty:
        raise ValueError(
            f"Mapping missing handwash={p_target}"
        )

    sub = sub.sort_values("beta_abm")

    x = sub["beta_abm"].to_numpy(dtype=float)
    y = sub["theta_hat"].to_numpy(dtype=float)

    if (
        beta_abm_target < x.min()
        or beta_abm_target > x.max()
    ):
        print(
            "[WARNING] beta_abm_target "
            "outside mapping range"
        )

    # Original mapping interpolation
    beta_sm = np.interp(
        beta_abm_target,
        x,
        y
    )

    if (
        "theta_low" in sub.columns
        and "theta_high" in sub.columns
    ):

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

    return (
        float(beta_sm),
        float(beta_sm_low),
        float(beta_sm_high)
    )


def get_actual_abm_by_p(
    df_actual,
    p_target
):
    """
    Read actual ABM intervention result directly.

    No ABM interpolation.

    The original ABM file's beta column represents
    handwashing rate, not transmission beta.
    """

    sub = df_actual.loc[
        np.isclose(
            df_actual["p_wash"].astype(float),
            p_target
        )
    ].copy()

    if sub.empty:
        raise ValueError(
            f"Actual ABM missing p={p_target}"
        )

    row = sub.iloc[0]

    mean_vec = parse_vec(row["mean"])
    std_vec = parse_vec(row["std"])

    return mean_vec, std_vec


def run_sm_surrogate(
    beta_sm,
    p_wash,
    n_months
):
    """
    Run the original surrogate model.
    """

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
# 6. Load existing ABM and mapping files
# ============================================================

print("Reading actual ABM:")
print(actual_abm_csv)

df_actual = pd.read_csv(actual_abm_csv)

print("Reading mapping:")
print(mapping_csv)

df_map = pd.read_csv(mapping_csv)

# Original ABM file uses beta as handwashing p
df_actual = df_actual.rename(
    columns={"beta": "p_wash"}
)

df_actual["p_wash"] = (
    df_actual["p_wash"].astype(float)
)

print("\nAvailable ABM p values:")
print(df_actual["p_wash"].tolist())


# ============================================================
# 7. Actual ABM vs Surrogate Model
# ============================================================

comparison_rows = []
comparison_store = {}

for p in target_p_values:

    print("\n" + "=" * 60)
    print(f"Handwashing p = {p}")
    print("=" * 60)

    # 1. Actual ABM results
    abm_monthly, abm_std = get_actual_abm_by_p(
        df_actual=df_actual,
        p_target=p
    )

    n_months = len(abm_monthly)

    # 2. beta_ABM -> beta_SM mapping
    beta_sm, beta_sm_low, beta_sm_high = (
        get_beta_sm_from_mapping(
            df_map=df_map,
            beta_abm_target=beta_abm_target,
            p_target=p
        )
    )

    # 3. Run surrogate model
    sm_monthly = run_sm_surrogate(
        beta_sm=beta_sm,
        p_wash=p,
        n_months=n_months
    )

    # 4. Cumulative HAI
    abm_cum = np.cumsum(abm_monthly)
    sm_cum = np.cumsum(sm_monthly)

    # Original approximate cumulative SD
    abm_cum_std = np.sqrt(
        np.cumsum(abm_std ** 2)
    )

    # 5. Metrics
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

    total_diff = sm_total - abm_total

    total_rel_diff = (
        float(total_diff / abm_total * 100)
        if abm_total != 0
        else np.nan
    )

    comparison_rows.append({
        "p_wash": p,
        "beta_abm_target": beta_abm_target,
        "beta_sm": beta_sm,
        "beta_sm_low": beta_sm_low,
        "beta_sm_high": beta_sm_high,
        "actual_abm_total_hai": abm_total,
        "sm_total_hai": sm_total,
        "total_difference": total_diff,
        "total_relative_difference_percent":
            total_rel_diff,
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
        f"p={p}, "
        f"beta_ABM={beta_abm_target}, "
        f"beta_SM={beta_sm:.4f}"
    )

    print(
        f"ABM total={abm_total:.3f}, "
        f"SM total={sm_total:.3f}"
    )

    print(
        f"Monthly RMSE={monthly_rmse:.4f}, "
        f"Cumulative RMSE={cumulative_rmse:.4f}"
    )


# ============================================================
# 8. Save original comparison summary
# ============================================================

df_compare = pd.DataFrame(comparison_rows)

df_compare = (
    df_compare
    .sort_values("p_wash")
    .reset_index(drop=True)
)

out_compare_csv = os.path.join(
    smfit_dir,
    f"actual_intervention_vs_SM_betaABM"
    f"{str(beta_abm_target).replace('.', 'p')}_"
    f"{data_type}{init_env}{tau0}_"
    f"handwash_selected.csv"
)

df_compare.to_csv(
    out_compare_csv,
    index=False,
    encoding="utf-8"
)

print("\nSaved comparison summary:")
print(out_compare_csv)


# ============================================================
# 9. NEW: Save ABM and SM time-series CSV
# ============================================================

export_rows = []

for p in sorted(comparison_store.keys()):

    item = comparison_store[p]

    months = get_months(
        item["n_months"]
    )

    for i, month in enumerate(months):

        export_rows.append({
            "month": month.strftime("%Y-%m"),
            "p_wash": p,
            "beta_abm": beta_abm_target,
            "beta_sm": item["beta_sm"],

            "ABM_monthly_mean":
                item["abm_monthly"][i],

            "ABM_monthly_std":
                item["abm_std"][i],

            "SM_monthly":
                item["sm_monthly"][i],

            "ABM_cumulative":
                item["abm_cum"][i],

            "ABM_cumulative_std_approx":
                item["abm_cum_std"][i],

            "SM_cumulative":
                item["sm_cum"][i]
        })

df_export = pd.DataFrame(export_rows)

out_timeseries_csv = os.path.join(
    smfit_dir,
    "actual_ABM_vs_SM_handwashing_A_timeseries.csv"
)

df_export.to_csv(
    out_timeseries_csv,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved time-series CSV:")
print(out_timeseries_csv)

print("\nPreview:")
print(df_export.head())


# ============================================================
# 10. Monthly comparison plots
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 9)
)

axes = axes.flatten()

for ax, p in zip(
    axes,
    sorted(comparison_store.keys())
):

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
        f"p_wash={p}, "
        f"beta_ABM={beta_abm_target:.5f}\n"
        f"beta_SM={item['beta_sm']:.4f}"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly HAI")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(
    "Actual ABM intervention vs SM: Monthly HAI",
    fontsize=16
)

plt.tight_layout()

out_monthly_png = os.path.join(
    fig_dir,
    "actual_ABM_vs_SM_handwashing_A_monthly.png"
)

plt.savefig(
    out_monthly_png,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. Cumulative comparison plots
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 9)
)

axes = axes.flatten()

for ax, p in zip(
    axes,
    sorted(comparison_store.keys())
):

    item = comparison_store[p]
    months = get_months(item["n_months"])

    ax.plot(
        months,
        item["abm_cum"],
        marker="o",
        linewidth=2,
        label="Actual ABM cumulative"
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
        label="SM cumulative"
    )

    ax.set_title(
        f"p_wash={p}, "
        f"beta_ABM={beta_abm_target:.5f}\n"
        f"beta_SM={item['beta_sm']:.4f}"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative HAI")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(
    "Actual ABM intervention vs SM: Cumulative HAI",
    fontsize=16
)

plt.tight_layout()

out_cumulative_png = os.path.join(
    fig_dir,
    "actual_ABM_vs_SM_handwashing_A_cumulative.png"
)

plt.savefig(
    out_cumulative_png,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 12. Finished
# ============================================================

print("\n" + "=" * 60)
print("HANDWASHING A COMPLETED")
print("=" * 60)

print("Comparison summary:")
print(out_compare_csv)

print("\nTime-series CSV:")
print(out_timeseries_csv)

print("\nMonthly plot:")
print(out_monthly_png)

print("\nCumulative plot:")
print(out_cumulative_png)

# %%
