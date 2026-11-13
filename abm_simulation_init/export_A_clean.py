
# %% ==========================================================
# Cleaning A - Actual ABM vs mapped SM
# Original model + comparison + time-series CSV export
# No mapping refitting
# =============================================================

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================
# 1. Period A imported colonized patient data
# =============================================================

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


# =============================================================
# 2. Original Cleaning Surrogate Model
# =============================================================

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
    Original difference-equation surrogate model.

    beta      : beta_SM
    init_env  : initial environmental contamination
    clean_day : deep cleaning period
    tau0      : cleaning phase offset
    p_wash    : fixed HCW handwashing rate
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

    # Preserve the value actually used in the original code
    isol_time = 7.0
    sigma = 1.0 / isol_time

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

    for t in range(T):

        # Deep cleaning
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            Env_C[t] = (1 - cleaning_eff) * Env_C[t]

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
                (PH_sh + iso_factor * PH_iso + PI) / B_tot
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

            # Fixed handwashing rate
            new_H = (
                (lam_PH + lam_EH) * (N_H - HC) * dt
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


# =============================================================
# 3. Original settings
# =============================================================

data_type = "A"
init_env = 9
first_clean_day = 20

cleaning_values = [60, 90, 180, 360]

p_wash_fixed = 0.9
beta_abm_target = 0.03847

days_per_month = 30
n_months = 19
start_month = "2017-01"

smfit_dir = "sm_fit"
result_dir = "../result"

fig_dir = os.path.join(smfit_dir, "figures")

os.makedirs(smfit_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

mapping_csv = os.path.join(
    smfit_dir,
    "theta_pairs_subset_cumGaussian_A9_firstclean20_cleaningALL.csv"
)

actual_raw_csv = os.path.join(
    result_dir,
    "interv_prob_transmission_LONG_A9_firstclean20_0.03847-0.03847_cleaningALL.csv"
)


# =============================================================
# 4. Original helper functions
# =============================================================

def parse_vec(x):

    if isinstance(x, list):
        return np.array(x, dtype=float)

    if isinstance(x, np.ndarray):
        return x.astype(float)

    if pd.isna(x):
        return np.array([], dtype=float)

    return np.array(
        ast.literal_eval(x),
        dtype=float
    )


def model_monthly_and_cum(
    theta,
    clean_day,
    tau0,
    months,
    init_env=init_env,
    p_wash=p_wash_fixed
):

    days, daily_inc, monthly_df, comp_df = (
        simulate_theta(
            beta=theta,
            init_env=init_env,
            clean_day=clean_day,
            tau0=tau0,
            p_wash=p_wash
        )
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

    cum = np.cumsum(monthly)

    return monthly, cum


def daily_to_monthly_local(
    daily_series,
    days_per_month=30,
    n_months=19
):

    arr = np.array(daily_series, dtype=float)

    needed = days_per_month * n_months
    arr = arr[:needed]

    m = len(arr) // days_per_month
    arr = arr[:m * days_per_month]

    return arr.reshape(
        m, days_per_month
    ).sum(axis=1)


def summarize_actual_abm_cleaning(df_raw, clean_day):

    sub = df_raw[
        df_raw["cleaningDay"].astype(float)
        == float(clean_day)
    ].copy()

    if sub.empty:
        raise ValueError(
            f"Actual ABM file missing cleaningDay={clean_day}"
        )

    monthly_runs = []

    for s in sub["HCW_related_infecs"].dropna():

        daily = parse_vec(s)

        monthly = daily_to_monthly_local(
            daily,
            days_per_month=days_per_month,
            n_months=n_months
        )

        monthly_runs.append(monthly)

    monthly_arr = np.array(
        monthly_runs,
        dtype=float
    )

    mean_ = monthly_arr.mean(axis=0)
    std_ = monthly_arr.std(axis=0, ddof=0)

    return mean_, std_, len(monthly_runs)


# =============================================================
# 5. Read existing Mapping and Actual ABM files
# =============================================================

if not os.path.exists(mapping_csv):
    raise FileNotFoundError(mapping_csv)

if not os.path.exists(actual_raw_csv):
    raise FileNotFoundError(actual_raw_csv)

print("Reading mapping:", mapping_csv)
df_map = pd.read_csv(mapping_csv)

print("Reading actual ABM:", actual_raw_csv)
df_actual_raw = pd.read_csv(actual_raw_csv)


# =============================================================
# 6. Original cleaning-specific beta_ABM -> beta_SM mapping
# =============================================================

mapped_rows = []

plt.figure(figsize=(8, 6))

for clean_day in cleaning_values:

    sub = (
        df_map[
            df_map["cleaningDay"].astype(float)
            == float(clean_day)
        ]
        .sort_values("beta_abm")
        .reset_index(drop=True)
    )

    if sub.empty:
        print(
            f"[WARNING] No mapping for cleaningDay={clean_day}"
        )
        continue

    x = sub["beta_abm"].to_numpy(float)
    y = sub["theta_hat"].to_numpy(float)

    has_ci = (
        "theta_low" in sub.columns
        and "theta_high" in sub.columns
    )

    # Original beta_SM mapping interpolation
    beta_sm_target = float(
        np.interp(beta_abm_target, x, y)
    )

    mapped_rows.append({
        "cleaningDay": clean_day,
        "first_clean_day": first_clean_day,
        "tau_offset_days": clean_day - first_clean_day,
        "beta_abm_target": beta_abm_target,
        "beta_sm_mapped": beta_sm_target
    })

    plt.plot(
        x, y,
        "o-",
        linewidth=2,
        markersize=5,
        label=f"cleaningDay={clean_day}"
    )

    if has_ci:
        yl = sub["theta_low"].to_numpy(float)
        yh = sub["theta_high"].to_numpy(float)

        plt.fill_between(
            x, yl, yh, alpha=0.12
        )

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

plt.xlabel(r"$\beta_{\mathrm{ABM}}$")
plt.ylabel(r"$\beta_{\mathrm{SM}}$")

plt.title(
    "Cleaning-specific beta_ABM to beta_SM mapping"
)

plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()

mapping_png = os.path.join(
    fig_dir,
    "mapping_cleaning_betaABM_to_betaSM_betaABM0p03847.png"
)

plt.savefig(
    mapping_png,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

df_mapped = pd.DataFrame(mapped_rows)

mapped_csv = os.path.join(
    smfit_dir,
    "mapped_betaSM_betaABM0p03847_A9_firstclean20_cleaning.csv"
)

df_mapped.to_csv(
    mapped_csv,
    index=False,
    encoding="utf-8"
)

print("\nMapped beta_SM:")
print(df_mapped.to_string(index=False))


# =============================================================
# 7. Actual ABM vs mapped SM comparison
# =============================================================

compare_rows = []
timeseries_rows = []

fig_m, axes_m = plt.subplots(
    2, 2, figsize=(12, 8), sharex=True
)

axes_m = axes_m.ravel()

fig_c, axes_c = plt.subplots(
    2, 2, figsize=(12, 8), sharex=True
)

axes_c = axes_c.ravel()

for i, clean_day in enumerate(cleaning_values):

    print("\n" + "=" * 60)
    print(f"Cleaning Day = {clean_day}")
    print("=" * 60)

    tau0 = clean_day - first_clean_day

    # Actual ABM beta_ABM=0.03847
    # Original method: daily runs -> 30-day monthly
    abm_mean, abm_std, n_runs = (
        summarize_actual_abm_cleaning(
            df_actual_raw,
            clean_day
        )
    )

    beta_used = beta_abm_target

    # Cleaning-specific mapped beta_SM
    row_map = df_mapped[
        df_mapped["cleaningDay"] == clean_day
    ]

    if row_map.empty:
        print(
            f"[WARNING] No mapped beta_SM for {clean_day}"
        )
        continue

    beta_sm = float(
        row_map.iloc[0]["beta_sm_mapped"]
    )

    months = pd.period_range(
        start=start_month,
        periods=n_months,
        freq="M"
    ).to_timestamp()

    # Run original SM
    sm_monthly, sm_cum = model_monthly_and_cum(
        theta=beta_sm,
        clean_day=clean_day,
        tau0=tau0,
        months=months,
        init_env=init_env,
        p_wash=p_wash_fixed
    )

    sm_monthly = np.array(
        sm_monthly, dtype=float
    )

    sm_cum = np.array(
        sm_cum, dtype=float
    )

    abm_cum = np.cumsum(abm_mean)

    # Original approximate cumulative SD
    abm_cum_std = np.sqrt(
        np.cumsum(abm_std ** 2)
    )

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

    rmse_monthly = float(
        np.sqrt(
            np.mean((abm_mean - sm_monthly) ** 2)
        )
    )

    rmse_cum = float(
        np.sqrt(
            np.mean((abm_cum - sm_cum) ** 2)
        )
    )

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

    # =========================================================
    # NEW: Store exact values used for the comparison plots
    # =========================================================

    for j in range(m):

        timeseries_rows.append({
            "month": months[j].strftime("%Y-%m"),
            "cleaningDay": clean_day,
            "first_clean_day": first_clean_day,
            "tau_offset_days": tau0,
            "p_wash": p_wash_fixed,
            "beta_abm": beta_abm_target,
            "beta_sm": beta_sm,
            "n_runs": n_runs,

            "ABM_monthly_mean": abm_mean[j],
            "ABM_monthly_std": abm_std[j],
            "SM_monthly": sm_monthly[j],

            "ABM_cumulative": abm_cum[j],
            "ABM_cumulative_std_approx":
                abm_cum_std[j],
            "SM_cumulative": sm_cum[j]
        })

    # =========================================================
    # Original monthly comparison plot
    # =========================================================

    axm = axes_m[i]

    axm.plot(
        xx,
        abm_mean,
        "o-",
        linewidth=2,
        label=(
            f"Actual ABM mean "
            f"(beta_ABM={beta_used:.5f})"
        )
    )

    axm.fill_between(
        xx,
        abm_mean - abm_std,
        abm_mean + abm_std,
        alpha=0.2,
        label="ABM +/- 1 SD"
    )

    axm.plot(
        xx,
        sm_monthly,
        "s--",
        linewidth=2,
        label=f"Mapped SM (beta_SM={beta_sm:.3f})"
    )

    axm.set_title(
        f"cleaningDay={clean_day}, "
        f"tau0={tau0}, n={n_runs}",
        fontsize=12
    )

    axm.set_xlabel("Month")
    axm.set_ylabel("Monthly HAI")
    axm.grid(alpha=0.3)
    axm.legend(fontsize=8)

    # =========================================================
    # Original cumulative comparison plot
    # =========================================================

    axc = axes_c[i]

    axc.plot(
        xx,
        abm_cum,
        "o-",
        linewidth=2,
        label=(
            f"Actual ABM cumulative "
            f"(beta_ABM={beta_used:.5f})"
        )
    )

    axc.fill_between(
        xx,
        abm_cum - abm_cum_std,
        abm_cum + abm_cum_std,
        alpha=0.2,
        label="ABM cumulative +/- approx SD"
    )

    axc.plot(
        xx,
        sm_cum,
        "s--",
        linewidth=2,
        label=f"Mapped SM (beta_SM={beta_sm:.3f})"
    )

    axc.set_title(
        f"cleaningDay={clean_day}, "
        f"tau0={tau0}, n={n_runs}",
        fontsize=12
    )

    axc.set_xlabel("Month")
    axc.set_ylabel("Cumulative HAI")
    axc.grid(alpha=0.3)
    axc.legend(fontsize=8)

    print(
        f"beta_SM={beta_sm:.6f}, "
        f"ABM total={abm_cum[-1]:.4f}, "
        f"SM total={sm_cum[-1]:.4f}"
    )

    print(
        f"Monthly RMSE={rmse_monthly:.4f}, "
        f"Cumulative RMSE={rmse_cum:.4f}"
    )


# =============================================================
# 8. Save monthly comparison figure
# =============================================================

fig_m.suptitle(
    "Actual ABM vs mapped SM monthly trajectories "
    "at beta_ABM=0.03847",
    fontsize=14
)

fig_m.tight_layout(rect=[0, 0, 1, 0.96])

monthly_png = os.path.join(
    fig_dir,
    "actual_cleaning_ABM_vs_mappedSM_monthly_betaABM0p03847.png"
)

fig_m.savefig(
    monthly_png,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved:", monthly_png)


# =============================================================
# 9. Save cumulative comparison figure
# =============================================================

fig_c.suptitle(
    "Actual ABM vs mapped SM cumulative trajectories "
    "at beta_ABM=0.03847",
    fontsize=14
)

fig_c.tight_layout(rect=[0, 0, 1, 0.96])

cum_png = os.path.join(
    fig_dir,
    "actual_cleaning_ABM_vs_mappedSM_cumulative_betaABM0p03847.png"
)

fig_c.savefig(
    cum_png,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved:", cum_png)


# =============================================================
# 10. Save original comparison summary CSV
# =============================================================

df_compare = pd.DataFrame(compare_rows)

compare_csv = os.path.join(
    smfit_dir,
    "actual_cleaning_ABM_vs_mappedSM_betaABM0p03847.csv"
)

df_compare.to_csv(
    compare_csv,
    index=False,
    encoding="utf-8"
)

print("\nSaved comparison summary:")
print(compare_csv)
print(df_compare.to_string(index=False))


# =============================================================
# 11. NEW: Save monthly and cumulative time-series CSV
# =============================================================

df_timeseries = pd.DataFrame(timeseries_rows)

timeseries_csv = os.path.join(
    smfit_dir,
    "actual_ABM_vs_SM_cleaning_A_timeseries.csv"
)

df_timeseries.to_csv(
    timeseries_csv,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved time-series CSV:")
print(timeseries_csv)

print("\nPreview:")
print(df_timeseries.head())


# =============================================================
# 12. Finished
# =============================================================

print("\n" + "=" * 60)
print("CLEANING A COMPLETED")
print("=" * 60)

print("Mapping CSV:", mapped_csv)
print("Comparison CSV:", compare_csv)
print("Time-series CSV:", timeseries_csv)
print("Monthly plot:", monthly_png)
print("Cumulative plot:", cum_png)

# %%
