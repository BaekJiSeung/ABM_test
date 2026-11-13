
# %% ==========================================================
# Isolation A - Actual ABM vs Mapped SM
# Original surrogate model + CSV time-series export
# Mapping refitting is NOT performed
# =============================================================

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================
# 1. Period A imported colonized patients
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
    "2018-05-25"
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
# 2. Original Isolation A surrogate model
# =============================================================

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
    beta          : beta_SM
    init_env      : initial environmental contamination
    tau0          : cleaning phase offset
    isol_time_abm : ABM isolation time upper bound

    Original assumption:
        SM mean isolation time = isol_time_abm / 2
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

    # Original isolation conversion
    isol_time_mean = isol_time_abm / 2.0
    sigma = 1.0 / isol_time_mean

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

        # Fixed environmental deep cleaning
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

            total_P = PS_sh + PH_sh + PH_iso + PI

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

            # Fixed handwashing
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
        "Env_C": Env_C
    }).set_index("date")

    return days, NewHAI_day, monthly, comp_df


# =============================================================
# 3. Original settings and paths
# =============================================================

data_type = "A"

init_env = 9
tau0 = 140

cleaningDay_fixed = 180
p_wash_fixed = 0.9

isoltime_values = [6, 14, 20, 28]

beta_abm_target = 0.03847
start_month = "2017-01"

days_per_month = 30
n_months = 19

scenario_tag = f"{data_type}{init_env}{tau0}"

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

result_dir = os.path.join(base_dir, "..", "result")
smfit_dir = os.path.join(base_dir, "sm_fit")
fig_dir = os.path.join(smfit_dir, "figures")

os.makedirs(smfit_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

mapping_csv = os.path.join(
    smfit_dir,
    f"theta_pairs_subset_cumGaussian_{scenario_tag}_isoltimeALL.csv"
)

actual_raw_csv = os.path.join(
    result_dir,
    f"interv_prob_transmission_LONG_{scenario_tag}_0.03847-0.03847_isoltimeALL.csv"
)


# =============================================================
# 4. Original helper functions
# =============================================================

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

    raise ValueError(
        f"Isolation time column not found: {list(df.columns)}"
    )


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


def summarize_actual_abm_isol(df_raw, isol_time):
    """
    Original method:
    Daily ABM runs -> 30-day monthly sums -> mean/std
    """

    isol_col = find_isol_col(df_raw)

    sub = df_raw[
        df_raw[isol_col].astype(float) == float(isol_time)
    ].copy()

    if sub.empty:
        raise ValueError(
            f"Actual ABM missing isolation_time={isol_time}"
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

    monthly_arr = np.array(
        monthly_runs,
        dtype=float
    )

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
            f"Mapping missing isolation_time={isol_time}"
        )

    return sub.sort_values(
        "beta_abm"
    ).reset_index(drop=True)


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


# =============================================================
# 5. Load existing Mapping and Actual ABM
# =============================================================

if not os.path.exists(mapping_csv):
    raise FileNotFoundError(mapping_csv)

if not os.path.exists(actual_raw_csv):
    raise FileNotFoundError(actual_raw_csv)

print("Reading Mapping:", mapping_csv)
df_map = pd.read_csv(mapping_csv)

print("Reading Actual ABM:", actual_raw_csv)
df_actual = pd.read_csv(actual_raw_csv)


# =============================================================
# 6. Original isolation-specific Mapping plot
# =============================================================

mapped_rows = []

plt.figure(figsize=(8, 6))

for isol_time in isoltime_values:

    sub = get_mapping_sub(df_map, isol_time)

    x = sub["beta_abm"].to_numpy(float)
    y = sub["theta_hat"].to_numpy(float)

    # Original beta_ABM -> beta_SM interpolation
    beta_sm = float(
        np.interp(beta_abm_target, x, y)
    )

    mapped_rows.append({
        "isolation_time_abm": isol_time,
        "isolation_time_sm_mean": isol_time / 2.0,
        "beta_abm_target": beta_abm_target,
        "beta_sm_mapped": beta_sm
    })

    plt.plot(
        x, y,
        "o-",
        linewidth=2,
        markersize=5,
        label=(
            f"ABM isol={isol_time}, "
            f"SM mean={isol_time / 2.0:.1f}"
        )
    )

    if (
        "theta_low" in sub.columns
        and "theta_high" in sub.columns
    ):
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
    r"Isolation-specific mapping: "
    r"$\beta_{\mathrm{ABM}} \rightarrow \beta_{\mathrm{SM}}$",
    fontsize=15
)

plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()

out_mapping_png = os.path.join(
    fig_dir,
    f"mapping_isoltime_betaABM_to_betaSM_betaABM0p03847_{scenario_tag}.png"
)

plt.savefig(
    out_mapping_png,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved:", out_mapping_png)

df_mapped = pd.DataFrame(mapped_rows)

out_mapped_csv = os.path.join(
    smfit_dir,
    f"mapped_betaSM_betaABM0p03847_{scenario_tag}_isoltime.csv"
)

df_mapped.to_csv(
    out_mapped_csv,
    index=False,
    encoding="utf-8"
)

print("\nMapped beta_SM:")
print(df_mapped.to_string(index=False))


# =============================================================
# 7. Actual ABM vs Mapped SM comparison
# =============================================================

compare_rows = []
timeseries_rows = []

fig_m, axes_m = plt.subplots(
    2, 2,
    figsize=(12, 8),
    sharex=True
)

axes_m = axes_m.ravel()

fig_c, axes_c = plt.subplots(
    2, 2,
    figsize=(12, 8),
    sharex=True
)

axes_c = axes_c.ravel()

for i, isol_time in enumerate(isoltime_values):

    print("\n" + "=" * 60)
    print(f"ABM Isolation Time = {isol_time}")
    print("=" * 60)

    # Actual ABM data at beta_ABM=0.03847
    abm_mean, abm_std, n_runs = (
        summarize_actual_abm_isol(
            df_actual,
            isol_time
        )
    )

    row_map = df_mapped[
        df_mapped["isolation_time_abm"] == isol_time
    ]

    if row_map.empty:
        print(
            f"[WARNING] No mapped beta_SM for {isol_time}"
        )
        continue

    beta_sm = float(
        row_map.iloc[0]["beta_sm_mapped"]
    )

    months = get_months(len(abm_mean))

    # Original SM run:
    # Pass ABM isolation time unchanged.
    # simulate_theta internally uses isol_time_abm / 2.
    sm_monthly, sm_cum = model_monthly_and_cum(
        theta=beta_sm,
        isol_time_abm=isol_time,
        months=months,
        init_env=init_env,
        tau0=tau0
    )

    sm_monthly = np.array(
        sm_monthly,
        dtype=float
    )

    sm_cum = np.array(
        sm_cum,
        dtype=float
    )

    abm_cum = np.cumsum(abm_mean)

    # Original approximate cumulative SD
    abm_cum_std = np.sqrt(
        np.cumsum(abm_std ** 2)
    )

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

    # =========================================================
    # NEW: Save exact monthly/cumulative plot data
    # =========================================================

    for j in range(m):
        timeseries_rows.append({
            "month": months[j].strftime("%Y-%m"),

            "isolation_time_abm": isol_time,
            "isolation_time_sm_mean": isol_time / 2.0,

            "beta_abm": beta_abm_target,
            "beta_sm": beta_sm,

            "init_env": init_env,
            "tau0": tau0,
            "cleaningDay": cleaningDay_fixed,
            "p_wash": p_wash_fixed,
            "n_runs": n_runs,

            "ABM_monthly_mean": abm_mean[j],
            "ABM_monthly_std": abm_std[j],
            "SM_monthly": sm_monthly[j],

            "ABM_cumulative": abm_cum[j],
            "ABM_cumulative_std_approx": abm_cum_std[j],
            "SM_cumulative": sm_cum[j]
        })

    # =========================================================
    # Original monthly plot
    # =========================================================

    axm = axes_m[i]

    axm.plot(
        xx,
        abm_mean,
        "o-",
        linewidth=2,
        markersize=4,
        label=(
            f"Actual ABM mean "
            f"(beta_ABM={beta_abm_target:.5f})"
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
        markersize=4,
        label=f"Mapped SM (beta_SM={beta_sm:.3f})"
    )

    axm.set_title(
        f"ABM isol={isol_time}, "
        f"SM mean={isol_time / 2.0:.1f}, "
        f"n={n_runs}",
        fontsize=12
    )

    axm.set_xlabel("Month")
    axm.set_ylabel("Monthly HAI")
    axm.grid(alpha=0.3)
    axm.legend(fontsize=8)

    # =========================================================
    # Original cumulative plot
    # =========================================================

    axc = axes_c[i]

    axc.plot(
        xx,
        abm_cum,
        "o-",
        linewidth=2,
        markersize=4,
        label=(
            f"Actual ABM cumulative "
            f"(beta_ABM={beta_abm_target:.5f})"
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
        markersize=4,
        label=f"Mapped SM (beta_SM={beta_sm:.3f})"
    )

    axc.set_title(
        f"ABM isol={isol_time}, "
        f"SM mean={isol_time / 2.0:.1f}, "
        f"n={n_runs}",
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
# 8. Save original monthly comparison plot
# =============================================================

fig_m.suptitle(
    "Actual isolation ABM vs mapped SM monthly trajectories "
    "at beta_ABM=0.03847",
    fontsize=14
)

fig_m.tight_layout(rect=[0, 0, 1, 0.96])

out_monthly = os.path.join(
    fig_dir,
    f"actual_isoltime_ABM_vs_mappedSM_monthly_betaABM0p03847_{scenario_tag}.png"
)

fig_m.savefig(
    out_monthly,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved:", out_monthly)


# =============================================================
# 9. Save original cumulative comparison plot
# =============================================================

fig_c.suptitle(
    "Actual isolation ABM vs mapped SM cumulative trajectories "
    "at beta_ABM=0.03847",
    fontsize=14
)

fig_c.tight_layout(rect=[0, 0, 1, 0.96])

out_cum = os.path.join(
    fig_dir,
    f"actual_isoltime_ABM_vs_mappedSM_cumulative_betaABM0p03847_{scenario_tag}.png"
)

fig_c.savefig(
    out_cum,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved:", out_cum)


# =============================================================
# 10. Save original comparison summary CSV
# =============================================================

df_compare = pd.DataFrame(compare_rows)

out_compare_csv = os.path.join(
    smfit_dir,
    f"actual_isoltime_ABM_vs_mappedSM_betaABM0p03847_{scenario_tag}.csv"
)

df_compare.to_csv(
    out_compare_csv,
    index=False,
    encoding="utf-8"
)

print("\nSaved comparison summary:")
print(out_compare_csv)
print(df_compare.to_string(index=False))


# =============================================================
# 11. NEW: Save ABM/SM monthly and cumulative time series
# =============================================================

df_timeseries = pd.DataFrame(timeseries_rows)

out_timeseries_csv = os.path.join(
    smfit_dir,
    "actual_ABM_vs_SM_isolation_A_timeseries.csv"
)

df_timeseries.to_csv(
    out_timeseries_csv,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved time-series CSV:")
print(out_timeseries_csv)

print("\nPreview:")
print(df_timeseries.head())


# =============================================================
# 12. Finished
# =============================================================

print("\n" + "=" * 60)
print("ISOLATION A COMPLETED")
print("=" * 60)

print("Mapping CSV:", out_mapped_csv)
print("Comparison CSV:", out_compare_csv)
print("Time-series CSV:", out_timeseries_csv)
print("Mapping plot:", out_mapping_png)
print("Monthly plot:", out_monthly)
print("Cumulative plot:", out_cum)

# %%
