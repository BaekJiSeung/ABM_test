
# %% ============================================================
# Period B: Actual ABM vs Mapped SM
# Handwashing / Cleaning / Isolation
#
# Actual ABM beta = 0.04136
# NO ABM interpolation
# Reuse existing beta_ABM -> beta_SM mapping
# Save monthly/cumulative CSVs, summaries, figures
# ===============================================================

import os
import ast
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===============================================================
# 1. SETTINGS
# ===============================================================

BETA_ABM = 0.04136

N_MONTHS = 36
INIT_ENV = 2
START_MONTH = "2021-01"

try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()

RESULT_DIR = os.path.normpath(
    os.path.join(BASE, "..", "result")
)

SM_DIR = os.path.join(BASE, "sm_fit")
FIG_DIR = os.path.join(SM_DIR, "figures")

os.makedirs(SM_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ===============================================================
# 2. ORIGINAL PERIOD B IMPORTED PATIENT DATA
# ===============================================================

PI_dates = [
    "2021-01-08", "2021-01-29",
    "2021-02-17", "2021-02-19",
    "2021-03-02", "2021-03-11", "2021-03-28",
    "2021-04-04",
    "2021-06-16", "2021-06-28", "2021-06-30",
    "2021-07-06", "2021-07-07", "2021-07-20",
    "2021-08-10", "2021-08-14", "2021-08-20",
    "2021-08-21", "2021-08-23",
    "2021-09-15", "2021-09-18",
    "2021-10-04", "2021-10-20", "2021-10-27",
    "2021-10-28", "2021-10-29",
    "2021-12-27", "2021-12-27",

    "2022-01-04", "2022-01-05", "2022-01-07",
    "2022-01-14",
    "2022-02-18",
    "2022-03-12",
    "2022-04-15", "2022-04-15", "2022-04-17",
    "2022-05-06", "2022-05-14", "2022-05-16",
    "2022-05-21", "2022-05-24",
    "2022-06-27", "2022-06-30",
    "2022-07-03", "2022-07-20",
    "2022-08-10", "2022-08-15", "2022-08-16",
    "2022-08-18", "2022-08-25",
    "2022-09-20",
    "2022-10-02", "2022-10-11", "2022-10-15",
    "2022-10-21",
    "2022-11-08", "2022-11-13", "2022-11-15",
    "2022-11-17", "2022-11-20", "2022-11-21",
    "2022-11-23",
    "2022-12-21", "2022-12-29",

    "2023-01-16", "2023-01-26", "2023-01-30",
    "2023-02-03", "2023-02-13",
    "2023-03-29",
    "2023-05-10", "2023-05-10", "2023-05-10",
    "2023-05-15", "2023-05-23",
    "2023-06-07", "2023-06-12", "2023-06-15",
    "2023-07-01",
    "2023-08-04", "2023-08-10", "2023-08-16",
    "2023-09-10", "2023-09-17", "2023-09-24",
    "2023-09-28",
    "2023-10-05", "2023-10-06", "2023-10-08",
    "2023-10-10", "2023-10-14", "2023-10-20",
    "2023-10-31",
    "2023-11-08", "2023-11-08", "2023-11-13",
    "2023-11-17", "2023-11-19", "2023-11-23",
    "2023-11-28",
    "2023-12-04", "2023-12-09", "2023-12-09"
]


# ===============================================================
# 3. INTERVENTION CONFIGURATION
# ===============================================================

CONFIG = {
    "handwashing": {
        "tag": "B260",
        "suffix": "handwashALL",
        "raw_col": "hcw_wash_rate",
        "map_col": "handwash",
        "values": [0.8, 0.9, 0.95, 0.99],
    },

    "cleaning": {
        "tag": "B2_firstclean20",
        "suffix": "cleaningALL",
        "raw_col": "cleaningDay",
        "map_col": "cleaningDay",
        "values": [60, 90, 180, 360],
    },

    "isolation": {
        "tag": "B260",
        "suffix": "isoltimeALL",
        "raw_col": "isolation_time",
        "map_col": "isolation_time",
        "values": [6, 14, 20, 28],
    },
}


# ===============================================================
# 4. ORIGINAL PERIOD B SURROGATE MODEL
# ===============================================================

def simulate_theta(
    beta,
    init_env,
    tau0,
    p_wash=0.9,
    deep_clean_period=180,
    isol_time=7.0
):
    """
    Period B difference-equation surrogate model.

    beta : beta_SM

    Handwashing:
        p_wash changes.

    Cleaning:
        deep_clean_period and tau0 change.

    Isolation:
        isol_time changes.

    Isolation B passes isol_time directly,
    following the original B model.
    """

    C_total = 30
    C_iso = 30

    N_H = 19
    N_E = 30

    mu_S = 1 / 14
    mu_HAI = 1 / 21
    mu_I = 1 / 14

    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    cleaning_eff = 0.90
    iso_factor = 0.75

    sigma = 1.0 / isol_time

    days = pd.date_range(
        "2021-01-01",
        "2023-12-31",
        freq="D"
    )

    T = len(days)

    A_I_day = np.zeros(T)

    idx = {
        d: i
        for i, d in enumerate(days)
    }

    for d in PI_dates:
        A_I_day[idx[pd.Timestamp(d)]] += 1.0

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
        if (
            t > 0
            and (t + tau0) % deep_clean_period == 0
        ):
            Env_C[t] *= (1 - cleaning_eff)

        PS_sh = P_S_sh[t]
        PH_sh = P_HAI_sh[t]
        PH_iso = P_HAI_iso[t]
        PI = P_I[t]

        HC = H_C[t]
        EC = Env_C[t]

        # Imported colonized patients
        inc = A_I_day[t]

        if inc > 0:

            total_P = (
                PS_sh + PH_sh + PH_iso + PI
            )

            stay_free = max(
                0.0,
                C_total - total_P
            )

            inc_eff = min(inc, stay_free)
            taken = min(PS_sh, inc_eff)

            PS_sh -= taken
            PI += taken

        for _ in range(contacts_per_day):

            B_tot = max(
                PS_sh + PH_sh + PH_iso + PI,
                1e-9
            )

            # HCW -> susceptible patient
            lam_HP_sh = beta * (HC / N_H)

            # Patient -> HCW
            lam_PH = beta * (
                (
                    PH_sh
                    + iso_factor * PH_iso
                    + PI
                ) / B_tot
            )

            # Environment <-> HCW
            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            # New HAI
            hai_sh = (
                lam_HP_sh * PS_sh * dt
            )

            # Isolation transition
            move_HA = (
                sigma * PH_sh * dt
            )

            # Patient discharge
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

            PH_sh += (
                hai_sh - outH_sh - move_HA
            )

            PH_iso += move_HA - outH_iso

            PI += -outI

            PS_sh = np.clip(
                PS_sh, 0, C_total
            )

            PH_sh = np.clip(
                PH_sh, 0, C_total
            )

            PH_iso = np.clip(
                PH_iso, 0, C_iso
            )

            PI = np.clip(
                PI, 0, C_total
            )

            # Handwashing
            new_H = (
                (lam_PH + lam_EH)
                * (N_H - HC)
                * dt
            )

            HC = (
                HC + new_H
            ) * (1 - p_wash)

            HC = np.clip(HC, 0, N_H)

            # Environmental contamination
            EC += (
                lam_HE * (N_E - EC) * dt
            )

            EC = np.clip(EC, 0, N_E)

            NewHAI_day[t] += hai_sh

        if t < T - 1:

            P_S_sh[t + 1] = PS_sh
            P_HAI_sh[t + 1] = PH_sh
            P_HAI_iso[t + 1] = PH_iso
            P_I[t + 1] = PI

            H_C[t + 1] = HC
            Env_C[t + 1] = EC

    # Calendar-month SM incidence
    df = pd.DataFrame({
        "date": days,
        "NewHAI": NewHAI_day
    })

    monthly = df.groupby(
        df["date"].dt.to_period("M")
    )["NewHAI"].sum()

    months = pd.period_range(
        START_MONTH,
        periods=N_MONTHS,
        freq="M"
    )

    monthly = monthly.reindex(
        months,
        fill_value=0.0
    ).to_numpy(float)

    return monthly, np.cumsum(monthly)


# ===============================================================
# 5. INTERVENTION-SPECIFIC SM PARAMETERS
# ===============================================================

def sm_parameters(kind, value):

    pars = {
        "init_env": INIT_ENV,
        "tau0": 60,
        "p_wash": 0.9,
        "deep_clean_period": 180,
        "isol_time": 7.0,
    }

    if kind == "handwashing":

        pars["p_wash"] = float(value)

    elif kind == "cleaning":

        pars["deep_clean_period"] = int(value)
        pars["tau0"] = int(value) - 20

    elif kind == "isolation":

        # Original B isolation SM
        pars["isol_time"] = float(value)

    else:
        raise ValueError(kind)

    return pars


# ===============================================================
# 6. PARSE ACTUAL ABM DAILY TRAJECTORIES
# ===============================================================

def parse_daily(x):

    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=float)

    return np.asarray(
        ast.literal_eval(x),
        dtype=float
    )


def get_raw_path(kind):
    """
    Prefer the exact 0.04136-0.04136 LONG CSV.

    If its filename differs, search other LONG files
    and verify the actual beta values inside.

    No ABM interpolation.
    """

    cfg = CONFIG[kind]

    exact_name = (
        f"interv_prob_transmission_LONG_"
        f"{cfg['tag']}_"
        f"0.04136-0.04136_"
        f"{cfg['suffix']}.csv"
    )

    exact_path = os.path.join(
        RESULT_DIR,
        exact_name
    )

    if os.path.isfile(exact_path):
        return exact_path

    pattern = os.path.join(
        RESULT_DIR,
        "**",
        f"*LONG*{cfg['tag']}*{cfg['suffix']}.csv"
    )

    candidates = sorted(
        glob.glob(pattern, recursive=True)
    )

    matching = []

    for path in candidates:

        try:
            beta_col = pd.read_csv(
                path,
                usecols=["prob_transmission"]
            )

        except (ValueError, OSError):
            continue

        vals = pd.to_numeric(
            beta_col["prob_transmission"],
            errors="coerce"
        )

        has_target = np.isclose(
            vals,
            BETA_ABM,
            rtol=0,
            atol=1e-9
        ).any()

        if has_target:
            matching.append(path)

    if len(matching) == 1:
        return matching[0]

    if len(matching) > 1:
        raise RuntimeError(
            f"Multiple candidate ABM files "
            f"for {kind}: {matching}"
        )

    raise FileNotFoundError(
        f"Actual ABM beta={BETA_ABM} "
        f"file not found for {kind}.\n"
        f"Expected: {exact_path}"
    )


def load_actual_abm(kind):

    cfg = CONFIG[kind]
    path = get_raw_path(kind)

    df = pd.read_csv(path)

    required = [
        "prob_transmission",
        cfg["raw_col"],
        "HCW_related_infecs"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}; "
            f"found {list(df.columns)}"
        )

    betas = pd.to_numeric(
        df["prob_transmission"],
        errors="coerce"
    )

    df = df.loc[
        np.isclose(
            betas,
            BETA_ABM,
            rtol=0,
            atol=1e-9
        )
    ].copy()

    if df.empty:
        raise ValueError(
            f"{path}: no rows at "
            f"beta_ABM={BETA_ABM}"
        )

    print(
        "Actual ABM input:",
        path,
        "| exact beta rows:",
        len(df)
    )

    return df


# ===============================================================
# 7. ACTUAL ABM MONTHLY SUMMARY
# ===============================================================

def abm_summary(df, kind, value):
    """
    Actual ABM daily trajectories
      -> 30-day monthly totals
      -> mean and sample SD (ddof=1)

    No interpolation of ABM values.
    """

    col = CONFIG[kind]["raw_col"]

    vals = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    sub = df.loc[
        np.isclose(
            vals,
            value,
            rtol=0,
            atol=1e-9
        )
    ]

    if sub.empty:
        raise ValueError(
            f"No actual ABM trajectories "
            f"for {kind}={value}"
        )

    trajectories = []

    for s in sub["HCW_related_infecs"].dropna():

        daily = parse_daily(s)

        if len(daily) < N_MONTHS * 30:
            raise ValueError(
                f"{kind}={value}: "
                f"daily trajectory too short: "
                f"{len(daily)}"
            )

        monthly = (
            daily[:N_MONTHS * 30]
            .reshape(N_MONTHS, 30)
            .sum(axis=1)
        )

        trajectories.append(monthly)

    arr = np.vstack(trajectories)

    if len(arr) < 2:
        raise ValueError(
            f"{kind}={value}: need at least "
            f"2 ABM runs for ddof=1 SD"
        )

    return (
        arr.mean(axis=0),
        arr.std(axis=0, ddof=1),
        len(arr)
    )


# ===============================================================
# 8. EXISTING BETA_ABM -> BETA_SM MAPPING
# ===============================================================

def mapped_beta(kind, value, df_map):

    col = CONFIG[kind]["map_col"]

    required = [
        col,
        "beta_abm",
        "theta_hat"
    ]

    for c in required:

        if c not in df_map.columns:
            raise ValueError(
                f"{kind} mapping lacks '{c}'. "
                f"Columns: {list(df_map.columns)}"
            )

    vals = pd.to_numeric(
        df_map[col],
        errors="coerce"
    )

    sub = df_map.loc[
        np.isclose(
            vals,
            value,
            rtol=0,
            atol=1e-9
        )
    ].copy()

    if sub.empty:
        raise ValueError(
            f"{kind}={value} not found "
            f"in mapping CSV"
        )

    sub = sub.sort_values("beta_abm")

    x = sub["beta_abm"].to_numpy(float)
    y = sub["theta_hat"].to_numpy(float)

    if not (
        x.min() <= BETA_ABM <= x.max()
    ):
        raise ValueError(
            f"beta_ABM={BETA_ABM} "
            f"outside mapping range"
        )

    # Interpolation is ONLY for beta_SM mapping.
    beta_sm = float(
        np.interp(BETA_ABM, x, y)
    )

    return beta_sm, sub


# ===============================================================
# 9. RUN ONE INTERVENTION
# ===============================================================

def run_one(kind):

    cfg = CONFIG[kind]

    map_path = os.path.join(
        SM_DIR,
        f"theta_pairs_subset_cumGaussian_"
        f"{cfg['tag']}_{cfg['suffix']}.csv"
    )

    if not os.path.isfile(map_path):
        raise FileNotFoundError(
            f"Mapping CSV not found: {map_path}"
        )

    df_map = pd.read_csv(map_path)

    # Actual ABM data at beta_ABM=0.04136
    df_abm = load_actual_abm(kind)

    ts_rows = []
    sum_rows = []
    map_rows = []

    months = pd.period_range(
        START_MONTH,
        periods=N_MONTHS,
        freq="M"
    )

    x = np.arange(1, N_MONTHS + 1)

    fig_m, axes_m = plt.subplots(
        2, 2,
        figsize=(13, 9),
        sharex=True
    )

    fig_c, axes_c = plt.subplots(
        2, 2,
        figsize=(13, 9),
        sharex=True
    )

    fig_map, ax_map = plt.subplots(
        figsize=(8, 6)
    )

    for i, value in enumerate(cfg["values"]):

        print(
            f"\n{kind} = {value}"
        )

        # -------------------------------------------
        # Existing beta mapping
        # -------------------------------------------

        beta_sm, sub_map = mapped_beta(
            kind,
            value,
            df_map
        )

        # -------------------------------------------
        # Actual ABM at exact beta_ABM
        # -------------------------------------------

        abm_m, abm_sd, n_runs = abm_summary(
            df_abm,
            kind,
            value
        )

        # -------------------------------------------
        # Original Period B surrogate model
        # -------------------------------------------

        pars = sm_parameters(kind, value)

        sm_m, sm_c = simulate_theta(
            beta=beta_sm,
            **pars
        )

        # -------------------------------------------
        # Cumulative trajectories
        # -------------------------------------------

        abm_c = np.cumsum(abm_m)

        abm_c_sd_approx = np.sqrt(
            np.cumsum(abm_sd ** 2)
        )

        # -------------------------------------------
        # RMSE
        # -------------------------------------------

        rmse_m = float(
            np.sqrt(
                np.mean(
                    (abm_m - sm_m) ** 2
                )
            )
        )

        rmse_c = float(
            np.sqrt(
                np.mean(
                    (abm_c - sm_c) ** 2
                )
            )
        )

        # -------------------------------------------
        # Mapping summary
        # -------------------------------------------

        map_rows.append({
            cfg["map_col"]: value,
            "beta_abm": BETA_ABM,
            "beta_sm": beta_sm
        })

        # -------------------------------------------
        # Comparison summary
        # -------------------------------------------

        sum_rows.append({
            cfg["map_col"]: value,
            "beta_abm": BETA_ABM,
            "beta_sm": beta_sm,
            "n_runs": n_runs,

            "ABM_final_cumulative": abm_c[-1],
            "SM_final_cumulative": sm_c[-1],

            "RMSE_monthly": rmse_m,
            "RMSE_cumulative": rmse_c
        })

        # -------------------------------------------
        # Monthly/cumulative CSV rows
        # -------------------------------------------

        for j in range(N_MONTHS):

            ts_rows.append({
                "month": str(months[j]),
                "intervention": kind,

                cfg["map_col"]: value,

                "beta_abm": BETA_ABM,
                "beta_sm": beta_sm,
                "n_runs": n_runs,

                "init_env": INIT_ENV,
                "tau0": pars["tau0"],

                "p_wash": pars["p_wash"],

                "cleaningDay": (
                    pars["deep_clean_period"]
                ),

                "SM_isol_time": (
                    pars["isol_time"]
                ),

                "ABM_monthly_mean": abm_m[j],
                "ABM_monthly_std": abm_sd[j],

                "SM_monthly": sm_m[j],

                "ABM_cumulative": abm_c[j],

                "ABM_cumulative_std_approx": (
                    abm_c_sd_approx[j]
                ),

                "SM_cumulative": sm_c[j]
            })

        # -------------------------------------------
        # Monthly plot
        # -------------------------------------------

        am = axes_m.flat[i]

        am.plot(
            x,
            abm_m,
            "o-",
            markersize=3,
            label="Actual ABM"
        )

        am.fill_between(
            x,
            abm_m - abm_sd,
            abm_m + abm_sd,
            alpha=0.2,
            label="ABM +/- 1 SD"
        )

        am.plot(
            x,
            sm_m,
            "s--",
            markersize=3,
            label=f"SM beta={beta_sm:.4f}"
        )

        # -------------------------------------------
        # Cumulative plot
        # -------------------------------------------

        ac = axes_c.flat[i]

        ac.plot(
            x,
            abm_c,
            "o-",
            markersize=3,
            label="Actual ABM"
        )

        ac.fill_between(
            x,
            abm_c - abm_c_sd_approx,
            abm_c + abm_c_sd_approx,
            alpha=0.2,
            label="ABM approx +/- 1 SD"
        )

        ac.plot(
            x,
            sm_c,
            "s--",
            markersize=3,
            label=f"SM beta={beta_sm:.4f}"
        )

        for ax, ylabel in [
            (am, "Monthly HAI"),
            (ac, "Cumulative HAI")
        ]:

            ax.set_title(
                f"{cfg['map_col']}={value} "
                f"| n={n_runs}"
            )

            ax.set_xlabel("Month")
            ax.set_ylabel(ylabel)

            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        # -------------------------------------------
        # Mapping plot
        # -------------------------------------------

        ax_map.plot(
            sub_map["beta_abm"],
            sub_map["theta_hat"],
            "o-",
            label=f"{cfg['map_col']}={value}"
        )

        ax_map.scatter(
            [BETA_ABM],
            [beta_sm],
            marker="*",
            s=100
        )

        print(
            f"beta_SM={beta_sm:.6f}, "
            f"n={n_runs}, "
            f"ABM_total={abm_c[-1]:.3f}, "
            f"SM_total={sm_c[-1]:.3f}"
        )

    # ===========================================================
    # 10. SAVE CSVs
    # ===========================================================

    prefix = f"actual_ABM_vs_SM_{kind}_B"

    ts_path = os.path.join(
        SM_DIR,
        prefix + "_timeseries.csv"
    )

    summary_path = os.path.join(
        SM_DIR,
        prefix + "_summary.csv"
    )

    mapped_path = os.path.join(
        SM_DIR,
        f"mapped_betaSM_{kind}_B.csv"
    )

    pd.DataFrame(ts_rows).to_csv(
        ts_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(sum_rows).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(map_rows).to_csv(
        mapped_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ===========================================================
    # 11. SAVE FIGURES
    # ===========================================================

    for fig, suffix, title in [
        (fig_m, "monthly", "Monthly"),
        (fig_c, "cumulative", "Cumulative")
    ]:

        fig.suptitle(
            f"Period B {kind}: "
            f"Actual ABM vs Mapped SM ({title})"
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        fig.savefig(
            os.path.join(
                FIG_DIR,
                prefix + "_" + suffix + ".png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

    ax_map.axvline(
        BETA_ABM,
        linestyle="--",
        color="gray"
    )

    ax_map.set_xlabel("beta_ABM")
    ax_map.set_ylabel("beta_SM")

    ax_map.set_title(
        f"Period B {kind} beta mapping"
    )

    ax_map.grid(alpha=0.3)
    ax_map.legend()

    fig_map.tight_layout()

    fig_map.savefig(
        os.path.join(
            FIG_DIR,
            prefix + "_mapping.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig_m)
    plt.close(fig_c)
    plt.close(fig_map)

    print("\nSAVED:", ts_path)
    print("Rows:", len(ts_rows))

    return ts_path


# ===============================================================
# 12. RUN ALL THREE
# ===============================================================

if __name__ == "__main__":

    success = {}
    errors = {}

    for kind in [
        "handwashing",
        "cleaning",
        "isolation"
    ]:

        print(
            "\n" + "=" * 65
            + "\n" + kind.upper()
        )

        try:

            success[kind] = run_one(kind)

        except Exception as exc:

            errors[kind] = str(exc)

            print(
                f"[FAILED] {kind}: {exc}"
            )

    # ===========================================================
    # 13. FINAL REPORT
    # ===========================================================

    print("\n=== FINAL REPORT ===")

    for kind in CONFIG:

        print(
            kind,
            ":",
            success.get(
                kind,
                "FAILED: "
                + errors.get(kind, "unknown")
            )
        )

# %%
