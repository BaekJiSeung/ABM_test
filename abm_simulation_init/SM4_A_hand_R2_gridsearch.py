
# ============================================================
# SM4_A_hand_R3.py
#
# Period A: 2D Grid Search
#
# ABM inputs:
#   13 beta conditions (p_ABM = 0.90)
#   4 handwashing conditions (beta_ABM = 0.03847)
#
# Surrogate parameters:
#   beta_SM: 0.5 ~ 12.0 (200 points)
#   p_SM:    0.3 ~ 1.0  (70 points)
#
# 14,000 grid evaluations per ABM condition
# 238,000 grid evaluations in total
#
# No Local Optimization
# Select the three lowest-NLL grid points
#
# Features:
# - 17 parallel workers
# - Progress messages
# - Intermediate CSV checkpoints
# - Resume interrupted grid searches
# - Wide and Long result CSV files
# ============================================================

import os
import ast
import time
import traceback

import numpy as np
import pandas as pd

from joblib import Parallel, delayed


# ============================================================
# 1. SETTINGS
# ============================================================

BETA_CSV = (
    "../result/"
    "interv_prob_transmission_summary_A9140_0.01-0.07.csv"
)

HANDWASH_CSV = (
    "../result/"
    "interv_prob_transmission_summary_A9140_0.03847-0.03847_handwashALL.csv"
)

# Parallel processing
N_JOBS = 17

# Fixed surrogate parameters
INIT_ENV = 9
TAU0 = 140

# Grid Search
BETA_GRID = np.linspace(
    0.5,
    12.0,
    200
)

P_GRID = np.linspace(
    0.3,
    1.0,
    70
)

# Select exactly 3 grid candidates
N_BEST = 3

# CSV matching tolerance
ATOL = 1e-7

# Save a checkpoint every 5 beta-grid rows
# Each beta-grid row contains 70 evaluations.
CHECKPOINT_EVERY = 5

# Print progress every 5 beta-grid rows
PROGRESS_EVERY = 5

# Fixed folder name allows interrupted runs to resume.
OUTPUT_DIR = os.path.join(
    "sm_fit",
    "mapping_2D_17cases_GRID200x70"
)


# ============================================================
# 2. PERIOD A INPUT DATA
# ============================================================

MONTHLY_PI = pd.Series({
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

PI_DATES = [
    "2017-03-01",
    "2017-05-30",
    "2017-09-17",
    "2017-09-29",
    "2017-11-14",
    "2017-11-20",
    "2018-05-25"
]


def make_imported_days(days):

    A = np.zeros(len(days))

    idx_map = {
        d: i
        for i, d in enumerate(days)
    }

    for d in PI_DATES:

        ts = pd.Timestamp(d)

        idx = idx_map.get(ts)

        if idx is not None:
            A[idx] += 1.0

    return A


# ============================================================
# 3. SURROGATE MODEL
# ============================================================

def simulate_theta(
    beta,
    init_env,
    tau0,
    p_wash
):

    C_total = 30

    N_H = 19
    N_E = 30

    mu_S = 1 / 7
    mu_HAI = 1 / 14
    mu_I = 1 / 7

    sigma = 1 / 7

    iso_factor = 0.75

    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    cleaning_eff = 0.90

    start = pd.Period(
        MONTHLY_PI.index.min(),
        freq="M"
    ).to_timestamp(how="start")

    end = pd.Period(
        MONTHLY_PI.index.max(),
        freq="M"
    ).to_timestamp(how="end")

    days = pd.date_range(
        start,
        end,
        freq="D"
    )

    T = len(days)

    A_I_day = make_imported_days(days)

    # Initial states
    PS_sh = float(C_total - 1)

    PH_sh = 0.0
    PH_iso = 0.0
    PI = 1.0

    HC = 0.0
    EC = float(init_env)

    NewHAI_day = np.zeros(T)

    for t in range(T):

        # ----------------------------------
        # Environmental deep cleaning
        # ----------------------------------

        if (
            t > 0
            and (t + tau0) % deep_clean_period == 0
        ):

            EC *= (1 - cleaning_eff)

        # ----------------------------------
        # Imported patient admission
        # ----------------------------------

        inc = A_I_day[t]

        if inc > 0:

            taken = min(PS_sh, inc)

            PS_sh -= taken
            PI += taken

        # ----------------------------------
        # Intraday transmission
        # ----------------------------------

        for _ in range(contacts_per_day):

            B_tot = max(
                PS_sh + PH_sh + PH_iso + PI,
                1e-9
            )

            # Transmission pressures
            lam_HP_sh = (
                beta * (HC / N_H)
            )

            lam_PH = beta * (
                (
                    PH_sh
                    + iso_factor * PH_iso
                    + PI
                ) / B_tot
            )

            lam_EH = (
                beta * (EC / N_E)
            )

            lam_HE = (
                beta * (HC / N_H)
            )

            # New HAI incidence
            hai_sh = (
                lam_HP_sh * PS_sh * dt
            )

            # Isolation transitions
            move_HA = (
                sigma * PH_sh * dt
            )

            move_I = (
                sigma * PI * dt
            )

            # Discharges
            outS_sh = (
                mu_S * PS_sh * dt
            )

            outH_sh = (
                mu_HAI * PH_sh * dt
            )

            outH_iso = (
                mu_HAI * PH_iso * dt
            )

            outI = (
                mu_I * PI * dt
            )

            leaving = (
                outS_sh
                + outH_sh
                + outH_iso
                + outI
            )

            total_P = (
                PS_sh
                + PH_sh
                + PH_iso
                + PI
            )

            # Susceptible admissions
            AS_tot = max(
                0.0,
                C_total - (total_P - leaving)
            )

            # Patient updates
            PS_sh = np.clip(
                PS_sh
                + AS_tot
                - outS_sh
                - hai_sh,
                0,
                C_total
            )

            PH_sh = np.clip(
                PH_sh
                + hai_sh
                - outH_sh
                - move_HA,
                0,
                C_total
            )

            PH_iso = np.clip(
                PH_iso
                + move_HA
                + move_I
                - outH_iso,
                0,
                C_total
            )

            PI = np.clip(
                PI
                - outI
                - move_I,
                0,
                C_total
            )

            # HCW contamination
            new_H = (
                (lam_PH + lam_EH)
                * (N_H - HC)
                * dt
            )

            HC = np.clip(
                (HC + new_H) * (1 - p_wash),
                0,
                N_H
            )

            # Environmental contamination
            EC = np.clip(
                EC
                + lam_HE * (N_E - EC) * dt,
                0,
                N_E
            )

            NewHAI_day[t] += hai_sh

    # ----------------------------------
    # Monthly cumulative HAI
    # ----------------------------------

    monthly_hai = (
        pd.Series(
            NewHAI_day,
            index=days
        )
        .groupby(days.to_period("M"))
        .sum()
        .to_numpy(dtype=float)
    )

    return np.cumsum(monthly_hai)


# ============================================================
# 4. LOAD EXACTLY 17 ABM CONDITIONS
# ============================================================

def parse_vector(value):

    if isinstance(value, str):

        value = ast.literal_eval(value)

    return np.asarray(
        value,
        dtype=float
    )


def build_cases():

    beta_df = pd.read_csv(
        BETA_CSV
    )

    hw_df = pd.read_csv(
        HANDWASH_CSV
    )

    beta_targets = np.linspace(
        0.01,
        0.07,
        13
    )

    handwash_targets = [
        0.80,
        0.90,
        0.95,
        0.99
    ]

    cases = []

    # ----------------------------------
    # Find one matching row
    # ----------------------------------

    def find_one(
        df,
        conditions,
        description
    ):

        mask = np.ones(
            len(df),
            dtype=bool
        )

        for col, target in conditions.items():

            values = pd.to_numeric(
                df[col],
                errors="coerce"
            ).to_numpy(dtype=float)

            mask &= np.isclose(
                values,
                target,
                atol=ATOL,
                rtol=0
            )

        matched = df.loc[mask]

        if matched.empty:

            raise ValueError(
                f"Missing ABM condition: "
                f"{description}"
            )

        if len(matched) > 1:

            print(
                f"[WARNING] {description}: "
                f"{len(matched)} matches. "
                "Using first row.",
                flush=True
            )

        return matched.iloc[0]

    # ----------------------------------
    # 13 beta sweep cases
    # ----------------------------------

    for beta in beta_targets:

        row = find_one(
            beta_df,
            {"beta": float(beta)},
            f"beta={beta:.5f}"
        )

        cases.append({
            "source": "beta_sweep",
            "beta_abm": float(beta),
            "p_abm": 0.90,
            "y_mean": parse_vector(
                row["mean"]
            ),
            "y_std": parse_vector(
                row["std"]
            )
        })

    # ----------------------------------
    # 4 handwashing sweep cases
    # ----------------------------------

    for p in handwash_targets:

        row = find_one(
            hw_df,
            {
                "beta": 0.03847,
                "handwash": p
            },
            (
                "beta=0.03847, "
                f"handwash={p:.2f}"
            )
        )

        cases.append({
            "source": "handwash_sweep",
            "beta_abm": 0.03847,
            "p_abm": float(p),
            "y_mean": parse_vector(
                row["mean"]
            ),
            "y_std": parse_vector(
                row["std"]
            )
        })

    print(
        f"Input CSV rows: "
        f"beta={len(beta_df)}, "
        f"handwash={len(hw_df)}",
        flush=True
    )

    print(
        f"Selected ABM conditions: "
        f"{len(cases)}",
        flush=True
    )

    if len(cases) != 17:

        raise ValueError(
            f"Expected 17 cases, "
            f"found {len(cases)}."
        )

    for case in cases:

        y_mean = case["y_mean"]
        y_std = case["y_std"]

        if (
            len(y_mean) != 19
            or len(y_std) != 19
        ):

            raise ValueError(
                "Expected 19 monthly values "
                f"for {case['source']} "
                f"beta={case['beta_abm']}, "
                f"p={case['p_abm']}"
            )

        if (
            not np.all(np.isfinite(y_mean))
            or not np.all(np.isfinite(y_std))
            or np.any(y_std < 0)
        ):

            raise ValueError(
                "Invalid ABM mean/std data."
            )

    return cases


# ============================================================
# 5. CHECKPOINT UTILITIES
# ============================================================

def atomic_save_csv(
    df,
    path
):

    """
    Write a temporary CSV and replace the original.
    This reduces the chance of corrupting a
    checkpoint when execution is interrupted.
    """

    temp_path = path + ".tmp"

    df.to_csv(
        temp_path,
        index=False,
        encoding="utf-8-sig"
    )

    os.replace(
        temp_path,
        path
    )


def load_grid_checkpoint(path):

    """
    Load previously calculated grid points.
    Returns a dictionary keyed by (beta_idx, p_idx).
    """

    records = {}

    if not os.path.exists(path):
        return records

    df = pd.read_csv(path)

    required = {
        "beta_idx",
        "p_idx",
        "beta_sm",
        "p_sm",
        "nll"
    }

    if not required.issubset(df.columns):

        raise ValueError(
            f"Invalid checkpoint format: {path}"
        )

    for row in df.itertuples(index=False):

        bi = int(row.beta_idx)
        pi = int(row.p_idx)

        if (
            bi < 0
            or bi >= len(BETA_GRID)
            or pi < 0
            or pi >= len(P_GRID)
        ):

            raise ValueError(
                "Checkpoint grid dimensions "
                "do not match current settings."
            )

        beta_expected = BETA_GRID[bi]
        p_expected = P_GRID[pi]

        if not (
            np.isclose(
                row.beta_sm,
                beta_expected,
                atol=1e-9,
                rtol=0
            )
            and np.isclose(
                row.p_sm,
                p_expected,
                atol=1e-9,
                rtol=0
            )
        ):

            raise ValueError(
                "Checkpoint parameters do not "
                "match the current grid."
            )

        if not np.isfinite(row.nll):

            raise ValueError(
                "Non-finite NLL in checkpoint."
            )

        key = (bi, pi)

        records[key] = {
            "beta_idx": bi,
            "p_idx": pi,
            "beta_sm": float(row.beta_sm),
            "p_sm": float(row.p_sm),
            "nll": float(row.nll)
        }

    return records


# ============================================================
# 6. GRID SEARCH FOR ONE ABM CONDITION
# ============================================================

def grid_search_one_case(
    case_id,
    case,
    output_dir
):

    start_time = time.perf_counter()

    beta_abm = case["beta_abm"]
    p_abm = case["p_abm"]
    source = case["source"]

    # ----------------------------------
    # ABM cumulative HAI
    # ----------------------------------

    cum_obs = np.cumsum(
        case["y_mean"]
    )

    cum_std = np.maximum(
        np.sqrt(
            np.cumsum(
                case["y_std"] ** 2
            )
        ),
        1e-6
    )

    # ----------------------------------
    # Gaussian NLL
    # ----------------------------------

    def gaussian_nll(
        beta_sm,
        p_sm
    ):

        cum_model = simulate_theta(
            beta=float(beta_sm),
            init_env=INIT_ENV,
            tau0=TAU0,
            p_wash=float(p_sm)
        )

        if len(cum_model) != len(cum_obs):

            raise ValueError(
                "ABM and SM output lengths differ."
            )

        residual = (
            cum_obs - cum_model
        )

        nll = 0.5 * np.sum(
            np.log(
                2 * np.pi * cum_std ** 2
            )
            + (
                residual / cum_std
            ) ** 2
        )

        if not np.isfinite(nll):

            raise ValueError(
                "Non-finite NLL encountered."
            )

        return float(nll)

    # ========================================================
    # 6-1. Load checkpoint
    # ========================================================

    grid_path = os.path.join(
        output_dir,
        "grids",
        f"grid_case_{case_id:02d}.csv"
    )

    records = load_grid_checkpoint(
        grid_path
    )

    n_beta = len(BETA_GRID)
    n_p = len(P_GRID)

    total_points = (
        n_beta * n_p
    )

    resumed_points = len(records)
    new_evaluations = 0

    print(
        f"[CASE {case_id:02d}/17 START] "
        f"ABM=({beta_abm:.5f}, {p_abm:.2f}) "
        f"Resumed={resumed_points}/{total_points}",
        flush=True
    )

    # ========================================================
    # 6-2. Exhaustive Grid Search
    # ========================================================

    for bi, beta_sm in enumerate(BETA_GRID):

        for pi, p_sm in enumerate(P_GRID):

            key = (bi, pi)

            # Skip previously completed points
            if key in records:
                continue

            nll = gaussian_nll(
                beta_sm,
                p_sm
            )

            records[key] = {
                "beta_idx": bi,
                "p_idx": pi,
                "beta_sm": float(beta_sm),
                "p_sm": float(p_sm),
                "nll": nll
            }

            new_evaluations += 1

        # ----------------------------------
        # Progress
        # ----------------------------------

        completed_rows = bi + 1

        if (
            completed_rows % PROGRESS_EVERY == 0
            or completed_rows == n_beta
        ):

            completed_points = len(records)

            progress = (
                100 * completed_points / total_points
            )

            elapsed_min = (
                time.perf_counter() - start_time
            ) / 60

            print(
                f"[CASE {case_id:02d}/17 GRID] "
                f"{completed_points:,}/{total_points:,} "
                f"({progress:.1f}%) "
                f"elapsed={elapsed_min:.1f} min",
                flush=True
            )

        # ----------------------------------
        # Checkpoint save
        # ----------------------------------

        if (
            completed_rows % CHECKPOINT_EVERY == 0
            or completed_rows == n_beta
        ):

            df_partial = pd.DataFrame(
                list(records.values())
            )

            df_partial = (
                df_partial.sort_values(
                    ["beta_idx", "p_idx"]
                )
                .reset_index(drop=True)
            )

            atomic_save_csv(
                df_partial,
                grid_path
            )

    # ========================================================
    # 6-3. Verify complete grid
    # ========================================================

    if len(records) != total_points:

        raise RuntimeError(
            f"Grid incomplete: "
            f"{len(records)} / {total_points}"
        )

    df_grid = pd.DataFrame(
        list(records.values())
    )

    df_grid = (
        df_grid.sort_values(
            ["nll", "beta_idx", "p_idx"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 6-4. Select TOP 3 directly from grid
    #
    # No separation constraint
    # No Local Optimization
    # ========================================================

    df_top3 = (
        df_grid.head(N_BEST)
        .copy()
        .reset_index(drop=True)
    )

    df_top3.insert(
        0,
        "rank",
        np.arange(
            1,
            len(df_top3) + 1
        )
    )

    df_top3.insert(
        0,
        "case_id",
        case_id
    )

    df_top3.insert(
        1,
        "source",
        source
    )

    df_top3.insert(
        2,
        "beta_abm",
        beta_abm
    )

    df_top3.insert(
        3,
        "p_abm",
        p_abm
    )

    elapsed_min = (
        time.perf_counter() - start_time
    ) / 60

    df_top3["elapsed_min"] = (
        elapsed_min
    )

    # ========================================================
    # 6-5. Save TOP 3
    # ========================================================

    top3_path = os.path.join(
        output_dir,
        "cases",
        f"case_{case_id:02d}_top3.csv"
    )

    atomic_save_csv(
        df_top3,
        top3_path
    )

    # ========================================================
    # 6-6. Wide result
    # ========================================================

    best = df_top3.iloc[0]

    result = {
        "case_id": case_id,
        "source": source,

        "beta_abm": beta_abm,
        "p_abm": p_abm,

        "best_beta_sm": float(
            best["beta_sm"]
        ),

        "best_p_sm": float(
            best["p_sm"]
        ),

        "best_nll": float(
            best["nll"]
        ),

        "selected": "Grid Rank 1",

        "grid_points": total_points,

        "new_evaluations": new_evaluations,

        "resumed_points": resumed_points,

        "elapsed_min": elapsed_min
    }

    for _, row in df_top3.iterrows():

        rank = int(
            row["rank"]
        )

        prefix = f"grid{rank}_"

        result[
            prefix + "beta_sm"
        ] = float(row["beta_sm"])

        result[
            prefix + "p_sm"
        ] = float(row["p_sm"])

        result[
            prefix + "nll"
        ] = float(row["nll"])

        result[
            prefix + "beta_idx"
        ] = int(row["beta_idx"])

        result[
            prefix + "p_idx"
        ] = int(row["p_idx"])

    wide_path = os.path.join(
        output_dir,
        "cases",
        f"case_{case_id:02d}_result.csv"
    )

    atomic_save_csv(
        pd.DataFrame([result]),
        wide_path
    )

    # ========================================================
    # 6-7. Print result
    # ========================================================

    print(
        f"\n[DONE CASE {case_id:02d}/17] "
        f"ABM=({beta_abm:.5f}, {p_abm:.2f})",
        flush=True
    )

    for _, row in df_top3.iterrows():

        print(
            f"  Rank {int(row['rank'])}: "
            f"beta_SM={row['beta_sm']:.6f}, "
            f"p_SM={row['p_sm']:.6f}, "
            f"NLL={row['nll']:.6f}",
            flush=True
        )

    print(
        f"  Time={elapsed_min:.2f} min\n",
        flush=True
    )

    return result, df_top3.to_dict(
        orient="records"
    )


# ============================================================
# 7. SAFE PARALLEL WORKER
# ============================================================

def safe_grid_search(
    case_id,
    case,
    output_dir
):

    try:

        result, top3 = grid_search_one_case(
            case_id,
            case,
            output_dir
        )

        return {
            "ok": True,
            "case_id": case_id,
            "result": result,
            "top3": top3
        }

    except Exception:

        error_text = (
            traceback.format_exc()
        )

        error_path = os.path.join(
            output_dir,
            "cases",
            f"case_{case_id:02d}_ERROR.txt"
        )

        with open(
            error_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(error_text)

        print(
            f"[ERROR CASE {case_id:02d}]\n"
            f"{error_text}",
            flush=True
        )

        return {
            "ok": False,
            "case_id": case_id,
            "error": error_text
        }


# ============================================================
# 8. MAIN
# ============================================================

def main():

    start_time = time.perf_counter()

    cases = build_cases()

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "cases"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "grids"
        ),
        exist_ok=True
    )

    n_beta = len(BETA_GRID)
    n_p = len(P_GRID)

    points_per_case = (
        n_beta * n_p
    )

    total_points = (
        points_per_case * len(cases)
    )

    print(
        "\n========== SETTINGS =========="
    )

    print(
        "ABM conditions:",
        len(cases)
    )

    print(
        "Parallel processes:",
        N_JOBS
    )

    print(
        "Beta grid:",
        n_beta,
        f"[{BETA_GRID[0]:.4f}, "
        f"{BETA_GRID[-1]:.4f}]"
    )

    print(
        "P grid:",
        n_p,
        f"[{P_GRID[0]:.4f}, "
        f"{P_GRID[-1]:.4f}]"
    )

    print(
        "Grid evaluations per case:",
        f"{points_per_case:,}"
    )

    print(
        "Total grid evaluations:",
        f"{total_points:,}"
    )

    print(
        "Selected grid candidates:",
        N_BEST
    )

    print(
        "Local Optimization:",
        "NONE"
    )

    print(
        "CSV ATOL:",
        ATOL
    )

    print(
        "Checkpoint interval:",
        CHECKPOINT_EVERY,
        "beta rows"
    )

    print(
        "Output directory:",
        os.path.abspath(OUTPUT_DIR)
    )

    print(
        "\n========== ABM CASES =========="
    )

    for i, case in enumerate(
        cases,
        1
    ):

        print(
            f"{i:02d} | "
            f"{case['source']:15s} | "
            f"beta_ABM={case['beta_abm']:.5f} | "
            f"p_ABM={case['p_abm']:.2f}"
        )

    # ========================================================
    # 8-1. Parallel Grid Search
    # ========================================================

    print(
        "\n========== PARALLEL GRID START =========="
    )

    jobs = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        verbose=10,
        pre_dispatch="n_jobs"
    )(
        delayed(safe_grid_search)(
            i,
            case,
            OUTPUT_DIR
        )
        for i, case in enumerate(
            cases,
            1
        )
    )

    wall_time = (
        time.perf_counter() - start_time
    ) / 60

    # ========================================================
    # 8-2. Collect completed cases
    # ========================================================

    completed = [
        item
        for item in jobs
        if item["ok"]
    ]

    failed = [
        item["case_id"]
        for item in jobs
        if not item["ok"]
    ]

    df_wide = pd.DataFrame([
        item["result"]
        for item in completed
    ])

    df_long = pd.DataFrame([
        row
        for item in completed
        for row in item["top3"]
    ])

    if not df_wide.empty:

        df_wide = (
            df_wide.sort_values(
                "case_id"
            )
            .reset_index(drop=True)
        )

    if not df_long.empty:

        df_long = (
            df_long.sort_values(
                ["case_id", "rank"]
            )
            .reset_index(drop=True)
        )

    # ========================================================
    # 8-3. Save combined results
    # ========================================================

    wide_path = os.path.join(
        OUTPUT_DIR,
        "mapping_2D_17cases_GRID200x70_wide.csv"
    )

    long_path = os.path.join(
        OUTPUT_DIR,
        "mapping_2D_17cases_GRID200x70_top3_long.csv"
    )

    atomic_save_csv(
        df_wide,
        wide_path
    )

    atomic_save_csv(
        df_long,
        long_path
    )

    # ========================================================
    # 8-4. Run summary
    # ========================================================

    summary = {
        "total_cases": len(cases),
        "completed_cases": len(completed),
        "failed_cases": len(failed),

        "failed_ids": ",".join(
            map(str, failed)
        ),

        "n_jobs": N_JOBS,

        "beta_grid_points": n_beta,
        "p_grid_points": n_p,

        "beta_min": float(
            BETA_GRID[0]
        ),

        "beta_max": float(
            BETA_GRID[-1]
        ),

        "p_min": float(
            P_GRID[0]
        ),

        "p_max": float(
            P_GRID[-1]
        ),

        "grid_points_per_case": (
            points_per_case
        ),

        "total_grid_points": (
            total_points
        ),

        "top_n": N_BEST,

        "local_optimization": False,

        "csv_atol": ATOL,

        "wall_time_min": wall_time
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "run_summary.csv"
    )

    atomic_save_csv(
        pd.DataFrame([summary]),
        summary_path
    )

    # ========================================================
    # 8-5. Print TOP 3 for all cases
    # ========================================================

    print(
        "\n========== ALL GRID TOP 3 =========="
    )

    if not df_long.empty:

        columns = [
            "case_id",
            "beta_abm",
            "p_abm",
            "rank",
            "beta_sm",
            "p_sm",
            "nll"
        ]

        print(
            df_long[
                columns
            ].to_string(index=False),
            flush=True
        )

    # ========================================================
    # 8-6. Print best mapping
    # ========================================================

    print(
        "\n========== BEST GRID MAPPING =========="
    )

    if not df_wide.empty:

        columns = [
            "case_id",
            "source",
            "beta_abm",
            "p_abm",
            "best_beta_sm",
            "best_p_sm",
            "best_nll"
        ]

        print(
            df_wide[
                columns
            ].to_string(index=False),
            flush=True
        )

    # ========================================================
    # 8-7. Final summary
    # ========================================================

    print(
        "\n========== FINAL SUMMARY =========="
    )

    print(
        "Completed:",
        len(completed),
        "/17"
    )

    print(
        "Failed:",
        failed
    )

    print(
        f"Total wall time: "
        f"{wall_time:.2f} min"
    )

    print(
        "\n========== SAVED FILES =========="
    )

    print(
        "WIDE:",
        wide_path
    )

    print(
        "LONG:",
        long_path
    )

    print(
        "SUMMARY:",
        summary_path
    )

    print(
        "GRID DIRECTORY:",
        os.path.join(
            OUTPUT_DIR,
            "grids"
        )
    )

    print(
        "CASE DIRECTORY:",
        os.path.join(
            OUTPUT_DIR,
            "cases"
        )
    )

    print(
        "\nFinished.",
        flush=True
    )


# ============================================================
# 9. RUN
# ============================================================

if __name__ == "__main__":

    main()
