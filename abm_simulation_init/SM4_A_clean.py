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