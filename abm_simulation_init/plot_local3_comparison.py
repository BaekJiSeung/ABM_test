"""Compare Period A ABM cumulative HAI with three local optimization results.

Place beside SM4_A_hand_R2.py and run:
    python plot_local3_comparison.py

If ABM CSV is in another location:
    python plot_local3_comparison.py --abm-csv "C:/path/to/interv_prob_transmission_summary_A9140_0.01-0.07.csv"

Outputs: local3_HAI_comparison.png, local3_HAI_comparison.pdf,
         local3_HAI_predictions.csv, local3_HAI_metrics.csv
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAIRS = [
    ("Local 1", 2.314142, 0.800182, 60.643248, True),
    ("Local 2", 3.491119, 0.901164, 60.637264, True),
    ("Local 3", 3.073892, 0.876072, 60.638853, False),
]


def read_vector(v):
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return np.asarray(v, dtype=float)


def load_model(path):
    spec = importlib.util.spec_from_file_location("sm4_r2_for_plot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "simulate_theta"):
        raise AttributeError(f"simulate_theta() not found in {path}")
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abm-csv", type=Path, default=Path("../result/interv_prob_transmission_summary_A9140_0.01-0.07.csv"))
    parser.add_argument("--model-script", type=Path, default=Path("SM4_A_hand_R2.py"))
    parser.add_argument("--out-dir", type=Path, default=Path("local3_comparison"))
    args = parser.parse_args()

    if not args.abm_csv.is_file():
        raise FileNotFoundError(
            f"ABM CSV not found: {args.abm_csv}\n"
            "Run with --abm-csv pointing to the beta sweep CSV."
        )
    if not args.model_script.is_file():
        raise FileNotFoundError(f"Model script not found: {args.model_script}")

    df = pd.read_csv(args.abm_csv)
    for col in ("beta", "mean", "std"):
        if col not in df.columns:
            raise ValueError(f"Missing required ABM CSV column: {col}")

    mask = np.isclose(pd.to_numeric(df["beta"], errors="coerce"), 0.040, rtol=0, atol=1e-7)
    if "handwash" in df.columns:
        mask &= np.isclose(pd.to_numeric(df["handwash"], errors="coerce"), 0.90, rtol=0, atol=1e-7)
    subset = df.loc[mask]
    if subset.empty:
        raise ValueError("No ABM row found for beta_ABM=0.040 (p_ABM=0.90 if column exists)")
    if len(subset) != 1:
        raise ValueError(f"Found {len(subset)} matching ABM rows; need a unique condition")

    row = subset.iloc[0]
    abm_monthly_mean = read_vector(row["mean"])
    abm_monthly_std = read_vector(row["std"])
    if len(abm_monthly_mean) != 19 or len(abm_monthly_std) != 19:
        raise ValueError("Expected 19 months of ABM mean and std")
    if np.any(abm_monthly_std < 0):
        raise ValueError("ABM std cannot be negative")

    abm_cumulative = np.cumsum(abm_monthly_mean)
    # Match the exact weighting used by the R2 Gaussian NLL calculation.
    abm_cumulative_sd = np.maximum(np.sqrt(np.cumsum(abm_monthly_std ** 2)), 1e-6)
    months = np.arange(1, 20)

    model = load_model(args.model_script)
    init_env = getattr(model, "INIT_ENV", 9)
    tau0 = getattr(model, "TAU0", 140)

    prediction_columns = {
        "month": months,
        "abm_mean_cumulative": abm_cumulative,
        "abm_mean_minus_1sd": abm_cumulative - abm_cumulative_sd,
        "abm_mean_plus_1sd": abm_cumulative + abm_cumulative_sd,
    }
    results = []
    for name, beta_sm, p_sm, reported_nll, converged in PAIRS:
        prediction = np.asarray(model.simulate_theta(beta_sm, init_env, tau0, p_sm), dtype=float)
        if prediction.shape != (19,):
            raise ValueError(f"{name}: expected 19 predicted cumulative HAI values")
        # Match exactly the NLL definition used for local optimization.
        nll = float(0.5 * np.sum(
            np.log(2 * np.pi * abm_cumulative_sd ** 2)
            + ((abm_cumulative - prediction) / abm_cumulative_sd) ** 2
        ))
        rmse = float(np.sqrt(np.mean((abm_cumulative - prediction) ** 2)))
        prediction_columns[name.lower().replace(" ", "_")] = prediction
        results.append({
            "local": name, "beta_sm": beta_sm, "p_sm": p_sm,
            "converged": converged, "reported_nll": reported_nll,
            "recalculated_nll": nll, "nll_difference": nll - reported_nll,
            "rmse_cumulative_hai": rmse,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_columns).to_csv(args.out_dir / "local3_HAI_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(args.out_dir / "local3_HAI_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.fill_between(months, abm_cumulative - abm_cumulative_sd,
                    abm_cumulative + abm_cumulative_sd,
                    alpha=0.17, label="ABM mean ± cumulative SD")
    ax.plot(months, abm_cumulative, color="black", linewidth=2.4,
            marker="o", markersize=4, label="ABM mean")
    for name, beta_sm, p_sm, _, converged in PAIRS:
        curve = prediction_columns[name.lower().replace(" ", "_")]
        ax.plot(months, curve, linewidth=2, linestyle="--" if converged else ":",
                label=f"{name}: β={beta_sm:.3f}, p={p_sm:.3f}"
                      + (" (not converged)" if not converged else ""))
    ax.set_title("Comparison of Three Local Optimization Results\n"
                 r"ABM: $\beta_{ABM}=0.040$, $p_{ABM}=0.90$")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative HAI incidence")
    ax.set_xticks(months)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(args.out_dir / "local3_HAI_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out_dir / "local3_HAI_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    print("\nResults (local parameter pairs are FINAL optimization results):")
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nOutput folder:", args.out_dir.resolve())
    if any(abs(r["nll_difference"]) > 0.05 for r in results):
        print("\nWARNING: Recalculated NLL differs from reported NLL by >0.05.")
        print("Check that the ABM CSV, model script, and NLL definition match the original fit.")


if __name__ == "__main__":
    main()
