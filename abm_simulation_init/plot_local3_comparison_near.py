
"""
Compare Period A ABM cumulative HAI with three parameter pairs
near the original local optimization results.

The parameter pairs are slightly perturbed for sensitivity analysis.
They are NOT three newly optimized parameter estimates.

Place this file beside SM4_A_hand_R2.py and run:

    python plot_near_local3_comparison.py

Optional:
    python plot_near_local3_comparison.py --abm-csv "C:/path/to/abm.csv"

Outputs:
    near_local3_HAI_comparison.png
    near_local3_HAI_comparison.pdf
    near_local3_HAI_predictions.csv
    near_local3_HAI_metrics.csv
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Parameter settings
# ============================================================

ABM_BETA = 0.040
ABM_P = 0.90

INIT_ENV = 9
TAU0 = 140

# Original local optimization results (reference only)
ORIGINAL_PAIRS = [
    ("Local 1", 2.314142, 0.800182),
    ("Local 2", 3.491119, 0.901164),
    ("Local 3", 3.073892, 0.876072),
]

# Slightly perturbed pairs for visualization
# These are NOT the original optimized parameter pairs.
PAIRS = [
    ("Pair 1", 2.314142, 0.797182),
    ("Pair 2", 3.491119, 0.901164),
    ("Pair 3", 3.073892, 0.879072),
]


# ============================================================
# 2. Utility functions
# ============================================================

def read_vector(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=float)


def load_model(path):
    spec = importlib.util.spec_from_file_location(
        "sm4_r2_near_local_plot",
        str(path.resolve())
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load model from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "simulate_theta"):
        raise AttributeError(
            f"simulate_theta() not found in {path}"
        )

    return module


def calculate_nll(observed, predicted, sd):
    return float(
        0.5 * np.sum(
            np.log(2 * np.pi * sd**2)
            + ((observed - predicted) / sd)**2
        )
    )


def calculate_rmse(observed, predicted):
    return float(
        np.sqrt(np.mean((observed - predicted)**2))
    )


def effective_parameter(beta, p):
    return beta**2 * (1 - p) / p


# ============================================================
# 3. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--abm-csv",
        type=Path,
        default=Path(
            "../result/"
            "interv_prob_transmission_summary_A9140_0.01-0.07.csv"
        )
    )

    parser.add_argument(
        "--model-script",
        type=Path,
        default=Path("SM4_A_hand_R2.py")
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("near_local3_comparison")
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not args.abm_csv.is_file():
        raise FileNotFoundError(
            f"ABM CSV not found: {args.abm_csv}\n"
            "Specify --abm-csv if necessary."
        )

    if not args.model_script.is_file():
        raise FileNotFoundError(
            f"Model script not found: {args.model_script}"
        )

    # --------------------------------------------------------
    # Load ABM results
    # --------------------------------------------------------

    df = pd.read_csv(args.abm_csv)

    for col in ("beta", "mean", "std"):
        if col not in df.columns:
            raise ValueError(
                f"Missing ABM CSV column: {col}"
            )

    beta_values = pd.to_numeric(
        df["beta"],
        errors="coerce"
    )

    mask = np.isclose(
        beta_values,
        ABM_BETA,
        rtol=0,
        atol=1e-7
    )

    if "handwash" in df.columns:
        p_values = pd.to_numeric(
            df["handwash"],
            errors="coerce"
        )

        mask &= np.isclose(
            p_values,
            ABM_P,
            rtol=0,
            atol=1e-7
        )

    subset = df.loc[mask]

    if subset.empty:
        raise ValueError(
            "No matching ABM condition found."
        )

    if len(subset) != 1:
        raise ValueError(
            f"Expected one ABM row, found {len(subset)}"
        )

    row = subset.iloc[0]

    abm_monthly_mean = read_vector(
        row["mean"]
    )

    abm_monthly_std = read_vector(
        row["std"]
    )

    if (
        len(abm_monthly_mean) != 19
        or len(abm_monthly_std) != 19
    ):
        raise ValueError(
            "Expected 19 monthly ABM values."
        )

    if np.any(abm_monthly_std < 0):
        raise ValueError(
            "ABM standard deviations must be nonnegative."
        )

    # Cumulative ABM mean
    abm_cumulative = np.cumsum(
        abm_monthly_mean
    )

    # Same approximate uncertainty calculation as R2
    abm_cumulative_sd = np.maximum(
        np.sqrt(np.cumsum(abm_monthly_std**2)),
        1e-6
    )

    months = np.arange(1, 20)

    # --------------------------------------------------------
    # Load surrogate
    # --------------------------------------------------------

    model = load_model(args.model_script)

    # --------------------------------------------------------
    # Predict using the three perturbed parameter pairs
    # --------------------------------------------------------

    prediction_columns = {
        "month": months,
        "abm_mean_cumulative": abm_cumulative,
        "abm_mean_minus_1sd":
            abm_cumulative - abm_cumulative_sd,
        "abm_mean_plus_1sd":
            abm_cumulative + abm_cumulative_sd,
    }

    results = []
    curves = {}

    for i, (name, beta_sm, p_sm) in enumerate(
        PAIRS
    ):

        original_name, original_beta, original_p = (
            ORIGINAL_PAIRS[i]
        )

        prediction = np.asarray(
            model.simulate_theta(
                beta_sm,
                INIT_ENV,
                TAU0,
                p_sm
            ),
            dtype=float
        )

        if prediction.shape != (19,):
            raise ValueError(
                f"{name}: expected 19 predictions."
            )

        if not np.all(np.isfinite(prediction)):
            raise ValueError(
                f"{name}: non-finite prediction."
            )

        nll = calculate_nll(
            abm_cumulative,
            prediction,
            abm_cumulative_sd
        )

        rmse = calculate_rmse(
            abm_cumulative,
            prediction
        )

        kappa = effective_parameter(
            beta_sm,
            p_sm
        )

        curves[name] = prediction

        column_name = name.lower().replace(" ", "_")

        prediction_columns[column_name] = prediction

        results.append({
            "pair": name,
            "beta_sm": beta_sm,
            "p_sm": p_sm,
            "original_beta_sm": original_beta,
            "original_p_sm": original_p,
            "delta_beta": beta_sm - original_beta,
            "delta_p": p_sm - original_p,
            "kappa": kappa,
            "nll": nll,
            "rmse_cumulative_hai": rmse,
            "final_cumulative_hai": prediction[-1],
            "parameter_type": (
                "near-local sensitivity pair"
            ),
        })

    # --------------------------------------------------------
    # Save CSV outputs
    # --------------------------------------------------------

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_df = pd.DataFrame(
        prediction_columns
    )

    metrics_df = pd.DataFrame(
        results
    )

    predictions_df.to_csv(
        args.out_dir /
        "near_local3_HAI_predictions.csv",
        index=False
    )

    metrics_df.to_csv(
        args.out_dir /
        "near_local3_HAI_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10.5, 6)
    )

    # ABM uncertainty band
    ax.fill_between(
        months,
        abm_cumulative - abm_cumulative_sd,
        abm_cumulative + abm_cumulative_sd,
        color="gray",
        alpha=0.17,
        label="ABM mean ± cumulative SD"
    )

    # ABM mean
    ax.plot(
        months,
        abm_cumulative,
        color="black",
        linewidth=2.4,
        marker="o",
        markersize=4,
        label="ABM mean",
        zorder=5
    )

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green"
    ]

    styles = [
        "-",
        "--",
        "-."
    ]

    for i, (name, beta_sm, p_sm) in enumerate(
        PAIRS
    ):

        ax.plot(
            months,
            curves[name],
            color=colors[i],
            linewidth=2.2,
            linestyle=styles[i],
            marker=["s", "^", "D"][i],
            markersize=3.5,
            label=(
                f"{name}: "
                f"β={beta_sm:.4f}, "
                f"p={p_sm:.4f}"
            ),
            zorder=3 + i
        )

    ax.set_title(
        "Sensitivity Analysis: Parameter Pairs Near Local Optima\n"
        r"ABM: $\beta_{ABM}=0.040$, $p_{ABM}=0.90$",
        fontsize=13
    )

    ax.set_xlabel(
        "Month",
        fontsize=12
    )

    ax.set_ylabel(
        "Cumulative HAI incidence",
        fontsize=12
    )

    ax.set_xticks(months)

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=9,
        loc="best"
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save figures
    # --------------------------------------------------------

    fig.savefig(
        args.out_dir /
        "near_local3_HAI_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        args.out_dir /
        "near_local3_HAI_comparison.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("NEAR-LOCAL PARAMETER SENSITIVITY COMPARISON")
    print("=" * 75)

    print(
        f"ABM condition: "
        f"beta={ABM_BETA:.5f}, "
        f"p={ABM_P:.2f}"
    )

    print(
        f"Surrogate: "
        f"init_env={INIT_ENV}, "
        f"tau0={TAU0}"
    )

    print("\nParameter pairs:")
    print(
        metrics_df[
            [
                "pair",
                "beta_sm",
                "p_sm",
                "delta_p",
                "kappa",
                "nll",
                "rmse_cumulative_hai",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nFinal cumulative HAI:")
    for name, _, _ in PAIRS:
        print(
            f"  {name}: "
            f"{curves[name][-1]:.6f}"
        )

    print(
        f"  ABM mean: "
        f"{abm_cumulative[-1]:.6f}"
    )

    print("\nOutput folder:")
    print(args.out_dir.resolve())

    print(
        "\nNOTE: These are manually perturbed "
        "parameter pairs, NOT new local optima."
    )


if __name__ == "__main__":
    main()
