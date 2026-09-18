# %% ================== run_direct_abm_mle_bounded_cumulative.py ==================
# Direct ABM MLE using BatchRunnerMP
#
# beta_ABM is optimized in [0.03, 0.05]
# each likelihood evaluation uses 50 stochastic ABM simulations
# CPU processes = 25
#
# For each beta:
# daily HAI incidence
# -> monthly HAI incidence
# -> monthly cumulative HAI
# -> mean/std from 50 ABM runs
# -> Gaussian NLL against observed cumulative HAI
#
# Optimization:
# scipy.optimize.minimize_scalar(method="bounded")

from model.cpe_model_month_lambda import CPE_Model_month
from model.cpe_model_month_lambda import getTotalInfec

from mesa.batchrunner import BatchRunnerMP
from mesa.datacollection import DataCollector

from multiprocessing import freeze_support
from scipy.optimize import minimize_scalar

import os
import ast
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Mesa BatchRunnerMP에서 agent reporter 없어서 터지는 문제 방지
# --------------------------------------------------
def _safe_get_agent_vars_dataframe(self):
    return pd.DataFrame()


DataCollector.get_agent_vars_dataframe = _safe_get_agent_vars_dataframe

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="No agent reporters*")


# ==================================================
# Helper functions
# ==================================================

def parse_series(x):
    """
    BatchRunnerMP output에서 HCW_related_infecs가
    list, ndarray, string 중 어떤 형태로 들어와도 list로 변환.
    """

    if isinstance(x, list):
        return x

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            raise ValueError(f"Cannot parse series string: {x[:200]}")

    raise TypeError(f"Unsupported series type: {type(x)}")


def daily_incidence_to_monthly_incidence(
    daily_series,
    n_months,
    days_per_month=30
):
    """
    daily HAI incidence list를 monthly HAI incidence로 변환.
    """

    daily_series = parse_series(daily_series)
    daily_series = np.asarray(daily_series, dtype=float)

    monthly_incidence = []

    for m in range(n_months):
        start = days_per_month * m
        end = days_per_month * (m + 1)

        month_sum = np.sum(daily_series[start:end])
        monthly_incidence.append(month_sum)

    return np.asarray(monthly_incidence, dtype=float)


def daily_incidence_to_monthly_cumulative(
    daily_series,
    n_months,
    days_per_month=30
):
    """
    daily HAI incidence
    -> monthly HAI incidence
    -> monthly cumulative HAI
    """

    monthly_incidence = daily_incidence_to_monthly_incidence(
        daily_series=daily_series,
        n_months=n_months,
        days_per_month=days_per_month,
    )

    monthly_cumulative = np.cumsum(monthly_incidence)

    return monthly_incidence, monthly_cumulative


def gaussian_nll(obs, mean, sd, sd_floor=1e-6):
    """
    Gaussian negative log-likelihood.

    obs:
        observed monthly cumulative HAI

    mean:
        mean monthly cumulative HAI from 50 ABM simulations

    sd:
        std monthly cumulative HAI from 50 ABM simulations

    sd_floor:
        division by zero 방지용 최소 SD
    """

    obs = np.asarray(obs, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)

    sigma = np.maximum(sd, sd_floor)
    var = sigma ** 2

    nll = 0.5 * np.sum(
        np.log(2 * np.pi * var)
        + ((obs - mean) ** 2) / var
    )

    return float(nll), sigma


def sse_loss(obs, mean):
    obs = np.asarray(obs, dtype=float)
    mean = np.asarray(mean, dtype=float)

    return float(np.sum((obs - mean) ** 2))


def check_cumulative(arr):
    """
    cumulative trajectory가 감소하는지 확인.
    """

    arr = np.asarray(arr, dtype=float)

    return not np.any(np.diff(arr) < -1e-12)


# ==================================================
# Main
# ==================================================

def main():

    # --------------------------------------------------
    # Basic settings
    # --------------------------------------------------
    data_type = "A"

    n_months = 19
    days_per_month = 30
    runtime_days = days_per_month * n_months

    num_iter = 50
    nr_processes = 25

    probNewPatient = 0.003
    isolationFactor = 0.75

    cleaningDay = 180
    washrate = 0.9
    isolationTime = 14

    init_envc = 9
    tau_offset_days = 140

    height = 11
    width = 32

    # --------------------------------------------------
    # Observed monthly HAI: Period A
    # --------------------------------------------------
    y_month = np.array([
        5, 2, 0, 2, 1, 1, 2, 2, 6,
        1, 1, 0, 2, 1, 1, 2, 5, 2, 2
    ], dtype=float)

    y_cum = np.cumsum(y_month)

    # --------------------------------------------------
    # Optimization setting
    # --------------------------------------------------
    beta_init = 0.04

    beta_lower = 0.03
    beta_upper = 0.05

    sd_floor = 1e-6

    # bounded scalar optimization
    maxiter = 20

    # beta 방향 종료 tolerance
    # 더 세밀하게 찾으려면 1e-5 정도 사용
    xatol = 0.001

    print("=" * 80)
    print("START direct ABM MLE with bounded scalar optimization")
    print("=" * 80)

    print("data_type:", data_type)
    print("beta initial comparison value:", beta_init)
    print("beta bounds:", beta_lower, beta_upper)

    print("num_iter per beta:", num_iter)
    print("nr_processes:", nr_processes)

    print("observed cumulative:", y_cum)

    print("sd_floor:", sd_floor)
    print("xatol:", xatol)

    print("=" * 80)

    # --------------------------------------------------
    # result directory
    # --------------------------------------------------
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    result_dir = os.path.join(
        base_dir,
        "..",
        "result"
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # max_steps 계산
    # --------------------------------------------------
    temp_model = CPE_Model_month(
        data_type=data_type,

        prob_new_patient=probNewPatient,
        prob_transmission=beta_init,

        isolation_factor=isolationFactor,

        cleaningDay=cleaningDay,
        hcw_wash_rate=washrate,
        isolation_time=isolationTime,

        height=height,
        width=width,

        init_env=init_envc,
        tau_offset_days=tau_offset_days,
    )

    max_steps = (
        temp_model.ticks_in_day
        * runtime_days
    )

    print(
        "ticks_in_day:",
        temp_model.ticks_in_day
    )

    print(
        "runtime_days:",
        runtime_days
    )

    print(
        "max_steps:",
        max_steps
    )

    fixed_params = {

        "data_type": data_type,

        "prob_new_patient": probNewPatient,

        "isolation_factor": isolationFactor,

        "cleaningDay": cleaningDay,

        "hcw_wash_rate": washrate,

        "isolation_time": isolationTime,

        "height": height,
        "width": width,

        "init_env": init_envc,

        "tau_offset_days": tau_offset_days,
    }

    # --------------------------------------------------
    # cache
    # --------------------------------------------------
    cache = {}

    raw_rows = []

    trajectory_rows = []

    # ==================================================
    # evaluate beta
    # ==================================================
    def evaluate_beta_scalar(beta_value):

        # 실제 ABM 평가 beta는 소수점 5자리까지
        beta = round(
            float(beta_value),
            5
        )

        # bounds 밖이면 큰 penalty
        if (
            beta < beta_lower
            or beta > beta_upper
        ):
            return 1e100

        # ----------------------------------------------
        # 이미 평가한 beta면 다시 ABM 돌리지 않음
        # ----------------------------------------------
        if beta in cache:

            print(
                f"[CACHE] "
                f"beta={beta:.5f}, "
                f"nll={cache[beta]['nll']:.6f}"
            )

            return cache[beta]["nll"]

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Evaluating beta = {beta:.5f}"
        )

        print(
            "=" * 80
        )

        variable_params = {

            "prob_transmission": [beta],

        }

        start_time = time.time()

        # ----------------------------------------------
        # ABM 50 repeated stochastic simulations
        # ----------------------------------------------
        batch_run = BatchRunnerMP(

            CPE_Model_month,

            nr_processes=nr_processes,

            variable_parameters=variable_params,

            fixed_parameters=fixed_params,

            iterations=num_iter,

            max_steps=max_steps,

            display_progress=True,

            model_reporters={

                "HCW_related_infecs":
                    getTotalInfec

            }

        )

        batch_run.run_all()

        run_data = (
            batch_run
            .get_model_vars_dataframe()
        )

        run_data = (
            run_data
            .reset_index(drop=True)
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            "run_data shape:",
            run_data.shape
        )

        if (
            run_data.shape[0]
            != num_iter
        ):

            print(
                "[WARNING] expected runs:",
                num_iter
            )

            print(
                "[WARNING] actual runs:",
                run_data.shape[0]
            )

        monthly_inc_sims = []

        monthly_cum_sims = []

        # ----------------------------------------------
        # 각 stochastic ABM run 처리
        # ----------------------------------------------
        for i, row in run_data.iterrows():

            daily_series = (
                row["HCW_related_infecs"]
            )

            (
                monthly_inc,
                monthly_cum

            ) = daily_incidence_to_monthly_cumulative(

                daily_series=daily_series,

                n_months=n_months,

                days_per_month=days_per_month,

            )

            if not check_cumulative(
                monthly_cum
            ):

                print(
                    "[WARNING] "
                    "cumulative trajectory decreases."
                )

                print(
                    "beta:",
                    beta,
                    "iteration:",
                    i
                )

                print(
                    monthly_cum
                )

            monthly_inc_sims.append(
                monthly_inc
            )

            monthly_cum_sims.append(
                monthly_cum
            )

            # ------------------------------------------
            # raw simulation 저장
            # ------------------------------------------
            raw_row = {

                "beta": beta,

                "iteration": i,

            }

            for m in range(n_months):

                raw_row[
                    f"monthly_incidence_m{m+1}"
                ] = monthly_inc[m]

                raw_row[
                    f"monthly_cumulative_m{m+1}"
                ] = monthly_cum[m]

            raw_rows.append(
                raw_row
            )

        # ----------------------------------------------
        # 50 simulations -> array
        # ----------------------------------------------
        monthly_inc_sims = np.asarray(
            monthly_inc_sims,
            dtype=float
        )

        monthly_cum_sims = np.asarray(
            monthly_cum_sims,
            dtype=float
        )

        # ----------------------------------------------
        # mean / std
        # ----------------------------------------------
        mean_monthly = (
            monthly_inc_sims
            .mean(axis=0)
        )

        sd_monthly = (
            monthly_inc_sims
            .std(
                axis=0,
                ddof=1
            )
        )

        mean_cum = (
            monthly_cum_sims
            .mean(axis=0)
        )

        sd_cum = (
            monthly_cum_sims
            .std(
                axis=0,
                ddof=1
            )
        )

        # ----------------------------------------------
        # Gaussian NLL
        # ----------------------------------------------
        nll, sigma_used = gaussian_nll(

            obs=y_cum,

            mean=mean_cum,

            sd=sd_cum,

            sd_floor=sd_floor,

        )

        # ----------------------------------------------
        # SSE도 참고용 저장
        # ----------------------------------------------
        sse = sse_loss(

            obs=y_cum,

            mean=mean_cum,

        )

        # ----------------------------------------------
        # cache 저장
        # ----------------------------------------------
        cache[beta] = {

            "beta":
                beta,

            "nll":
                nll,

            "sse":
                sse,

            "mean_monthly":
                mean_monthly,

            "sd_monthly":
                sd_monthly,

            "mean_cum":
                mean_cum,

            "sd_cum":
                sd_cum,

            "sigma_used":
                sigma_used,

            "elapsed_seconds":
                elapsed,

            "n_abm_runs":
                num_iter,

        }

        # ----------------------------------------------
        # trajectory summary 저장
        # ----------------------------------------------
        for m in range(n_months):

            trajectory_rows.append({

                "beta":
                    beta,

                "month":
                    m + 1,

                "obs_monthly":
                    y_month[m],

                "obs_cumulative":
                    y_cum[m],

                "abm_mean_monthly":
                    mean_monthly[m],

                "abm_sd_monthly":
                    sd_monthly[m],

                "abm_mean_cumulative":
                    mean_cum[m],

                "abm_sd_cumulative":
                    sd_cum[m],

                "sigma_used":
                    sigma_used[m],

                "residual_cumulative":
                    y_cum[m]
                    - mean_cum[m],

            })

        print(
            "beta:",
            beta
        )

        print(
            "NLL:",
            nll
        )

        print(
            "SSE:",
            sse
        )

        print(
            "ABM mean cumulative:",
            mean_cum
        )

        print(
            "ABM sd cumulative:",
            sd_cum
        )

        print(
            "min sd cumulative:",
            np.min(sd_cum)
        )

        print(
            "elapsed seconds:",
            elapsed
        )

        print(
            "total beta evaluations:",
            len(cache)
        )

        print(
            "total ABM simulations:",
            len(cache) * num_iter
        )

        return nll

    # --------------------------------------------------
    # beta = 0.04 먼저 한 번 평가
    # 기존 결과와 비교하기 위한 reference
    # --------------------------------------------------
    evaluate_beta_scalar(
        beta_init
    )

    # --------------------------------------------------
    # bounded scalar optimization
    # --------------------------------------------------
    opt_res = minimize_scalar(

        evaluate_beta_scalar,

        bounds=(
            beta_lower,
            beta_upper
        ),

        method="bounded",

        options={

            "xatol":
                xatol,

            "maxiter":
                maxiter,

            "disp":
                3,

        },

    )

    beta_hat = round(
        float(opt_res.x),
        5
    )

    # 혹시 모를 numerical error 방지
    beta_hat = min(
        max(
            beta_hat,
            beta_lower
        ),
        beta_upper
    )

    # optimizer 최종값도 실제 평가
    nll_hat = evaluate_beta_scalar(
        beta_hat
    )

    # --------------------------------------------------
    # 실제 평가된 beta 정리
    # --------------------------------------------------
    profile_rows = []

    for beta, res in cache.items():

        profile_rows.append({

            "beta":
                beta,

            "nll":
                res["nll"],

            "sse":
                res["sse"],

            "n_abm_runs":
                res["n_abm_runs"],

            "elapsed_seconds":
                res["elapsed_seconds"],

            "min_sd_cum":
                float(
                    np.min(
                        res["sd_cum"]
                    )
                ),

            "mean_sd_cum":
                float(
                    np.mean(
                        res["sd_cum"]
                    )
                ),

        })

    df_profile = pd.DataFrame(
        profile_rows
    )

    df_profile = (
        df_profile
        .sort_values("beta")
        .reset_index(drop=True)
    )

    # ----------------------------------------------
    # 실제 평가된 beta 중 NLL 최소
    # ----------------------------------------------
    best_idx = (
        df_profile["nll"]
        .idxmin()
    )

    best_beta = (
        df_profile.loc[
            best_idx,
            "beta"
        ]
    )

    best_nll = (
        df_profile.loc[
            best_idx,
            "nll"
        ]
    )

    best_sse = (
        df_profile.loc[
            best_idx,
            "sse"
        ]
    )

    total_abm_simulations = (
        len(cache)
        * num_iter
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Optimization summary"
    )

    print(
        "=" * 80
    )

    print(
        "optimizer success:",
        opt_res.success
    )

    print(
        "optimizer message:",
        opt_res.message
    )

    print(
        "optimizer beta_hat:",
        beta_hat
    )

    print(
        "optimizer nll_hat:",
        nll_hat
    )

    print(
        "best evaluated beta:",
        best_beta
    )

    print(
        "best evaluated nll:",
        best_nll
    )

    print(
        "best evaluated sse:",
        best_sse
    )

    print(
        "unique beta evaluations:",
        len(cache)
    )

    print(
        "total ABM simulations:",
        total_abm_simulations
    )

    print(
        "=" * 80
    )

    # ==================================================
    # Save files
    # ==================================================

    df_traj = pd.DataFrame(
        trajectory_rows
    )

    df_traj = (
        df_traj
        .sort_values(
            ["beta", "month"]
        )
        .reset_index(drop=True)
    )

    df_raw = pd.DataFrame(
        raw_rows
    )

    df_raw = (
        df_raw
        .sort_values(
            ["beta", "iteration"]
        )
        .reset_index(drop=True)
    )

    df_cost = pd.DataFrame([
        {

            "method":
                "Direct ABM MLE with bounded minimize_scalar",

            "data_type":
                data_type,

            "likelihood":
                "Gaussian NLL",

            "data_used":
                "monthly cumulative HAI",

            "std_source":
                "std from 50 ABM cumulative trajectories for each beta",

            "sd_floor":
                sd_floor,

            "beta_init_reference":
                beta_init,

            "beta_lower":
                beta_lower,

            "beta_upper":
                beta_upper,

            "xatol":
                xatol,

            "optimizer_success":
                opt_res.success,

            "optimizer_beta_hat":
                beta_hat,

            "optimizer_nll_hat":
                nll_hat,

            "best_evaluated_beta":
                best_beta,

            "best_evaluated_nll":
                best_nll,

            "best_evaluated_sse":
                best_sse,

            "unique_beta_evaluations":
                len(cache),

            "repetitions_per_beta":
                num_iter,

            "cpu_processes":
                nr_processes,

            "total_abm_simulations":
                total_abm_simulations,

        }
    ])

    # --------------------------------------------------
    # 파일명: fmin -> bounded 로 변경
    # --------------------------------------------------

    profile_csv = os.path.join(
        result_dir,
        "direct_abm_mle_bounded_cumulative_profile.csv"
    )

    traj_csv = os.path.join(
        result_dir,
        "direct_abm_mle_bounded_cumulative_trajectory_summary.csv"
    )

    raw_csv = os.path.join(
        result_dir,
        "direct_abm_mle_bounded_cumulative_raw_iterations.csv"
    )

    cost_csv = os.path.join(
        result_dir,
        "direct_abm_mle_bounded_cumulative_cost_summary.csv"
    )

    df_profile.to_csv(
        profile_csv,
        index=False,
        encoding="utf-8"
    )

    df_traj.to_csv(
        traj_csv,
        index=False,
        encoding="utf-8"
    )

    df_raw.to_csv(
        raw_csv,
        index=False,
        encoding="utf-8"
    )

    df_cost.to_csv(
        cost_csv,
        index=False,
        encoding="utf-8"
    )

    print(
        "saved:",
        profile_csv
    )

    print(
        "saved:",
        traj_csv
    )

    print(
        "saved:",
        raw_csv
    )

    print(
        "saved:",
        cost_csv
    )

    # ==================================================
    # Plot NLL profile
    # ==================================================

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(

        df_profile["beta"],

        df_profile["nll"],

        marker="o",

        linewidth=1.8,

    )

    plt.axvline(

        best_beta,

        linestyle="--",

        label=(
            fr"Best evaluated "
            fr"$\beta_{{ABM}}$ = "
            fr"{best_beta:.5f}"
        ),

    )

    plt.xlabel(
        r"$\beta_{\mathrm{ABM}}$"
    )

    plt.ylabel(
        "Gaussian negative log-likelihood"
    )

    plt.title(
        "Direct ABM MLE profile "
        "(bounded scalar optimization)"
    )

    plt.legend()

    plt.tight_layout()

    profile_fig = os.path.join(
        result_dir,
        "direct_abm_mle_bounded_cumulative_profile.png"
    )

    plt.savefig(
        profile_fig,
        dpi=300
    )

    plt.show()

    print(
        "saved:",
        profile_fig
    )

    # ==================================================
    # Plot observed vs best ABM cumulative fit
    # ==================================================

    best_traj = (
        df_traj[
            df_traj["beta"]
            == best_beta
        ]
        .copy()
    )

    months = (
        best_traj["month"]
        .values
    )

    obs_cumulative = (
        best_traj[
            "obs_cumulative"
        ]
        .values
    )

    abm_mean_cumulative = (
        best_traj[
            "abm_mean_cumulative"
        ]
        .values
    )

    abm_sd_cumulative = (
        best_traj[
            "abm_sd_cumulative"
        ]
        .values
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(

        months,

        obs_cumulative,

        marker="o",

        linewidth=1.8,

        label=(
            "Observed cumulative HAI"
        ),

    )

    plt.plot(

        months,

        abm_mean_cumulative,

        marker="s",

        linewidth=1.8,

        label=(
            fr"ABM mean, "
            fr"$\beta_{{ABM}}$="
            fr"{best_beta:.5f}"
        ),

    )

    plt.fill_between(

        months,

        (
            abm_mean_cumulative
            - abm_sd_cumulative
        ),

        (
            abm_mean_cumulative
            + abm_sd_cumulative
        ),

        alpha=0.2,

        label="ABM mean ± 1 SD",

    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Cumulative HAI"
    )

    plt.title(
        "Observed vs direct ABM-fitted cumulative HAI"
    )

    plt.legend()

    plt.tight_layout()

    fit_fig = os.path.join(
        result_dir,
        "direct_abm_mle_bounded_cumulative_best_fit.png"
    )

    plt.savefig(
        fit_fig,
        dpi=300
    )

    plt.show()

    print(
        "saved:",
        fit_fig
    )

    print(
        "\nDONE"
    )


if __name__ == "__main__":

    freeze_support()

    main()