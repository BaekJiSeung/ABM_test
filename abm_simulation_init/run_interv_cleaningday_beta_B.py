# %% ================== run_interv_cleaningday_beta_B.py ==================

from model.cpe_model_month_lambda import CPE_Model_month
from model.cpe_model_month_lambda import getTotalInfec
from mesa.batchrunner import BatchRunnerMP
from multiprocessing import freeze_support
import os
import time
import pandas as pd
import numpy as np
import warnings
from mesa.datacollection import DataCollector


# --------------------------------------------------
# Mesa BatchRunnerMP에서 agent reporter 없어서 터지는 문제 방지
# --------------------------------------------------
def _safe_get_agent_vars_dataframe(self):
    return pd.DataFrame()


DataCollector.get_agent_vars_dataframe = _safe_get_agent_vars_dataframe

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="No agent reporters*")


def main():

    # %% Basic settings
    data_type = "B"
    num_iter = 50
    nr_processes = 24

    # selected initial condition setting for Period B
    init_envc = 2

    # cleaning intervention setting
    first_clean_day = 20

    # baseline intervention settings
    washrate = 0.9
    isolationTime = 14

    # B period: 2021 Jan. – 2023 Dec. = 36 months
    runtime = 30 * 36

    # fixed model parameters
    probNewPatient = 0.003
    isolationFactor = 0.75

    height = 11
    width = 32

    # %% Variable settings
    variable_name = "prob_transmission"

    # Step4 mapping용 beta_ABM grid
    beta_values = np.round(np.arange(0.02, 0.0601, 0.005), 5)
    beta_values = [0.04136]
    # cleaningDay intervention values
    cleaning_values = [60, 90, 180, 360]

    beta_tag1 = f"{beta_values[0]:.5f}"
    beta_tag2 = f"{beta_values[-1]:.5f}"

    scenario_tag = f"{data_type}{init_envc}_firstclean{first_clean_day}"

    print("=" * 80)
    print("START cleaningDay intervention ABM run - Period B")
    print("data_type:", data_type)
    print("init_envc:", init_envc)
    print("first_clean_day:", first_clean_day)
    print("washrate:", washrate)
    print("isolationTime:", isolationTime)
    print("runtime days:", runtime)
    print("num_iter:", num_iter)
    print("nr_processes:", nr_processes)
    print("beta values:", beta_values)
    print("cleaning values:", cleaning_values)
    print("number of beta values:", len(beta_values))
    print("number of cleaning values:", len(cleaning_values))
    print("total parameter settings:", len(beta_values) * len(cleaning_values))
    print("total simulations:", len(beta_values) * len(cleaning_values) * num_iter)
    print("=" * 80)

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    result_dir = os.path.join(base_dir, "..", "result")
    os.makedirs(result_dir, exist_ok=True)

    all_long = []
    start_time_all = time.time()

    # --------------------------------------------------
    # cleaningDay별로 따로 실행
    # cleanDay마다 tau_offset_days가 달라져야 첫 청소가 day 20에 맞춰짐
    # --------------------------------------------------
    for clean_day in cleaning_values:

        tau_offset_days = clean_day - first_clean_day

        print("\n" + "=" * 80)
        print("RUN cleaningDay =", clean_day)
        print("first_clean_day:", first_clean_day)
        print("tau_offset_days:", tau_offset_days)
        print("=" * 80)

        fixed_params = {
            "data_type": data_type,

            "prob_new_patient": probNewPatient,
            "isolation_factor": isolationFactor,

            "cleaningDay": clean_day,
            "hcw_wash_rate": washrate,
            "isolation_time": isolationTime,

            "height": height,
            "width": width,

            "init_env": init_envc,
            "tau_offset_days": tau_offset_days,
        }

        variable_params = {
            "prob_transmission": beta_values,
        }

        model = CPE_Model_month(
            data_type=data_type,
            prob_new_patient=probNewPatient,
            prob_transmission=float(beta_values[0]),
            isolation_factor=isolationFactor,
            cleaningDay=clean_day,
            hcw_wash_rate=washrate,
            isolation_time=isolationTime,
            height=height,
            width=width,
            init_env=init_envc,
            tau_offset_days=tau_offset_days,
        )

        max_steps = model.ticks_in_day * runtime
        print("\nmax_steps:", max_steps)

        start_time = time.time()

        print("\nnow run")

        batch_run = BatchRunnerMP(
            CPE_Model_month,
            nr_processes=nr_processes,
            variable_parameters=variable_params,
            fixed_parameters=fixed_params,
            iterations=num_iter,
            max_steps=max_steps,
            display_progress=True,
            model_reporters={
                "HCW_related_infecs": getTotalInfec
            }
        )

        batch_run.run_all()

        run_data = batch_run.get_model_vars_dataframe()

        elapsed = time.time() - start_time

        print("\ndone running cleaningDay =", clean_day)
        print("elapsed seconds:", elapsed)
        print("run_data shape before reset:", run_data.shape)
        print("cols:", list(run_data.columns))
        print(run_data.head())

        run_data = run_data.reset_index(drop=True)

        run_data["cleaningDay"] = clean_day
        run_data["init_env_used"] = init_envc
        run_data["first_clean_day"] = first_clean_day
        run_data["tau_offset_days_used"] = tau_offset_days

        run_data["my_iteration"] = (
            run_data
            .groupby(["cleaningDay", "prob_transmission"])
            .cumcount()
        )

        iter_col = "my_iteration"

        check_counts = (
            run_data
            .groupby(["cleaningDay", "prob_transmission"])
            .size()
            .reset_index(name="n")
        )

        print("\ncheck counts:")
        print(check_counts)
        print("min n:", check_counts["n"].min())
        print("max n:", check_counts["n"].max())

        if check_counts["n"].min() != num_iter or check_counts["n"].max() != num_iter:
            print("[WARNING] Some parameter settings do not have num_iter runs.")
        else:
            print("[OK] Every parameter setting has num_iter runs.")

        all_long.append(run_data.copy())

        df_wide = run_data.pivot_table(
            index=iter_col,
            columns=variable_name,
            values="HCW_related_infecs",
            aggfunc="first"
        ).reset_index(drop=True)

        df_wide.columns.name = None
        df_wide = df_wide.reindex(columns=beta_values)

        csv_path = os.path.join(
            result_dir,
            f"interv_{variable_name}_"
            f"{scenario_tag}_"
            f"{beta_tag1}-{beta_tag2}_cleaning{clean_day}.csv"
        )

        df_wide.to_csv(csv_path, index=False, encoding="utf-8")

        print("df_wide shape:", df_wide.shape)
        print(df_wide.head())
        print("saved ->", csv_path)

        expected_shape = (num_iter, len(beta_values))

        if df_wide.shape != expected_shape:
            print("[WARNING] Saved file shape is not expected.")
            print("expected:", expected_shape)
            print("actual:", df_wide.shape)
        else:
            print("[OK] Saved file shape is correct.")

    # --------------------------------------------------
    # LONG raw 파일 저장
    # --------------------------------------------------
    long_df = pd.concat(all_long, ignore_index=True)

    long_csv_path = os.path.join(
        result_dir,
        f"interv_{variable_name}_LONG_"
        f"{scenario_tag}_"
        f"{beta_tag1}-{beta_tag2}_cleaningALL.csv"
    )

    long_df.to_csv(long_csv_path, index=False, encoding="utf-8")
    print("\nsaved LONG raw ->", long_csv_path)

    elapsed_all = time.time() - start_time_all
    print("\nDONE cleaningDay intervention ABM run - Period B")
    print("total elapsed seconds:", elapsed_all)


if __name__ == "__main__":
    freeze_support()
    main()