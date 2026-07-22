# %% ================== run_interv_cleaning_beta.py ==================

import os
import pandas as pd
import numpy as np
from multiprocessing import freeze_support

from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunnerMP

from model.cpe_model_month_lambda import CPE_Model_month, getTotalInfec


def _safe_get_agent_vars_dataframe(self):
    return pd.DataFrame()

DataCollector.get_agent_vars_dataframe = _safe_get_agent_vars_dataframe


def main():

    data_type = "A"

    init_envc = 9
    first_clean_day = 20

    runtime = 30 * 19

    num_iter = 50
    nr_processes = 16

    variable_name = "prob_transmission"

    beta_values = np.round(
        np.arange(0.02, 0.0601, 0.005),
        5
    )

    cleaning_values = [60, 90, 180, 360]

    beta_tag1 = beta_values[0]
    beta_tag2 = beta_values[-1]

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    result_dir = os.path.join(base_dir, "..", "result")
    os.makedirs(result_dir, exist_ok=True)

    scenario_tag = f"{data_type}{init_envc}_firstclean{first_clean_day}"

    all_long = []

    print("=" * 80)
    print("START cleaningDay intervention ABM run")
    print("data_type:", data_type)
    print("init_envc:", init_envc)
    print("first_clean_day:", first_clean_day)
    print("cleaning_values:", cleaning_values)
    print("beta_values:", beta_values)
    print("num_iter:", num_iter)
    print("runtime:", runtime)
    print("=" * 80)

    for clean_day in cleaning_values:

        init_tau0 = clean_day - first_clean_day

        print("\n" + "=" * 80)
        print("RUN cleanDay =", clean_day)
        print("first cleaning day =", first_clean_day)
        print("init_tau0 =", init_tau0)
        print("=" * 80)

        fixed_params = {
            "data_type": data_type,
            "init_envc": init_envc,
            "init_tau0": init_tau0,

            "height": 11,
            "width": 32,

            "runtime": runtime,

            "probNewPatient": 0.003,
            "probTransmission": 0.0866,

            "washrate": 0.9,
            "isolationTime": 14,
            "isolationFactor": 0.75,

            "cleanDay": clean_day,
        }

        variable_params = {
            "probTransmission": beta_values,
        }

        model_reporters = {
            "TotalInfec": getTotalInfec,
        }

        batch_run = BatchRunnerMP(
            CPE_Model_month,
            variable_parameters=variable_params,
            fixed_parameters=fixed_params,
            iterations=num_iter,
            max_steps=runtime,
            model_reporters=model_reporters,
            nr_processes=nr_processes
        )

        batch_run.run_all()

        print("done running cleanDay =", clean_day)

        run_data = batch_run.get_model_vars_dataframe()

        print("cols:", list(run_data.columns))
        print("run_data shape before reset:", run_data.shape)

        run_data = run_data.reset_index(drop=True)

        run_data["cleanDay"] = clean_day
        run_data["first_clean_day"] = first_clean_day
        run_data["init_tau0_used"] = init_tau0

        run_data["my_iteration"] = (
            run_data
            .groupby(["cleanDay", "probTransmission"])
            .cumcount()
        )

        check_counts = (
            run_data
            .groupby(["cleanDay", "probTransmission"])
            .size()
            .reset_index(name="n")
        )

        print("min n:", check_counts["n"].min())
        print("max n:", check_counts["n"].max())

        all_long.append(run_data.copy())

        df_wide = run_data.pivot_table(
            index="my_iteration",
            columns="probTransmission",
            values="TotalInfec",
            aggfunc="first"
        )

        df_wide = df_wide.sort_index(axis=0).sort_index(axis=1)

        print("df_wide shape:", df_wide.shape)
        print(df_wide.head())

        out_csv = os.path.join(
            result_dir,
            f"interv_{variable_name}_"
            f"{scenario_tag}_{beta_tag1}-{beta_tag2}_cleaning{clean_day}.csv"
        )

        df_wide.to_csv(out_csv, index=False, encoding="utf-8")
        print("saved:", out_csv)

    long_df = pd.concat(all_long, ignore_index=True)

    long_csv_path = os.path.join(
        result_dir,
        f"interv_{variable_name}_LONG_"
        f"{scenario_tag}_{beta_tag1}-{beta_tag2}_cleaningALL.csv"
    )

    long_df.to_csv(long_csv_path, index=False, encoding="utf-8")
    print("\nsaved LONG raw:", long_csv_path)

    print("\nDONE cleaningDay intervention ABM run")


if __name__ == "__main__":
    freeze_support()
    main()