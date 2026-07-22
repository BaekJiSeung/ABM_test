# %% ================== run_interv_isoltime_beta.py ==================

import os
import pandas as pd
import numpy as np
from multiprocessing import freeze_support

from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunnerMP

from model.cpe_model_month_lambda import CPE_Model_month, getTotalInfec


# --------------------------------------------------
# Mesa DataCollector monkey patch
# --------------------------------------------------
def _safe_get_agent_vars_dataframe(self):
    return pd.DataFrame()

DataCollector.get_agent_vars_dataframe = _safe_get_agent_vars_dataframe


# --------------------------------------------------
# main
# --------------------------------------------------
def main():

    # -----------------------------
    # settings
    # -----------------------------
    data_type = "A"

    init_envc = 9
    init_tau0 = 140

    runtime = 30 * 19

    num_iter = 50
    nr_processes = 16

    variable_name = "prob_transmission"

    beta_values = np.round(
        np.arange(0.02, 0.0601, 0.005),
        5
    )

    # isolationTime intervention values
    isoltime_values = [6, 14, 20, 28]

    beta_tag1 = beta_values[0]
    beta_tag2 = beta_values[-1]

    # -----------------------------
    # fixed baseline parameters
    # -----------------------------
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
        "cleanDay": 180,
        "isolationFactor": 0.75,

        # isolationTime은 variable_params에서 바꿈
        # "isolationTime": 14,
    }

    variable_params = {
        "probTransmission": beta_values,
        "isolationTime": isoltime_values,
    }

    model_reporters = {
        "TotalInfec": getTotalInfec,
    }

    print("=" * 80)
    print("START isolationTime intervention ABM run")
    print("data_type:", data_type)
    print("beta_values:", beta_values)
    print("isoltime_values:", isoltime_values)
    print("num_iter:", num_iter)
    print("runtime:", runtime)
    print("=" * 80)

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

    print("done running")

    run_data = batch_run.get_model_vars_dataframe()

    print("cols:", list(run_data.columns))
    print("run_data shape before reset:", run_data.shape)

    # --------------------------------------------------
    # IMPORTANT:
    # BatchRunnerMP의 Run column을 iteration으로 쓰면 안 됨.
    # 직접 groupby cumcount로 iteration 생성.
    # --------------------------------------------------
    run_data = run_data.reset_index(drop=True)

    run_data["my_iteration"] = (
        run_data
        .groupby(["isolationTime", "probTransmission"])
        .cumcount()
    )

    check_counts = (
        run_data
        .groupby(["isolationTime", "probTransmission"])
        .size()
        .reset_index(name="n")
    )

    print("check_counts head:")
    print(check_counts.head())

    print("min n:", check_counts["n"].min())
    print("max n:", check_counts["n"].max())

    # -----------------------------
    # result directory
    # -----------------------------
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    result_dir = os.path.join(base_dir, "..", "result")
    os.makedirs(result_dir, exist_ok=True)

    # -----------------------------
    # save LONG raw
    # -----------------------------
    long_csv_path = os.path.join(
        result_dir,
        f"interv_{variable_name}_LONG_"
        f"{data_type}{init_envc}{init_tau0}_"
        f"{beta_tag1}-{beta_tag2}_isoltimeALL.csv"
    )

    run_data.to_csv(long_csv_path, index=False, encoding="utf-8")
    print("saved LONG raw:", long_csv_path)

    # -----------------------------
    # save wide csv by isolationTime
    # rows = iteration
    # columns = beta
    # values = TotalInfec daily list
    # -----------------------------
    for isol_time in isoltime_values:

        print("\nSaving isolationTime =", isol_time)

        sub = run_data.loc[
            run_data["isolationTime"] == isol_time
        ].copy()

        df_wide = sub.pivot_table(
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
            f"{data_type}{init_envc}{init_tau0}_"
            f"{beta_tag1}-{beta_tag2}_isoltime{isol_time}.csv"
        )

        df_wide.to_csv(out_csv, index=False, encoding="utf-8")
        print("saved:", out_csv)

    print("\nDONE isolationTime intervention ABM run")


if __name__ == "__main__":
    freeze_support()
    main()