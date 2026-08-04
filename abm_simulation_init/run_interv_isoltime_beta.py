# %% ================== run_interv_isoltime_beta.py ==================

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
    data_type = "A"
    num_iter = 50
    nr_processes = 16

    init_envc = 9
    tau_offset_days = 140

    runtime = 30 * 19   # A: 30*19, B: 30*36

    probNewPatient = 0.003
    isolationFactor = 0.75

    washrate = 0.9
    cleaningDay = 180

    height = 11
    width = 32

    # %% Variable settings
    variable_name = "prob_transmission"

    beta_values = np.round(
        np.arange(0.02, 0.0601, 0.005),
        5
    )

    isoltime_values = [6, 14, 20, 28]

    beta_tag1 = beta_values[0]
    beta_tag2 = beta_values[-1]

    print("=" * 80)
    print("START isolationTime intervention ABM run")
    print("data_type:", data_type)
    print("init_envc:", init_envc)
    print("tau_offset_days:", tau_offset_days)
    print("beta_values:", beta_values)
    print("isoltime_values:", isoltime_values)
    print("num_iter:", num_iter)
    print("runtime days:", runtime)
    print("nr_processes:", nr_processes)
    print("=" * 80)

    # -----------------------------
    # fixed baseline parameters
    # -----------------------------
    fixed_params = {
        "data_type": data_type,

        "prob_new_patient": probNewPatient,
        "isolation_factor": isolationFactor,

        "cleaningDay": cleaningDay,
        "hcw_wash_rate": washrate,

        "height": height,
        "width": width,

        "init_env": init_envc,
        "tau_offset_days": tau_offset_days,
    }

    # prob_transmission, isolation_time은 variable로 돌림
    variable_params = {
        "prob_transmission": beta_values,
        "isolation_time": isoltime_values,
    }

    # -----------------------------
    # max_steps 계산
    # -----------------------------
    model = CPE_Model_month(
        data_type=data_type,
        prob_new_patient=probNewPatient,
        prob_transmission=float(beta_values[0]),
        isolation_factor=isolationFactor,
        cleaningDay=cleaningDay,
        hcw_wash_rate=washrate,
        isolation_time=isoltime_values[0],
        height=height,
        width=width,
        init_env=init_envc,
        tau_offset_days=tau_offset_days
    )

    max_steps = model.ticks_in_day * runtime

    print("max_steps:", max_steps)

    # -----------------------------
    # Run
    # -----------------------------
    start_time = time.time()

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

    print("done running")
    print("elapsed seconds:", elapsed)
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
        .groupby(["isolation_time", "prob_transmission"])
        .cumcount()
    )

    # 각 isolation_time-beta 조합마다 num_iter개씩 있는지 확인
    check_counts = (
        run_data
        .groupby(["isolation_time", "prob_transmission"])
        .size()
        .reset_index(name="n")
    )

    print("\ncheck counts:")
    print(check_counts.head(40))
    print("min n:", check_counts["n"].min())
    print("max n:", check_counts["n"].max())

    if check_counts["n"].min() != num_iter or check_counts["n"].max() != num_iter:
        print("[WARNING] Some parameter settings do not have num_iter runs.")
    else:
        print("[OK] Every parameter setting has num_iter runs.")

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
        f"{data_type}{init_envc}{tau_offset_days}_"
        f"{beta_tag1}-{beta_tag2}_isoltimeALL.csv"
    )

    run_data.to_csv(long_csv_path, index=False, encoding="utf-8")
    print("\nsaved LONG raw:", long_csv_path)

    # -----------------------------
    # save wide csv by isolation_time
    # row = iteration
    # column = beta
    # value = HCW_related_infecs daily list
    # 정상 shape = (50, 9)
    # -----------------------------
    for isol_time in isoltime_values:

        print("\n" + "=" * 60)
        print("Saving isolation_time =", isol_time)
        print("=" * 60)

        sub = run_data.loc[
            run_data["isolation_time"] == isol_time
        ].copy()

        sub_counts = (
            sub
            .groupby("prob_transmission")
            .size()
            .reset_index(name="n")
        )

        print("sub counts:")
        print(sub_counts)
        print("sub min n:", sub_counts["n"].min())
        print("sub max n:", sub_counts["n"].max())

        df_wide = sub.pivot_table(
            index="my_iteration",
            columns="prob_transmission",
            values="HCW_related_infecs",
            aggfunc="first"
        ).reset_index(drop=True)

        df_wide.columns.name = None
        df_wide = df_wide.sort_index(axis=1)

        print("df_wide shape:", df_wide.shape)
        print(df_wide.head())

        out_csv = os.path.join(
            result_dir,
            f"interv_{variable_name}_"
            f"{data_type}{init_envc}{tau_offset_days}_"
            f"{beta_tag1}-{beta_tag2}_isoltime{isol_time}.csv"
        )

        df_wide.to_csv(out_csv, index=False, encoding="utf-8")
        print("saved:", out_csv)

        if df_wide.shape != (num_iter, len(beta_values)):
            print("[WARNING] Saved file shape is not expected.")
            print("expected:", (num_iter, len(beta_values)))
            print("actual:", df_wide.shape)
        else:
            print("[OK] Saved file shape is correct.")

    print("\nDONE isolationTime intervention ABM run")


if __name__ == "__main__":
    freeze_support()
    main()