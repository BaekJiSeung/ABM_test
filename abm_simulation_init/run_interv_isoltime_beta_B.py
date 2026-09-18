# %% ================== run_interv_isoltime_beta_B.py ==================

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
    init_tau0 = 60

    # baseline intervention settings
    cleanDay = 180
    washrate = 0.9

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
    # isolation intervention values
    # paper에서는 average isolation delay = 3, 7, 10, 14 days로 설명
    isolation_values = [6, 14, 20, 28]

    beta_tag1 = f"{beta_values[0]:.5f}"
    beta_tag2 = f"{beta_values[-1]:.5f}"

    # max_steps용 dummy value
    probTransmission_dummy = float(beta_values[0])
    isolationTime_dummy = int(isolation_values[0])

    print("=" * 80)
    print("START isolation intervention ABM run - Period B")
    print("data_type:", data_type)
    print("init_envc:", init_envc)
    print("tau_offset_days:", init_tau0)
    print("cleaningDay:", cleanDay)
    print("washrate:", washrate)
    print("runtime days:", runtime)
    print("num_iter:", num_iter)
    print("nr_processes:", nr_processes)
    print("beta values:", beta_values)
    print("isolation values:", isolation_values)
    print("average isolation delays:", [x / 2 for x in isolation_values])
    print("number of beta values:", len(beta_values))
    print("number of isolation values:", len(isolation_values))
    print("total parameter settings:", len(beta_values) * len(isolation_values))
    print("total simulations:", len(beta_values) * len(isolation_values) * num_iter)
    print("=" * 80)

    # --------------------------------------------------
    # fixed baseline parameters
    # --------------------------------------------------
    fixed_params = {
        "data_type": data_type,

        "prob_new_patient": probNewPatient,
        "isolation_factor": isolationFactor,

        "cleaningDay": cleanDay,
        "hcw_wash_rate": washrate,

        "height": height,
        "width": width,

        "init_env": init_envc,
        "tau_offset_days": init_tau0,
    }

    # prob_transmission, isolation_time은 variable로 돌림
    variable_params = {
        "prob_transmission": beta_values,
        "isolation_time": isolation_values,
    }

    # --------------------------------------------------
    # max_steps 계산용 model
    # --------------------------------------------------
    model = CPE_Model_month(
        data_type=data_type,
        prob_new_patient=probNewPatient,
        prob_transmission=probTransmission_dummy,
        isolation_factor=isolationFactor,
        cleaningDay=cleanDay,
        hcw_wash_rate=washrate,
        isolation_time=isolationTime_dummy,
        height=height,
        width=width,
        init_env=init_envc,
        tau_offset_days=init_tau0,
    )

    max_steps = model.ticks_in_day * runtime

    print("\nmax_steps:", max_steps)

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
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

    print("\ndone running")
    print("elapsed seconds:", elapsed)
    print("run_data shape before reset:", run_data.shape)
    print("cols:", list(run_data.columns))
    print(run_data.head())

    run_data = run_data.reset_index(drop=True)

    run_data["init_env_used"] = init_envc
    run_data["tau_offset_days_used"] = init_tau0
    run_data["avg_isolation_delay"] = run_data["isolation_time"].astype(float) / 2

    run_data["my_iteration"] = (
        run_data
        .groupby(["isolation_time", "prob_transmission"])
        .cumcount()
    )

    iter_col = "my_iteration"

    print("\niteration column:", iter_col)

    check_counts = (
        run_data
        .groupby(["isolation_time", "prob_transmission"])
        .size()
        .reset_index(name="n")
    )

    print("\ncheck counts:")
    print(check_counts.head(50))
    print("min n:", check_counts["n"].min())
    print("max n:", check_counts["n"].max())

    if check_counts["n"].min() != num_iter or check_counts["n"].max() != num_iter:
        print("[WARNING] Some parameter settings do not have num_iter runs.")
    else:
        print("[OK] Every parameter setting has num_iter runs.")

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    result_dir = os.path.join(base_dir, "..", "result")
    os.makedirs(result_dir, exist_ok=True)

    # --------------------------------------------------
    # 1) LONG raw 파일 저장
    # --------------------------------------------------
    long_csv_path = os.path.join(
        result_dir,
        f"interv_{variable_name}_LONG_"
        f"{data_type}{init_envc}{init_tau0}_"
        f"{beta_tag1}-{beta_tag2}_isoltimeALL.csv"
    )

    run_data.to_csv(long_csv_path, index=False, encoding="utf-8")
    print("\nsaved LONG raw ->", long_csv_path)

    # --------------------------------------------------
    # 2) isolation_time별 wide csv 저장
    # --------------------------------------------------
    for isol_time in isolation_values:

        avg_delay = isol_time / 2

        print("\n" + "=" * 60)
        print(f"Saving isolation_time = {isol_time}, average delay = {avg_delay}")
        print("=" * 60)

        sub = run_data.loc[
            run_data["isolation_time"].astype(int) == isol_time
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
            f"{data_type}{init_envc}{init_tau0}_"
            f"{beta_tag1}-{beta_tag2}_isoltime{isol_time}.csv"
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

    print("\nDONE isolation intervention ABM run - Period B")


if __name__ == "__main__":
    freeze_support()
    main()