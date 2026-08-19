# %% ================== run_interv_handwash_beta_B.py ==================

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
    nr_processes = 16

    # selected initial condition setting
    init_envc = 9
    init_tau0 = 140

    # baseline intervention settings
    cleanDay = 180
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

    # handwashing intervention values
    handwash_values = [0.8, 0.9, 0.95, 0.99]

    beta_tag1 = f"{beta_values[0]:.2f}"
    beta_tag2 = f"{beta_values[-1]:.2f}"

    # max_steps용 dummy value
    # 실제 run에서는 variable_params의 prob_transmission이 사용됨
    probTransmission_dummy = float(beta_values[0])
    washrate_dummy = float(handwash_values[0])

    print("=" * 80)
    print("START handwash intervention ABM run - Period B")
    print("data_type:", data_type)
    print("init_envc:", init_envc)
    print("tau_offset_days:", init_tau0)
    print("cleaningDay:", cleanDay)
    print("isolationTime:", isolationTime)
    print("runtime days:", runtime)
    print("num_iter:", num_iter)
    print("nr_processes:", nr_processes)
    print("beta values:", beta_values)
    print("handwash values:", handwash_values)
    print("number of beta values:", len(beta_values))
    print("number of handwash values:", len(handwash_values))
    print("total parameter settings:", len(beta_values) * len(handwash_values))
    print("total simulations:", len(beta_values) * len(handwash_values) * num_iter)
    print("=" * 80)

    # --------------------------------------------------
    # fixed baseline parameters
    # --------------------------------------------------
    fixed_params = {
        "data_type": data_type,

        "prob_new_patient": probNewPatient,
        "isolation_factor": isolationFactor,

        "cleaningDay": cleanDay,
        "isolation_time": isolationTime,

        "height": height,
        "width": width,

        "init_env": init_envc,
        "tau_offset_days": init_tau0,
    }

    # prob_transmission, hcw_wash_rate는 variable로 돌림
    variable_params = {
        "prob_transmission": beta_values,
        "hcw_wash_rate": handwash_values,
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
        hcw_wash_rate=washrate_dummy,
        isolation_time=isolationTime,
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

    # --------------------------------------------------
    # IMPORTANT:
    # BatchRunnerMP의 Run column은 iteration 번호로 믿으면 안 됨.
    # handwash-beta 조합별로 my_iteration을 직접 생성.
    # --------------------------------------------------
    run_data = run_data.reset_index(drop=True)

    run_data["my_iteration"] = (
        run_data
        .groupby(["hcw_wash_rate", "prob_transmission"])
        .cumcount()
    )

    iter_col = "my_iteration"

    print("\niteration column:", iter_col)

    # 각 handwash-beta 조합마다 num_iter개씩 있는지 확인
    check_counts = (
        run_data
        .groupby(["hcw_wash_rate", "prob_transmission"])
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

    # --------------------------------------------------
    # result directory
    # --------------------------------------------------
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    result_dir = os.path.join(base_dir, "..", "result")
    os.makedirs(result_dir, exist_ok=True)

    # --------------------------------------------------
    # 1) LONG raw 파일 저장
    # 이 파일이 있으면 wide 저장이 꼬여도 복구 가능
    # --------------------------------------------------
    long_csv_path = os.path.join(
        result_dir,
        f"interv_{variable_name}_LONG_"
        f"{data_type}{init_envc}{init_tau0}_"
        f"{beta_tag1}-{beta_tag2}_handwashALL.csv"
    )

    run_data.to_csv(long_csv_path, index=False, encoding="utf-8")
    print("\nsaved LONG raw ->", long_csv_path)

    # --------------------------------------------------
    # 2) handwash별 wide csv 저장
    # row = iteration
    # column = beta_ABM
    # value = HCW_related_infecs daily list
    # 정상 shape = (50, 9)
    # --------------------------------------------------
    for wash in handwash_values:

        print("\n" + "=" * 60)
        print(f"Saving handwash = {wash}")
        print("=" * 60)

        sub = run_data.loc[
            np.isclose(run_data["hcw_wash_rate"].astype(float), wash)
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
        df_wide = df_wide.sort_index(axis=1)

        wash_tag = str(wash).replace(".", "p")

        csv_path = os.path.join(
            result_dir,
            f"interv_{variable_name}_"
            f"{data_type}{init_envc}{init_tau0}_"
            f"{beta_tag1}-{beta_tag2}_handwash{wash_tag}.csv"
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

    print("\nDONE handwash intervention ABM run - Period B")


if __name__ == "__main__":
    freeze_support()
    main()