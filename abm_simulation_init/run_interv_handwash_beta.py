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
    data_type = 'A'
    num_iter = 50

    init_envc = 9
    init_tau0 = 140

    cleanDay = 180
    washrate = 0.9
    isolationTime = 14

    runtime = 30 * 19   # A: 30*19, B: 30*36

    probNewPatient = 0.003
    probTransmission = 0.0866
    isolationFactor = 0.75

    height = 11
    width = 32

    # %% Variable settings
    variable_name = 'prob_transmission'

    beta_values = np.round(np.arange(0.01, 0.0601, 0.005), 5)
    handwash_values = [0.8, 0.9, 0.95, 0.99]

    beta_tag1 = beta_values[0]
    beta_tag2 = beta_values[-1]

    print("beta values:", beta_values)
    print("handwash values:", handwash_values)
    print("number of beta values:", len(beta_values))
    print("number of handwash values:", len(handwash_values))
    print("total parameter settings:", len(beta_values) * len(handwash_values))
    print("total simulations:", len(beta_values) * len(handwash_values) * num_iter)

    fixed_params = {
        "data_type": data_type,
        "prob_new_patient": probNewPatient,
        "prob_transmission": probTransmission,
        "isolation_factor": isolationFactor,
        "cleaningDay": cleanDay,
        "hcw_wash_rate": washrate,
        "isolation_time": isolationTime,
        "height": height,
        "width": width,
        "init_env": init_envc,
        "tau_offset_days": init_tau0
    }

    # prob_transmission과 hcw_wash_rate는 variable로 돌릴 것이므로 fixed_params에서 제거
    for p in ["prob_transmission", "hcw_wash_rate"]:
        if p in fixed_params:
            del fixed_params[p]

    variable_params = {
        "prob_transmission": beta_values,
        "hcw_wash_rate": handwash_values
    }

    # %% Model for max_steps
    model = CPE_Model_month(
        data_type=data_type,
        prob_new_patient=probNewPatient,
        prob_transmission=probTransmission,
        isolation_factor=isolationFactor,
        cleaningDay=cleanDay,
        hcw_wash_rate=washrate,
        isolation_time=isolationTime,
        height=height,
        width=width,
        init_env=init_envc,
        tau_offset_days=init_tau0
    )

    max_steps = model.ticks_in_day * runtime

    # %% Run
    start_time = time.time()

    print("now run")

    batch_run = BatchRunnerMP(
        CPE_Model_month,
        nr_processes=16,
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
    print(run_data.head())
    print("cols:", list(run_data.columns))

    # --------------------------------------------------
    # 중요 수정:
    # Run 컬럼은 반복번호가 아님.
    # 무조건 handwash-beta 조합별로 my_iteration 새로 만든다.
    # --------------------------------------------------
    run_data = run_data.reset_index(drop=True)

    run_data["my_iteration"] = (
        run_data
        .groupby(["hcw_wash_rate", "prob_transmission"])
        .cumcount()
    )

    iter_col = "my_iteration"

    print("iteration column:", iter_col)

    # 각 handwash-beta 조합마다 num_iter개씩 있는지 확인
    check_counts = (
        run_data
        .groupby(["hcw_wash_rate", "prob_transmission"])
        .size()
        .reset_index(name="n")
    )

    print("\ncheck counts:")
    print(check_counts.head(30))
    print("min n:", check_counts["n"].min())
    print("max n:", check_counts["n"].max())

    if check_counts["n"].min() != num_iter or check_counts["n"].max() != num_iter:
        print("[WARNING] Some parameter settings do not have num_iter runs.")
    else:
        print("[OK] Every parameter setting has num_iter runs.")

    # %% Save
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    result_dir = os.path.join(base_dir, '..', 'result')
    os.makedirs(result_dir, exist_ok=True)

    # --------------------------------------------------
    # 1) LONG raw 파일 저장
    # 이 파일이 있으면 pivot 저장이 꼬여도 복구 가능
    # --------------------------------------------------
    long_csv_path = os.path.join(
        result_dir,
        f'interv_{variable_name}_LONG_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}_handwashALL.csv'
    )

    run_data.to_csv(long_csv_path, index=False)
    print("\nsaved long raw ->", long_csv_path)

    # --------------------------------------------------
    # 2) 기존 방식처럼 handwash별 wide 파일 저장
    # row = iteration, column = beta
    # 정상이면 df shape = (50, 11)
    # --------------------------------------------------
    for wash in handwash_values:

        print("\n" + "=" * 60)
        print(f"Saving handwash = {wash}")
        print("=" * 60)

        sub = run_data[run_data["hcw_wash_rate"] == wash].copy()

        # 저장 전 확인
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

        df = sub.pivot_table(
            index=iter_col,
            columns=variable_name,
            values="HCW_related_infecs",
            aggfunc="first"
        ).reset_index(drop=True)

        df.columns.name = None

        wash_tag = str(wash).replace(".", "p")

        csv_path = os.path.join(
            result_dir,
            f'interv_{variable_name}_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}_handwash{wash_tag}.csv'
        )

        df.to_csv(csv_path, index=False)

        print("df shape:", df.shape)
        print(df.head())
        print("done!! ->", csv_path)

        if df.shape != (num_iter, len(beta_values)):
            print("[WARNING] Saved file shape is not expected.")
            print("expected:", (num_iter, len(beta_values)))
            print("actual:", df.shape)
        else:
            print("[OK] Saved file shape is correct.")


if __name__ == "__main__":
    freeze_support()
    main()