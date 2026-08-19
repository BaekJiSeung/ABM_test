# %%

from model.cpe_model_month_lambda import CPE_Model_month
from model.cpe_model_month_lambda import getHCWInfec
from model.cpe_model_month_lambda import getTotalInfec
from mesa.batchrunner import BatchRunner
from mesa.batchrunner import BatchRunnerMP
from multiprocessing import freeze_support
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="No agent reporters*")


# %% Basic settings

data_type = 'A'
num_iter = 1

init_envc = 9
init_tau0 = 140

cleanDay = 180
washrate = 0.9              # default value, but hcw_wash_rate will be variable
isolationTime = 14

runtime = 30 * 19           # A: 30*19, B: 30*36

probNewPatient = 0.003
probTransmission = 0.0866   # default value, but prob_transmission will be variable
isolationFactor = 0.75

height = 11
width = 32


# %% Variable settings

variable_name = 'prob_transmission'

beta_values = np.round(np.arange(0.01, 0.0601, 0.005), 5)
handwash_values = [0.8, 0.9,0.95, 0.99]

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

start_time = time.time()
# %%
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


# %% Run all combinations at once using CPU multiprocessing

start_time = time.time()

print("now run")

batch_run = BatchRunnerMP(
    CPE_Model_month,
    nr_processes=8,   # 전체 CPU 코어 사용. 너무 무거우면 4, 6, 8 등으로 직접 지정
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



# %% Iteration column correction

ITER_CANDIDATES = ["iteration", "Iteration", "Run", "run", "run_id"]
iter_col = None

for c in ITER_CANDIDATES:
    if c in run_data.columns:
        iter_col = c
        break

# iteration column이 없으면 beta, handwash 조합별로 0,1,2,... 생성
if iter_col is None:
    run_data = run_data.reset_index(drop=True)

    run_data["iteration"] = (
        run_data
        .groupby(["hcw_wash_rate", "prob_transmission"])
        .cumcount()
    )

    iter_col = "iteration"

print("iteration column:", iter_col)


# %% Save separately by handwash value

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

result_dir = os.path.join(base_dir, '..', 'result')
os.makedirs(result_dir, exist_ok=True)


for wash in handwash_values:

    print("\n" + "="*60)
    print(f"Saving handwash = {wash}")
    print("="*60)

    sub = run_data[run_data["hcw_wash_rate"] == wash].copy()

    # 기존 방식처럼 row = iteration, column = beta
    df = sub.pivot_table(
        index=iter_col,
        columns=variable_name,
        values="HCW_related_infecs",
        aggfunc="first"
    ).reset_index(drop=True)

    # column 이름 정리
    df.columns.name = None

    wash_tag = str(wash).replace(".", "p")

    csv_path = os.path.join(
        result_dir,
        f'interv_{variable_name}_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}_handwash{wash_tag}.csv'
    )

    df.to_csv(csv_path, index=False)

    print(df.head())
    print("done!! ->", csv_path)











# %% interv raw 파일을 Summary (월별로 나오게끔)
# %% interv raw 파일 3개를 Summary 파일로 변환
# raw: interv_prob_transmission_A9140_0.01-0.06_handwash0p8.csv
# out: interv_prob_transmission_summary_A9140_0.01-0.06_handwash0p8.csv

import os
import ast
import numpy as np
import pandas as pd

# -----------------------------
# 설정
# -----------------------------
data_type = "A"          # A or B
days_per_month = 30

variable_name = "prob_transmission"

init_envc = 9
init_tau0 = 140
beta_values = [0.03847]
beta_tag1 = beta_values[0]
beta_tag2 = beta_values[-1]

handwash_values = [0.8, 0.9,0.95, 0.99]

if data_type == "A":
    n_months = 19
elif data_type == "B":
    n_months = 36
else:
    raise ValueError("data_type must be 'A' or 'B'")

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

result_dir = os.path.join(base_dir, "..", "result")


# -----------------------------
# 문자열 리스트 -> 파이썬 리스트 변환 함수
# -----------------------------
def parse_series(x):
    if pd.isna(x):
        return None
    if isinstance(x, list):
        return x
    return ast.literal_eval(x)


# -----------------------------
# 일별 -> 월별 변환
# -----------------------------
def daily_to_monthly(daily_series, days_per_month=30, n_months=None):
    arr = np.array(daily_series, dtype=float)

    if n_months is not None:
        needed = days_per_month * n_months
        arr = arr[:needed]

    m = len(arr) // days_per_month
    arr = arr[:m * days_per_month]

    monthly = arr.reshape(m, days_per_month).sum(axis=1)
    return monthly.tolist()


# -----------------------------
# handwash별 raw -> summary
# -----------------------------
for wash in handwash_values:

    wash_tag = str(wash).replace(".", "p")

    csv_path = os.path.join(
        result_dir,
        f"interv_{variable_name}_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}_handwash{wash_tag}.csv"
    )

    print("\n" + "=" * 70)
    print("raw file:", csv_path)

    raw_df = pd.read_csv(csv_path)

    rows = []

    for beta in raw_df.columns:

        # beta 열에서 iteration별 daily list 가져오기
        series_list = raw_df[beta].dropna().apply(parse_series).tolist()

        # 각 iteration의 daily series -> monthly series
        monthly_runs = [
            daily_to_monthly(
                s,
                days_per_month=days_per_month,
                n_months=n_months
            )
            for s in series_list
        ]

        monthly_arr = np.array(monthly_runs, dtype=float)

        mean_ = monthly_arr.mean(axis=0).tolist()
        std_ = monthly_arr.std(axis=0, ddof=0).tolist()
        max_ = monthly_arr.max(axis=0).tolist()
        median_ = np.median(monthly_arr, axis=0).tolist()

        nonzero_mean_ = []
        disease_free_ = []

        for j in range(monthly_arr.shape[1]):
            col = monthly_arr[:, j]
            nz = col[col > 0]

            if len(nz) == 0:
                nonzero_mean_.append(0.0)
            else:
                nonzero_mean_.append(float(nz.mean()))

            disease_free_.append(float((col == 0).mean() * 100.0))

        rows.append({
            "beta": float(beta),
            "handwash": wash,
            "mean": mean_,
            "std": std_,
            "n": int(monthly_arr.shape[0]),
            "max": max_,
            "median": median_,
            "nonzero_mean": nonzero_mean_,
            "disease_free(%)": disease_free_
        })

    summary_df = (
        pd.DataFrame(rows)
        .sort_values("beta")
        .reset_index(drop=True)
    )

    print(summary_df.head())

    out_path = os.path.join(
        result_dir,
        f"interv_{variable_name}_summary_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}_handwash{wash_tag}.csv"
    )

    summary_df.to_csv(out_path, index=False)

    print("saved ->", out_path)










# %% 밑의 셀들은 동일하게하지만 컴파트먼트가 전부 나오는버전, 출력결과가 동일함은 확인완료







import os
import time
import numpy as np
import pandas as pd

data_type = 'A'
num_iter = 10

init_envc = 9
init_tau0 = 140

# Parameters
cleanDay = 180
washrate = 0.9
isolationTime = 14

runtime = 30 * 19   # A
probNewPatient = 0.003
probTransmission = 0.0866
isolationFactor = 0.75
height = 11
width = 32

variable_name = 'prob_transmission'
variable_value = [0.03802]

start_time = time.time()

all_histories = []

# 저장 폴더
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

result_dir = os.path.join(base_dir, '..', 'result')
os.makedirs(result_dir, exist_ok=True)

print("Current working directory:")
print(os.getcwd())

for b in variable_value:
    print(f"\nRunning beta = {b}")

    for it in range(num_iter):
        print(f"  iteration {it+1}/{num_iter} running...")

        model = CPE_Model_month(
            data_type=data_type,
            prob_new_patient=probNewPatient,
            prob_transmission=b,
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

        for step in range(max_steps):
            model.step()
            # 둘 동일한지 체크만 살짝
            print(
            f"  check iteration {it+1}:",
            sum(model.totalHCWinf),
            model.cumul_sick_patients_by_HCW
)
        df_hist = model.get_history_dataframe().copy()
        df_hist["prob_transmission"] = b
        df_hist["iteration"] = it + 1

        all_histories.append(df_hist)

        print(f"  iteration {it+1}/{num_iter} finished")

elapsed = time.time() - start_time
print(f"\nDone. Elapsed time = {elapsed:.2f} sec")

# --- 모든 trajectory 합치기 ---
traj_df = pd.concat(all_histories, ignore_index=True)

# --- day별 평균 trajectory ---
mean_traj = (
    traj_df
    .groupby(["prob_transmission", "day"], as_index=False)
    .mean(numeric_only=True)
)

# --- 각 iteration 마지막 날 요약 ---
final_summary = (
    traj_df
    .sort_values(["prob_transmission", "iteration", "day"])
    .groupby(["prob_transmission", "iteration"], as_index=False)
    .tail(1)
    [[
        "prob_transmission",
        "iteration",
        "patients",
        "patient_C",
        "patient_S",
        "patient_isolated",
        "patient_positive",
        "patient_preinfection",
        "hcws",
        "hcw_C",
        "goo",
        "goo_C",
        "beds",
        "filled_beds",
        "filled_sick_beds",
        "empty_isolated_beds",
        "daily_hcw_infections",
        "cumulative_sick_patients",
        "cumulative_sick_patients_by_HCW",
        "cumulative_patients",
        "move2isol"
    ]]
)

# --- 마지막 날 평균 ---
final_mean = (
    final_summary
    .groupby("prob_transmission", as_index=False)
    .mean(numeric_only=True)
)

beta_tag1 = variable_value[0]
beta_tag2 = variable_value[-1]

# --- 저장 파일명 ---
tag = f"{data_type}_env{init_envc}_tau{init_tau0}_{beta_tag1}-{beta_tag2}"
print(tag)
traj_path = os.path.join(result_dir, f"traj_all_iterations_{tag}.csv")
mean_path = os.path.join(result_dir, f"traj_mean_{tag}.csv")
final_summary_path = os.path.join(result_dir, f"traj_final_summary_{tag}.csv")
final_mean_path = os.path.join(result_dir, f"traj_final_mean_{tag}.csv")

traj_df.to_csv(traj_path, index=False)
mean_traj.to_csv(mean_path, index=False)
final_summary.to_csv(final_summary_path, index=False)
final_mean.to_csv(final_mean_path, index=False)

print("\nSaved files:")
print(traj_path)
print(mean_path)
print(final_summary_path)
print(final_mean_path)

# %%

# %%
# %% 환자 환경



plt.figure(figsize=(10, 5))
for b in variable_value:
    temp = mean_traj[mean_traj["prob_transmission"] == b]
    plt.plot(temp["day"], temp["patient_C"], label=f"beta={b}")

plt.xlabel("Day")
plt.ylabel("Colonized patients")
plt.title("Daily trajectory of colonized patients")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
for b in variable_value:
    temp = mean_traj[mean_traj["prob_transmission"] == b]
    plt.plot(temp["day"], temp["goo_C"], label=f"beta={b}")

plt.xlabel("Day")
plt.ylabel("Colonized Goo")
plt.title("Daily trajectory of contaminated environment")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
for b in variable_value:
    temp = mean_traj[mean_traj["prob_transmission"] == b]
    plt.plot(temp["day"], temp["cumulative_sick_patients_by_HCW"], label=f"beta={b}")

plt.xlabel("Day")
plt.ylabel("cumulative_sick_patients_by_HCW")
plt.title("cumulative_sick_patients_by_HCW")
plt.legend()
plt.grid(True)
plt.show()