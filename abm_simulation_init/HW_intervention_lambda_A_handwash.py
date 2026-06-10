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
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="No agent reporters*")


# %% 람다 수정/stay값 수정 버전
data_type = 'A'
num_iter = 50
init_envc = 9
init_tau0 = 140

# Parameters
cleanDay = 180
washrate = 0.9
isolationTime = 14

runtime = 30 * 19
probNewPatient = 0.003
probTransmission = 0.0866
isolationFactor = 0.75
height = 11
width = 32

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

# 여기를 수정하면 원하는 파라미터에 대한 인터벤션 가능
variable_name = 'hcw_wash_rate'
variable_value = [0.3,0.5,0.7, 0.9, 0.93, 0.95, 0.98, 1]

beta_tag1 = variable_value[0]
beta_tag2 = variable_value[-1]

start_time = time.time()

# =========================================================
# [1] posterior beta에서 비복원추출 후 저장
# =========================================================
np.random.seed(3905)

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

result_dir = os.path.abspath(os.path.join(base_dir, '..', 'result'))
data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))

os.makedirs(result_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

beta_csv = os.path.join(data_dir, 'beta_posterior_samples_15_0.0025_20000.csv')
beta_all = pd.read_csv(beta_csv, header=None).iloc[:, 0].astype(float).values

if len(beta_all) < num_iter:
    raise ValueError(f"beta sample count ({len(beta_all)}) is smaller than num_iter ({num_iter})")

beta_draws = np.random.choice(beta_all, size=num_iter, replace=False)

beta_draw_df = pd.DataFrame({
    "iteration": np.arange(1, num_iter + 1),
    "beta_draw": beta_draws
})

beta_draw_save_path = os.path.join(
    result_dir,
    f'interv_beta_draws_{data_type}{init_envc}{init_tau0}.csv'
)
beta_draw_df.to_csv(beta_draw_save_path, index=False)

# =========================================================
# [ADD] 뽑은 beta 50개 분포 그림
# =========================================================

plt.figure(figsize=(10,6))

plt.hist(
    beta_draws,
    bins=10,
    color="skyblue",
    edgecolor="black"
)

plt.axvline(
    np.mean(beta_draws),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"mean = {np.mean(beta_draws):.5f}"
)

plt.xlabel("Beta")
plt.ylabel("Frequency")
plt.title("Sampled beta draws (n=50)")
plt.legend()

plt.tight_layout()

beta_fig_path = os.path.join(
    result_dir,
    f'beta_draw_hist_{data_type}{init_envc}{init_tau0}.png'
)

plt.savefig(beta_fig_path, dpi=150)
plt.close()

print("saved beta histogram ->", beta_fig_path)
print("saved beta draws ->", beta_draw_save_path)
print(beta_draw_df.head())

# =========================================================
# [2] 같은 beta 50개를 모든 washrate 값에 대해 공통 사용
# =========================================================
print("loading...\n")
print("now run")

all_rows = []

tasks = [(wash, it) for wash in variable_value for it in range(num_iter)]

for wash, it in tqdm(tasks):
    beta_now = beta_draws[it]

    model = CPE_Model_month(
        data_type=data_type,
        prob_new_patient=probNewPatient,
        prob_transmission=beta_now,
        isolation_factor=isolationFactor,
        cleaningDay=cleanDay,
        hcw_wash_rate=wash,
        isolation_time=isolationTime,
        height=height,
        width=width,
        init_env=init_envc,
        tau_offset_days=init_tau0
    )

    max_steps = model.ticks_in_day * runtime

    for _ in range(max_steps):
        model.step()

    all_rows.append({
        variable_name: wash,
        "iteration": it,
        "beta_draw": beta_now,
        "HCW_related_infecs": getTotalInfec(model)
    })

elapsed = time.time() - start_time
print(f"\nDone. Elapsed time = {elapsed:.2f} sec")

# =========================================================
# [3] run_data 생성
# =========================================================
run_data = pd.DataFrame(all_rows)

print(run_data.head())
print("cols:", list(run_data.columns))

# --- iteration 보정 (원래 형식 최대한 유지) ---
ITER_CANDIDATES = ["iteration", "Iteration", "Run", "run", "run_id"]
iter_col = None

for c in ITER_CANDIDATES:
    if c in run_data.columns:
        iter_col = c
        break

if iter_col is None:
    run_data = run_data.reset_index(drop=True)
    if variable_name in run_data.columns:
        run_data["iteration"] = run_data.groupby(variable_name).cumcount()
    else:
        run_data["iteration"] = np.arange(len(run_data))
    iter_col = "iteration"

# =========================================================
# [4] 피벗 (원래 방식 그대로)
# =========================================================
df = run_data.pivot_table(
    index=iter_col,
    columns=variable_name,
    values="HCW_related_infecs",
    aggfunc="first"
).reset_index(drop=True)

print(df.head())

# =========================================================
# [5] 저장 (원래 방식 그대로)
# =========================================================
csv_path = os.path.join(
    result_dir,
    f'interv_{variable_name}_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}.csv'
)

df.to_csv(csv_path, index=False)

print("done!! ->", csv_path)


# %% interv raw 파일을 Summary (월별로 나오게끔)

import os
import ast
import numpy as np
import pandas as pd

# -----------------------------
# 설정
# -----------------------------
# 파일명 맞게 수정, 수정없을경우 위의 셀에서 만든 파일 사용
#csv_path = "../result/interv_hcw_wash_rate_A9140_0.3-1_18.csv"   
data_type = "A"                                 # A or B
days_per_month = 30

if data_type == "A":
    n_months = 19
elif data_type == "B":
    n_months = 36
else:
    raise ValueError("data_type must be 'A' or 'B'")

# -----------------------------
# raw data 읽기
# -----------------------------
raw_df = pd.read_csv(csv_path)

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
# beta별 summary 계산
# -----------------------------
rows = []

for beta in raw_df.columns:
    # 해당 beta 열에서 non-null iteration만 가져오기
    series_list = raw_df[beta].dropna().apply(parse_series).tolist()

    # 각 iteration의 일별 series -> 월별 series
    monthly_runs = [
        daily_to_monthly(s, days_per_month=days_per_month, n_months=n_months)
        for s in series_list
    ]

    monthly_arr = np.array(monthly_runs, dtype=float)   # shape = (n_iter, n_months)

    # 월별 통계
    mean_ = monthly_arr.mean(axis=0).tolist()
    std_ = monthly_arr.std(axis=0, ddof=0).tolist()
    max_ = monthly_arr.max(axis=0).tolist()
    median_ = np.median(monthly_arr, axis=0).tolist()

    # nonzero_mean
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
        "mean": mean_,
        "std": std_,
        "n": int(monthly_arr.shape[0]),
        "max": max_,
        "median": median_,
        "nonzero_mean": nonzero_mean_,
        "disease_free(%)": disease_free_
    })

summary_df = pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)

print(summary_df.head())

# -----------------------------
# 저장
# -----------------------------

out_path = f"../result/interv_{variable_name}_summary_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}.csv"
summary_df.to_csv(out_path, index=False)
print("saved ->", out_path)
#이름예시 interv_prob_transmission_summary_A10140_0.02-0.05















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


# %% 람다 수정/stay값 수정 버전
data_type = 'A'
num_iter = 50; np.int64(num_iter)
init_envc = 9
init_tau0 = 140
# Parameters
cleanDay = 180
washrate = 0.9
isolationTime = 14

runtime = 30*19 # dont forget change A : 30 * 19, B : 30 * 36
probNewPatient = 0.003 # 0.053, Old Calibration # 1/2000, 2592 ticks per day
probTransmission = 0.0866 # calibration result
isolationFactor = 0.75 # fix
height=11
width=32


fixed_params = {
    "data_type" : data_type , 
    "prob_new_patient" : probNewPatient, 
    "prob_transmission" :probTransmission,
    "isolation_factor" : isolationFactor, 
    "cleaningDay" : cleanDay,
    "hcw_wash_rate" : washrate, 
    "isolation_time" : isolationTime, 
    "height" : height, "width" : width,
    "init_env": init_envc,
    "tau_offset_days": init_tau0
    }


# 여기를 수정하면 원하는 파라미터에 대한 인터벤션 가능
# 저장도
variable_name = 'hcw_wash_rate'
variable_value = [0.7,0.8,0.9,0.91,0.93,0.95,0.97,0.99]
beta_tag1 = variable_value[0]
beta_tag2 = variable_value[-1]
del fixed_params[variable_name]
variable_params = {variable_name : variable_value}

start_time = time.time()

# STEP4
model = CPE_Model_month(
    data_type=data_type,
    prob_new_patient=probNewPatient, 
    prob_transmission=probTransmission, 
    isolation_factor=isolationFactor,
    cleaningDay=cleanDay,
    hcw_wash_rate=washrate,
    isolation_time=isolationTime,
    height=height, width=width,
    init_env=init_envc,                 # [ADD] 초기 오염 개수 (예: 10)
    tau_offset_days=init_tau0
    )
print('loading...\n\n')


batch_run = BatchRunner(
    CPE_Model_month,
    variable_parameters = variable_params,
    fixed_parameters = fixed_params,
    iterations = num_iter,
    max_steps = model.ticks_in_day * runtime,
    display_progress=True,
    model_reporters={"HCW_related_infecs" : getTotalInfec}
                    #  ,"Num_move_Patients": getNumIsol}
                    #  "Number_of_Patients_sick":getNumSick}
)


# ... 위는 동일 ...

print("now run")
batch_run.run_all()

run_data = batch_run.get_model_vars_dataframe()
print(run_data.head())
print("cols:", list(run_data.columns))

# --- iteration 보정 ---
ITER_CANDIDATES = ["iteration", "Iteration", "Run", "run", "run_id"]
iter_col = None
for c in ITER_CANDIDATES:
    if c in run_data.columns:
        iter_col = c
        break

# 없으면 variable_name별로 0,1,2,... 붙여서 만들어준다
if iter_col is None:
    run_data = run_data.reset_index(drop=True)
    # 같은 변수값(b)끼리 0,1,2,... 부여
    if variable_name in run_data.columns:
        run_data["iteration"] = run_data.groupby(variable_name).cumcount()
    else:
        # 변수컬럼조차 없다면 전체에 0,1,2,...
        run_data["iteration"] = np.arange(len(run_data))
    iter_col = "iteration"

# --- 피벗 ---
df = run_data.pivot_table(
    index=iter_col,                  # 실행 반복 구분
    columns=variable_name,           # ex: cleaningDay
    values="HCW_related_infecs",     # reporter 결과
    aggfunc="first"
).reset_index(drop=True)

print(df.head())

# --- 저장 (경로 안전 처리) ---
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

os.makedirs(os.path.join(base_dir, '..', 'result'), exist_ok=True)
csv_path = os.path.join(base_dir, '..', f'result/interv_{variable_name}_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}.csv')
df.to_csv(csv_path, index=False)
print("done!! ->", csv_path)
#이름 예시 interv_prob_transmission_A10140_0.02-0.05














# %% interv raw 파일을 Summary (월별로 나오게끔)

import os
import ast
import numpy as np
import pandas as pd

# -----------------------------
# 설정
# -----------------------------
# 파일명 맞게 수정, 수정없을경우 위의 셀에서 만든 파일 사용
# csv_path = "../result/emulation_beta_lambdaA10140_0.02-0.05.csv"   
data_type = "A"                                 # A or B
days_per_month = 30

if data_type == "A":
    n_months = 19
elif data_type == "B":
    n_months = 36
else:
    raise ValueError("data_type must be 'A' or 'B'")

# -----------------------------
# raw data 읽기
# -----------------------------
raw_df = pd.read_csv(csv_path)

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
# beta별 summary 계산
# -----------------------------
rows = []

for beta in raw_df.columns:
    # 해당 beta 열에서 non-null iteration만 가져오기
    series_list = raw_df[beta].dropna().apply(parse_series).tolist()

    # 각 iteration의 일별 series -> 월별 series
    monthly_runs = [
        daily_to_monthly(s, days_per_month=days_per_month, n_months=n_months)
        for s in series_list
    ]

    monthly_arr = np.array(monthly_runs, dtype=float)   # shape = (n_iter, n_months)

    # 월별 통계
    mean_ = monthly_arr.mean(axis=0).tolist()
    std_ = monthly_arr.std(axis=0, ddof=0).tolist()
    max_ = monthly_arr.max(axis=0).tolist()
    median_ = np.median(monthly_arr, axis=0).tolist()

    # nonzero_mean
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
        "mean": mean_,
        "std": std_,
        "n": int(monthly_arr.shape[0]),
        "max": max_,
        "median": median_,
        "nonzero_mean": nonzero_mean_,
        "disease_free(%)": disease_free_
    })

summary_df = pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)

print(summary_df.head())

# -----------------------------
# 저장
# -----------------------------

out_path = f"../result/interv_{variable_name}_summary_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}.csv"
summary_df.to_csv(out_path, index=False)
print("saved ->", out_path)
#이름예시 interv_prob_transmission_summary_A10140_0.02-0.05












# %% 밑의 셀들은 동일하게하지만 컴파트먼트가 전부 나오는버전, 출력결과가 동일함은 확인완료







import os
import time
import numpy as np
import pandas as pd

data_type = 'A'
num_iter = 1

init_envc = 10
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
variable_value = [0.05]

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