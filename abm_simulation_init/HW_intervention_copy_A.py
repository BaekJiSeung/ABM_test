# %%

from model.cpe_model_month_copy import CPE_Model_month
from model.cpe_model_month_copy import getHCWInfec
from model.cpe_model_month_copy import getTotalInfec
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


# %%
data_type = 'A'
num_iter = 10; np.int64(num_iter)
init_envc = 10
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

variable_name = 'prob_transmission'
variable_value = [0.01,0.02,0.03,0.04,0.05]

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
csv_path = os.path.join(base_dir, '..', f'result/emulation_beta_{data_type}{init_envc}{init_tau0}.csv')
df.to_csv(csv_path, index=False)
print("done!! ->", csv_path)
# %% RAW Summarry 인데 람다수정버전(일별로나오네 이걸 월별로바꿔보자)
# %%
import os
import ast
import numpy as np
import pandas as pd

# -----------------------------
# 설정
# -----------------------------
csv_path = "../result/emulation_beta_A10140_lambda.csv"   # 파일명 맞게 수정
                            # A or B
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
out_path = f"../result/monthly_summary_{data_type}.csv"
summary_df.to_csv(out_path, index=False)
print("saved ->", out_path)







# %%  ===================== RAW + SUMMARY 저장 =====================
import os, math, ast
import pandas as pd
import numpy as np

# df : 기존 batch_run 결과를 모은 DataFrame (열 = beta, 값 = [월별 감염수 리스트])

def _to_list(x):
    """셀 값이 문자열이면 리스트로 변환, 아니면 기존 로직 유지"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)  # 문자열 "[0,1,2]" → [0,1,2]
        except:
            return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]

def _series_to_matrix(series):
    """한 beta의 시뮬 결과(series: 각 row가 리스트)를 (n_iter, n_months) 행렬로 변환"""
    lists = series.apply(_to_list)
    maxlen = max((len(L) for L in lists), default=0)
    mat = np.zeros((len(lists), maxlen))
    for i, L in enumerate(lists):
        if len(L) > 0:
            mat[i, :len(L)] = np.array(L, dtype=float)
    return mat  # shape = (n_iter, n_months)

summary_rows = []
raw_out = {}

for b in variable_value:
    if b not in df.columns:
        continue
    mat = _series_to_matrix(df[b])  # (n_iter, n_months)
    if mat.size == 0:
        continue

    # 월별 mean, std, max, median, nonzero_mean, disease_free(%)
    mean_vals = mat.mean(axis=0).tolist()
    std_vals = mat.std(axis=0, ddof=1).tolist()
    max_vals = mat.max(axis=0).tolist()
    median_vals = np.median(mat, axis=0).tolist()

    nonzero_means = []
    disease_free = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        nonzero = col[col > 0]
        nonzero_means.append(nonzero.mean() if len(nonzero) else 0.0)
        disease_free.append((col == 0).sum() / len(col) * 100.0)

    summary_rows.append({
        "beta": b,
        "mean": mean_vals,
        "std": std_vals,
        "n": int(mat.shape[0]),
        "max": max_vals,
        "median": median_vals,
        "nonzero_mean": nonzero_means,
        "disease_free(%)": disease_free
    })

    raw_out[str(b)] = mat.tolist()

# --- RAW 저장 ---
raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    f'result/interv_{variable_name}_{data_type}_raw.csv')
pd.DataFrame(raw_out).to_csv(raw_path, index=False)
print("raw saved ->", raw_path)

# --- SUMMARY 저장 ---
summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    f'result/interv_{variable_name}_{data_type}_summary.csv')

if os.path.isfile(summary_path):
    old = pd.read_csv(summary_path)
    out = pd.concat([old, summary_df], ignore_index=True)
    out.to_csv(summary_path, index=False)
else:
    summary_df.to_csv(summary_path, index=False)

print("summary saved ->", summary_path)

# %% ==============================================================







# %%

# ====== (붙여넣기 시작) : RAW 저장 + 베타별 히스토그램/빈도표 ======
# 0) RAW 저장 (열: beta, 행: 반복 실행의 HCW_related_infecs)
raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    f'result/interv_{variable_name}_{data_type}_raw.csv')
df.to_csv(raw_path, index=False)
print("raw saved ->", raw_path)

# 1) 출력 폴더 준비
hist_outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    f'result/hists_{variable_name}_{data_type}')
os.makedirs(hist_outdir, exist_ok=True)

# 2) 베타별 히스토그램 & 빈도표 생성
import matplotlib.pyplot as plt

def _beta_to_safe_name(b):
    # 파일명에 쓰기 좋은 형태로 변환 (예: 0.003 -> 0_003)
    return str(b).replace('.', '_')

for b in variable_value:
    series = df[b].dropna().astype(int)  # 정수 값으로 가정(감염자 수)
    if series.empty:
        print(f"[warn] beta={b} : empty series, skip.")
        continue

    vmax = int(series.max())
    # 0 ~ vmax까지 빠짐없이 인덱스 구성
    idx = list(range(0, vmax + 1))
    freq = series.value_counts().sort_index()
    freq = freq.reindex(idx, fill_value=0)

    # --- 빈도표 저장 ---
    freq_df = freq.reset_index()
    freq_df.columns = ['value', 'count']
    freq_csv = os.path.join(hist_outdir, f'hist_beta_{_beta_to_safe_name(b)}.csv')
    freq_df.to_csv(freq_csv, index=False)

    # --- 히스토그램(막대그래프) 저장 ---
    plt.figure(figsize=(6,4), dpi=150)
    plt.bar(freq_df['value'], freq_df['count'])
    plt.title(f'P_HAI Distribution (β = {b})')
    plt.xlabel('P_HAI')
    plt.ylabel('Frequency')
    plt.tight_layout()
    fig_path = os.path.join(hist_outdir, f'hist_beta_{_beta_to_safe_name(b)}.png')
    plt.savefig(fig_path)
    plt.close()

    print(f"saved -> {fig_path} / {freq_csv}")

print("done : histograms & frequency tables")
# ====== (붙여넣기 끝) ======


# %%





# %% ===================== RAW + SUMMARY 저장 =====================
import os, math, ast
import pandas as pd
import numpy as np

# df : 기존 batch_run 결과를 모은 DataFrame (열 = beta, 값 = [월별 감염수 리스트])

def _to_list(x):
    """셀 값이 문자열이면 리스트로 변환, 아니면 기존 로직 유지"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)  # 문자열 "[0,1,2]" → [0,1,2]
        except:
            return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]

def _series_to_matrix(series):
    """한 beta의 시뮬 결과(series: 각 row가 리스트)를 (n_iter, n_months) 행렬로 변환"""
    lists = series.apply(_to_list)
    maxlen = max((len(L) for L in lists), default=0)
    mat = np.zeros((len(lists), maxlen))
    for i, L in enumerate(lists):
        if len(L) > 0:
            mat[i, :len(L)] = np.array(L, dtype=float)
    return mat  # shape = (n_iter, n_months)

summary_rows = []
raw_out = {}

for b in variable_value:
    if b not in df.columns:
        continue
    mat = _series_to_matrix(df[b])  # (n_iter, n_months)
    if mat.size == 0:
        continue

    # 월별 mean, std, max, median, nonzero_mean, disease_free(%)
    mean_vals = mat.mean(axis=0).tolist()
    std_vals = mat.std(axis=0, ddof=1).tolist()
    max_vals = mat.max(axis=0).tolist()
    median_vals = np.median(mat, axis=0).tolist()

    nonzero_means = []
    disease_free = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        nonzero = col[col > 0]
        nonzero_means.append(nonzero.mean() if len(nonzero) else 0.0)
        disease_free.append((col == 0).sum() / len(col) * 100.0)

    summary_rows.append({
        "beta": b,
        "mean": mean_vals,
        "std": std_vals,
        "n": int(mat.shape[0]),
        "max": max_vals,
        "median": median_vals,
        "nonzero_mean": nonzero_means,
        "disease_free(%)": disease_free
    })

    raw_out[str(b)] = mat.tolist()

# --- RAW 저장 ---
raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    f'result/interv_{variable_name}_{data_type}_raw.csv')
pd.DataFrame(raw_out).to_csv(raw_path, index=False)
print("raw saved ->", raw_path)

# --- SUMMARY 저장 ---
summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    f'result/interv_{variable_name}_{data_type}_summary.csv')

if os.path.isfile(summary_path):
    old = pd.read_csv(summary_path)
    out = pd.concat([old, summary_df], ignore_index=True)
    out.to_csv(summary_path, index=False)
else:
    summary_df.to_csv(summary_path, index=False)

print("summary saved ->", summary_path)























# %% 여기서부터 B
data_type = 'B'
num_iter = 50; np.int64(num_iter)

# Parameters
cleanDay = 180
washrate = 0.9
isolationTime = 14
init_envc = 2
init_tau0 = 40

runtime = 30*36 # dont forget change A : 30 * 19, B : 30 * 36
probNewPatient = 0.003 # 0.053, Old Calibration # 1/2000, 2592 ticks per day
probTransmission = 0.00005 # calibration result
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

variable_name = 'prob_transmission'
variable_value = [0.07]

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
    init_env= init_envc,
    tau_offset_days= init_tau0
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
print('now run')


# for _ in range(num_iter):
batch_run.run_all()
run_data = batch_run.get_model_vars_dataframe()

# 베타별 series 뽑기
series_map = {}
lengths = []
for b in variable_value:
    s = run_data.query(f"{variable_name}=={b}")["HCW_related_infecs"].reset_index(drop=True)
    series_map[b] = s
    lengths.append(len(s))

# "블록 스택" 형태로 DataFrame 만들기: (다른 열은 빈칸)
total_rows = sum(lengths)
out_df = pd.DataFrame(index=range(total_rows), columns=variable_value, dtype=object)

start = 0
for b in variable_value:
    s = series_map[b]
    out_df.loc[start:start+len(s)-1, b] = s.values
    start += len(s)

# 저장
#%%
csv_path = os.path.join(base_dir, '..', f'result/emulation_beta_{data_type}{init_envc}{init_tau0}.csv')

out_df.to_csv(csv_path, index=False)
print("done!! ->", csv_path)



# %%
df = pd.DataFrame()
for value in variable_value:
    temp = run_data.query('{}=={}'.format(variable_name,value))['HCW_related_infecs']
    df[value] = temp.reset_index(drop = True)
emulated_data = df.values[0][0]

print(emulated_data)
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

os.makedirs(os.path.join(base_dir, '..', 'result'), exist_ok=True)
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    'result/emulation_beta_{}.csv'.format(data_type))
csv_path = os.path.join(base_dir, '..', f'result/emulation_beta_{data_type}{init_envc}{init_tau0}.csv')

if os.path.isfile(csv_path):
    saved_df = pd.read_csv(csv_path,index_col=0)
    saved_df.columns = variable_value
    df = pd.concat([saved_df,df], ignore_index=True)
df.to_csv(csv_path)

print("done!!")









# %% 컴파트먼트까지 출력되는 버전

import os
import time
import pandas as pd

data_type = 'B'
num_iter = 1

# Parameters
cleanDay = 180
washrate = 0.9
isolationTime = 14
init_envc = 2
init_tau0 = 40

runtime = 30 * 36
probNewPatient = 0.003
probTransmission = 0.00005
isolationFactor = 0.75
height = 11
width = 32

variable_name = 'prob_transmission'
variable_value = [0.04]   # 필요하면 여러 값 넣어도 됨

start_time = time.time()

all_histories = []

# --- 저장 폴더: 한 칸 상위 result ---
result_dir = "../result"
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

            # 선택: 진행상황 보고 싶으면 활성화
            # if step % model.ticks_in_day == 0:
            #     day_now = step // model.ticks_in_day
            #     print(f"    day {day_now}")

        df_hist = model.get_history_dataframe().copy()

        df_hist["prob_transmission"] = b
        df_hist["iteration"] = it + 1

        all_histories.append(df_hist)

        print(f"  iteration {it+1}/{num_iter} finished")

elapsed = time.time() - start_time
print(f"\nDone. Elapsed time = {elapsed:.2f} sec")

# --- trajectory 합치기 ---
traj_df = pd.concat(all_histories, ignore_index=True)

# --- 평균 trajectory ---
mean_traj = (
    traj_df
    .groupby(["prob_transmission", "day"], as_index=False)
    .mean(numeric_only=True)
)

# --- 저장 ---
traj_path = f"{result_dir}/traj_all_iterations.csv"
mean_path = f"{result_dir}/traj_mean.csv"

traj_df.to_csv(traj_path, index=False)
mean_traj.to_csv(mean_path, index=False)

print("\nSaved files:")
print(traj_path)
print(mean_path)
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
# %% A기간에서하는걸로 컴파트먼트나오게 








# %%
import os
import time
import numpy as np
import pandas as pd

data_type = 'A'
num_iter = 10

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
variable_value = [0.05,0.06]

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
    sum(model.totalHWCinf),
    model.cumul_sick_patients
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
