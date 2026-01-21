# %%
from model.cpe_model_month import CPE_Model_month
from model.cpe_model_month import getHCWInfec
from model.cpe_model_month import getTotalInfec
from mesa.batchrunner import BatchRunner
from mesa.batchrunner import BatchRunnerMP
from multiprocessing import freeze_support
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mesa; print(mesa.__version__)


# %%
data_type = 'A'
num_iter = 50; np.int64(num_iter)

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
    "init_env": 10,
    "tau_offset_days": 140 
    }

variable_name = 'prob_transmission'
variable_value = [0.07,0.08,0.09,0.10]

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
    init_env=10,                 # [ADD] 초기 오염 개수 (예: 9)
    tau_offset_days=140
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
csv_path = os.path.join(base_dir, '..', f'result/emulation_beta_{data_type}.csv')
df.to_csv(csv_path, index=False)
print("done!! ->", csv_path)


#  ===================== RAW + SUMMARY 저장 =====================
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




















# %%
data_type = 'B'
num_iter = 50; np.int64(num_iter)

# Parameters
cleanDay = 180
washrate = 0.9
isolationTime = 14

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
    "height" : height, "width" : width 
    }

variable_name = 'prob_transmission'
variable_value = [0.08,0.082,0.083,0.084,0.086,0.088]

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
    height=height, width=width
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
df = pd.DataFrame()
for value in variable_value:
    temp = run_data.query('{}=={}'.format(variable_name,value))['HCW_related_infecs']
    df[value] = temp.reset_index(drop = True)
emulated_data = df.values[0][0]

print(emulated_data)

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    'result/emulation_beta_{}.csv'.format(data_type))
if os.path.isfile(csv_path):
    saved_df = pd.read_csv(csv_path,index_col=0)
    saved_df.columns = variable_value
    df = pd.concat([saved_df,df], ignore_index=True)
df.to_csv(csv_path)

print("done!!")
# %%

# %%
