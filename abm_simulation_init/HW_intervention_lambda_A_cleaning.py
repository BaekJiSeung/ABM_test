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


# %% CleaningDay intervention using saved beta_draws

import os
import time
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# =========================================================
# 기본 설정
# =========================================================

data_type = 'A'
num_iter = 50
init_envc = 9
init_tau0 = 140

# Parameters fixed
cleanDay = 180
washrate = 0.9
isolationTime = 14

runtime = 30 * 19
probNewPatient = 0.003
probTransmission = 0.0866
isolationFactor = 0.75
height = 11
width = 32

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

result_dir = os.path.abspath(os.path.join(base_dir, '..', 'result'))
data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))

os.makedirs(result_dir, exist_ok=True)

# =========================================================
# [1] 이미 저장된 beta_draws 불러오기
# =========================================================

beta_draw_load_path = os.path.join(
    result_dir,
    f'interv_beta_draws_{data_type}{init_envc}{init_tau0}.csv'
)

beta_draw_df = pd.read_csv(beta_draw_load_path)

if "beta_draw" not in beta_draw_df.columns:
    raise ValueError("beta_draw column not found in saved beta draw file.")

beta_draws = beta_draw_df["beta_draw"].astype(float).values

if len(beta_draws) < num_iter:
    raise ValueError(
        f"saved beta_draws count ({len(beta_draws)}) is smaller than num_iter ({num_iter})"
    )

beta_draws = beta_draws[:num_iter]

print("loaded beta draws ->", beta_draw_load_path)
print(beta_draw_df.head())
# %%
# =========================================================
# [2] CleaningDay intervention 설정
# =========================================================

variable_name = "cleaningDay"

# 원하는 cleaning 주기들
# 단위는 기존 코드와 동일하게 day 기준
variable_value = [30, 60, 90, 120,180,360]

beta_tag1 = variable_value[0]
beta_tag2 = variable_value[-1]
first_clean_after_days = 10
start_time = time.time()

print("CleaningDay intervention loading...\n")
print("now run")

all_rows = []

tasks = [
    (clean_day, it)
    for clean_day in variable_value
    for it in range(num_iter)
]

for clean_day, it in tqdm(tasks):

    beta_now = beta_draws[it]
    tau_for_clean = clean_day - first_clean_after_days
    model = CPE_Model_month(
        data_type=data_type,
        prob_new_patient=probNewPatient,
        prob_transmission=beta_now,
        isolation_factor=isolationFactor,   # fixed
        cleaningDay=clean_day,              # intervention
        hcw_wash_rate=washrate,             # fixed
        isolation_time=isolationTime,
        height=height,
        width=width,
        init_env=init_envc,
        tau_offset_days=tau_for_clean
    )

    max_steps = model.ticks_in_day * runtime

    for _ in range(max_steps):
        model.step()

    all_rows.append({
        variable_name: clean_day,
        "iteration": it,
        "beta_draw": beta_now,
        "HCW_related_infecs": getTotalInfec(model)
    })

elapsed = time.time() - start_time
print(f"\nCleaningDay intervention done. Elapsed time = {elapsed:.2f} sec")

# =========================================================
# [3] run_data 생성
# =========================================================

run_data = pd.DataFrame(all_rows)

print(run_data.head())
print("cols:", list(run_data.columns))

# =========================================================
# [4] pivot 저장
# =========================================================

df = run_data.pivot_table(
    index="iteration",
    columns=variable_name,
    values="HCW_related_infecs",
    aggfunc="first"
).reset_index(drop=True)

print(df.head())

csv_path = os.path.join(
    result_dir,
    f'interv_{variable_name}_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}.csv'
)

df.to_csv(csv_path, index=False)

print("raw saved ->", csv_path)
# %%

# %% intervention raw 파일을 월별 Summary로 변환
data_type = "A"     # A or B
init_envc = 9
init_tau0 = 140

days_per_month = 30

if data_type == "A":
    n_months = 19
elif data_type == "B":
    n_months = 36
else:
    raise ValueError("data_type must be 'A' or 'B'")
import os
import ast
import numpy as np
import pandas as pd
out_path = f"../result/interv_{variable_name}_summary_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}.csv"

# =========================================================
# 설정

# =========================================================
# raw data 읽기
# =========================================================

raw_df = pd.read_csv(csv_path)

print("raw columns:", raw_df.columns.tolist())
print(raw_df.head())


# =========================================================
# 문자열 리스트 -> 파이썬 리스트 변환 함수
# =========================================================

def parse_series(x):
    if pd.isna(x):
        return None

    if isinstance(x, list):
        return x

    if isinstance(x, np.ndarray):
        return x.tolist()

    # 문자열 "[...]" 형태
    if isinstance(x, str):
        return ast.literal_eval(x)

    raise ValueError(f"Cannot parse cell: {type(x)}, value={x}")


# =========================================================
# 일별 -> 월별 변환
# =========================================================

def daily_to_monthly(daily_series, days_per_month=30, n_months=None):
    arr = np.array(daily_series, dtype=float)

    # 이미 월별 길이인 경우
    if n_months is not None and len(arr) == n_months:
        return arr.tolist()

    if n_months is not None:
        needed = days_per_month * n_months
        arr = arr[:needed]

    m = len(arr) // days_per_month
    arr = arr[:m * days_per_month]

    monthly = arr.reshape(m, days_per_month).sum(axis=1)

    if n_months is not None:
        monthly = monthly[:n_months]

    return monthly.tolist()


# =========================================================
# intervention value별 summary 계산
# =========================================================

rows = []

for val in raw_df.columns:

    # 혹시 iteration, beta_draw 같은 컬럼이 섞여 있으면 제외
    if val in ["iteration", "Iteration", "Run", "run", "run_id", "beta_draw"]:
        continue

    series_list = raw_df[val].dropna().apply(parse_series).tolist()

    if len(series_list) == 0:
        print(f"[skip] {val}: no data")
        continue

    monthly_runs = [
        daily_to_monthly(
            s,
            days_per_month=days_per_month,
            n_months=n_months
        )
        for s in series_list
    ]

    monthly_arr = np.array(monthly_runs, dtype=float)

    # shape 체크
    if monthly_arr.shape[1] != n_months:
        print(f"[warning] {val}: expected {n_months} months, got {monthly_arr.shape[1]}")

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
        variable_name: float(val),
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
    .sort_values(variable_name)
    .reset_index(drop=True)
)

print(summary_df.head())


# =========================================================
# 저장
# =========================================================

summary_df.to_csv(out_path, index=False)

print("saved ->", out_path)

# %%
