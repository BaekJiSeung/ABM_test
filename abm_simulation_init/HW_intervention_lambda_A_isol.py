# %% Isolation time intervention using saved beta_draws

from model.cpe_model_month_lambda import CPE_Model_month
from model.cpe_model_month_lambda import getHCWInfec
from model.cpe_model_month_lambda import getTotalInfec

import os
import time
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="No agent reporters*")

# %%
# =========================================================
# [0] 기본 설정
# =========================================================

data_type = 'A'
num_iter = 50

init_envc = 9
init_tau0 = 140

# Fixed parameters
cleanDay = 180
washrate = 0.9
isolationTime = 14      # baseline
isolationFactor = 0.75

runtime = 30 * 19

probNewPatient = 0.003
probTransmission = 0.0866   # 실제 run에서는 beta_draws[it] 사용
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
    raise ValueError(
        f"'beta_draw' column not found in {beta_draw_load_path}. "
        f"Current columns: {list(beta_draw_df.columns)}"
    )

beta_draws = beta_draw_df["beta_draw"].astype(float).values

if len(beta_draws) < num_iter:
    raise ValueError(
        f"saved beta_draws count ({len(beta_draws)}) is smaller than num_iter ({num_iter})"
    )

beta_draws = beta_draws[:num_iter]

print("loaded beta draws ->", beta_draw_load_path)
print(beta_draw_df.head())


# =========================================================
# [2] isolation_time intervention 설정
# =========================================================

variable_name = "isolation_time"

# baseline = 14
# 낮을수록 더 빨리 isolation으로 이동
# 주의: 1은 np.random.randint(1, 1) 때문에 에러 가능
variable_value = [2,6, 10, 14,20,28]

beta_tag1 = variable_value[0]
beta_tag2 = variable_value[-1]

start_time = time.time()

print("\nIsolation time intervention")
print("variable values:", variable_value)
print("now run")


# =========================================================
# [3] 같은 beta 50개를 모든 isolation_time 값에 대해 공통 사용
# =========================================================

all_rows = []

tasks = [
    (iso_time, it)
    for iso_time in variable_value
    for it in range(num_iter)
]

for iso_time, it in tqdm(tasks):

    beta_now = beta_draws[it]

    model = CPE_Model_month(
        data_type=data_type,
        prob_new_patient=probNewPatient,
        prob_transmission=beta_now,
        isolation_factor=isolationFactor,   # fixed
        cleaningDay=cleanDay,               # fixed
        hcw_wash_rate=washrate,             # fixed
        isolation_time=iso_time,            # intervention
        height=height,
        width=width,
        init_env=init_envc,
        tau_offset_days=init_tau0
    )

    max_steps = model.ticks_in_day * runtime

    for _ in range(max_steps):
        model.step()

    all_rows.append({
        variable_name: iso_time,
        "iteration": it,
        "beta_draw": beta_now,
        "HCW_related_infecs": getTotalInfec(model)
    })

elapsed = time.time() - start_time
print(f"\nDone. Elapsed time = {elapsed:.2f} sec")


# =========================================================
# [4] run_data 생성
# =========================================================

run_data = pd.DataFrame(all_rows)

print(run_data.head())
print("cols:", list(run_data.columns))


# =========================================================
# [5] 피벗 저장
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
# %% interv raw 파일을 Summary (월별로 나오게끔)

import os
import ast
import numpy as np
import pandas as pd

# -----------------------------
# 설정
# -----------------------------
data_type = "A"          # A or B
days_per_month = 30

# 예시
# variable_name = "isolation_time"
# csv_path = f"../result/interv_{variable_name}_{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}.csv"

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
    """
    raw cell이 문자열 list, list, np.ndarray 어느 형태든 daily series로 변환
    """

    # 이미 list
    if isinstance(x, list):
        return x

    # numpy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    # 문자열
    if isinstance(x, str):
        xs = x.strip()

        if xs == "" or xs.lower() == "nan":
            return None

        # "[1,2,3]" 형태
        if xs.startswith("[") and xs.endswith("]"):
            return ast.literal_eval(xs)

        # 혹시 숫자 하나만 들어온 경우
        return [float(xs)]

    # 결측값
    if pd.isna(x):
        return None

    # 그 외 숫자 하나
    return [float(x)]


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

    # 혹시 길이 부족하면 n_months에 맞춰 padding
    if n_months is not None and len(monthly) < n_months:
        monthly = np.pad(
            monthly,
            (0, n_months - len(monthly)),
            mode="constant",
            constant_values=0
        )

    return monthly.tolist()


# -----------------------------
# parameter별 summary 계산
# -----------------------------
rows = []

skip_cols = ["iteration", "Iteration", "Run", "run", "run_id", "beta_draw"]

for val in raw_df.columns:

    if val in skip_cols:
        continue

    # 해당 parameter value 열에서 non-null iteration만 가져오기
    series_list = (
        raw_df[val]
        .dropna()
        .apply(parse_series)
        .dropna()
        .tolist()
    )

    if len(series_list) == 0:
        print(f"[skip] {val}: no valid series")
        continue

    # 각 iteration의 일별 series -> 월별 series
    monthly_runs = [
        daily_to_monthly(
            s,
            days_per_month=days_per_month,
            n_months=n_months
        )
        for s in series_list
    ]

    monthly_arr = np.array(monthly_runs, dtype=float)   # shape = (n_iter, n_months)

    # 월별 통계
    mean_ = monthly_arr.mean(axis=0).tolist()
    std_ = monthly_arr.std(axis=0, ddof=0).tolist()
    max_ = monthly_arr.max(axis=0).tolist()
    median_ = np.median(monthly_arr, axis=0).tolist()

    # nonzero_mean, disease_free
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


# -----------------------------
# 저장
# -----------------------------
out_path = (
    f"../result/interv_{variable_name}_summary_"
    f"{data_type}{init_envc}{init_tau0}_{beta_tag1}-{beta_tag2}.csv"
)

summary_df.to_csv(out_path, index=False)

print("saved ->", out_path)
# %%
