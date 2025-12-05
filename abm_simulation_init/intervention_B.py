# %%
from model.cpe_model_month import CPE_Model_month
from model.cpe_model_month import getTotalInfec
from mesa.batchrunner import BatchRunner

import os, time, math, ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
# -------------------- 유틸 --------------------
def _fmt_secs(s):
    m, s = divmod(float(s), 60.0); h, m = divmod(m, 60.0)
    if h >= 1: return f"{int(h)}h {int(m)}m {s:0.1f}s"
    if m >= 1: return f"{int(m)}m {s:0.1f}s"
    return f"{s:0.2f}s"

def _to_list(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return []
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except: return []
    if isinstance(x, (list, tuple, np.ndarray)): return list(x)
    return [x]

def _series_to_matrix(series):
    lists = series.apply(_to_list)
    maxlen = max((len(L) for L in lists), default=0)
    mat = np.zeros((len(lists), maxlen))
    for i, L in enumerate(lists):
        if len(L) > 0: mat[i, :len(L)] = np.array(L, dtype=float)
    return mat  # (n_iter, n_months)

# -------------------- 경로 --------------------
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

out_dir = os.path.join(base_dir, '..', 'result')
os.makedirs(out_dir, exist_ok=True)
# %%
# -------------------- 실험 설정 --------------------
data_type = 'B'
num_iter  = 50             # β 샘플 개수(행 개수) — 테스트면 2로
cleanDay       = 180
washrate_base  = 0.90
isolationTime  = 14
runtime        = 30*36     # B기준36개월
probNewPatient = 0.003
isolationFactor= 0.75
height, width  = 11, 32

# 열 = handwash 값들
variable_name  = 'hcw_wash_rate'
variable_value = [0.91,0.93,0.97,0.99]   # 테스트면 [0.90, 0.95]

# -------------------- beta 샘플링 --------------------
beta_csv = os.path.join(base_dir, '..', 'data', 'B_samp_b.csv')
try:
    s = pd.read_csv(beta_csv, header=None, comment='#').iloc[:, 0]
except Exception:
    df_tmp = pd.read_csv(beta_csv, comment='#')
    s = df_tmp.select_dtypes(include=[np.number]).iloc[:, 0] if not df_tmp.empty else pd.Series(dtype=float)

s = pd.to_numeric(s, errors='coerce').dropna().reset_index(drop=True)
s_tail = s.iloc[2000:]  # 앞 2000 제외

# num_iter 개만 비복원 추출
sampled_betas = (
    s_tail.sample(n=num_iter, random_state=42)
          .sort_values()
          .reset_index(drop=True)
)

# 샘플 저장 + 히스토그램
betas_path = os.path.join(out_dir, f'sampled_betas_{data_type}.csv')
pd.DataFrame({'beta': sampled_betas}).to_csv(betas_path, index=False)

plt.figure(figsize=(6,4))
plt.hist(s_tail, bins='auto')
for b in sampled_betas:
    plt.axvline(b, linewidth=0.5)
plt.xlabel('beta (prob_transmission)'); plt.ylabel('count'); plt.tight_layout()
hist_path = os.path.join(out_dir, f'sampled_betas_{data_type}_hist.png')
plt.savefig(hist_path, dpi=150); plt.close()

print(f"[beta sampling] total={len(s)}, used={len(s_tail)}, sampled={len(sampled_betas)}", flush=True)
print(f"saved -> {betas_path}", flush=True)
print(f"saved -> {hist_path}", flush=True)
# %%
# -------------------- max_steps 계산 --------------------
tmp_model = CPE_Model_month(
    data_type=data_type,
    prob_new_patient=probNewPatient,
    prob_transmission=float(sampled_betas.iloc[0]),
    isolation_factor=isolationFactor,
    cleaningDay=cleanDay,
    hcw_wash_rate=washrate_base,
    isolation_time=isolationTime,
    height=height, width=width
)
max_steps = tmp_model.ticks_in_day * runtime

# -------------------- 누적 컨테이너 --------------------
fixed_params_base = {
    "data_type": data_type,
    "prob_new_patient": probNewPatient,
    "isolation_factor": isolationFactor,
    "cleaningDay": cleanDay,
    "isolation_time": isolationTime,
    "height": height, "width": width
}

# (beta, wash) → 월별 리스트
cell = []
summary_rows = []

# -------------------- 실행 (행=beta, 열=wash) --------------------
overall_t0 = time.time()
start = time.time()

TOTAL_RUNS = len(sampled_betas) * len(variable_value)
run_idx = 0

for wi, w in enumerate(variable_value, 1):
    wash_t0 = time.time()
    print(f"\n[wash {wi}/{len(variable_value)}] hcw_wash_rate={w:.2f} >>> start", flush=True)

    for bi, beta in enumerate(sampled_betas.tolist(), 1):
        run_idx += 1
        print(f"[{run_idx:04d}/{TOTAL_RUNS:04d}] start  | wash={w:.2f}  beta={beta:.6f}", flush=True)
        run_t0 = time.time()

        fixed_params = dict(fixed_params_base)
        fixed_params["prob_transmission"] = float(beta)

        # 변수는 반드시 variable_parameters로 전달(빈 dict 금지)
        variable_params = {variable_name: [float(w)]}

        # --- 여기부터 예전 느낌 출력 ---
        print("loading...\n\n", flush=True)

        batch_run = BatchRunner(
            CPE_Model_month,
            variable_parameters=variable_params,
            fixed_parameters=fixed_params,
            iterations=1,              # 조합당 1회
            max_steps=max_steps,
            display_progress=True,     # tqdm 표시
            model_reporters={"HCW_related_infecs": getTotalInfec}
        )

        print("now run", flush=True)
        batch_run.run_all()           # tqdm가 "0it [00:00, ?it/s]" 등 출력
        # --- 여기까지 ---

        run_data = batch_run.get_model_vars_dataframe()

        # reporter 값(월별 리스트) 한 개만 추출
        if "HCW_related_infecs" in run_data.columns:
            monthly_list = run_data.iloc[0]["HCW_related_infecs"]
        else:
            monthly_list = next((run_data.iloc[0][c] for c in run_data.columns
                                 if isinstance(run_data.iloc[0][c], (list, tuple))), [])

        # 저장
        cell.append((float(beta), float(w), monthly_list))

        # SUMMARY (조합당 1회지만 포맷 유지)
        mat = _series_to_matrix(pd.Series([monthly_list]))
        if mat.size > 0:
            mean_vals   = mat.mean(axis=0).tolist()
            std_vals    = [0.0]*mat.shape[1]  # n=1 → 0
            max_vals    = mat.max(axis=0).tolist()
            median_vals = np.median(mat, axis=0).tolist()
            nonzero_means, disease_free = [], []
            for j in range(mat.shape[1]):
                col = mat[:, j]
                nz = col[col > 0]
                nonzero_means.append(nz.mean() if len(nz) else 0.0)
                disease_free.append((col == 0).sum() / len(col) * 100.0)
        else:
            mean_vals = std_vals = max_vals = median_vals = nonzero_means = disease_free = []

        summary_rows.append({
            "beta": float(beta),
            variable_name: float(w),
            "n": 1,
            "mean": mean_vals,
            "std": std_vals,
            "max": max_vals,
            "median": median_vals,
            "nonzero_mean": nonzero_means,
            "disease_free(%)": disease_free
        })

        print(f"[{run_idx:04d}/{TOTAL_RUNS:04d}] done   | took {_fmt_secs(time.time()-run_t0)}", flush=True)

    print(f"[wash {wi}/{len(variable_value)}] hcw_wash_rate={w:.2f} <<< "
          f"took {_fmt_secs(time.time()-wash_t0)}", flush=True)

elapsed = time.time() - start
print(f"\n[done] total betas={len(sampled_betas)}, handwash={len(variable_value)}, "
      f"runs={len(sampled_betas)*len(variable_value)}, elapsed={_fmt_secs(elapsed)}", flush=True)
print(f"[TOTAL] elapsed = {_fmt_secs(time.time() - overall_t0)}", flush=True)
# %%
# -------------------- 저장 (행=beta, 열=wash) --------------------
beta_index = pd.Index(sampled_betas.tolist(), name="beta")
wash_cols  = pd.Index([f"{w:.2f}" for w in variable_value], name="hcw_wash_rate")

emul_df = pd.DataFrame(index=beta_index, columns=wash_cols, dtype=object)
for b, w, lst in cell:
    w_str = f"{w:.2f}"
    emul_df.loc[b, w_str] = lst if isinstance(lst, (list, tuple, np.ndarray)) else []

emul_path = os.path.join(out_dir, f'emulation_beta_{data_type}.csv')
emul_df.to_csv(emul_path, index=True)
print("emulation saved ->", emul_path)

raw_path = os.path.join(out_dir, f'interv_{variable_name}_{data_type}_raw.csv')
emul_df.to_csv(raw_path, index=True)
print("raw saved       ->", raw_path)

summary_path = os.path.join(out_dir, f'interv_{variable_name}_{data_type}_summary.csv')
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
print("summary saved   ->", summary_path)


# -------------------- (추가) handwash 값별로 β 평균/표준편차 집계 --------------------
# emul_df: 행=beta, 열=wash("0.90" 등 문자열), 셀=월별 리스트
wash_summary = []

for w_str in emul_df.columns:
    series = emul_df[w_str]                 # 해당 handwash 열(모든 β에 대한 월별 리스트)
    mat = _series_to_matrix(series)         # (n_betas, n_months)로 변환
    if mat.size == 0:
        continue

    # β(=행) 방향 통계
    mean_vals   = mat.mean(axis=0).tolist()
    std_vals    = mat.std(axis=0, ddof=1).tolist() if mat.shape[0] > 1 else [0.0] * mat.shape[1]
    max_vals    = mat.max(axis=0).tolist()
    median_vals = np.median(mat, axis=0).tolist()

    nonzero_means = []
    disease_free  = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        nz = col[col > 0]
        nonzero_means.append(nz.mean() if len(nz) else 0.0)
        disease_free.append((col == 0).sum() / len(col) * 100.0)

    wash_summary.append({
        "hcw_wash_rate": float(w_str),
        "n_betas": int(mat.shape[0]),   # = num_iter (예: 50)
        "mean": mean_vals,
        "std": std_vals,
        "max": max_vals,
        "median": median_vals,
        "nonzero_mean": nonzero_means,
        "disease_free(%)": disease_free,
    })

wash_summary_df = pd.DataFrame(wash_summary).sort_values("hcw_wash_rate").reset_index(drop=True)

# handwash 기준 요약 저장
wash_sum_path = os.path.join(out_dir, f'interv_{variable_name}_{data_type}_by_wash_summary.csv')
wash_summary_df.to_csv(wash_sum_path, index=False)
print("wash-by-wash summary saved ->", wash_sum_path)
# %%
# ==== SAVE-ONLY RECOVERY (순서 기반으로 재구성; β 중복 OK) ====
import os, sys, pandas as pd, numpy as np

def _to_list(x):
    import ast, math
    if x is None or (isinstance(x, float) and math.isnan(x)): return []
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except: return []
    if isinstance(x, (list, tuple, np.ndarray)): return list(x)
    return [x]

def _series_to_matrix(series):
    lists = series.apply(_to_list)
    maxlen = max((len(L) for L in lists), default=0)
    mat = np.zeros((len(lists), maxlen))
    for i, L in enumerate(lists):
        if len(L) > 0:
            mat[i, :len(L)] = np.array(L, dtype=float)
    return mat

try:
    # 0) 필요한 변수 확인
    needed = ["cell", "sampled_betas", "variable_value", "out_dir", "data_type", "variable_name", "summary_rows"]
    missing = [k for k in needed if k not in globals()]
    if missing:
        raise RuntimeError(f"메모리에 없는 변수: {missing}")

    n_w = len(variable_value)
    n_b = len(sampled_betas)
    if len(cell) != n_w * n_b:
        print(f"[WARN] cell 길이({len(cell)}) != n_w*n_b({n_w*n_b}). 일부 조합이 누락되었을 수 있습니다.")

    # 1) 순서 기반으로 2D object 배열 채우기: 행=β 순서, 열=wash 순서
    arr = np.empty((n_b, n_w), dtype=object)
    p = 0
    for wi in range(n_w):
        for bi in range(n_b):
            # cell[wi*n_b + bi] 가 (beta, wash, list)
            _, _, lst = cell[p]
            p += 1
            if not isinstance(lst, (list, tuple, np.ndarray)):
                lst = []
            else:
                lst = list(lst)
            arr[bi, wi] = lst

    # 2) DataFrame 구성 (인덱스에 β 값 그대로 — 중복 허용)
    beta_index = pd.Index(list(sampled_betas), name="beta")
    wash_cols  = pd.Index([f"{float(w):.2f}" for w in variable_value], name="hcw_wash_rate")
    emul_df = pd.DataFrame(arr, index=beta_index, columns=wash_cols)

    # 3) 저장 경로
    os.makedirs(out_dir, exist_ok=True)
    emul_path    = os.path.abspath(os.path.join(out_dir, f'emulation_beta_{data_type}.csv'))
    raw_path     = os.path.abspath(os.path.join(out_dir, f'interv_{variable_name}_{data_type}_raw.csv'))
    summary_path = os.path.abspath(os.path.join(out_dir, f'interv_{variable_name}_{data_type}_summary.csv'))

    # 4) 저장 (emulation과 raw는 동일 형식 유지)
    emul_df.to_csv(emul_path, index=True)
    print(f"[SAVE] emulation -> {emul_path}")
    emul_df.to_csv(raw_path, index=True)
    print(f"[SAVE] raw       -> {raw_path}")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[SAVE] summary   -> {summary_path}")

    # 5) wash별(열별) β-평균 요약 저장
    wash_summary = []
    for w_str in emul_df.columns:
        series = emul_df[w_str]
        mat = _series_to_matrix(series)
        if mat.size == 0:
            continue
        mean_vals   = mat.mean(axis=0).tolist()
        std_vals    = mat.std(axis=0, ddof=1).tolist() if mat.shape[0] > 1 else [0.0]*mat.shape[1]
        max_vals    = mat.max(axis=0).tolist()
        median_vals = np.median(mat, axis=0).tolist()

        nonzero_means, disease_free = [], []
        for j in range(mat.shape[1]):
            col = mat[:, j]
            nz = col[col > 0]
            nonzero_means.append(nz.mean() if len(nz) else 0.0)
            disease_free.append((col == 0).sum() / len(col) * 100.0)

        wash_summary.append({
            "hcw_wash_rate": float(w_str),
            "n_betas": int(mat.shape[0]),   # = num_iter
            "mean": mean_vals,
            "std": std_vals,
            "max": max_vals,
            "median": median_vals,
            "nonzero_mean": nonzero_means,
            "disease_free(%)": disease_free,
        })

    wash_sum_path = os.path.abspath(os.path.join(out_dir, f'interv_{variable_name}_{data_type}_by_wash_summary.csv'))
    pd.DataFrame(wash_summary).sort_values("hcw_wash_rate").to_csv(wash_sum_path, index=False)
    print(f"[SAVE] wash-by-wash summary -> {wash_sum_path}")

    print(f"[CHECK] β={n_b}, wash={n_w}, cell={len(cell)} (expected={n_b*n_w})")

except Exception as e:
    import traceback
    print("[ERROR] 저장 중 문제가 발생했습니다:", e, file=sys.stderr)
    traceback.print_exc()

# %%
