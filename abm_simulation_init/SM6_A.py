# %%
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator

def infer_abm_beta_from_sm(sm_hat, sm_low=None, sm_high=None,
                           pairs_csv="sm_fit/theta_pairs_subset_cumPoisson_A9140.csv"):
    """
    Step6: (β_ABM, β_SM) 매핑으로 데이터에서 얻은 β_SM → β_ABM 역보간
    """

    # 1) 페어 로드 & 열 이름 유연 처리  -----------------------------
    df_raw = pd.read_csv(pairs_csv)
    # 열 이름이 'beta_sm_hat' 이거나 'theta_hat'일 수 있음 -> 통일
    if "beta_sm_hat" in df_raw.columns:
        col_sm = "beta_sm_hat"
    elif "theta_hat" in df_raw.columns:          # ← [NEW] fallback
        col_sm = "theta_hat"
    else:
        raise ValueError("CSV에 beta_sm_hat(또는 theta_hat) 열이 없습니다.")

    if "beta_abm" not in df_raw.columns:
        raise ValueError("CSV에 beta_abm 열이 없습니다.")

    df = df_raw[["beta_abm", col_sm]].dropna().rename(columns={col_sm: "beta_sm_hat"})
    df = df.sort_values("beta_abm").reset_index(drop=True)
    x_abm = df["beta_abm"].to_numpy(float)
    y_sm  = df["beta_sm_hat"].to_numpy(float)

    # 2) ABM → SM 보간기 (단조 보존)  ------------------------------
    f_abm2sm = PchipInterpolator(x_abm, y_sm)

    # 촘촘한 그리드에서 SM 값 생성
    abm_grid = np.linspace(x_abm.min(), x_abm.max(), 2001)
    sm_grid  = f_abm2sm(abm_grid)

    # 단조 증가 방향 맞추기
    if sm_grid[-1] < sm_grid[0]:
        sm_grid  = sm_grid[::-1]
        abm_grid = abm_grid[::-1]

    # 3) SM → ABM 역보간 준비: x가 strictly increasing 되도록 압축 ----
    # (동일한 sm 값들이 있으면 PCHIP이 거부하므로 고유값만 사용)
    order = np.argsort(sm_grid)
    sm_sorted  = sm_grid[order]
    abm_sorted = abm_grid[order]

    sm_unique, idx_unique = np.unique(sm_sorted, return_index=True)  # ← [KEY]
    abm_unique = abm_sorted[idx_unique]

    # 이제 sm_unique는 strictly increasing
    g_sm2abm = PchipInterpolator(sm_unique, abm_unique)

    # 4) 점추정 및 구간 역보간 ---------------------------------------
    # 범위 밖 값은 클리핑
    sm_min, sm_max = float(sm_unique.min()), float(sm_unique.max())
    sm_hat_c = np.clip(sm_hat, sm_min, sm_max)
    abm_hat = float(g_sm2abm(sm_hat_c))

    if sm_low is None or sm_high is None:
        return {
            "beta_abm_hat": abm_hat,
            "beta_abm_low": None,
            "beta_abm_high": None,
            "range_used": (sm_min, sm_max)
        }

    sm_lo_c = np.clip(sm_low,  sm_min, sm_max)
    sm_hi_c = np.clip(sm_high, sm_min, sm_max)
    abm_low  = float(g_sm2abm(sm_lo_c))
    abm_high = float(g_sm2abm(sm_hi_c))
    if abm_low > abm_high:
        abm_low, abm_high = abm_high, abm_low

    return {
        "beta_abm_hat": abm_hat,
        "beta_abm_low": abm_low,
        "beta_abm_high": abm_high,
        "range_used": (sm_min, sm_max)
    }

# %%
# [CHANGED] 여기 예시 호출도 동일. 다만 pairs_csv가 여전히
#           'theta_pairs_subset_cumPoisson.csv' 파일이라면,
#           그 파일 안의 열 이름이 반드시 'beta_sm_hat'이어야 함!
#           (아직 'theta_hat'이면 CSV를 미리 rename 하거나 위 read 구문을 고쳐줘.)
res = infer_abm_beta_from_sm(
    sm_hat=3.374,         # 데이터에서 뽑은 SM β (누적 MLE)
    sm_low=3.315,          # CI 있으면 넣기
    sm_high=3.431,
    pairs_csv="sm_fit/theta_pairs_subset_cumPoisson_A9140.csv"
)
print(res)

# %%
# %% imports
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

# ==== 설정 ====
csv_path = "../result/interv_prob_transmission_summary_A9140_0.03802-0.03802.csv"
start_month = "2017-01"

# 색상
COLOR_OBS = "black"
COLOR_ABM = "blue"

# ==== CSV 읽기 ====
df = pd.read_csv(csv_path)

print("columns:", df.columns.tolist())
print(df.head())

# 첫 행 사용 (beta 하나만 있는 파일이라고 가정)
row = df.iloc[0]

# beta 값
beta_val = float(row["beta"]) if "beta" in df.columns else 0.03802

# mean / std 파싱
m_mean = np.array(ast.literal_eval(row["mean"]), dtype=float)
m_std  = np.array(ast.literal_eval(row["std"]), dtype=float)

# 길이 확인
n_months = len(m_mean)
print("n_months:", n_months)

# observed 데이터
y_month = np.array([5,2,0,2,1,1,2,2,6,1,1,0,2,1,1,2,5,2,2], dtype=float)

if len(y_month) != n_months:
    raise ValueError(f"Observed data length ({len(y_month)}) != mean length ({n_months})")

# ==== 월 인덱스 ====
months = pd.period_range(start_month, periods=n_months, freq="M").to_timestamp()

x = np.arange(n_months)
date_labels = pd.date_range(start=f"{start_month}-01", periods=n_months, freq="MS")
date_str = [d.strftime("%Y-%m-%d") for d in date_labels]

major_idx = np.arange(0, n_months, 6)
minor_idx = np.arange(0, n_months, 3)

# ==== cumulative ====
c_mean = np.cumsum(m_mean)
y_cum  = np.cumsum(y_month)

# 주의:
# 월별 std를 단순 누적해서 cumulative std로 보는 건 엄밀하지 않음.
# 독립 가정하에 대략적인 cumulative band를 만들려면 아래처럼 계산 가능.
c_std = np.sqrt(np.cumsum(m_std**2))

# ==== 플롯: 월별 ====
plt.figure(figsize=(9, 6))

plt.plot(
    x, m_mean,
    "o-",
    color=COLOR_ABM,
    linewidth=2,
    label=f"ABM mean (beta_ABM={beta_val:.5f})"
)

plt.fill_between(
    x,
    m_mean - m_std,
    m_mean + m_std,
    color=COLOR_ABM,
    alpha=0.2,
    label="ABM ±1 SD"
)

plt.plot(
    x, y_month,
    "s-",
    color=COLOR_OBS,
    linewidth=2.5,
    label="Observed data"
)

plt.ylabel("HAI counts", fontsize=18)

ax = plt.gca()
ax.set_xticks(major_idx)
ax.set_xticks(minor_idx, minor=True)
ax.set_xticklabels([date_str[i] for i in major_idx], fontsize=18)

plt.yticks(fontsize=18)

ax.grid(True, which="major", axis="x", linestyle="-")
ax.grid(True, which="minor", axis="x", linestyle="--")
ax.grid(True, axis="y", linestyle="-")

plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

# ==== 플롯: 누적 ====
plt.figure(figsize=(9, 6))

plt.plot(
    x, c_mean,
    "o-",
    color=COLOR_ABM,
    linewidth=2,
    label=f"ABM cumulative (beta_ABM={beta_val:.5f})"
)

plt.fill_between(
    x,
    c_mean - c_std,
    c_mean + c_std,
    color=COLOR_ABM,
    alpha=0.2,
    label="ABM cumulative ± approx. SD"
)

plt.plot(
    x, y_cum,
    "s-",
    color=COLOR_OBS,
    linewidth=2.5,
    label="Observed cumulative"
)

plt.ylabel("Cumulative HAI", fontsize=18)

ax = plt.gca()
ax.set_xticks(major_idx)
ax.set_xticks(minor_idx, minor=True)
ax.set_xticklabels([date_str[i] for i in major_idx], fontsize=18)

plt.yticks(fontsize=18)

ax.grid(True, which="major", axis="x", linestyle="-")
ax.grid(True, which="minor", axis="x", linestyle="--")
ax.grid(True, axis="y", linestyle="-")
ax.set_ylim(bottom=0)

plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

# ==== 표 출력 ====
summary_df = pd.DataFrame({
    "month": months,
    "monthly_mean": m_mean,
    "monthly_sd": m_std,
    "observed_monthly": y_month,
    "cum_mean": c_mean,
    "cum_sd_approx": c_std,
    "observed_cum": y_cum
})

print(summary_df.round(3).to_string(index=False))
# %%
