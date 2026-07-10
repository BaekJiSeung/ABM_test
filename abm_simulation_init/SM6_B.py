# %% Step6: B version, infer ABM beta from SM beta

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

def infer_abm_beta_from_sm(
    sm_hat,
    sm_low=None,
    sm_high=None,
    pairs_csv="sm_fit/theta_pairs_subset_cumGaussian_B260.csv"
):
    """
    Step6:
    B period에서 만든 (beta_ABM, theta_SM) mapping을 이용해서
    observed data에서 얻은 beta_SM(theta_SM)을 beta_ABM으로 역보간.
    """

    # 1) mapping pair 로드
    df_raw = pd.read_csv(pairs_csv)

    # theta column 이름 유연 처리
    if "beta_sm_hat" in df_raw.columns:
        col_sm = "beta_sm_hat"
    elif "theta_hat" in df_raw.columns:
        col_sm = "theta_hat"
    else:
        raise ValueError("CSV에 beta_sm_hat 또는 theta_hat 열이 없습니다.")

    if "beta_abm" not in df_raw.columns:
        raise ValueError("CSV에 beta_abm 열이 없습니다.")

    df = (
        df_raw[["beta_abm", col_sm]]
        .dropna()
        .rename(columns={col_sm: "beta_sm_hat"})
        .sort_values("beta_abm")
        .reset_index(drop=True)
    )

    x_abm = df["beta_abm"].to_numpy(float)
    y_sm  = df["beta_sm_hat"].to_numpy(float)

    # 2) ABM beta -> SM theta 보간
    f_abm2sm = PchipInterpolator(x_abm, y_sm)

    abm_grid = np.linspace(x_abm.min(), x_abm.max(), 2001)
    sm_grid  = f_abm2sm(abm_grid)

    # 3) SM -> ABM 역보간 준비
    # sm_grid가 감소 방향이면 뒤집기
    if sm_grid[-1] < sm_grid[0]:
        sm_grid  = sm_grid[::-1]
        abm_grid = abm_grid[::-1]

    # 혹시 non-monotone 구간이 있거나 같은 값이 있으면 정렬 후 unique 처리
    order = np.argsort(sm_grid)
    sm_sorted  = sm_grid[order]
    abm_sorted = abm_grid[order]

    sm_unique, idx_unique = np.unique(sm_sorted, return_index=True)
    abm_unique = abm_sorted[idx_unique]

    g_sm2abm = PchipInterpolator(sm_unique, abm_unique)

    # 4) 범위 확인 및 clipping
    sm_min, sm_max = float(sm_unique.min()), float(sm_unique.max())

    sm_hat_c = np.clip(sm_hat, sm_min, sm_max)
    abm_hat = float(g_sm2abm(sm_hat_c))

    if sm_low is None or sm_high is None:
        return {
            "beta_abm_hat": abm_hat,
            "beta_abm_low": None,
            "beta_abm_high": None,
            "range_used": (sm_min, sm_max),
            "was_clipped": sm_hat != sm_hat_c
        }

    sm_lo_c = np.clip(sm_low, sm_min, sm_max)
    sm_hi_c = np.clip(sm_high, sm_min, sm_max)

    abm_low  = float(g_sm2abm(sm_lo_c))
    abm_high = float(g_sm2abm(sm_hi_c))

    if abm_low > abm_high:
        abm_low, abm_high = abm_high, abm_low

    return {
        "beta_abm_hat": abm_hat,
        "beta_abm_low": abm_low,
        "beta_abm_high": abm_high,
        "range_used": (sm_min, sm_max),
        "was_clipped": (sm_hat != sm_hat_c) or (sm_low != sm_lo_c) or (sm_high != sm_hi_c)
    }


# =========================
# B observed SM result
# =========================

res_B = infer_abm_beta_from_sm(
    sm_hat=3.3148,
    sm_low=3.2936,
    sm_high=3.3360,
    pairs_csv="sm_fit/theta_pairs_subset_cumGaussian_B260.csv"
)

print(res_B)
# %%


# %% imports
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

# ==== 설정 ====
csv_path = "../result/interv_prob_transmission_summary_B260_0.041361-0.043161.csv"
start_month = "2021-01"

# 색상
COLOR_OBS = "black"
COLOR_ABM = "blue"

# ==== CSV 읽기 ====
df = pd.read_csv(csv_path)

print("columns:", df.columns.tolist())
print(df.head())

# 첫 행 사용
# beta 하나만 있는 파일이라고 가정
row = df.iloc[0]

# beta 값
beta_val = float(row["beta"]) if "beta" in df.columns else 0.03788

# mean / std 파싱
m_mean = np.array(ast.literal_eval(row["mean"]), dtype=float)
m_std  = np.array(ast.literal_eval(row["std"]), dtype=float)

# 길이 확인
n_months = len(m_mean)
print("n_months:", n_months)

# ==== B period observed data: 2021-01 ~ 2023-12 ====
y_month = np.array(
    [
        1,2,2,2,1,0,0,3,2,2,2,0,
        3,0,1,2,0,1,4,5,4,2,4,1,
        0,0,1,0,1,1,0,3,1,0,3,0
    ],
    dtype=float
)

if len(y_month) != n_months:
    raise ValueError(
        f"Observed data length ({len(y_month)}) != mean length ({n_months})"
    )

# ==== 월 인덱스 ====
months = pd.period_range(
    start_month,
    periods=n_months,
    freq="M"
).to_timestamp()

x = np.arange(n_months)

date_labels = pd.date_range(
    start=f"{start_month}-01",
    periods=n_months,
    freq="MS"
)

date_str = [d.strftime("%Y-%m-%d") for d in date_labels]

major_idx = np.arange(0, n_months, 6)
minor_idx = np.arange(0, n_months, 3)

# ==== cumulative ====
c_mean = np.cumsum(m_mean)
y_cum  = np.cumsum(y_month)

# 월별 std를 cumulative std로 근사
# 독립 가정:
# Var(cumsum) ≈ cumsum(Var)
c_std = np.sqrt(np.cumsum(m_std ** 2))

# ==== 플롯 1: 월별 incidence ====
plt.figure(figsize=(9, 6))

plt.plot(
    x,
    m_mean,
    "o-",
    color=COLOR_ABM,
    linewidth=2,
    label=fr"ABM mean ($\beta_{{ABM}}$={beta_val:.5f})"
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
    x,
    y_month,
    "s-",
    color=COLOR_OBS,
    linewidth=2.5,
    label="Observed data"
)

plt.ylabel("Monthly HAI counts", fontsize=18)

ax = plt.gca()
ax.set_xticks(major_idx)
ax.set_xticks(minor_idx, minor=True)
ax.set_xticklabels([date_str[i] for i in major_idx], fontsize=14, rotation=30)

plt.yticks(fontsize=16)

ax.grid(True, which="major", axis="x", linestyle="-", alpha=0.4)
ax.grid(True, which="minor", axis="x", linestyle="--", alpha=0.3)
ax.grid(True, axis="y", linestyle="-", alpha=0.4)

plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# ==== 플롯 2: 누적 incidence ====
plt.figure(figsize=(9, 6))

plt.plot(
    x,
    c_mean,
    "o-",
    color=COLOR_ABM,
    linewidth=2,
    label=fr"ABM cumulative ($\beta_{{ABM}}$={beta_val:.5f})"
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
    x,
    y_cum,
    "s-",
    color=COLOR_OBS,
    linewidth=2.5,
    label="Observed cumulative"
)

plt.ylabel("Cumulative HAI", fontsize=18)

ax = plt.gca()
ax.set_xticks(major_idx)
ax.set_xticks(minor_idx, minor=True)
ax.set_xticklabels([date_str[i] for i in major_idx], fontsize=14, rotation=30)

plt.yticks(fontsize=16)

ax.grid(True, which="major", axis="x", linestyle="-", alpha=0.4)
ax.grid(True, which="minor", axis="x", linestyle="--", alpha=0.3)
ax.grid(True, axis="y", linestyle="-", alpha=0.4)
ax.set_ylim(bottom=0)

plt.legend(fontsize=14)
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