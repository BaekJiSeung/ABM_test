# %% ================== 준비 ==================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast, os
from scipy.optimize import brentq


# ---- 관측 데이터 ----
y_month = np.array(
    [1,2,2,2,1,0,0,3,2,2,2,0,3,0,1,2,0,1,
     4,5,4,2,4,1,0,0,1,0,1,1,0,3,1,0,3,0],
    dtype=float
)
y_cum = np.cumsum(y_month)


# ---- 기간 정의용 (2021-01 ~ 2023-12) ----
monthly_PI = pd.Series({
    "2021-01":2,"2021-02":2,"2021-03":3,"2021-04":1,"2021-05":0,"2021-06":3,
    "2021-07":3,"2021-08":5,"2021-09":2,"2021-10":5,"2021-11":0,"2021-12":2,
    "2022-01":4,"2022-02":1,"2022-03":1,"2022-04":3,"2022-05":5,"2022-06":2,
    "2022-07":2,"2022-08":5,"2022-09":1,"2022-10":4,"2022-11":7,"2022-12":2,
    "2023-01":3,"2023-02":2,"2023-03":1,"2023-04":0,"2023-05":5,"2023-06":3,
    "2023-07":1,"2023-08":3,"2023-09":4,"2023-10":7,"2023-11":7,"2023-12":3
})

# ---- 실제 P_I 입원 날짜 ----
PI_dates = [
    "2021-01-08","2021-01-29",
    "2021-02-17","2021-02-19",
    "2021-03-02","2021-03-11","2021-03-28",
    "2021-04-04",
    "2021-06-16","2021-06-28","2021-06-30",
    "2021-07-06","2021-07-07","2021-07-20",
    "2021-08-10","2021-08-14","2021-08-20","2021-08-21","2021-08-23",
    "2021-09-15","2021-09-18",
    "2021-10-04","2021-10-20","2021-10-27","2021-10-28","2021-10-29",
    "2021-12-27","2021-12-27",

    "2022-01-04","2022-01-05","2022-01-07","2022-01-14",
    "2022-02-18",
    "2022-03-12",
    "2022-04-15","2022-04-15","2022-04-17",
    "2022-05-06","2022-05-14","2022-05-16","2022-05-21","2022-05-24",
    "2022-06-27","2022-06-30",
    "2022-07-03","2022-07-20",
    "2022-08-10","2022-08-15","2022-08-16","2022-08-18","2022-08-25",
    "2022-09-20",
    "2022-10-02","2022-10-11","2022-10-15","2022-10-21",
    "2022-11-08","2022-11-13","2022-11-15","2022-11-17","2022-11-20","2022-11-21","2022-11-23",
    "2022-12-21","2022-12-29",

    "2023-01-16","2023-01-26","2023-01-30",
    "2023-02-03","2023-02-13",
    "2023-03-29",
    "2023-05-10","2023-05-10","2023-05-10","2023-05-15","2023-05-23",
    "2023-06-07","2023-06-12","2023-06-15",
    "2023-07-01",
    "2023-08-04","2023-08-10","2023-08-16",
    "2023-09-10","2023-09-17","2023-09-24","2023-09-28",
    "2023-10-05","2023-10-06","2023-10-08","2023-10-10","2023-10-14","2023-10-20","2023-10-31",
    "2023-11-08","2023-11-08","2023-11-13","2023-11-17","2023-11-19","2023-11-23","2023-11-28",
    "2023-12-04","2023-12-09","2023-12-09"
]


def _make_AI_from_dates(pi_dates, days):
    T = len(days)
    A = np.zeros(T)
    idx_map = {d: i for i, d in enumerate(days)}

    for d in pi_dates:
        ts = pd.to_datetime(d)
        idx = idx_map.get(ts, None)
        if idx is not None:
            A[idx] += 1.0

    return A


def simulate_theta(beta, init_env, tau0,
                   monthly_PI=monthly_PI,
                   pi_dates=PI_dates):
    """
    B period SM/ABM-style simulation.
    상태:
    P_S_sh, P_HAI_sh, P_HAI_iso, P_I, H_C, Env_C
    """

    # ---- 파라미터 ----
    C_total = 30
    C_iso = 30
    C_sh = C_total

    N_H, N_E = 19, 30

    # B기간: hospital stay 14일 기반
    mu_S = 1/7
    mu_HAI = 1/14
    mu_I = 1/7

    p_wash = 0.90
    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    iso_factor = 0.75

    isol_time = 7.0
    sigma = 1.0 / isol_time

    # ---- 시간축 ----
    start = pd.Period(monthly_PI.index.min(), freq="M").to_timestamp(how="start")
    end = pd.Period(monthly_PI.index.max(), freq="M").to_timestamp(how="end")

    days = pd.date_range(start, end, freq="D")
    T = len(days)

    # ---- P_I 입원 패턴 ----
    A_I_day = _make_AI_from_dates(pi_dates, days)

    # ---- 상태 ----
    P_S_sh = np.zeros(T)
    P_HAI_sh = np.zeros(T)
    P_HAI_iso = np.zeros(T)
    P_I = np.zeros(T)
    H_C = np.zeros(T)
    Env_C = np.zeros(T)
    NewHAI_day = np.zeros(T)

    # 초기조건
    P_S_sh[0] = C_total - 1
    P_I[0] = 1
    Env_C[0] = init_env

    for t in range(T):

        # tau0 반영 대청소
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            xx = Env_C[t]
            Env_C[t] = 0.1 * xx

        PS_sh, PH_sh = P_S_sh[t], P_HAI_sh[t]
        PH_iso = P_HAI_iso[t]
        PI, HC, EC = P_I[t], H_C[t], Env_C[t]

        # 오늘 입원하는 P_I
        inc = A_I_day[t]

        if inc > 0:
            total_P = PS_sh + PH_sh + PH_iso + PI
            stay_free = max(0.0, C_total - total_P)
            inc_eff = min(inc, stay_free)

            taken = min(PS_sh, inc_eff)

            PS_sh -= taken
            PI += taken

        for _ in range(contacts_per_day):

            B_tot = max(PS_sh + PH_sh + PH_iso + PI, 1e-9)

            # shared에서만 새 HAI 발생
            lam_HP_sh = beta * (HC / N_H)

            # 환자 → HCW
            lam_PH = beta * ((PH_sh + iso_factor * PH_iso + PI) / B_tot)

            # Env ↔ HCW
            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            # 새 HAI
            hai_sh = lam_HP_sh * PS_sh * dt

            # shared HAI → iso HAI
            move_HA = sigma * PH_sh * dt

            # 퇴원
            outS_sh = mu_S * PS_sh * dt
            outH_sh = mu_HAI * PH_sh * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI = mu_I * PI * dt

            leaving = outS_sh + outH_sh + outH_iso + outI
            total_P = PS_sh + PH_sh + PH_iso + PI

            AS_tot = max(0.0, C_total - (total_P - leaving))
            AS_sh = AS_tot

            # shared
            PS_sh += AS_sh - outS_sh - hai_sh
            PH_sh += hai_sh - outH_sh - move_HA

            # iso
            PH_iso += move_HA - outH_iso

            # P_I
            PI += -outI

            # clip
            PS_sh = np.clip(PS_sh, 0, C_sh)
            PH_sh = np.clip(PH_sh, 0, C_sh)
            PH_iso = np.clip(PH_iso, 0, C_iso)
            PI = np.clip(PI, 0, C_total)

            # HCW
            new_H = (lam_PH + lam_EH) * (N_H - HC) * dt
            HC = (HC + new_H) * (1 - p_wash)
            HC = np.clip(HC, 0, N_H)

            # Env
            EC += lam_HE * (N_E - EC) * dt
            EC = np.clip(EC, 0, N_E)

            # incidence
            NewHAI_day[t] += hai_sh

        if t < T - 1:
            P_S_sh[t+1] = PS_sh
            P_HAI_sh[t+1] = PH_sh
            P_HAI_iso[t+1] = PH_iso
            P_I[t+1] = PI
            H_C[t+1] = HC
            Env_C[t+1] = EC

    # ---- 월별 합계 ----
    df = pd.DataFrame({
        "date": days,
        "NewHAI": NewHAI_day
    })

    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["NewHAI"]
          .sum()
          .reset_index()
          .rename(columns={
              "date": "month",
              "NewHAI": "NewHAI_month"
          })
    )

    monthly["cum_NewHAI"] = monthly["NewHAI_month"].cumsum()

    # ---- 컴파트먼트 DF ----
    H_S = N_H - H_C
    Env_S = N_E - Env_C

    comp_df = pd.DataFrame({
        "date": days,
        "P_S_sh": P_S_sh,
        "P_HAI_sh": P_HAI_sh,
        "P_HAI_iso": P_HAI_iso,
        "P_I": P_I,
        "H_S": H_S,
        "H_C": H_C,
        "Env_S": Env_S,
        "Env_C": Env_C,
    }).set_index("date")

    return days, NewHAI_day, monthly, comp_df























# %% ================== Step4: B기간 전체 36개월 누적 Gaussian MLE + 95% CI ==================

# --- 설정 ---
abm_csv = "../result/interv_prob_transmission_summary_B260_0.01-0.07.csv"

init_env = 2
tau0 = 60

theta_min, theta_max = 0.5, 6.0

beta_list = [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07
]
    
# --- ABM CSV 읽고 mean/std 벡터 만들기 ---
df_abm = pd.read_csv(abm_csv)

df_abm["mean_vec"] = df_abm["mean"].apply(
    lambda s: np.array(ast.literal_eval(s), dtype=float)
)

df_abm["std_vec"] = df_abm["std"].apply(
    lambda s: np.array(ast.literal_eval(s), dtype=float)
)

# --- 월 index: B기간 전체 36개월 사용 ---
n_months = len(df_abm["mean_vec"].iloc[0])
start_month = "2021-01"

months = pd.period_range(
    start_month,
    periods=n_months,
    freq="M"
).to_timestamp()

print("Used months:", len(months), months[0], "to", months[-1])


def model_monthly_and_cum(theta, init_env=init_env, tau0=tau0):
    """
    theta를 넣고 simulate_theta 실행.
    B기간 전체 36개월의 월별 예측값과 누적 예측값 반환.
    """

    days, daily_inc, monthly_df, comp_df = simulate_theta(
        theta,
        init_env,
        tau0
    )

    mdf = monthly_df.copy()
    mdf["month"] = pd.to_datetime(mdf["month"].astype(str))
    mdf = mdf.set_index("month")

    monthly = np.array([
        mdf["NewHAI_month"].get(m, 0.0)
        for m in months
    ])

    cum = np.cumsum(monthly)

    return monthly, cum


def negloglik_theta_cum_gaussian(theta, cum_obs, init_env, tau0):
    """
    누적 Gaussian negative log-likelihood.

    cum_obs_t = cum_model_t(theta) + error_t
    error_t ~ N(0, sigma^2)

    sigma^2는 theta마다 residual variance MLE로 추정.
    """

    _, cum_model = model_monthly_and_cum(
        theta,
        init_env=init_env,
        tau0=tau0
    )

    y = np.asarray(cum_obs, dtype=float)
    mu = np.asarray(cum_model, dtype=float)

    m = min(len(y), len(mu))
    y = y[:m]
    mu = mu[:m]

    resid = y - mu
    n = len(resid)

    sigma2_hat = np.mean(resid ** 2)
    sigma2_hat = max(sigma2_hat, 1e-12)

    nll = 0.5 * n * (np.log(2 * np.pi * sigma2_hat) + 1)

    return float(nll)


def ci95_profile_theta_gaussian(
    cum_obs,
    init_env,
    tau0,
    theta_hat,
    nll_hat,
    bounds=(0.5, 6.0),
    grid_n=200
):
    """
    Gaussian profile likelihood 95% CI.

    -2 log Lambda(theta) ~ chi-square_1(0.95)=3.84

    NLL(theta) = NLL(theta_hat) + 3.84/2
               = NLL(theta_hat) + 1.92
    """

    thr = nll_hat + 1.92

    a, b = bounds
    grid = np.linspace(a, b, grid_n)

    vals = np.array([
        negloglik_theta_cum_gaussian(
            th,
            cum_obs,
            init_env,
            tau0
        )
        for th in grid
    ])

    g = vals - thr

    i_hat = np.searchsorted(grid, theta_hat)

    left = np.nan
    right = np.nan

    # 왼쪽 경계
    for i in range(i_hat, 0, -1):
        if g[i-1] > 0 and g[i] <= 0:
            left = brentq(
                lambda x: negloglik_theta_cum_gaussian(
                    x,
                    cum_obs,
                    init_env,
                    tau0
                ) - thr,
                grid[i-1],
                grid[i]
            )
            break

    # 오른쪽 경계
    for i in range(i_hat, len(grid)-1):
        if g[i] <= 0 and g[i+1] > 0:
            right = brentq(
                lambda x: negloglik_theta_cum_gaussian(
                    x,
                    cum_obs,
                    init_env,
                    tau0
                ) - thr,
                grid[i],
                grid[i+1]
            )
            break

    return float(left), float(right)


def fit_theta_cum_gaussian_for_one(beta_abm, y_mean):
    """
    beta_ABM 하나에 대해:
    B기간 전체 36개월 mean trajectory를 누적으로 바꾸고,
    theta_hat, 95% CI, NLL_min, sigma_hat 반환.
    """

    cum_obs = np.cumsum(y_mean)

    theta_grid = np.linspace(theta_min, theta_max, 200)

    vals = np.array([
        negloglik_theta_cum_gaussian(
            th,
            cum_obs,
            init_env,
            tau0
        )
        for th in theta_grid
    ])

    idx = vals.argmin()

    theta_hat = float(theta_grid[idx])
    nll_min = float(vals[idx])

    # sigma_hat 계산
    _, cum_model_hat = model_monthly_and_cum(
        theta_hat,
        init_env=init_env,
        tau0=tau0
    )

    m = min(len(cum_obs), len(cum_model_hat))
    resid = cum_obs[:m] - cum_model_hat[:m]

    sigma_hat = float(np.sqrt(np.mean(resid ** 2)))

    # 95% CI
    theta_low, theta_high = ci95_profile_theta_gaussian(
        cum_obs,
        init_env,
        tau0,
        theta_hat,
        nll_min,
        bounds=(theta_min, theta_max),
        grid_n=200
    )

    return theta_hat, theta_low, theta_high, nll_min, sigma_hat


# %% ================== 루프 실행: 각 beta_ABM별 theta_hat, CI 출력/저장 ==================

results = []

for b in beta_list:

    sub = df_abm.loc[np.isclose(df_abm["beta"], b)]

    if sub.empty:
        print(f"[경고] beta={b} 인 행이 CSV에 없음, 스킵")
        continue

    row = sub.iloc[0]

    beta_abm = float(row["beta"])
    y_mean = row["mean_vec"]

    theta_hat, theta_low, theta_high, nll_min, sigma_hat = fit_theta_cum_gaussian_for_one(
        beta_abm,
        y_mean
    )

    print(f"=== B CUM-Gaussian MLE for beta_ABM = {beta_abm:0.4f} ===")
    print(
        f"  theta_hat = {theta_hat:.4f}, "
        f"95% CI = [{theta_low:.4f}, {theta_high:.4f}], "
        f"sigma_hat = {sigma_hat:.4f}, "
        f"NLL_cum_min = {nll_min:.2f}"
    )

    results.append({
        "beta_abm": beta_abm,
        "theta_hat": theta_hat,
        "theta_low": theta_low,
        "theta_high": theta_high,
        "sigma_hat": sigma_hat,
        "neg_loglik_cum_min": nll_min,
        "init_env": init_env,
        "tau0": tau0,
        "used_months": len(months)
    })


# %% ================== 저장 ==================

df_res = pd.DataFrame(results)

os.makedirs("sm_fit", exist_ok=True)

out_csv = "sm_fit/theta_pairs_subset_cumGaussian_B260.csv"

df_res.to_csv(
    out_csv,
    index=False,
    encoding="utf-8"
)

print("\n저장 완료 →", out_csv)
print(df_res.head())
# %%

# %% B version: ABM cumulative vs SM cumulative comparison
# 비교 와 매핑 그리기




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# B setting
# =========================

theta_hat = 3.28985507
  # 여기에 B direct fitting 또는 mapping에서 얻은 theta 넣기

init_env = 9
tau0 = 140

# B버전 simulate_theta가 이미 정의되어 있어야 함
days, daily_inc_hat, monthly_hat_df, comp_hat_df = simulate_theta(
    theta_hat,
    init_env,
    tau0
)

# =========================
# 1) ABM 쪽: beta_ABM 행 뽑기
# =========================

beta_test = 0.04

sub = df_abm.loc[np.isclose(df_abm["beta"], beta_test)]

if sub.empty:
    raise ValueError(f"beta={beta_test} 인 행이 df_abm에 없습니다.")

row = sub.iloc[0]

# B는 길이 36이어야 함
y_mean = row["mean_vec"]

# =========================
# 2) SM 쪽: theta_hat으로 월별 HAI 벡터 만들기
# =========================

model_month = monthly_hat_df.copy()
model_month["month"] = pd.to_datetime(model_month["month"].astype(str))
model_month = model_month.set_index("month")

start_month = "2021-01"

months = pd.period_range(
    start_month,
    periods=len(y_mean),
    freq="M"
).to_timestamp()

y_model = np.array([
    model_month["NewHAI_month"].get(m, 0.0)
    for m in months
])

# =========================
# 3) 월별 → 누적합
# =========================

cum_abm = np.cumsum(y_mean)
cum_model = np.cumsum(y_model)

# =========================
# 4) 누적 비교 plot
# =========================

plt.figure(figsize=(11, 6))

plt.plot(
    months,
    cum_abm,
    "o-",
    color="blue",
    label=f"ABM cumulative (beta_ABM={beta_test})"
)

plt.plot(
    months,
    cum_model,
    "s-",
    color="orange",
    label=f"SM cumulative (theta_SM={theta_hat:.4f})"
)

plt.ylabel("Cumulative HAI", fontsize=18)

ax = plt.gca()

# B는 36개월이라 6개월/3개월 간격 그대로 괜찮음
ax.set_xticks(months[::6])
ax.set_xticks(months[::3], minor=True)

plt.yticks(fontsize=18)
plt.xticks(fontsize=18)

ax.grid(True, which="major", axis="x", linestyle="-")
ax.grid(True, which="minor", axis="x", linestyle="--")
ax.grid(True, axis="y", linestyle="-")

ax.set_ylim(bottom=0)

plt.title("B period: ABM cumulative vs SM cumulative", fontsize=18)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

# %% Step4: B version β_ABM → θ_SM with 95% CI plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 설정
# =========================

csv_path = Path("sm_fit/theta_pairs_subset_cumGaussian_B9140.csv")
out_png  = Path("sm_fit/step4_betaabm_to_thetasm_plotB9140_gaussian.png")

# 만약 Poisson 결과를 쓰고 싶으면 위 두 줄을 이렇게 바꿔:
# csv_path = Path("sm_fit/theta_pairs_subset_cumPoisson_B260.csv")
# out_png  = Path("sm_fit/step4_betaabm_to_thetasm_plotB260_poisson.png")

# =========================
# 데이터 읽기
# =========================

df = pd.read_csv(csv_path)

need = {"beta_abm", "theta_hat", "theta_low", "theta_high"}

if not need.issubset(df.columns):
    raise ValueError(
        f"CSV에 다음 컬럼이 필요합니다: {sorted(need)} "
        f"(현재: {list(df.columns)})"
    )

# 정렬
df = df.sort_values("beta_abm").reset_index(drop=True)

x  = df["beta_abm"].to_numpy(float)
y  = df["theta_hat"].to_numpy(float)
yl = df["theta_low"].to_numpy(float)
yh = df["theta_high"].to_numpy(float)

# 에러바 길이
yerr = np.vstack([
    y - yl,
    yh - y
])

# =========================
# Plot
# =========================

plt.figure(figsize=(8.2, 4.8))

plt.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="o",
    capsize=3,
    lw=1.2,
    label=r"$\theta_{\mathrm{SM}}$ MLE with 95% CI"
)

plt.plot(
    x,
    y,
    "-",
    lw=1.2,
    label=r"$\theta_{\mathrm{SM}}$ trend"
)

plt.fill_between(
    x,
    yl,
    yh,
    alpha=0.15,
    label="95% CI band"
)

plt.xlabel(r"$\beta_{\mathrm{ABM}}$", fontsize=14)
plt.ylabel(r"$\theta_{\mathrm{SM}}$ fitted", fontsize=14)
plt.title("B period Step 4: Mapping ABM β → SM θ", fontsize=15)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(out_png, dpi=200)
plt.show()

print(f"saved plot -> {out_png}")
print(df[["beta_abm", "theta_hat", "theta_low", "theta_high"]].to_string(index=False))








# %%
