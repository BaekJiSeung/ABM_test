# %% ================== 준비 ==================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
from scipy.optimize import brentq

# ---- 기간 정의용 ----
monthly_PI = pd.Series({
    "2017-01":0,"2017-02":0,"2017-03":1,"2017-04":0,"2017-05":1,"2017-06":0,
    "2017-07":0,"2017-08":0,"2017-09":2,"2017-10":0,"2017-11":2,"2017-12":0,
    "2018-01":0,"2018-02":0,"2018-03":0,"2018-04":0,"2018-05":1,"2018-06":0,"2018-07":0
})

PI_dates = [
    "2017-03-01","2017-05-30","2017-09-17","2017-09-29",
    "2017-11-14","2017-11-20","2018-05-25",
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


# %% ================== SM simulation: p_wash fitting version ==================

def simulate_pwash(p_wash, init_env, tau0,
                   beta_SM=3.374,
                   monthly_PI=monthly_PI,
                   pi_dates=PI_dates):
    """
    SM beta는 beta_SM=3.374로 고정.
    p_wash만 바꿔가며 surrogate output을 생성한다.

    상태:
    P_S_sh, P_HAI_sh, P_HAI_iso, P_I, H_C, Env_C
    """

    # ---- 파라미터 ----
    C_total = 30
    C_iso = 30
    C_sh = C_total

    N_H, N_E = 19, 30

    mu_S = 1 / 7
    mu_HAI = 1 / 14
    mu_I = 1 / 7

    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    iso_factor = 0.75

    isol_time = 14.0
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

        PS_sh = P_S_sh[t]
        PH_sh = P_HAI_sh[t]
        PH_iso = P_HAI_iso[t]
        PI = P_I[t]
        HC = H_C[t]
        EC = Env_C[t]

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

            # beta는 고정: beta_SM = 3.374
            lam_HP_sh = beta_SM * (HC / N_H)

            lam_PH = beta_SM * (
                (PH_sh + iso_factor * PH_iso + PI) / B_tot
            )

            lam_EH = beta_SM * (EC / N_E)
            lam_HE = beta_SM * (HC / N_H)

            # 새 HAI
            hai_sh = lam_HP_sh * PS_sh * dt
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
        .rename(columns={"date": "month", "NewHAI": "NewHAI_month"})
    )

    monthly["cum_NewHAI"] = monthly["NewHAI_month"].cumsum()

    # ---- compartment DF ----
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


# %% ================== Step4: p_wash 누적 Poisson MLE + 95% CI ==================

# --- 설정 ---
# ABM의 p_wash intervention output summary 파일
abm_csv = "../result/interv_hcw_wash_rate_summary_A9140_0.7-0.99.csv"

init_env = 9
tau0 = 140

# SM beta 고정
beta_SM = 3.374

# p_wash 탐색 범위: 0~1
p_min = 0.8
p_max = 0.9
grid_n = 101

# ABM에서 실험한 p값들
p_list = [0.7,0.8]

# --- ABM CSV 읽기 ---
df_abm = pd.read_csv(abm_csv)

df_abm["mean_vec"] = df_abm["mean"].apply(
    lambda s: np.array(ast.literal_eval(s), dtype=float)
)

df_abm["std_vec"] = df_abm["std"].apply(
    lambda s: np.array(ast.literal_eval(s), dtype=float)
)

# p값 컬럼 찾기
if "hcw_wash_rate" in df_abm.columns:
    p_col = "hcw_wash_rate"
elif "p" in df_abm.columns:
    p_col = "p"
elif "beta" in df_abm.columns:
    # 예전 저장 방식에서 p 값이 beta 컬럼에 들어간 경우
    p_col = "beta"
else:
    raise ValueError("p값 컬럼을 찾지 못했습니다. hcw_wash_rate, p, beta 중 하나가 필요합니다.")

print("p column =", p_col)

# 월 index
n_months = len(df_abm["mean_vec"].iloc[0])
start_month = "2017-01"
months = pd.period_range(start_month, periods=n_months, freq="M").to_timestamp()


def model_monthly_and_cum_pwash(p_wash, init_env=init_env, tau0=tau0, beta_SM=beta_SM):
    days, daily_inc, monthly_df, comp_df = simulate_pwash(
        p_wash=p_wash,
        init_env=init_env,
        tau0=tau0,
        beta_SM=beta_SM
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


def negloglik_pwash_cum_poisson(p_wash, cum_obs, init_env, tau0, beta_SM):
    _, cum_model = model_monthly_and_cum_pwash(
        p_wash=p_wash,
        init_env=init_env,
        tau0=tau0,
        beta_SM=beta_SM
    )

    lam = np.clip(cum_model, 1e-12, None)
    y = np.asarray(cum_obs, dtype=float)

    # -log L(p) = sum(lambda - y log lambda) + const
    nll = np.sum(lam - y * np.log(lam))

    return nll


def ci95_profile_pwash(cum_obs, init_env, tau0, p_hat, nll_hat,
                       beta_SM=beta_SM,
                       bounds=(p_min, p_max),
                       grid_n=grid_n):

    # -2 log Lambda ~ chi-square_1
    # NLL threshold = NLL_min + 1.92
    thr = nll_hat + 1.92

    a, b = bounds
    grid = np.linspace(a, b, grid_n)

    vals = np.array([
        negloglik_pwash_cum_poisson(
            p,
            cum_obs,
            init_env,
            tau0,
            beta_SM
        )
        for p in grid
    ])

    g = vals - thr

    i_hat = np.searchsorted(grid, p_hat)

    # left boundary
    left = a
    for i in range(i_hat, 0, -1):
        if g[i-1] > 0 and g[i] <= 0:
            left = brentq(
                lambda x: negloglik_pwash_cum_poisson(
                    x, cum_obs, init_env, tau0, beta_SM
                ) - thr,
                grid[i-1],
                grid[i]
            )
            break

    # right boundary
    right = b
    for i in range(i_hat, len(grid)-1):
        if g[i] <= 0 and g[i+1] > 0:
            right = brentq(
                lambda x: negloglik_pwash_cum_poisson(
                    x, cum_obs, init_env, tau0, beta_SM
                ) - thr,
                grid[i],
                grid[i+1]
            )
            break

    return float(left), float(right)


def fit_pwash_cum_poisson_for_one(p_abm, y_mean):
    """
    ABM p_wash 결과 y_mean에 대해
    SM p_wash_hat을 누적 Poisson MLE로 추정.
    """
    cum_obs = np.cumsum(y_mean)

    p_grid = np.linspace(p_min, p_max, grid_n)

    vals = np.array([
        negloglik_pwash_cum_poisson(
            p,
            cum_obs,
            init_env,
            tau0,
            beta_SM
        )
        for p in p_grid
    ])

    idx = vals.argmin()

    p_hat = float(p_grid[idx])
    nll_min = float(vals[idx])

    p_low, p_high = ci95_profile_pwash(
        cum_obs=cum_obs,
        init_env=init_env,
        tau0=tau0,
        p_hat=p_hat,
        nll_hat=nll_min,
        beta_SM=beta_SM,
        bounds=(p_min, p_max),
        grid_n=grid_n
    )

    return p_hat, p_low, p_high, nll_min


# %% ================== 실행 ==================

results = []

for p in p_list:

    sub = df_abm.loc[np.isclose(df_abm[p_col].astype(float), p)]

    if sub.empty:
        print(f"[경고] p={p} 인 행이 CSV에 없음, 스킵")
        continue

    row = sub.iloc[0]

    p_abm = float(row[p_col])
    y_mean = row["mean_vec"]

    p_hat, p_low, p_high, nll_min = fit_pwash_cum_poisson_for_one(
        p_abm=p_abm,
        y_mean=y_mean
    )

    print(f"=== CUM-Poisson MLE for p_ABM = {p_abm:0.3f} ===")
    print(
        f"  p_hat = {p_hat:.4f}, "
        f"95% CI = [{p_low:.4f}, {p_high:.4f}], "
        f"NLL_cum_min = {nll_min:.2f}"
    )

    results.append({
        "p_abm": p_abm,
        "p_hat": p_hat,
        "p_low": p_low,
        "p_high": p_high,
        "neg_loglik_cum_min": nll_min,
        "beta_SM_fixed": beta_SM,
        "init_env": init_env,
        "tau0": tau0,
    })


# %% ================== 저장 ==================

df_res = pd.DataFrame(results)

os.makedirs("sm_fit", exist_ok=True)

out_csv = "sm_fit/pwash_pairs_cumPoisson_A9140_betaSM4p89.csv"

df_res.to_csv(out_csv, index=False, encoding="utf-8")

print("\n저장 완료 →", out_csv)
print(df_res)


# %% ================== 결과 플롯 ==================

if not df_res.empty:

    plt.figure(figsize=(7, 6))

    yerr = np.vstack([
        df_res["p_hat"] - df_res["p_low"],
        df_res["p_high"] - df_res["p_hat"]
    ])

    plt.errorbar(
        df_res["p_abm"],
        df_res["p_hat"],
        yerr=yerr,
        fmt="o",
        capsize=4,
        label="SM fitted p_hat with 95% CI"
    )

    # y=x 기준선
    plt.plot([0, 1], [0, 1], "--", label="y = x")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel("ABM handwash probability p")
    plt.ylabel("SM fitted handwash probability p_hat")
    plt.title("Mapping ABM p_wash to SM p_wash\n(beta_SM fixed at 3.374)")
    plt.legend()
    plt.grid(True)

    fig_path = "sm_fit/pwash_mapping_A9140_betaSM3p374.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.show()

    print("그림 저장 완료 →", fig_path)
# %%
# ================================
# fitted p 값 (SM에서 얻은 결과)
# ================================

p_hat = 0.917   # 예: fitting 결과

days, daily_inc_hat, monthly_hat_df, comp_hat_df = simulate_pwash(
    p_hat,
    init_env,
    tau0
)

# ================================
# 1) ABM 쪽: p 값 행 뽑기
# ================================

p_test = 0.95

# p 컬럼 이름 확인 (중요)
if "hcw_wash_rate" in df_abm.columns:
    p_col = "hcw_wash_rate"
elif "p" in df_abm.columns:
    p_col = "p"
elif "beta" in df_abm.columns:
    p_col = "beta"
else:
    raise ValueError("p column not found")

sub = df_abm.loc[np.isclose(df_abm[p_col], p_test)]

row = sub.iloc[0]

y_mean = row["mean_vec"]   # 월별 incidence (길이 19)

# ================================
# 2) 모델 쪽: p_hat으로 월별 HAI 벡터 만들기
# ================================

model_month = monthly_hat_df.copy()

model_month["month"] = pd.to_datetime(
    model_month["month"].astype(str)
)

model_month = model_month.set_index("month")

start_month = "2017-01"

months = pd.period_range(
    start_month,
    periods=len(y_mean),
    freq="M"
).to_timestamp()

y_model = np.array([
    model_month["NewHAI_month"].get(m, 0.0)
    for m in months
])

# ================================
# 3) 월별 → 누적합
# ================================

cum_abm = np.cumsum(y_mean)

cum_model = np.cumsum(y_model)

# ================================
# 4) 누적 비교 플롯
# ================================

plt.figure(figsize=(9,6))

plt.plot(
    months,
    cum_abm,
    "o-",
    color="blue",
    label=f"ABM cumulative (p_ABM={p_test})"
)

plt.plot(
    months,
    cum_model,
    "s-",
    color="orange",
    label=f"SM cumulative (p_SM={p_hat:.4f})"
)

plt.ylabel("Cumulative HAI", fontsize=18)

ax = plt.gca()

# x-axis ticks
ax.set_xticks(months[::6])
ax.set_xticks(months[::3], minor=True)

plt.yticks(fontsize=18)
plt.xticks(fontsize=18)

ax.grid(True, which='major', axis='x', linestyle='-')
ax.grid(True, which='minor', axis='x', linestyle='--')
ax.grid(True, axis='y', linestyle='-')

ax.set_ylim(bottom=0)

plt.legend(fontsize=20)

plt.tight_layout()

plt.show()
# %%
