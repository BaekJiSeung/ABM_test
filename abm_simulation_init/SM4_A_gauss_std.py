
# ABM summary의 std를 사용하는 weighted cumulative Gaussian version
# 저장명은 기존과 동일하게 유지


# %% ================== 준비(그대로 사용) ==================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast, os
from scipy.optimize import brentq

# ---- 기간 정의용 (그냥 날짜 범위용) ----
monthly_PI = pd.Series({
    "2017-01":0,"2017-02":0,"2017-03":1,"2017-04":0,"2017-05":1,"2017-06":0,
    "2017-07":0,"2017-08":0,"2017-09":2,"2017-10":0,"2017-11":2,"2017-12":0,
    "2018-01":0,"2018-02":0,"2018-03":0,"2018-04":0,"2018-05":1,"2018-06":0,"2018-07":0
})

# ---- 실제 P_I 입원 날짜들 ----
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


def simulate_theta(beta, init_env, tau0,
                   monthly_PI=monthly_PI, pi_dates=PI_dates):
    """
    격리=HAI만 모아두는 칸(P_HAI_iso만).
    상태: P_S_sh, P_HAI_sh, P_HAI_iso, P_I, H_C, Env_C
    """

    # ---- 파라미터 ----
    C_total = 30
    C_iso = 30
    C_sh = C_total

    N_H, N_E = 19, 30

    mu_S = 1/7
    mu_HAI = 1/14
    mu_I = 1/7

    p_wash = 0.90
    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    iso_factor = 0.75

    isol_time = 7.0
    sigma = 1.0 / isol_time   # shared HAI → iso HAI

    # ---- 시간축 ----
    start = pd.Period(monthly_PI.index.min(), freq="M").to_timestamp(how="start")
    end   = pd.Period(monthly_PI.index.max(), freq="M").to_timestamp(how="end")

    days = pd.date_range(start, end, freq="D")
    T = len(days)

    # ---- P_I 입원 패턴 ----
    A_I_day = _make_AI_from_dates(pi_dates, days)

    # ---- 상태 ----
    P_S_sh     = np.zeros(T)
    P_HAI_sh   = np.zeros(T)
    P_HAI_iso  = np.zeros(T)
    P_I        = np.zeros(T)
    H_C        = np.zeros(T)
    Env_C      = np.zeros(T)
    NewHAI_day = np.zeros(T)

    # 초기조건
    P_S_sh[0] = C_total - 1
    P_I[0]    = 1
    Env_C[0]  = init_env

    for t in range(T):

        # tau0 반영 대청소
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            xx = Env_C[t]
            Env_C[t] = (1 - p_wash) * xx

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
            outS_sh  = mu_S   * PS_sh  * dt
            outH_sh  = mu_HAI * PH_sh  * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI     = mu_I   * PI     * dt

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
            PS_sh  = np.clip(PS_sh,  0, C_sh)
            PH_sh  = np.clip(PH_sh,  0, C_sh)
            PH_iso = np.clip(PH_iso, 0, C_iso)
            PI     = np.clip(PI,     0, C_total)

            # HCW
            new_H = (lam_PH + lam_EH) * (N_H - HC) * dt
            HC = (HC + new_H) * (1 - p_wash)
            HC = np.clip(HC, 0, N_H)

            # Env
            EC += lam_HE * (N_E - EC) * dt
            EC = np.clip(EC, 0, N_E)

            # 인시던스
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


# %% ================== Step4: 누적 Gaussian MLE + 95% CI ==================

# --- 설정 ---
abm_csv = "../result/interv_prob_transmission_summary_A9140_0.01-0.07.csv"

init_env = 9
tau0 = 140

theta_min, theta_max = 1, 6

beta_list = [
    0.01, 0.015, 0.02, 0.025, 0.03, 0.035,
    0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07
]

# --- ABM CSV 읽고 mean/std 벡터 만들기 ---
df_abm = pd.read_csv(abm_csv)

df_abm["mean_vec"] = df_abm["mean"].apply(
    lambda s: np.array(ast.literal_eval(s), dtype=float)
)

df_abm["std_vec"] = df_abm["std"].apply(
    lambda s: np.array(ast.literal_eval(s), dtype=float)
)

# 월 index (2017-01부터 n개월)
n_months = len(df_abm["mean_vec"].iloc[0])
start_month = "2017-01"

months = pd.period_range(
    start_month,
    periods=n_months,
    freq="M"
).to_timestamp()


def model_monthly_and_cum(theta, init_env=init_env, tau0=tau0):
    """
    theta를 넣고 simulate_theta 실행.
    월별 예측값과 누적 예측값 반환.
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


def make_cum_std_from_monthly_std(y_std):
    """
    월별 std를 누적 std로 근사.

    독립 가정:
    Var(C_1 + ... + C_t) = Var(C_1) + ... + Var(C_t)

    따라서:
    cum_std_t = sqrt(cumsum(monthly_std_t^2))
    """

    y_std = np.asarray(y_std, dtype=float)
    cum_std = np.sqrt(np.cumsum(y_std ** 2))

    # std가 0이면 likelihood가 터지므로 작은 값으로 보정
    cum_std = np.maximum(cum_std, 1e-6)

    return cum_std


def negloglik_theta_cum_gaussian(theta, cum_obs, cum_std, init_env, tau0):
    """
    누적 weighted Gaussian negative log-likelihood.

    cum_obs_t = cum_model_t(theta) + error_t
    error_t ~ N(0, cum_std_t^2)

    여기서 cum_std는 ABM 반복 simulation의 월별 std를 누적 std로 근사한 값.
    """

    _, cum_model = model_monthly_and_cum(
        theta,
        init_env=init_env,
        tau0=tau0
    )

    y = np.asarray(cum_obs, dtype=float)
    mu = np.asarray(cum_model, dtype=float)
    sd = np.asarray(cum_std, dtype=float)

    m = min(len(y), len(mu), len(sd))
    y = y[:m]
    mu = mu[:m]
    sd = sd[:m]

    sd = np.maximum(sd, 1e-6)

    resid = y - mu

    nll = 0.5 * np.sum(
        np.log(2 * np.pi * sd**2) + (resid**2) / (sd**2)
    )

    return float(nll)


def ci95_profile_theta_gaussian(cum_obs, cum_std, init_env, tau0,
                                theta_hat, nll_hat,
                                bounds=(0.5, 6.0),
                                grid_n=200):
    """
    Gaussian profile likelihood 95% CI.

    -2 log Lambda(theta) ~ chi-square_1(0.95)=3.84

    NLL(theta) = NLL(theta_hat) + 3.84/2
               = NLL(theta_hat) + 1.92

    여기서는 ABM std 기반 weighted Gaussian NLL 사용.
    """

    thr = nll_hat + 1.92

    a, b = bounds
    grid = np.linspace(a, b, grid_n)

    vals = np.array([
        negloglik_theta_cum_gaussian(th, cum_obs, cum_std, init_env, tau0)
        for th in grid
    ])

    g = vals - thr

    i_hat = np.searchsorted(grid, theta_hat)

    # 왼쪽 경계
    left = a

    for i in range(i_hat, 0, -1):
        if g[i-1] > 0 and g[i] <= 0:
            left = brentq(
                lambda x: negloglik_theta_cum_gaussian(
                    x,
                    cum_obs,
                    cum_std,
                    init_env,
                    tau0
                ) - thr,
                grid[i-1],
                grid[i]
            )
            break

    # 오른쪽 경계
    right = b

    for i in range(i_hat, len(grid)-1):
        if g[i] <= 0 and g[i+1] > 0:
            right = brentq(
                lambda x: negloglik_theta_cum_gaussian(
                    x,
                    cum_obs,
                    cum_std,
                    init_env,
                    tau0
                ) - thr,
                grid[i],
                grid[i+1]
            )
            break

    return float(left), float(right)


def fit_theta_cum_gaussian_for_one(beta_abm, y_mean, y_std):
    """
    beta_ABM 하나에 대해:
    y_mean trajectory를 누적으로 바꾸고,
    y_std도 누적 std로 근사한 뒤,
    weighted Gaussian likelihood로
    theta_hat, 95% CI, NLL_min, sigma_hat 반환.
    """

    # 월별 평균 -> 누적 평균
    cum_obs = np.cumsum(y_mean)

    # 월별 std -> 누적 std 근사
    cum_std = make_cum_std_from_monthly_std(y_std)

    theta_grid = np.linspace(theta_min, theta_max, 300)

    vals = np.array([
        negloglik_theta_cum_gaussian(th, cum_obs, cum_std, init_env, tau0)
        for th in theta_grid
    ])

    idx = vals.argmin()

    theta_hat = float(theta_grid[idx])
    nll_min = float(vals[idx])

    # sigma_hat은 기존처럼 참고용 RMSE로 계산
    _, cum_model_hat = model_monthly_and_cum(
        theta_hat,
        init_env=init_env,
        tau0=tau0
    )

    m = min(len(cum_obs), len(cum_model_hat))
    resid = cum_obs[:m] - cum_model_hat[:m]
    sigma_hat = float(np.sqrt(np.mean(resid ** 2)))

    # 참고용 weighted RMSE도 계산 가능
    weighted_rmse = float(np.sqrt(np.mean((resid / cum_std[:m]) ** 2)))

    # 95% CI
    theta_low, theta_high = ci95_profile_theta_gaussian(
        cum_obs,
        cum_std,
        init_env,
        tau0,
        theta_hat,
        nll_min,
        bounds=(theta_min, theta_max),
        grid_n=200
    )

    return theta_hat, theta_low, theta_high, nll_min, sigma_hat, weighted_rmse


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
    y_std = row["std_vec"]

    theta_hat, theta_low, theta_high, nll_min, sigma_hat, weighted_rmse = fit_theta_cum_gaussian_for_one(
        beta_abm,
        y_mean,
        y_std
    )

    print(f"=== CUM-Gaussian MLE with ABM std for beta_ABM = {beta_abm:0.3f} ===")
    print(
        f"  theta_hat = {theta_hat:.4f}, "
        f"95% CI = [{theta_low:.4f}, {theta_high:.4f}], "
        f"sigma_hat(RMSE) = {sigma_hat:.4f}, "
        f"weighted_RMSE = {weighted_rmse:.4f}, "
        f"NLL_cum_min = {nll_min:.2f}"
    )

    results.append({
        "beta_abm": beta_abm,
        "theta_hat": theta_hat,
        "theta_low": theta_low,
        "theta_high": theta_high,
        "sigma_hat": sigma_hat,
        "weighted_rmse": weighted_rmse,
        "neg_loglik_cum_min": nll_min,
        "init_env": init_env,
        "tau0": tau0,
    })


# %% ================== 저장 ==================

df_res = pd.DataFrame(results)

os.makedirs("sm_fit", exist_ok=True)

# 저장명은 기존과 동일하게 유지
out_csv = "sm_fit/theta_pairs_subset_cumGaussian_A9140.csv"

df_res.to_csv(
    out_csv,
    index=False,
    encoding="utf-8"
)

print("\n저장 완료 →", out_csv)
print(df_res.head())


# %% 가우시안 1개월제거버전

# %% ================== 준비 ==================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast, os
from scipy.optimize import brentq

# ---- 기간 정의용 ----
monthly_PI = pd.Series({
    "2017-01":0,"2017-02":0,"2017-03":1,"2017-04":0,"2017-05":1,"2017-06":0,
    "2017-07":0,"2017-08":0,"2017-09":2,"2017-10":0,"2017-11":2,"2017-12":0,
    "2018-01":0,"2018-02":0,"2018-03":0,"2018-04":0,"2018-05":1,"2018-06":0,"2018-07":0
})

# ---- 실제 P_I 입원 날짜들 ----
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


def simulate_theta(beta, init_env, tau0,
                   monthly_PI=monthly_PI, pi_dates=PI_dates):
    """
    격리=HAI만 모아두는 칸(P_HAI_iso만).
    상태:
    P_S_sh, P_HAI_sh, P_HAI_iso, P_I, H_C, Env_C
    """

    # ---- 파라미터 ----
    C_total = 30
    C_iso = 30
    C_sh = C_total

    N_H, N_E = 19, 30

    mu_S = 1/7
    mu_HAI = 1/14
    mu_I = 1/7

    p_wash = 0.90

    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    iso_factor = 0.75
    isol_time = 7.0
    sigma = 1.0 / isol_time   # shared HAI → iso HAI

    # ---- 시간축 ----
    start = pd.Period(monthly_PI.index.min(), freq="M").to_timestamp(how="start")
    end   = pd.Period(monthly_PI.index.max(), freq="M").to_timestamp(how="end")

    days = pd.date_range(start, end, freq="D")
    T = len(days)

    # ---- P_I 입원 패턴 ----
    A_I_day = _make_AI_from_dates(pi_dates, days)

    # ---- 상태 ----
    P_S_sh     = np.zeros(T)
    P_HAI_sh   = np.zeros(T)
    P_HAI_iso  = np.zeros(T)
    P_I        = np.zeros(T)
    H_C        = np.zeros(T)
    Env_C      = np.zeros(T)
    NewHAI_day = np.zeros(T)

    # 초기조건
    P_S_sh[0] = C_total - 1
    P_I[0]    = 1
    Env_C[0]  = init_env

    for t in range(T):

        # tau0 반영 대청소
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            xx = Env_C[t]
            Env_C[t] = (1 - p_wash) * xx

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
            outS_sh  = mu_S   * PS_sh  * dt
            outH_sh  = mu_HAI * PH_sh  * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI     = mu_I   * PI     * dt

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
            PS_sh  = np.clip(PS_sh,  0, C_sh)
            PH_sh  = np.clip(PH_sh,  0, C_sh)
            PH_iso = np.clip(PH_iso, 0, C_iso)
            PI     = np.clip(PI,     0, C_total)

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


# %% ================== Step4: 앞 1개월 제거 + 누적 Gaussian MLE + 95% CI ==================

# ---- 설정 ----
abm_csv = "../result/interv_prob_transmission_summary_A9140_0.01-0.07.csv"

init_env = 9
tau0 = 140

theta_min, theta_max = 0.5, 6.0

beta_list = [
    0.01, 0.015, 0.02, 0.025, 0.03, 0.035,
    0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07
]

# 앞 1개월 제거 고정
trim_front = 1


# ---- ABM CSV 읽고 mean/std 벡터 만들기 ----
df_abm = pd.read_csv(abm_csv)

df_abm["mean_vec"] = df_abm["mean"].apply(
    lambda s: np.array(ast.literal_eval(s), dtype=float)
)

df_abm["std_vec"] = df_abm["std"].apply(
    lambda s: np.array(ast.literal_eval(s), dtype=float)
)


# ---- 월 index 만들기 ----
n_months = len(df_abm["mean_vec"].iloc[0])
start_month = "2017-01"

months_full = pd.period_range(
    start_month,
    periods=n_months,
    freq="M"
).to_timestamp()

# 모델 prediction도 첫 1개월 제거된 월만 사용
months = months_full[trim_front:]

print("Full months:", len(months_full), months_full[0], "to", months_full[-1])
print("Used months:", len(months), months[0], "to", months[-1])
print("trim_front =", trim_front)


def model_monthly_and_cum(theta, init_env=init_env, tau0=tau0):
    """
    theta를 넣고 simulate_theta 실행.
    첫 1개월 제거 후 월별 예측값과 누적 예측값 반환.
    """

    days, daily_inc, monthly_df, comp_df = simulate_theta(
        theta,
        init_env,
        tau0
    )

    mdf = monthly_df.copy()
    mdf["month"] = pd.to_datetime(mdf["month"].astype(str))
    mdf = mdf.set_index("month")

    # months가 이미 첫 1개월 제거된 index
    monthly = np.array([
        mdf["NewHAI_month"].get(m, 0.0)
        for m in months
    ])

    # 제거된 구간 이후부터 새로 cumulative 계산
    cum = np.cumsum(monthly)

    return monthly, cum


def negloglik_theta_cum_gaussian(theta, cum_obs, init_env, tau0):
    """
    앞 1개월 제거 후 누적 Gaussian negative log-likelihood.

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

    # 왼쪽 경계
    left = a

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
    right = b

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
    y_mean trajectory의 첫 1개월을 제거하고,
    그 이후부터 새로 cumulative를 만든 뒤,
    theta_hat, 95% CI, NLL_min, sigma_hat 반환.
    """

    # 관측 ABM mean trajectory도 첫 1개월 제거
    y_mean_used = y_mean[trim_front:]

    # 제거된 구간 이후부터 새로 cumulative 계산
    cum_obs = np.cumsum(y_mean_used)

    theta_grid = np.linspace(theta_min, theta_max, 300)

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


# %% ================== 루프 실행 ==================

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

    print(f"=== DROP-FIRST1 CUM-Gaussian MLE for beta_ABM = {beta_abm:0.3f} ===")
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
        "trim_front": trim_front,
        "used_months": len(months)
    })


# %% ================== 저장 ==================

df_res = pd.DataFrame(results)

os.makedirs("sm_fit", exist_ok=True)

out_csv = "sm_fit/theta_pairs_subset_cumGaussian_A9140_dropFirst1.csv"

df_res.to_csv(
    out_csv,
    index=False,
    encoding="utf-8"
)

print("\n저장 완료 →", out_csv)
print(df_res.head())




# %%




# %% 비교 (누적 버전)
theta_hat = 3.64548494983277
 # 피팅 결과

days, daily_inc_hat, monthly_hat_df, comp_hat_df = simulate_theta(
    theta_hat,
    init_env,
    tau0
)

# 1) ABM 쪽: beta=0.10 행 뽑기  (지금은 0.100으로 되어 있음)
beta_test = 0.05
sub = df_abm.loc[np.isclose(df_abm["beta"], beta_test)]
row = sub.iloc[0]
y_mean = row["mean_vec"]   # 길이 19, 월별 평균 HAI

# 2) 모델 쪽: theta_hat으로 월별 HAI 벡터 만들기
model_month = monthly_hat_df.copy()
model_month["month"] = pd.to_datetime(model_month["month"].astype(str))
model_month = model_month.set_index("month")

start_month = "2017-01"
months = pd.period_range(start_month, periods=len(y_mean), freq="M").to_timestamp()

y_model = np.array([model_month["NewHAI_month"].get(m, 0.0) for m in months])

# 3) 월별 → 누적합으로 변환
cum_abm   = np.cumsum(y_mean)
cum_model = np.cumsum(y_model)

# 4) 누적 비교 플롯
plt.figure(figsize=(9,6))

plt.plot(months, cum_abm, "o-",color ='blue', label=f"ABM cumulative (beta_ABM={beta_test})")
plt.plot(months, cum_model, "s-", color='orange', label=f"SM cumulative (beta_SM={theta_hat:.4f})")


plt.ylabel("Cumulative HAI", fontsize=18)

ax = plt.gca()

# 🔥 핵심
ax.set_xticks(months[::6])                 # label (6개월)
ax.set_xticks(months[::3], minor=True)     # grid (3개월)
[5,2,0,2,1,1,2,2,6,1,1,0,2,1,1,2,5,2,2]
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

# %% Step4: β_ABM → θ_SM(=beta_sm) with 95% CI plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 설정 =====
csv_path = Path("sm_fit/theta_pairs_subset_cumGaussian_A9140_14.csv")
out_png  = Path("sm_fit/step4_betaabm_to_thetasm_plotA9140_14.png")

# ===== 데이터 읽기 =====
df = pd.read_csv(csv_path)

# 컬럼 이름 맞춤(네 CSV 스샷 기준)
# beta_abm, theta_hat, theta_low, theta_high, neg_loglik_cum_min, init_env, tau0
need = {"beta_abm","theta_hat","theta_low","theta_high"}
if not need.issubset(df.columns):
    raise ValueError(f"CSV에 다음 컬럼이 필요합니다: {sorted(need)}  (현재: {list(df.columns)})")

# 정렬
df = df.sort_values("beta_abm").reset_index(drop=True)

x  = df["beta_abm"].to_numpy(float)
y  = df["theta_hat"].to_numpy(float)
yl = df["theta_low"].to_numpy(float)
yh = df["theta_high"].to_numpy(float)

# 에러바 길이 (위/아래)
yerr = np.vstack([y - yl, yh - y])

# ===== 플롯 =====
plt.figure(figsize=(7.5, 4.5))

# 1) 점 + 에러바
plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, lw=1.2, label="θ_SM (MLE) with 95% CI")

# 2) 선형 연결(가독성용)
plt.plot(x, y, "-", lw=1.2, label="θ_SM (interp)")

# 3) CI 밴드 (선형으로 간단히)
#   x가 균일 간격이라면 아래로 충분. (비균일이어도 문제 없음)
plt.fill_between(x, yl, yh, alpha=0.15, label="95% CI band")

plt.xlabel(r"$\beta_{\mathrm{ABM}}$")
plt.ylabel(r"$\theta_{\mathrm{SM}}$ (fitted)")
plt.title("Step 4: Mapping ABM β → SM θ (with 95% CI)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.show()

print(f"saved plot -> {out_png}")
print(df[["beta_abm","theta_hat","theta_low","theta_high"]].to_string(index=False))



# %%
