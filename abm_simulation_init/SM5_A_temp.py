# %% 기본 세팅
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ---- 기간 정의용 (그냥 날짜 범위용) ----
monthly_PI = pd.Series({
    "2017-01":0,"2017-02":0,"2017-03":1,"2017-04":0,"2017-05":1,"2017-06":0,
    "2017-07":0,"2017-08":0,"2017-09":2,"2017-10":0,"2017-11":2,"2017-12":0,
    "2018-01":0,"2018-02":0,"2018-03":0,"2018-04":0,"2018-05":1,"2018-06":0,"2018-07":0
})

# ---- 실제 P_I 입원 날짜들 ----
PI_dates = [
    "2017-03-01",
    "2017-05-30",
    "2017-09-17",
    "2017-09-29",
    "2017-11-14",
    "2017-11-20",
    "2018-05-25",
]

def _make_AI_from_dates(pi_dates, days):
    """실제 감염 입원 날짜 리스트 → 일별 입원수 A_I_day"""
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
    격리 = HAI만 모아두는 칸 (P_HAI_iso만 있음)
    상태변수: P_S_sh, P_HAI_sh, P_HAI_iso, P_I, H_C, Env_C
    """

    # ---- 파라미터 ----
    C_total = 30
    C_iso   = 30
    C_sh    = C_total 

    N_H, N_E = 19, 30
    mu_S, mu_HAI, mu_I = 1/7, 1/14, 1/7
    p_wash = 0.90
    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    iso_factor        = 0.75
    isol_time         = 14.0
    sigma             = 1.0 / isol_time   # shared HAI → iso HAI

    # ---- 시간축 ----
    start = pd.Period(monthly_PI.index.min(), freq="M").to_timestamp(how="start")
    end   = pd.Period(monthly_PI.index.max(), freq="M").to_timestamp(how="end")
    days  = pd.date_range(start, end, freq="D")
    T     = len(days)

    # ---- P_I 입원 패턴 ----
    A_I_day = _make_AI_from_dates(pi_dates, days)

    # ---- 상태 ----
    P_S_sh    = np.zeros(T)   # shared S
    P_HAI_sh  = np.zeros(T)   # shared HAI
    P_HAI_iso = np.zeros(T)   # isolated HAI
    P_I       = np.zeros(T)   # colonized on admission
    H_C       = np.zeros(T)
    Env_C     = np.zeros(T)
    NewHAI_day = np.zeros(T)

    # 초기조건
    P_S_sh[0]   = C_total - 1
    P_HAI_sh[0] = 0
    P_HAI_iso[0]= 0
    P_I[0]      = 1
    Env_C[0]    = init_env   # 초기 환경오염

    for t in range(T):
        # τ0 반영 대청소: (t + tau0)가 180의 배수일 때 청소
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            xx = Env_C[t]
            Env_C[t] = 0.1*xx
            

        PS_sh, PH_sh  = P_S_sh[t],  P_HAI_sh[t]
        PH_iso        = P_HAI_iso[t]
        PI, HC, EC    = P_I[t],     H_C[t],       Env_C[t]

        # 오늘 입원하는 P_I
        inc = A_I_day[t]
        if inc > 0:
            total_P   = PS_sh + PH_sh + PH_iso + PI
            stay_free = max(0.0, C_total - total_P)
            inc_eff   = min(inc, stay_free)

            taken = min(PS_sh, inc_eff)
            PS_sh -= taken
            PI    += taken

        for _ in range(contacts_per_day):
            B_tot = max(PS_sh + PH_sh + PH_iso + PI, 1e-9)

            # shared에서만 새 HAI 발생 (격리는 이미 HAI만 있음)
            lam_HP_sh = beta * (HC / N_H)

            # 환자 → HCW (shared + iso + P_I)
            lam_PH = beta * ((PH_sh + iso_factor*PH_iso + PI) / B_tot)

            # Env ↔ HCW
            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            # 새 HAI (shared)
            hai_sh = lam_HP_sh * PS_sh * dt

            # shared HAI 중 일부 격리로 이동
            move_HA = sigma * PH_sh * dt

            # 퇴원
            outS_sh  = mu_S   * PS_sh * dt
            outH_sh  = mu_HAI * PH_sh * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI     = mu_I   * PI * dt

            leaving = outS_sh + outH_sh + outH_iso + outI
            total_P = PS_sh + PH_sh + PH_iso + PI
            AS_tot  = max(0.0, C_total - (total_P - leaving))
            AS_sh   = AS_tot   # 새 입원은 shared S로만

            # shared
            PS_sh += AS_sh - outS_sh - hai_sh
            PH_sh += hai_sh - outH_sh - move_HA

            # iso (HAI만 있음)
            PH_iso += move_HA - outH_iso

            # P_I
            PI += 0.0 - outI

            # clip
            PS_sh   = np.clip(PS_sh,   0, C_sh)
            PH_sh   = np.clip(PH_sh,   0, C_sh)
            PH_iso  = np.clip(PH_iso,  0, C_iso)
            PI      = np.clip(PI,      0, C_total)

            # HCW
            new_H = (lam_PH + lam_EH) * (N_H - HC) * dt
            HC = (HC + new_H) * (1 - p_wash)
            HC = np.clip(HC, 0, N_H)

            # Env
            EC += lam_HE * (N_E - EC) * dt
            EC = np.clip(EC, 0, N_E)

            # 인시던스: 새로 생긴 HAI = shared에서만
            NewHAI_day[t] += hai_sh

        if t < T-1:
            P_S_sh[t+1], P_HAI_sh[t+1]   = PS_sh, PH_sh
            P_HAI_iso[t+1]               = PH_iso
            P_I[t+1]                     = PI
            H_C[t+1], Env_C[t+1]         = HC, EC

    # ---- 월별 합계 ----
    df = pd.DataFrame({"date": days, "NewHAI": NewHAI_day})
    monthly = (df.groupby(df["date"].dt.to_period("M"))["NewHAI"]
                 .sum()
                 .reset_index()
                 .rename(columns={"date":"month","NewHAI":"NewHAI_month"}))
    monthly["cum_NewHAI"] = monthly["NewHAI_month"].cumsum()

    # ---- 컴파트먼트 DF ----
    H_S   = N_H - H_C
    Env_S = N_E - Env_C
    comp_df = pd.DataFrame({
        "date": days,
        "P_S_sh":   P_S_sh,
        "P_HAI_sh": P_HAI_sh,
        "P_HAI_iso":P_HAI_iso,
        "P_I":      P_I,
        "H_S":      H_S,   "H_C":   H_C,
        "Env_S":    Env_S, "Env_C": Env_C,
    }).set_index("date")

    return days, NewHAI_day, monthly, comp_df


# %% # %% Gaussian MLE for theta + 95% CI using profile likelihood

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, brentq

# ===== 관측(월별)과 누적 =====
y_month = np.array([5,2,0,2,1,1,2,2,6,1,1,0,2,1,1,2,5,2,2], dtype=float)
y_cum   = np.cumsum(y_month)

def cum_from_theta(theta, init_env, tau0):
    """
    theta -> 월별 예측 -> 누적 예측
    """
    days, daily_inc, monthly_df, comp_df = simulate_theta(theta, init_env, tau0)
    lam_month = monthly_df["NewHAI_month"].to_numpy(dtype=float)
    lam_cum   = np.cumsum(lam_month)

    m = min(len(y_cum), len(lam_cum))
    return lam_cum[:m]

def loglik_cum_gaussian(theta, init_env, tau0):
    """
    Gaussian profile log-likelihood.
    sigma^2는 각 theta에서 residual variance MLE로 추정함.

    y_t ~ N(mu_t(theta), sigma^2)
    """
    mu = cum_from_theta(theta, init_env, tau0)
    y  = y_cum[:len(mu)]

    resid = y - mu
    n = len(resid)

    # sigma^2 MLE
    sigma2_hat = np.mean(resid**2)

    # numerical safety
    sigma2_hat = max(sigma2_hat, 1e-12)

    ll = -0.5 * n * (np.log(2*np.pi*sigma2_hat) + 1)

    return float(ll)

def fit_theta_cum_gaussian_with_CI(init_env=10, tau0=140, bounds=(0.1, 6.0),
                                   xatol=1e-3, maxiter=800, grid_n=500,
                                   auto_expand=True):

    lo, hi = bounds

    # 1) MLE
    obj = lambda th: -loglik_cum_gaussian(th, init_env, tau0)

    res = minimize_scalar(
        obj,
        bounds=(lo, hi),
        method="bounded",
        options=dict(xatol=xatol, maxiter=maxiter)
    )

    theta_hat = float(res.x)
    ll_hat    = -float(res.fun)

    # 2) AIC/BIC
    # theta + sigma = 2 parameters
    k = 2
    n = len(y_cum)

    AIC = 2*k - 2*ll_hat
    BIC = k*np.log(n) - 2*ll_hat

    # 3) 95% CI by likelihood ratio
    # 2*(ll_hat - ll(theta)) <= 3.84
    # ll(theta) >= ll_hat - 1.92
    target = ll_hat - 1.92

    def profile_on_grid(lo, hi, grid_n):
        grid = np.linspace(lo, hi, grid_n)
        ll_vals = np.array([
            loglik_cum_gaussian(th, init_env, tau0)
            for th in grid
        ])
        g = ll_vals - target
        i_hat = int(np.argmin((grid - theta_hat)**2))
        return grid, ll_vals, g, i_hat

    # bounds 자동 확장
    if auto_expand:
        for _ in range(6):
            grid, ll_vals, g, i_hat = profile_on_grid(lo, hi, grid_n)

            if (g[0] < 0) and (g[i_hat] > 0) and (g[-1] < 0):
                break

            if g[0] >= 0:
                lo = max(1e-6, lo * 0.5)

            if g[-1] >= 0:
                hi = hi * 1.5

        grid, ll_vals, g, i_hat = profile_on_grid(lo, hi, grid_n)

    else:
        grid, ll_vals, g, i_hat = profile_on_grid(lo, hi, grid_n)

    # 왼쪽 CI
    theta_low = lo
    found_left = False

    for i in range(i_hat, 0, -1):
        if (g[i-1] < 0) and (g[i] >= 0):
            theta_low = brentq(
                lambda x: loglik_cum_gaussian(x, init_env, tau0) - target,
                grid[i-1],
                grid[i]
            )
            found_left = True
            break

    # 오른쪽 CI
    theta_high = hi
    found_right = False

    for i in range(i_hat, len(grid)-1):
        if (g[i] >= 0) and (g[i+1] < 0):
            theta_high = brentq(
                lambda x: loglik_cum_gaussian(x, init_env, tau0) - target,
                grid[i],
                grid[i+1]
            )
            found_right = True
            break

    # theta_hat에서 sigma도 계산
    mu_hat = cum_from_theta(theta_hat, init_env, tau0)
    y_hat  = y_cum[:len(mu_hat)]
    sigma_hat = np.sqrt(np.mean((y_hat - mu_hat)**2))

    return {
        "theta_hat": theta_hat,
        "theta_low": float(theta_low),
        "theta_high": float(theta_high),
        "sigma_hat": float(sigma_hat),
        "logLik_hat": ll_hat,
        "AIC": AIC,
        "BIC": BIC,
        "bounds_used": (float(lo), float(hi)),
        "target_logLik": float(target),
        "grid": grid,
        "ll_vals": ll_vals,
        "found_left": found_left,
        "found_right": found_right,
    }

# ===== 실행 =====
init_env = 9
tau0     = 140
bounds   = (3.2,3.7)
import time

t0 = time.time()
_ = cum_from_theta(3.4, init_env=10, tau0=140)
t1 = time.time()

print("One simulation time:", t1 - t0, "seconds")
res_g = fit_theta_cum_gaussian_with_CI(
    init_env=init_env,
    tau0=tau0,
    bounds=bounds,
    grid_n=50,
    auto_expand=True
)

print(
    f"theta_hat={res_g['theta_hat']:.4f}  "
    f"95% CI=({res_g['theta_low']:.4f}, {res_g['theta_high']:.4f})  "
    f"sigma_hat={res_g['sigma_hat']:.4f}  "
    f"logLik={res_g['logLik_hat']:.2f}  "
    f"AIC={res_g['AIC']:.2f}  "
    f"BIC={res_g['BIC']:.2f}"
)

print("bounds actually used:", res_g["bounds_used"],
      "| found_left:", res_g["found_left"],
      "| found_right:", res_g["found_right"])

# ===== 누적 관측 vs 예측 plot =====
lam_cum_hat = cum_from_theta(res_g["theta_hat"], init_env, tau0)
obs_cum = y_cum[:len(lam_cum_hat)]

plt.figure(figsize=(9,6))
plt.plot(obs_cum, "o-", color="black", label="Observed cumulative")
plt.plot(
    lam_cum_hat,
    "s-",
    color="orange",
    label=f"Gaussian fit, theta={res_g['theta_hat']:.4f}"
)

plt.xlabel("Month index")
plt.ylabel("Cumulative HAI")
plt.title("Cumulative Gaussian MLE fit")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ===== Profile log-likelihood plot =====
grid = res_g["grid"]
ll_vals = res_g["ll_vals"]
target = res_g["target_logLik"]

plt.figure(figsize=(7.2, 4.2))
plt.plot(grid, ll_vals, lw=2)
plt.axhline(target, ls="--", label="target = ll_hat - 1.92")
plt.axvline(res_g["theta_hat"], ls="--", label="theta_hat")
plt.axvline(res_g["theta_low"], ls=":", label="CI low")
plt.axvline(res_g["theta_high"], ls=":", label="CI high")
plt.xlabel("theta")
plt.ylabel("logLik")
plt.title("Gaussian profile log-likelihood")
plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()
plt.show()

# ===== LR plot =====
ll_hat = res_g["logLik_hat"]
LR = 2*(ll_hat - ll_vals)

plt.figure(figsize=(7.2, 4.2))
plt.plot(grid, LR, lw=2)
plt.axhline(3.84, ls="--", label="3.84")
plt.axvline(res_g["theta_hat"], ls="--", label="theta_hat")
plt.axvline(res_g["theta_low"], ls=":", label="CI low")
plt.axvline(res_g["theta_high"], ls=":", label="CI high")
plt.xlabel("theta")
plt.ylabel("LR(theta)")
plt.title("Gaussian profile likelihood check")
plt.grid(alpha=.3)
plt.legend()
plt.tight_layout()
plt.show()
# %% 1개월 뗸 가우시안







# Gaussian likelihood version
# Drop first 1 month
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar, brentq


# ============================================================
# 1. Observed data
# ============================================================

y_month_full = np.array(
    [5,2,0,2,1,1,2,2,6,1,1,0,2,1,1,2,5,2,2],
    dtype=float
)

trim_front = 1

# Drop first month
y_month = y_month_full[trim_front:]

# Recompute cumulative after dropping first month
y_cum = np.cumsum(y_month)


# ============================================================
# 2. Period information
# ============================================================

monthly_PI = pd.Series({
    "2017-01":0,"2017-02":0,"2017-03":1,"2017-04":0,"2017-05":1,"2017-06":0,
    "2017-07":0,"2017-08":0,"2017-09":2,"2017-10":0,"2017-11":2,"2017-12":0,
    "2018-01":0,"2018-02":0,"2018-03":0,"2018-04":0,"2018-05":1,"2018-06":0,"2018-07":0
})

PI_dates = [
    "2017-03-01",
    "2017-05-30",
    "2017-09-17",
    "2017-09-29",
    "2017-11-14",
    "2017-11-20",
    "2018-05-25",
]


def _make_AI_from_dates(pi_dates, days):
    """
    실제 P_I 입원 날짜 리스트를 일별 입원수 배열로 변환
    """
    T = len(days)
    A = np.zeros(T)

    idx_map = {d: i for i, d in enumerate(days)}

    for d in pi_dates:
        ts = pd.to_datetime(d)
        idx = idx_map.get(ts, None)

        if idx is not None:
            A[idx] += 1.0

    return A


# ============================================================
# 3. ABM simulation function
# ============================================================

def simulate_theta(beta, init_env, tau0,
                   monthly_PI=monthly_PI,
                   pi_dates=PI_dates):
    """
    상태변수:
    P_S_sh, P_HAI_sh, P_HAI_iso, P_I, H_C, Env_C
    """

    # -----------------------------
    # Parameters
    # -----------------------------

    C_total = 30
    C_iso   = 30
    C_sh    = C_total

    N_H, N_E = 19, 30

    mu_S   = 1/7
    mu_HAI = 1/14
    mu_I   = 1/7

    p_wash = 0.90

    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    iso_factor = 0.75

    isol_time = 14.0
    sigma = 1.0 / isol_time

    # -----------------------------
    # Time axis
    # -----------------------------

    start = pd.Period(
        monthly_PI.index.min(),
        freq="M"
    ).to_timestamp(how="start")

    end = pd.Period(
        monthly_PI.index.max(),
        freq="M"
    ).to_timestamp(how="end")

    days = pd.date_range(start, end, freq="D")
    T = len(days)

    # -----------------------------
    # Imported infected admission pattern
    # -----------------------------

    A_I_day = _make_AI_from_dates(pi_dates, days)

    # -----------------------------
    # State variables
    # -----------------------------

    P_S_sh     = np.zeros(T)
    P_HAI_sh   = np.zeros(T)
    P_HAI_iso  = np.zeros(T)
    P_I        = np.zeros(T)
    H_C        = np.zeros(T)
    Env_C      = np.zeros(T)
    NewHAI_day = np.zeros(T)

    # -----------------------------
    # Initial condition
    # -----------------------------

    P_S_sh[0] = C_total - 1
    P_HAI_sh[0] = 0
    P_HAI_iso[0] = 0
    P_I[0] = 1
    Env_C[0] = init_env

    # -----------------------------
    # Simulation loop
    # -----------------------------

    for t in range(T):

        # Deep cleaning
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            Env_C[t] = 0.1 * Env_C[t]

        PS_sh, PH_sh = P_S_sh[t], P_HAI_sh[t]
        PH_iso = P_HAI_iso[t]

        PI = P_I[t]
        HC = H_C[t]
        EC = Env_C[t]

        # Imported P_I admission
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

            # HCW -> patient
            lam_HP_sh = beta * (HC / N_H)

            # patient -> HCW
            lam_PH = beta * ((PH_sh + iso_factor * PH_iso + PI) / B_tot)

            # environment <-> HCW
            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            # new HAI in shared ward
            hai_sh = lam_HP_sh * PS_sh * dt

            # shared HAI -> isolated HAI
            move_HA = sigma * PH_sh * dt

            # discharge
            outS_sh  = mu_S   * PS_sh  * dt
            outH_sh  = mu_HAI * PH_sh  * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI     = mu_I   * PI     * dt

            leaving = outS_sh + outH_sh + outH_iso + outI
            total_P = PS_sh + PH_sh + PH_iso + PI

            AS_tot = max(0.0, C_total - (total_P - leaving))
            AS_sh = AS_tot

            # patient states
            PS_sh += AS_sh - outS_sh - hai_sh
            PH_sh += hai_sh - outH_sh - move_HA
            PH_iso += move_HA - outH_iso
            PI += -outI

            # clipping
            PS_sh  = np.clip(PS_sh,  0, C_sh)
            PH_sh  = np.clip(PH_sh,  0, C_sh)
            PH_iso = np.clip(PH_iso, 0, C_iso)
            PI     = np.clip(PI,     0, C_total)

            # HCW
            new_H = (lam_PH + lam_EH) * (N_H - HC) * dt
            HC = (HC + new_H) * (1 - p_wash)
            HC = np.clip(HC, 0, N_H)

            # Environment
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

    # -----------------------------
    # Monthly incidence
    # -----------------------------

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

    # -----------------------------
    # Compartment dataframe
    # -----------------------------

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


# ============================================================
# 4. Prediction after dropping first month
# ============================================================

def cum_from_theta_dropfirst(theta, init_env, tau0, trim_front=1):
    """
    theta -> monthly prediction -> drop first month -> cumulative prediction
    """

    days, daily_inc, monthly_df, comp_df = simulate_theta(
        theta,
        init_env,
        tau0
    )

    lam_month_full = monthly_df["NewHAI_month"].to_numpy(dtype=float)

    # Drop first month from model prediction
    lam_month = lam_month_full[trim_front:]

    # Recompute cumulative after dropping first month
    lam_cum = np.cumsum(lam_month)

    m = min(len(y_cum), len(lam_cum))

    return lam_cum[:m]


# ============================================================
# 5. Gaussian likelihood
# ============================================================

def loglik_cum_gaussian_dropfirst(theta, init_env, tau0):
    """
    Gaussian profile log-likelihood.

    y_t ~ N(mu_t(theta), sigma^2)

    sigma^2 is estimated by MLE for each theta.
    """

    mu = cum_from_theta_dropfirst(
        theta,
        init_env,
        tau0,
        trim_front=trim_front
    )

    y = y_cum[:len(mu)]

    resid = y - mu
    n = len(resid)

    sigma2_hat = np.mean(resid ** 2)
    sigma2_hat = max(sigma2_hat, 1e-12)

    ll = -0.5 * n * (np.log(2 * np.pi * sigma2_hat) + 1)

    return float(ll)


# ============================================================
# 6. Gaussian MLE + profile likelihood CI
# ============================================================

def fit_theta_cum_gaussian_dropfirst_with_CI(
    init_env=9,
    tau0=140,
    bounds=(0.1, 6.0),
    xatol=1e-3,
    maxiter=800,
    grid_n=30,
    auto_expand=True
):

    lo, hi = bounds

    # -----------------------------
    # MLE
    # -----------------------------

    obj = lambda th: -loglik_cum_gaussian_dropfirst(
        th,
        init_env,
        tau0
    )

    res = minimize_scalar(
        obj,
        bounds=(lo, hi),
        method="bounded",
        options=dict(
            xatol=xatol,
            maxiter=maxiter
        )
    )

    theta_hat = float(res.x)
    ll_hat = -float(res.fun)

    # -----------------------------
    # AIC / BIC
    # theta + sigma = 2 parameters
    # -----------------------------

    k = 2
    n = len(y_cum)

    AIC = 2 * k - 2 * ll_hat
    BIC = k * np.log(n) - 2 * ll_hat

    # -----------------------------
    # 95% profile likelihood CI
    # 2 * (ll_hat - ll(theta)) <= 3.84
    # ll(theta) >= ll_hat - 1.92
    # -----------------------------

    target = ll_hat - 1.92

    def profile_on_grid(lo, hi, grid_n):
        grid = np.linspace(lo, hi, grid_n)

        ll_vals = np.array([
            loglik_cum_gaussian_dropfirst(th, init_env, tau0)
            for th in grid
        ])

        g = ll_vals - target
        i_hat = int(np.argmin((grid - theta_hat) ** 2))

        return grid, ll_vals, g, i_hat

    # -----------------------------
    # Auto expand bounds if CI is outside
    # -----------------------------

    if auto_expand:
        for _ in range(6):
            grid, ll_vals, g, i_hat = profile_on_grid(
                lo,
                hi,
                grid_n
            )

            if (g[0] < 0) and (g[i_hat] > 0) and (g[-1] < 0):
                break

            if g[0] >= 0:
                lo = max(1e-6, lo * 0.5)

            if g[-1] >= 0:
                hi = hi * 1.5

        grid, ll_vals, g, i_hat = profile_on_grid(
            lo,
            hi,
            grid_n
        )

    else:
        grid, ll_vals, g, i_hat = profile_on_grid(
            lo,
            hi,
            grid_n
        )

    # -----------------------------
    # Left CI
    # -----------------------------

    theta_low = lo
    found_left = False

    for i in range(i_hat, 0, -1):
        if (g[i-1] < 0) and (g[i] >= 0):
            theta_low = brentq(
                lambda x: loglik_cum_gaussian_dropfirst(
                    x,
                    init_env,
                    tau0
                ) - target,
                grid[i-1],
                grid[i]
            )
            found_left = True
            break

    # -----------------------------
    # Right CI
    # -----------------------------

    theta_high = hi
    found_right = False

    for i in range(i_hat, len(grid)-1):
        if (g[i] >= 0) and (g[i+1] < 0):
            theta_high = brentq(
                lambda x: loglik_cum_gaussian_dropfirst(
                    x,
                    init_env,
                    tau0
                ) - target,
                grid[i],
                grid[i+1]
            )
            found_right = True
            break

    # -----------------------------
    # Sigma at theta_hat
    # -----------------------------

    mu_hat = cum_from_theta_dropfirst(
        theta_hat,
        init_env,
        tau0,
        trim_front=trim_front
    )

    y_hat = y_cum[:len(mu_hat)]

    sigma_hat = np.sqrt(
        np.mean((y_hat - mu_hat) ** 2)
    )

    return {
        "theta_hat": theta_hat,
        "theta_low": float(theta_low),
        "theta_high": float(theta_high),
        "sigma_hat": float(sigma_hat),
        "logLik_hat": ll_hat,
        "AIC": AIC,
        "BIC": BIC,
        "bounds_used": (float(lo), float(hi)),
        "target_logLik": float(target),
        "grid": grid,
        "ll_vals": ll_vals,
        "found_left": found_left,
        "found_right": found_right,
    }


# ============================================================
# 7. Run Gaussian fitting
# ============================================================

init_env = 9
tau0 = 140

bounds = (3.2, 3.7)

res_g = fit_theta_cum_gaussian_dropfirst_with_CI(
    init_env=init_env,
    tau0=tau0,
    bounds=bounds,
    grid_n=30,
    auto_expand=True
)

print(
    f"theta_hat={res_g['theta_hat']:.4f}  "
    f"95% CI=({res_g['theta_low']:.4f}, {res_g['theta_high']:.4f})  "
    f"sigma_hat={res_g['sigma_hat']:.4f}  "
    f"logLik={res_g['logLik_hat']:.2f}  "
    f"AIC={res_g['AIC']:.2f}  "
    f"BIC={res_g['BIC']:.2f}"
)

print(
    "bounds actually used:",
    res_g["bounds_used"],
    "| found_left:",
    res_g["found_left"],
    "| found_right:",
    res_g["found_right"]
)


# ============================================================
# 8. Plot: observed vs fitted cumulative
# ============================================================

lam_cum_hat = cum_from_theta_dropfirst(
    res_g["theta_hat"],
    init_env,
    tau0,
    trim_front=trim_front
)

obs_cum = y_cum[:len(lam_cum_hat)]

plt.figure(figsize=(9, 6))

plt.plot(
    obs_cum,
    "o-",
    color="black",
    label="Observed cumulative, drop first 1 month"
)

plt.plot(
    lam_cum_hat,
    "s-",
    color="orange",
    label=f"Gaussian fit, theta={res_g['theta_hat']:.4f}"
)

plt.xlabel("Month index after dropping first month")
plt.ylabel("Cumulative HAI")
plt.title("Cumulative Gaussian MLE fit, drop first 1 month")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 9. Plot: profile log-likelihood
# ============================================================

grid = res_g["grid"]
ll_vals = res_g["ll_vals"]
target = res_g["target_logLik"]

plt.figure(figsize=(7.2, 4.2))

plt.plot(
    grid,
    ll_vals,
    lw=2
)

plt.axhline(
    target,
    ls="--",
    label="target = ll_hat - 1.92"
)

plt.axvline(
    res_g["theta_hat"],
    ls="--",
    label="theta_hat"
)

plt.axvline(
    res_g["theta_low"],
    ls=":",
    label="CI low"
)

plt.axvline(
    res_g["theta_high"],
    ls=":",
    label="CI high"
)

plt.xlabel("theta")
plt.ylabel("logLik")
plt.title("Gaussian profile log-likelihood, drop first 1 month")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. Plot: likelihood ratio
# ============================================================

ll_hat = res_g["logLik_hat"]
LR = 2 * (ll_hat - ll_vals)

plt.figure(figsize=(7.2, 4.2))

plt.plot(
    grid,
    LR,
    lw=2
)

plt.axhline(
    3.84,
    ls="--",
    label="3.84"
)

plt.axvline(
    res_g["theta_hat"],
    ls="--",
    label="theta_hat"
)

plt.axvline(
    res_g["theta_low"],
    ls=":",
    label="CI low"
)

plt.axvline(
    res_g["theta_high"],
    ls=":",
    label="CI high"
)

plt.xlabel("theta")
plt.ylabel("LR(theta)")
plt.title("Gaussian profile likelihood check, drop first 1 month")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()







# %% 1개월 뗀 푸아송

# %% ============================================================
# Poisson likelihood version
# Drop first 1 month
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar, brentq
from scipy.special import gammaln


# ============================================================
# 1. Observed data
# ============================================================

y_month_full = np.array(
    [5,2,0,2,1,1,2,2,6,1,1,0,2,1,1,2,5,2,2],
    dtype=float
)

trim_front = 1

# Drop first month
y_month = y_month_full[trim_front:]

# Recompute cumulative after dropping first month
y_cum = np.cumsum(y_month)


# ============================================================
# 2. Period information
# ============================================================

monthly_PI = pd.Series({
    "2017-01":0,"2017-02":0,"2017-03":1,"2017-04":0,"2017-05":1,"2017-06":0,
    "2017-07":0,"2017-08":0,"2017-09":2,"2017-10":0,"2017-11":2,"2017-12":0,
    "2018-01":0,"2018-02":0,"2018-03":0,"2018-04":0,"2018-05":1,"2018-06":0,"2018-07":0
})

PI_dates = [
    "2017-03-01",
    "2017-05-30",
    "2017-09-17",
    "2017-09-29",
    "2017-11-14",
    "2017-11-20",
    "2018-05-25",
]


def _make_AI_from_dates(pi_dates, days):
    """
    실제 P_I 입원 날짜 리스트를 일별 입원수 배열로 변환
    """
    T = len(days)
    A = np.zeros(T)

    idx_map = {d: i for i, d in enumerate(days)}

    for d in pi_dates:
        ts = pd.to_datetime(d)
        idx = idx_map.get(ts, None)

        if idx is not None:
            A[idx] += 1.0

    return A


# ============================================================
# 3. ABM simulation function
# ============================================================

def simulate_theta(beta, init_env, tau0,
                   monthly_PI=monthly_PI,
                   pi_dates=PI_dates):
    """
    상태변수:
    P_S_sh, P_HAI_sh, P_HAI_iso, P_I, H_C, Env_C
    """

    # -----------------------------
    # Parameters
    # -----------------------------

    C_total = 30
    C_iso   = 30
    C_sh    = C_total

    N_H, N_E = 19, 30

    mu_S   = 1/7
    mu_HAI = 1/14
    mu_I   = 1/7

    p_wash = 0.90

    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    iso_factor = 0.75

    isol_time = 14.0
    sigma = 1.0 / isol_time

    # -----------------------------
    # Time axis
    # -----------------------------

    start = pd.Period(
        monthly_PI.index.min(),
        freq="M"
    ).to_timestamp(how="start")

    end = pd.Period(
        monthly_PI.index.max(),
        freq="M"
    ).to_timestamp(how="end")

    days = pd.date_range(start, end, freq="D")
    T = len(days)

    # -----------------------------
    # Imported infected admission pattern
    # -----------------------------

    A_I_day = _make_AI_from_dates(pi_dates, days)

    # -----------------------------
    # State variables
    # -----------------------------

    P_S_sh     = np.zeros(T)
    P_HAI_sh   = np.zeros(T)
    P_HAI_iso  = np.zeros(T)
    P_I        = np.zeros(T)
    H_C        = np.zeros(T)
    Env_C      = np.zeros(T)
    NewHAI_day = np.zeros(T)

    # -----------------------------
    # Initial condition
    # -----------------------------

    P_S_sh[0] = C_total - 1
    P_HAI_sh[0] = 0
    P_HAI_iso[0] = 0
    P_I[0] = 1
    Env_C[0] = init_env

    # -----------------------------
    # Simulation loop
    # -----------------------------

    for t in range(T):

        # Deep cleaning
        if t > 0 and (t + tau0) % deep_clean_period == 0:
            Env_C[t] = 0.1 * Env_C[t]

        PS_sh, PH_sh = P_S_sh[t], P_HAI_sh[t]
        PH_iso = P_HAI_iso[t]

        PI = P_I[t]
        HC = H_C[t]
        EC = Env_C[t]

        # Imported P_I admission
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

            # HCW -> patient
            lam_HP_sh = beta * (HC / N_H)

            # patient -> HCW
            lam_PH = beta * ((PH_sh + iso_factor * PH_iso + PI) / B_tot)

            # environment <-> HCW
            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            # new HAI in shared ward
            hai_sh = lam_HP_sh * PS_sh * dt

            # shared HAI -> isolated HAI
            move_HA = sigma * PH_sh * dt

            # discharge
            outS_sh  = mu_S   * PS_sh  * dt
            outH_sh  = mu_HAI * PH_sh  * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI     = mu_I   * PI     * dt

            leaving = outS_sh + outH_sh + outH_iso + outI
            total_P = PS_sh + PH_sh + PH_iso + PI

            AS_tot = max(0.0, C_total - (total_P - leaving))
            AS_sh = AS_tot

            # patient states
            PS_sh += AS_sh - outS_sh - hai_sh
            PH_sh += hai_sh - outH_sh - move_HA
            PH_iso += move_HA - outH_iso
            PI += -outI

            # clipping
            PS_sh  = np.clip(PS_sh,  0, C_sh)
            PH_sh  = np.clip(PH_sh,  0, C_sh)
            PH_iso = np.clip(PH_iso, 0, C_iso)
            PI     = np.clip(PI,     0, C_total)

            # HCW
            new_H = (lam_PH + lam_EH) * (N_H - HC) * dt
            HC = (HC + new_H) * (1 - p_wash)
            HC = np.clip(HC, 0, N_H)

            # Environment
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

    # -----------------------------
    # Monthly incidence
    # -----------------------------

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

    # -----------------------------
    # Compartment dataframe
    # -----------------------------

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


# ============================================================
# 4. Prediction after dropping first month
# ============================================================

def cum_from_theta_dropfirst(theta, init_env, tau0, trim_front=1):
    """
    theta -> monthly prediction -> drop first month -> cumulative prediction
    """

    days, daily_inc, monthly_df, comp_df = simulate_theta(
        theta,
        init_env,
        tau0
    )

    lam_month_full = monthly_df["NewHAI_month"].to_numpy(dtype=float)

    # Drop first month from model prediction
    lam_month = lam_month_full[trim_front:]

    # Recompute cumulative after dropping first month
    lam_cum = np.cumsum(lam_month)

    m = min(len(y_cum), len(lam_cum))

    return lam_cum[:m]


# ============================================================
# 5. Poisson likelihood
# ============================================================

def loglik_cum_poisson_dropfirst(theta, init_env, tau0):
    """
    Cumulative Poisson log-likelihood.

    y_t ~ Poisson(mu_t(theta))
    """

    mu = cum_from_theta_dropfirst(
        theta,
        init_env,
        tau0,
        trim_front=trim_front
    )

    y = y_cum[:len(mu)]

    mu = np.clip(mu, 1e-12, None)

    ll = np.sum(
        y * np.log(mu)
        - mu
        - gammaln(y + 1)
    )

    return float(ll)


# ============================================================
# 6. Poisson MLE + profile likelihood CI
# ============================================================

def fit_theta_cum_poisson_dropfirst_with_CI(
    init_env=9,
    tau0=140,
    bounds=(0.1, 6.0),
    xatol=1e-3,
    maxiter=800,
    grid_n=30,
    auto_expand=True
):

    lo, hi = bounds

    # -----------------------------
    # MLE
    # -----------------------------

    obj = lambda th: -loglik_cum_poisson_dropfirst(
        th,
        init_env,
        tau0
    )

    res = minimize_scalar(
        obj,
        bounds=(lo, hi),
        method="bounded",
        options=dict(
            xatol=xatol,
            maxiter=maxiter
        )
    )

    theta_hat = float(res.x)
    ll_hat = -float(res.fun)

    # -----------------------------
    # AIC / BIC
    # theta only = 1 parameter
    # -----------------------------

    k = 1
    n = len(y_cum)

    AIC = 2 * k - 2 * ll_hat
    BIC = k * np.log(n) - 2 * ll_hat

    # -----------------------------
    # 95% profile likelihood CI
    # -----------------------------

    target = ll_hat - 1.92

    def profile_on_grid(lo, hi, grid_n):
        grid = np.linspace(lo, hi, grid_n)

        ll_vals = np.array([
            loglik_cum_poisson_dropfirst(th, init_env, tau0)
            for th in grid
        ])

        g = ll_vals - target
        i_hat = int(np.argmin((grid - theta_hat) ** 2))

        return grid, ll_vals, g, i_hat

    # -----------------------------
    # Auto expand bounds
    # -----------------------------

    if auto_expand:
        for _ in range(6):
            grid, ll_vals, g, i_hat = profile_on_grid(
                lo,
                hi,
                grid_n
            )

            if (g[0] < 0) and (g[i_hat] > 0) and (g[-1] < 0):
                break

            if g[0] >= 0:
                lo = max(1e-6, lo * 0.5)

            if g[-1] >= 0:
                hi = hi * 1.5

        grid, ll_vals, g, i_hat = profile_on_grid(
            lo,
            hi,
            grid_n
        )

    else:
        grid, ll_vals, g, i_hat = profile_on_grid(
            lo,
            hi,
            grid_n
        )

    # -----------------------------
    # Left CI
    # -----------------------------

    theta_low = lo
    found_left = False

    for i in range(i_hat, 0, -1):
        if (g[i-1] < 0) and (g[i] >= 0):
            theta_low = brentq(
                lambda x: loglik_cum_poisson_dropfirst(
                    x,
                    init_env,
                    tau0
                ) - target,
                grid[i-1],
                grid[i]
            )
            found_left = True
            break

    # -----------------------------
    # Right CI
    # -----------------------------

    theta_high = hi
    found_right = False

    for i in range(i_hat, len(grid)-1):
        if (g[i] >= 0) and (g[i+1] < 0):
            theta_high = brentq(
                lambda x: loglik_cum_poisson_dropfirst(
                    x,
                    init_env,
                    tau0
                ) - target,
                grid[i],
                grid[i+1]
            )
            found_right = True
            break

    return {
        "theta_hat": theta_hat,
        "theta_low": float(theta_low),
        "theta_high": float(theta_high),
        "logLik_hat": ll_hat,
        "AIC": AIC,
        "BIC": BIC,
        "bounds_used": (float(lo), float(hi)),
        "target_logLik": float(target),
        "grid": grid,
        "ll_vals": ll_vals,
        "found_left": found_left,
        "found_right": found_right,
    }


# ============================================================
# 7. Run Poisson fitting
# ============================================================

init_env = 9
tau0 = 140

bounds = (3.2, 3.7)

res_p = fit_theta_cum_poisson_dropfirst_with_CI(
    init_env=init_env,
    tau0=tau0,
    bounds=bounds,
    grid_n=30,
    auto_expand=True
)

print(
    f"theta_hat={res_p['theta_hat']:.4f}  "
    f"95% CI=({res_p['theta_low']:.4f}, {res_p['theta_high']:.4f})  "
    f"logLik={res_p['logLik_hat']:.2f}  "
    f"AIC={res_p['AIC']:.2f}  "
    f"BIC={res_p['BIC']:.2f}"
)

print(
    "bounds actually used:",
    res_p["bounds_used"],
    "| found_left:",
    res_p["found_left"],
    "| found_right:",
    res_p["found_right"]
)


# ============================================================
# 8. Plot: observed vs fitted cumulative
# ============================================================

lam_cum_hat = cum_from_theta_dropfirst(
    res_p["theta_hat"],
    init_env,
    tau0,
    trim_front=trim_front
)

obs_cum = y_cum[:len(lam_cum_hat)]

plt.figure(figsize=(9, 6))

plt.plot(
    obs_cum,
    "o-",
    color="black",
    label="Observed cumulative, drop first 1 month"
)

plt.plot(
    lam_cum_hat,
    "s-",
    color="orange",
    label=f"Poisson fit, theta={res_p['theta_hat']:.4f}"
)

plt.xlabel("Month index after dropping first month")
plt.ylabel("Cumulative HAI")
plt.title("Cumulative Poisson MLE fit, drop first 1 month")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 9. Plot: profile log-likelihood
# ============================================================

grid = res_p["grid"]
ll_vals = res_p["ll_vals"]
target = res_p["target_logLik"]

plt.figure(figsize=(7.2, 4.2))

plt.plot(
    grid,
    ll_vals,
    lw=2
)

plt.axhline(
    target,
    ls="--",
    label="target = ll_hat - 1.92"
)

plt.axvline(
    res_p["theta_hat"],
    ls="--",
    label="theta_hat"
)

plt.axvline(
    res_p["theta_low"],
    ls=":",
    label="CI low"
)

plt.axvline(
    res_p["theta_high"],
    ls=":",
    label="CI high"
)

plt.xlabel("theta")
plt.ylabel("logLik")
plt.title("Poisson profile log-likelihood, drop first 1 month")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. Plot: likelihood ratio
# ============================================================

ll_hat = res_p["logLik_hat"]
LR = 2 * (ll_hat - ll_vals)

plt.figure(figsize=(7.2, 4.2))

plt.plot(
    grid,
    LR,
    lw=2
)

plt.axhline(
    3.84,
    ls="--",
    label="3.84"
)

plt.axvline(
    res_p["theta_hat"],
    ls="--",
    label="theta_hat"
)

plt.axvline(
    res_p["theta_low"],
    ls=":",
    label="CI low"
)

plt.axvline(
    res_p["theta_high"],
    ls=":",
    label="CI high"
)

plt.xlabel("theta")
plt.ylabel("LR(theta)")
plt.title("Poisson profile likelihood check, drop first 1 month")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %%
