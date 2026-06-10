# %% ================== 준비(그대로 사용) ==================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import ast, os
from scipy.optimize import brentq


# ---- 관측 데이터 ----
y_month = np.array([1,2,2,2,1,0,0,3,2,2,2,0,3,0,1,2,0,1,4,5,4,2,4,1,0,0,1,0,1,1,0,3,1,0,3,0], dtype=float)
y_cum   = np.cumsum(y_month)

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
                   monthly_PI=monthly_PI, pi_dates=PI_dates):
    """
    격리=HAI만 모아두는 칸(P_HAI_iso만). 상태: P_S_sh, P_HAI_sh, P_HAI_iso, P_I, H_C, Env_C
    """
    # ---- 파라미터 ----
    C_total = 30; C_iso = 30; C_sh = C_total
    N_H, N_E = 19, 30
    mu_S, mu_HAI, mu_I = 1/14, 1/21, 1/14
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
    days  = pd.date_range(start, end, freq="D"); T = len(days)

    # ---- P_I 입원 패턴 ----
    A_I_day = _make_AI_from_dates(pi_dates, days)

    # ---- 상태 ----
    P_S_sh    = np.zeros(T)
    P_HAI_sh  = np.zeros(T)
    P_HAI_iso = np.zeros(T)
    P_I       = np.zeros(T)
    H_C       = np.zeros(T)
    Env_C     = np.zeros(T)
    NewHAI_day= np.zeros(T)

    # 초기조건
    P_S_sh[0] = C_total - 1
    P_I[0]    = 1
    Env_C[0]  = init_env

    for t in range(T):
        # τ0 반영 대청소
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
            taken     = min(PS_sh, inc_eff)
            PS_sh -= taken
            PI    += taken

        for _ in range(contacts_per_day):
            B_tot = max(PS_sh + PH_sh + PH_iso + PI, 1e-9)
            # shared에서만 새 HAI 발생
            lam_HP_sh = beta * (HC / N_H)
            # 환자→HCW (iso는 감염력 감소 반영)
            lam_PH = beta * ((PH_sh + iso_factor*PH_iso + PI) / B_tot)
            # Env↔HCW
            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            # 새 HAI (shared)
            hai_sh  = lam_HP_sh * PS_sh * dt
            move_HA = sigma * PH_sh * dt

            # 퇴원
            outS_sh  = mu_S   * PS_sh * dt
            outH_sh  = mu_HAI * PH_sh * dt
            outH_iso = mu_HAI * PH_iso* dt
            outI     = mu_I   * PI    * dt

            leaving = outS_sh + outH_sh + outH_iso + outI
            total_P = PS_sh + PH_sh + PH_iso + PI
            AS_tot  = max(0.0, C_total - (total_P - leaving))
            AS_sh   = AS_tot

            # shared
            PS_sh += AS_sh - outS_sh - hai_sh
            PH_sh += hai_sh - outH_sh - move_HA
            # iso
            PH_iso += move_HA - outH_iso
            # P_I
            PI += - outI

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





# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, brentq
from scipy.special import gammaln

# =========================
# B 관측 데이터: 2021-01 ~ 2023-12
# =========================

y_month = np.array(
    [1,2,2,2,1,0,0,3,2,2,2,0,
     3,0,1,2,0,1,4,5,4,2,4,1,
     0,0,1,0,1,1,0,3,1,0,3,0],
    dtype=float
)

y_cum = np.cumsum(y_month)


# =========================
# theta -> cumulative prediction
# =========================

def cum_from_theta(theta, init_env, tau0):
    """
    theta -> 월별 예측 -> 누적 예측
    관측 길이 36개월에 맞춰 자름.

    주의:
    여기서 simulate_theta는 B버전이어야 함.
    즉 B기간 monthly_PI, PI_dates,
    mu_S=1/14, mu_HAI=1/21, mu_I=1/14 세팅이 들어간 함수.
    """
    days, daily_inc, monthly_df, comp_df = simulate_theta(
        theta,
        init_env,
        tau0
    )

    lam_month = monthly_df["NewHAI_month"].to_numpy(dtype=float)
    lam_cum = np.cumsum(lam_month)

    m = min(len(y_cum), len(lam_cum))

    return lam_cum[:m]


# =========================
# Cumulative Poisson log-likelihood
# =========================

def loglik_cum_poisson(theta, init_env, tau0):
    """
    누적 Poisson 로그우도:

    sum[y log(mu) - mu - log(y!)]
    """
    mu = np.clip(
        cum_from_theta(theta, init_env, tau0),
        1e-12,
        None
    )

    y = y_cum[:len(mu)]

    ll = np.sum(
        y * np.log(mu)
        - mu
        - gammaln(y + 1)
    )

    return float(ll)


# =========================
# theta MLE + 95% CI
# =========================

def fit_theta_cum_with_CI(
    init_env=2,
    tau0=60,
    bounds=(3.0, 3.8),
    xatol=1e-3,
    maxiter=800,
    grid_n=500,
    auto_expand=True
):
    """
    theta MLE + 95% CI by profile likelihood.

    95% CI criterion:
    2 * (ll_hat - ll(theta)) <= 3.84

    equivalent:
    ll(theta) >= ll_hat - 1.92
    """

    lo, hi = bounds

    # -------------------------
    # 1. MLE
    # -------------------------

    obj = lambda th: -loglik_cum_poisson(
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

    # -------------------------
    # 2. AIC / BIC
    # -------------------------

    k = 1
    n = len(y_cum)

    AIC = 2 * k - 2 * ll_hat
    BIC = k * np.log(n) - 2 * ll_hat

    # -------------------------
    # 3. Profile likelihood CI
    # -------------------------

    target = ll_hat - 1.92

    def profile_on_grid(lo, hi, grid_n):
        grid = np.linspace(lo, hi, grid_n)

        ll_vals = np.array([
            loglik_cum_poisson(
                th,
                init_env,
                tau0
            )
            for th in grid
        ])

        g = ll_vals - target
        i_hat = int(np.argmin((grid - theta_hat) ** 2))

        return grid, ll_vals, g, i_hat

    # -------------------------
    # Optional auto-expand bounds
    # -------------------------

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

    # -------------------------
    # 4. Left CI
    # -------------------------

    theta_low = lo
    found_left = False

    for i in range(i_hat, 0, -1):
        if (g[i-1] < 0) and (g[i] >= 0):
            theta_low = brentq(
                lambda x: loglik_cum_poisson(
                    x,
                    init_env,
                    tau0
                ) - target,
                grid[i-1],
                grid[i]
            )
            found_left = True
            break

    # -------------------------
    # 5. Right CI
    # -------------------------

    theta_high = hi
    found_right = False

    for i in range(i_hat, len(grid)-1):
        if (g[i] >= 0) and (g[i+1] < 0):
            theta_high = brentq(
                lambda x: loglik_cum_poisson(
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


# =========================
# Run: B baseline
# =========================

init_env = 2
tau0 = 60

bounds = (3.0, 3.6)
grid_n = 61

res = fit_theta_cum_with_CI(
    init_env=init_env,
    tau0=tau0,
    bounds=bounds,
    grid_n=grid_n,
    auto_expand=True
)

print(
    f"theta_hat={res['theta_hat']:.4f}  "
    f"95% CI=({res['theta_low']:.4f}, {res['theta_high']:.4f})  "
    f"logLik={res['logLik_hat']:.2f}  "
    f"AIC={res['AIC']:.2f}  "
    f"BIC={res['BIC']:.2f}"
)

print(
    "bounds actually used:",
    res["bounds_used"],
    "| found_left:",
    res["found_left"],
    "| found_right:",
    res["found_right"]
)


# =========================
# Plot: observed vs fitted cumulative
# =========================

lam_cum_hat = cum_from_theta(
    res["theta_hat"],
    init_env,
    tau0
)

obs_cum = y_cum[:len(lam_cum_hat)]

plt.figure(figsize=(9, 6))

plt.plot(
    obs_cum,
    "o-",
    color="black",
    label="Observed cumulative, B"
)

plt.plot(
    lam_cum_hat,
    "s-",
    color="orange",
    label=f"Fitted model, theta={res['theta_hat']:.4f}"
)

plt.xlabel("Month index")
plt.ylabel("Cumulative HAI")
plt.title("B period cumulative Poisson fit")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# Plot: profile log-likelihood
# =========================

grid = res["grid"]
ll_vals = res["ll_vals"]
target = res["target_logLik"]

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
    res["theta_hat"],
    ls="--",
    label="theta_hat"
)

plt.axvline(
    res["theta_low"],
    ls=":",
    label="CI low"
)

plt.axvline(
    res["theta_high"],
    ls=":",
    label="CI high"
)

plt.xlabel("theta")
plt.ylabel("logLik")
plt.title("B period profile log-likelihood")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# Plot: likelihood ratio
# =========================

ll_hat = res["logLik_hat"]
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
    res["theta_hat"],
    ls="--",
    label="theta_hat"
)

plt.axvline(
    res["theta_low"],
    ls=":",
    label="CI low"
)

plt.axvline(
    res["theta_high"],
    ls=":",
    label="CI high"
)

plt.xlabel("theta")
plt.ylabel("LR(theta)")
plt.title("B period likelihood ratio check")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%
