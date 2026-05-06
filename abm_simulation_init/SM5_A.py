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

# %% MLE on cumulative Poisson (theta only)
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

# 관측(월별)과 누적
y_month = np.array([5,2,0,2,1,1,2,2,6,1,1,0,2,1,1,2,5,2,2], dtype=float)
y_cum   = np.cumsum(y_month)

# 누적 예측 생성기
def cum_from_theta(theta, init_env, tau0):
    days, daily_inc, monthly_df, comp_df = simulate_theta(theta, init_env, tau0)
    lam_month = monthly_df["NewHAI_month"].to_numpy(dtype=float)
    lam_cum   = np.cumsum(lam_month)
    # 관측 길이에 맞춰 자르기
    m = min(len(y_cum), len(lam_cum))
    return lam_cum[:m]

# 누적 포아송 로그우도
def loglik_cum_poisson(theta, init_env, tau0):
    mu = np.clip(cum_from_theta(theta, init_env, tau0), 1e-9, None)
    y  = y_cum[:len(mu)]
    # Σ [y log μ - μ - log(y!)]
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))

# 음의 로그우도 최소화 (θ만)
def fit_theta_cum(init_env=10, tau0=140, bounds=(2.5, 6.0)):
    obj = lambda th: -loglik_cum_poisson(th, init_env, tau0)
    res = minimize_scalar(obj, bounds=bounds, method="bounded",
                          options=dict(xatol=1e-3, maxiter=500))
    theta_hat = float(res.x)
    ll_hat    = -float(res.fun)
    k, n = 1, len(y_cum)
    AIC = 2*k - 2*ll_hat
    BIC = k*np.log(n) - 2*ll_hat
    return theta_hat, ll_hat, AIC, BIC

# === 실행 예 ===
theta_hat, ll_hat, AIC, BIC = fit_theta_cum(init_env=10, tau0=140, bounds=(2.5, 6.0))
print(f"theta_hat={theta_hat:.3f}, logLik={ll_hat:.2f}, AIC={AIC:.2f}, BIC={BIC:.2f}")

# 누적 비교용 간단 플롯 (선택)
import matplotlib.pyplot as plt
lam_cum_hat = cum_from_theta(theta_hat, init_env=10, tau0=140)
obs_cum = y_cum[:len(lam_cum_hat)]
plt.plot(obs_cum, "o-", label="Observed cumulative")
plt.plot(lam_cum_hat, "s-", label=f"Model cumulative (theta={theta_hat:.2f})")
plt.legend(); plt.title("Cumulative Poisson MLE fit"); plt.tight_layout(); plt.show()

# %%


# %% 누적 Poisson MLE for theta + 95% CI (profile likelihood)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.optimize import brentq
from scipy.special import gammaln

# ===== 관측(월별)과 누적 =====
y_month = np.array([5,2,0,2,1,1,2,2,6,1,1,0,2,1,1,2,5,2,2], dtype=float)
y_cum   = np.cumsum(y_month)

# ===== 주어진 simulate_theta를 그대로 사용 =====
# simulate_theta(theta, init_env, tau0) -> (days, daily_inc, monthly_df, comp_df)
# (네가 위에 정의한 simulate_theta를 그대로 둔다고 가정)

def cum_from_theta(theta, init_env, tau0):
    """θ -> 월별 예측 -> 누적 예측(관측 길이에 맞춰 자르기)"""
    days, daily_inc, monthly_df, comp_df = simulate_theta(theta, init_env, tau0)
    lam_month = monthly_df["NewHAI_month"].to_numpy(dtype=float)
    lam_cum   = np.cumsum(lam_month)
    m = min(len(y_cum), len(lam_cum))
    return lam_cum[:m]

def loglik_cum_poisson(theta, init_env, tau0):
    """누적 Poisson 로그우도: sum[y log μ - μ - log y!]"""
    mu = np.clip(cum_from_theta(theta, init_env, tau0), 1e-12, None)
    y  = y_cum[:len(mu)]
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))

def fit_theta_cum_with_CI(init_env=10, tau0=140, bounds=(2.5, 6.0),
                          xatol=1e-3, maxiter=600, grid_n=400):
    """
    θ MLE + 95% CI(프로파일 우도).
    - bounds: θ 탐색 경계
    - grid_n: CI용 브라켓 찾기 위해 그리는 격자 갯수
    """
    # 1) MLE
    obj = lambda th: -loglik_cum_poisson(th, init_env, tau0)
    res = minimize_scalar(obj, bounds=bounds, method="bounded",
                          options=dict(xatol=xatol, maxiter=maxiter))
    theta_hat = float(res.x)
    ll_hat    = -float(res.fun)

    # 2) AIC/BIC (k=1)
    k, n = 1, len(y_cum)
    AIC = 2*k - 2*ll_hat
    BIC = k*np.log(n) - 2*ll_hat

    # 3) 95% CI by likelihood ratio: ell(theta) >= ell_hat - 1.92
    target = ll_hat - 1.92

    # 격자로 좌/우 브라켓 찾기
    lo, hi = bounds
    grid = np.linspace(lo, hi, grid_n)
    ll_vals = np.array([loglik_cum_poisson(th, init_env, tau0) for th in grid])

    # 좌측 경계: (ll - target) 부호가 +에서 -로 넘어가는 지점 탐색
    g = ll_vals - target
    # hat 인덱스 (가장 높은 우도 근처)
    i_hat = int(np.argmin((grid - theta_hat)**2))

    # 왼쪽
    theta_low = lo
    for i in range(i_hat, 1, -1):
        if g[i-1] >= 0 and g[i] < 0:  # 교차 구간
            theta_low = brentq(lambda x: loglik_cum_poisson(x, init_env, tau0) - target,
                               grid[i-1], grid[i])
            break

    # 오른쪽
    theta_high = hi
    for i in range(i_hat, len(grid)-1):
        if g[i] >= 0 and g[i+1] < 0:  # 교차 구간
            theta_high = brentq(lambda x: loglik_cum_poisson(x, init_env, tau0) - target,
                                grid[i], grid[i+1])
            break

    return {
        "theta_hat": theta_hat,
        "theta_low": float(theta_low),
        "theta_high": float(theta_high),
        "logLik_hat": ll_hat,
        "AIC": AIC,
        "BIC": BIC,
        "bounds": bounds,
        "target_logLik": target
    }

# ===== 실행 예시 =====
init_env = 10
tau0     = 140
bounds   = (0.1, 6.0)
grid_n = 400
res = fit_theta_cum_with_CI(init_env=init_env, tau0=tau0, bounds=bounds, grid_n = grid_n)
print(
    f"theta_hat={res['theta_hat']:.3f}  "
    f"95% CI=({res['theta_low']:.3f}, {res['theta_high']:.3f})  "
    f"logLik={res['logLik_hat']:.2f}  AIC={res['AIC']:.2f}  BIC={res['BIC']:.2f}"
)

# ===== 진단 플롯(누적 관측 vs 예측) =====
lam_cum_hat = cum_from_theta(res["theta_hat"], init_env, tau0)
obs_cum = y_cum[:len(lam_cum_hat)]

plt.figure(figsize=(7.2, 4.2))
plt.plot(obs_cum, "o-", label="Observed cumulative")
plt.plot(lam_cum_hat, "s-", label=f"Model cumulative (θ={res['theta_hat']:.2f})")
plt.xlabel("Month index")
plt.ylabel("Cumulative HAI")
plt.title("Cumulative Poisson MLE fit (with 95% CI computed by profile LR)")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

# CI해결 Profile likelihood plot (diagnose why CI hits bound)
import numpy as np
import matplotlib.pyplot as plt

# 이미 위에서 계산된 것들:
# res = fit_theta_cum_with_CI(...)
# init_env, tau0, bounds
# loglik_cum_poisson(theta, init_env, tau0)

theta_hat = res["theta_hat"]
ll_hat    = res["logLik_hat"]
target    = res["target_logLik"]  # = ll_hat - 1.92
lo, hi    = res["bounds"]

# grid (진단용: 조금 촘촘히)
grid_n_plot = 250
theta_grid = np.linspace(lo, hi, grid_n_plot)
ll_grid = np.array([loglik_cum_poisson(th, init_env, tau0) for th in theta_grid])

# (핵심 진단 1) grid에서의 최대점이 theta_hat이랑 같은지 확인
imax = np.argmax(ll_grid)
theta_grid_hat = theta_grid[imax]
ll_grid_hat = ll_grid[imax]

print("=== CHECK: hat from optimizer vs hat from grid ===")
print(f"optimizer theta_hat = {theta_hat:.6f}, ll_hat = {ll_hat:.6f}")
print(f"grid      theta*    = {theta_grid_hat:.6f}, ll*   = {ll_grid_hat:.6f}")

# (핵심 진단 2) 왼쪽 bound에서 target을 이미 넘는지 확인 -> 그러면 CI가 lo에 붙을 수밖에 없음
print("\n=== CHECK: left bound behavior ===")
print(f"ll(lo={lo}) = {ll_grid[0]:.6f}, target = {target:.6f}, ll(lo)-target = {ll_grid[0]-target:.6f}")
print(f"ll(hi={hi}) = {ll_grid[-1]:.6f}, target = {target:.6f}, ll(hi)-target = {ll_grid[-1]-target:.6f}")

# -2 log Lambda 형태(일반적으로 이걸로 CI 판단)
# LR(θ) = 2*(ll_hat - ll(θ))  <= 3.84 가 95% CI
LR = 2*(ll_hat - ll_grid)

plt.figure(figsize=(7.6,4.6))
plt.plot(theta_grid, LR)
plt.axhline(3.84, linestyle="--", linewidth=1.2)   # chi-square 0.95, df=1
plt.axvline(theta_hat, linestyle="--", linewidth=1.2)

# 네 코드가 찾은 CI
plt.axvline(res["theta_low"], linestyle=":", linewidth=2)
plt.axvline(res["theta_high"], linestyle=":", linewidth=2)

plt.xlabel("theta")
plt.ylabel(r"LR(theta) = 2*(ll_hat - ll(theta))")
plt.title("Profile likelihood check (95% CI where LR <= 3.84)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# logLik 그래프도 같이 (직관용)
plt.figure(figsize=(7.6,4.6))
plt.plot(theta_grid, ll_grid)
plt.axhline(target, linestyle="--", linewidth=1.2)  # ll_hat - 1.92
plt.axvline(theta_hat, linestyle="--", linewidth=1.2)
plt.axvline(res["theta_low"], linestyle=":", linewidth=2)
plt.axvline(res["theta_high"], linestyle=":", linewidth=2)
plt.xlabel("theta")
plt.ylabel("logLik")
plt.title("Profile log-likelihood (target = ll_hat - 1.92)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% 다시만든버전







# 누적 Poisson MLE for theta + 95% CI (profile likelihood)  [FIXED + plots]
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, brentq
from scipy.special import gammaln

# ===== 관측(월별)과 누적 =====
y_month = np.array([5,2,0,2,1,1,2,2,6,1,1,0,2,1,1,2,5,2,2], dtype=float)
y_cum   = np.cumsum(y_month)

def cum_from_theta(theta, init_env, tau0):
    """θ -> 월별 예측 -> 누적 예측(관측 길이에 맞춰 자르기)"""
    days, daily_inc, monthly_df, comp_df = simulate_theta(theta, init_env, tau0)
    lam_month = monthly_df["NewHAI_month"].to_numpy(dtype=float)
    lam_cum   = np.cumsum(lam_month)
    m = min(len(y_cum), len(lam_cum))
    return lam_cum[:m]

def loglik_cum_poisson(theta, init_env, tau0):
    """누적 Poisson 로그우도: sum[y log μ - μ - log y!]"""
    mu = np.clip(cum_from_theta(theta, init_env, tau0), 1e-12, None)
    y  = y_cum[:len(mu)]
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))

def fit_theta_cum_with_CI(init_env=10, tau0=140, bounds=(0.1, 6.0),
                          xatol=1e-3, maxiter=800, grid_n=500,
                          auto_expand=True):
    """
    θ MLE + 95% CI(프로파일 우도).
    - 교정 포인트:
      (A) 왼쪽 CI 교차조건: g[i-1] < 0 and g[i] >= 0  (음수→양수)
      (B) CI가 bounds 밖이면 bounds 자동확장(선택)
    """
    lo, hi = bounds

    # 1) MLE
    obj = lambda th: -loglik_cum_poisson(th, init_env, tau0)
    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                          options=dict(xatol=xatol, maxiter=maxiter))
    theta_hat = float(res.x)
    ll_hat    = -float(res.fun)

    # 2) AIC/BIC (k=1)
    k, n = 1, len(y_cum)
    AIC = 2*k - 2*ll_hat
    BIC = k*np.log(n) - 2*ll_hat

    # 3) 95% CI by LR: ll(theta) >= ll_hat - 1.92
    target = ll_hat - 1.92

    def profile_on_grid(lo, hi, grid_n):
        grid = np.linspace(lo, hi, grid_n)
        ll_vals = np.array([loglik_cum_poisson(th, init_env, tau0) for th in grid])
        g = ll_vals - target
        i_hat = int(np.argmin((grid - theta_hat)**2))
        return grid, ll_vals, g, i_hat

    # (선택) bounds가 너무 좁아서 CI가 밖에 있으면 자동 확장
    # 목표: g(lo)<0, g(i_hat)>0, g(hi)<0 형태가 되게(양쪽에서 target 아래로 내려가야 CI가 잡힘)
    if auto_expand:
        for _ in range(6):  # 최대 6번만 확장
            grid, ll_vals, g, i_hat = profile_on_grid(lo, hi, grid_n)
            if (g[0] < 0) and (g[i_hat] > 0) and (g[-1] < 0):
                break
            # 왼쪽이 아직 target 위면(lo에서 g>=0) -> 더 왼쪽으로 확장
            if g[0] >= 0:
                lo = max(1e-6, lo * 0.5)
            # 오른쪽이 아직 target 위면(hi에서 g>=0) -> 더 오른쪽으로 확장
            if g[-1] >= 0:
                hi = hi * 1.5
        # 최종 grid 재계산
        grid, ll_vals, g, i_hat = profile_on_grid(lo, hi, grid_n)
    else:
        grid, ll_vals, g, i_hat = profile_on_grid(lo, hi, grid_n)

    # 4) CI root 찾기
    # 왼쪽: (음수 -> 양수)로 바뀌는 구간을 찾아야 함  ✅ (이게 네 코드의 핵심 버그)
    theta_low = lo
    found_left = False
    for i in range(i_hat, 0, -1):
        if (g[i-1] < 0) and (g[i] >= 0):
            theta_low = brentq(lambda x: loglik_cum_poisson(x, init_env, tau0) - target,
                               grid[i-1], grid[i])
            found_left = True
            break

    # 오른쪽: (양수 -> 음수)
    theta_high = hi
    found_right = False
    for i in range(i_hat, len(grid)-1):
        if (g[i] >= 0) and (g[i+1] < 0):
            theta_high = brentq(lambda x: loglik_cum_poisson(x, init_env, tau0) - target,
                                grid[i], grid[i+1])
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

# ===== 실행 =====
init_env = 90
tau0     = 140
bounds   = (3, 4)
grid_n   = 101

res = fit_theta_cum_with_CI(init_env=init_env, tau0=tau0, bounds=bounds, grid_n=grid_n, auto_expand=True)

print(
    f"theta_hat={res['theta_hat']:.3f}  "
    f"95% CI=({res['theta_low']:.3f}, {res['theta_high']:.3f})  "
    f"logLik={res['logLik_hat']:.2f}  AIC={res['AIC']:.2f}  BIC={res['BIC']:.2f}"
)
print("bounds actually used:", res["bounds_used"],
      "| found_left:", res["found_left"], "| found_right:", res["found_right"])
# %%
# ===== (1) 진단 플롯: 누적 관측 vs 예측 =====
# ===== (1) 진단 플롯: 누적 관측 vs 예측 =====
lam_cum_hat = cum_from_theta(res["theta_hat"], init_env, tau0)
obs_cum = y_cum[:len(lam_cum_hat)]

n = len(lam_cum_hat)
x = np.arange(n)

plt.figure(figsize=(9,6))
plt.plot(x, obs_cum, "o-",color = 'black', label="Observed data")
plt.plot(x, lam_cum_hat, "s-",color='orange', label=f"SM cumulative (beta_SM={res['theta_hat']:.4f})")


plt.ylabel("Cumulative HAI", fontsize=18)
plt.legend(fontsize=10)
ax = plt.gca()

# 아까 세팅
major_idx = np.arange(0, n, 6)
minor_idx = np.arange(0, n, 3)

ax.set_xticks(major_idx)
ax.set_xticks(minor_idx, minor=True)
date_labels = pd.date_range(start="2017-01-01", periods=n, freq="MS")
date_str = [d.strftime("%Y-%m-%d") for d in date_labels]

ax.set_xticks(major_idx)
ax.set_xticks(minor_idx, minor=True)
ax.set_xticklabels([date_str[i] for i in major_idx], fontsize=18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# 축 시작


# grid
ax.grid(True, which='major', axis='x', linestyle='-')
ax.grid(True, which='minor', axis='x', linestyle='--')
ax.grid(True, axis='y', linestyle='-')

plt.legend(fontsize=20)
plt.tight_layout()
plt.show()
# ===== (2) 프로파일 log-likelihood 플롯 =====
grid = res["grid"]
ll_vals = res["ll_vals"]
target = res["target_logLik"]

plt.figure(figsize=(7.2, 4.2))
plt.plot(grid, ll_vals, lw=2)
plt.axhline(target, ls="--", label="target = ll_hat - 1.92")
plt.axvline(res["theta_hat"], ls="--", label="theta_hat")
plt.axvline(res["theta_low"], ls=":", label="CI low")
plt.axvline(res["theta_high"], ls=":", label="CI high")
plt.xlabel("theta")
plt.ylabel("logLik")
plt.title("Profile log-likelihood (95% CI by LR)")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

# ===== (3) LR(theta) = 2*(ll_hat - ll(theta)) 플롯 =====
ll_hat = res["logLik_hat"]
LR = 2*(ll_hat - ll_vals)

plt.figure(figsize=(7.2, 4.2))
plt.plot(grid, LR, lw=2)
plt.axhline(3.84, ls="--", label="3.84 (chi2_1, 0.95)")
plt.axvline(res["theta_hat"], ls="--", label="theta_hat")
plt.axvline(res["theta_low"], ls=":", label="CI low")
plt.axvline(res["theta_high"], ls=":", label="CI high")
plt.xlabel("theta")
plt.ylabel("LR(theta)")
plt.title("Profile likelihood check (95% CI where LR <= 3.84)")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()


# %%
