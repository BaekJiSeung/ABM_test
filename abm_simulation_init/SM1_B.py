# %%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln

# ---- 관측 데이터 ----
y_month = np.array([1,2,2,2,1,0,0,3,2,2,2,0,3,0,1,2,0,1,4,5,4,2,4,1,0,0,1,0,1,1,0,3,1,0,3,0], dtype=float)
y_cum   = np.cumsum(y_month)

# ---- 기간 정의용 (2021-01 ~ 2023-12) ----
monthly_PI = pd.Series({
    "2021-01":2,"2021-02":2,"2021-03":3,"2021-04":1,"2021-05":0,"2021-06":3,
    "2021-07":3,"2021-08":5,"2021-09":2,"2021-10":5,"2021-11":0,"2021-12":2,
    "2022-01":4,"2022-02":1,"2022-03":1,"2022-04":3,"2022-05":5,"2022-06":2,
    "2022-07":2,"2022-08":5,"2022-09":1,"2022-10":4,"2022-11":7,"2022-12":2,
    "2023-01":0,"2023-02":0,"2023-03":1,"2023-04":0,"2023-05":1,"2023-06":0,
    "2023-07":0,"2023-08":0,"2023-09":2,"2023-10":0,"2023-11":2,"2023-12":0
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
    격리 = HAI만 모아두는 칸 (P_HAI_iso만 있음)
    상태변수: P_S_sh, P_HAI_sh, P_HAI_iso, P_I, H_C, Env_C
    """

    # ---- 파라미터 ----
    C_total = 30
    C_iso   = 30
    C_sh    = C_total 

    N_H, N_E = 19, 30
    mu_S, mu_HAI, mu_I = 1/14, 1/21, 1/14  # B구간이니까 7에서 14로 바꿔었다
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
            xx=Env_C[t]
            Env_C[t] = (1-p_wash)*xx

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



# ===== 테스트 =====
theta = 3.29

init_env = 2
tau0 = 40   # 청소 후 30일 지난 시점에서 시작한다고 가정

days, daily_inc, monthly_df, comp_df = simulate_theta(theta, init_env, tau0)


# 월별 표
disp = monthly_df.copy()
disp["month"] = disp["month"].astype(str)
disp["NewHAI_month"] = disp["NewHAI_month"].map(lambda x: f"{x:.4f}")
disp["cum_NewHAI"]   = disp["cum_NewHAI"].map(lambda x: f"{x:.4f}")
print(disp.to_string(index=False))

# 컴파트먼트
plt.figure(figsize=(12,6))
for col in comp_df.columns:
    plt.plot(comp_df.index, comp_df[col], label=col)
plt.title(f"Compartments (theta={theta})")
plt.legend(ncol=3); plt.tight_layout(); plt.show()

# 누적 HAI
pd.Series(daily_inc, index=days).cumsum().plot()
plt.title("Cumulative P_HAI"); plt.show()

y = [1,2,2,2,1,0,0,3,2,2,2,0,3,0,1,2,0,1,4,5,4,2,4,1,0,0,1,0,1,1,0,3,1,0,3,0]
pd.Series(y, index=pd.period_range('2021-01', periods=len(y), freq='M').to_timestamp()).cumsum()\
  .plot(marker='o'); plt.title("Cumulative P_HAI"); plt.show()

y = [2,2,3,1,0,3,3,5,2,5,0,2,4,1,1,3,5,2,2,5,1,4,7,2,3,2,1,0,5,3,1,3,4,7,7,3]
pd.Series(y, index=pd.period_range('2021-01', periods=len(y), freq='M').to_timestamp())\
  .plot(marker='o'); plt.title("monthly P_I"); plt.show()



# 1) 모델 누적 (일 단위)
cum_model = pd.Series(daily_inc, index=days).cumsum()

# 2) 관측 누적 (월 단위)
y = [1,2,2,2,1,0,0,3,2,2,2,0,3,0,1,2,0,1,4,5,4,2,4,1,0,0,1,0,1,1,0,3,1,0,3,0]
month_idx = pd.period_range('2021-01', periods=len(y), freq='M').to_timestamp()
cum_obs = pd.Series(y, index=month_idx).cumsum()

plt.figure(figsize=(10,5))

# plot
plt.plot(cum_model.index, cum_model.values, label="Model cumulative", linewidth=2)

# 
plt.plot(cum_obs.index, cum_obs.values, 'o', label="Observed cumulative", markersize=6)

plt.title(f"model vs observed : theta={theta},init_env={init_env},tau0={tau0}")
plt.xlabel("Date")
plt.ylabel("Cumulative cases")
plt.legend()
plt.tight_layout()
plt.show()

# %%
import numpy as np
from scipy.special import gammaln   

# 관측 월별 P_HAI
y_month = np.array([1,2,2,2,1,0,0,3,2,2,2,0,3,0,1,2,0,1,4,5,4,2,4,1,0,0,1,0,1,1,0,3,1,0,3,0], dtype=float)
y_cum   = np.cumsum(y_month)

def loglik_cum_poisson(theta, init_env, tau0):
    # simulate_theta(theta, init_env, tau0) 형태라고 가정
    days, daily_inc, monthly_df, comp_df = simulate_theta(theta, init_env, tau0)

    lam_month = monthly_df["NewHAI_month"].values.astype(float)
    lam_cum   = np.cumsum(lam_month)

    m  = min(len(y_cum), len(lam_cum))
    y  = y_cum[:m]
    mu = lam_cum[:m]
    mu = np.clip(mu, 1e-9, None)

    # log P(Y=y | μ) = y log μ - μ - log(y!)
    ll = np.sum(y * np.log(mu) - mu - gammaln(y + 1))
    return ll
# %%
# ====== 그리드 정의 (예시는 대충, 직접 조정하면 됨) ======
theta_grid = np.linspace(1.0, 5.0, 81)        # β 후보
init_grid  = np.arange(0, 18, 1)       # 초기 Env_C(0)
tau_grid   = np.arange(0, 181, 10)           # 청소 후 경과일: 0,30,60,...,180

best_params = None
best_ll = -1e18

for th in theta_grid:
    
    for init_env in init_grid:
        for tau0 in tau_grid:
            ll = loglik_cum_poisson(th, init_env, tau0)
            if ll > best_ll:
                best_ll = ll
                best_params = (th, init_env, tau0)
                print(th,init_env,tau0)

best_theta, best_init, best_tau0 = best_params
print("best theta:", best_theta)
print("best init_env:", best_init)
print("best tau0:", best_tau0)
print("best loglik:", best_ll)

# 최적 파라미터로 한 번 더 돌려보기
days, daily_inc, monthly_df, comp_df = simulate_theta(best_theta, best_init, best_tau0)
print(monthly_df[["month","NewHAI_month","cum_NewHAI"]])

# ===== AIC/BIC (파라미터 3개) =====
k = 3              # theta, init_env, tau0
n = len(y_cum)     # 관측 누적 데이터 개수

aic = 2*k - 2*best_ll
bic = k*np.log(n) - 2*best_ll

print("AIC:", aic)
print("BIC:", bic)



# %%
# ===== AIC/BIC (파라미터 3개) =====
k = 3              # theta, init_env, tau0
n = len(y_cum)     # 관측 누적 데이터 개수

def aic_bic(theta, init_env, tau0, k=3):
    """
    임의의 (theta, init_env, tau0)를 넣으면
    loglik, AIC, BIC를 반환.
    k = 추정 파라미터 개수 (기본 3개)
    """
    ll = loglik_cum_poisson(theta, init_env, tau0)

    aic = 2*k - 2*ll
    bic = k*np.log(n) - 2*ll
    return ll, aic, bic


# ===== 사용 예시 =====
theta_test   = 3.35
init_env_test = 10
tau0_test    = 140    # 마지막 청소 직후라고 가정

ll, aic, bic = aic_bic(theta_test, init_env_test, tau0_test)
print("loglik:", ll)
print("AIC   :", aic)
print("BIC   :", bic)


# %%
