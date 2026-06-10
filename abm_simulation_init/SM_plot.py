# %% ================== 준비(그대로 사용) ==================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
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
    격리=HAI만 모아두는 칸(P_HAI_iso만). 상태: P_S_sh, P_HAI_sh, P_HAI_iso, P_I, H_C, Env_C
    """
    # ---- 파라미터 ----
    C_total = 30; C_iso = 30; C_sh = C_total
    N_H, N_E = 19, 30
    mu_S, mu_HAI, mu_I = 1/7, 1/14, 1/7
    p_wash = 0.90
    contacts_per_day = 108
    dt = 1.0 / contacts_per_day
    deep_clean_period = 180
    iso_factor = 0.75
    isol_time = 14.0
    sigma = 1.0 / isol_time   # shared HAI → iso HAI
    isol_time = 14.0

# ABM: randint(1, isolation_time), so delay = 1,...,13
    mean_iso_delay = (1 + (isol_time - 1)) / 2
    
    sigma = 1.0 / mean_iso_delay
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
            Env_C[t] = (1 - p_wash)*xx

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

# %% ================== simple plot: monthly & cumulative by beta ==================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 설정
# =========================

init_env = 9
tau0 = 140

beta_list = [3.374]

# A observed monthly HAI
y_month_A = np.array(
    [5,2,0,2,1,1,2,2,6,1,1,0,2,1,1,2,5,2,2],
    dtype=float
)

# =========================
# 1) 월별 plot
# =========================

plt.figure(figsize=(10, 5.5))

for beta in beta_list:

    days, daily_inc, monthly_df, comp_df = simulate_theta(
        beta,
        init_env,
        tau0
    )

    plot_df = monthly_df.copy()
    plot_df["month"] = pd.to_datetime(plot_df["month"].astype(str))

    plt.plot(
        plot_df["month"],
        plot_df["NewHAI_month"],
        "o-",
        linewidth=2,
        markersize=5,
        label=f"beta_SM={beta:g}"
    )

# observed는 점만
months_obs = pd.period_range(
    "2017-01",
    periods=len(y_month_A),
    freq="M"
).to_timestamp()

plt.plot(
    months_obs,
    y_month_A,
    "o",
    color="black",
    markersize=7,
    label="Observed monthly"
)

plt.xlabel("Month", fontsize=14)
plt.ylabel("Monthly HAI", fontsize=14)
plt.title("Monthly HAI by SM beta", fontsize=16)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 2) 누적 plot
# =========================

plt.figure(figsize=(10, 5.5))

for beta in beta_list:

    days, daily_inc, monthly_df, comp_df = simulate_theta(
        beta,
        init_env,
        tau0
    )

    plot_df = monthly_df.copy()
    plot_df["month"] = pd.to_datetime(plot_df["month"].astype(str))

    plt.plot(
        plot_df["month"],
        plot_df["cum_NewHAI"],
        "o-",
        linewidth=2,
        markersize=5,
        label=f"beta_SM={beta:g}"
    )

# observed cumulative는 점만
y_cum_A = np.cumsum(y_month_A)

plt.plot(
    months_obs,
    y_cum_A,
    "o",
    color="black",
    markersize=7,
    label="Observed cumulative"
)

plt.xlabel("Month", fontsize=14)
plt.ylabel("Cumulative HAI", fontsize=14)
plt.title("Cumulative HAI by SM beta", fontsize=16)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%



# %% compare isolation-time handling in SM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_theta_iso_compare(
    beta,
    init_env,
    tau0,
    iso_mode="sm_original",   # "sm_original" or "abm_mean"
    monthly_PI=monthly_PI,
    pi_dates=PI_dates
):
    """
    iso_mode:
    - "sm_original": sigma = 1 / 14
    - "abm_mean":    ABM uniform delay 평균에 맞춰 sigma = 1 / 7
    """

    # ---- 파라미터 ----
    C_total = 30
    C_iso = 30
    C_sh = C_total

    N_H, N_E = 19, 30

    mu_S = 1 / 7
    mu_HAI = 1 / 14
    mu_I = 1 / 7

    p_wash = 0.90
    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    iso_factor = 0.75

    isol_time = 14.0

    if iso_mode == "sm_original":
        sigma = 1.0 / isol_time

    elif iso_mode == "abm_mean":
        # ABM: randint(1, isolation_time) = 1,...,13
        # mean delay = 7
        mean_iso_delay = (1 + (isol_time - 1)) / 2
        sigma = 1.0 / mean_iso_delay

    else:
        raise ValueError("iso_mode must be 'sm_original' or 'abm_mean'")

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
            Env_C[t] = (1 - p_wash) * xx

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

            lam_HP_sh = beta * (HC / N_H)

            lam_PH = beta * (
                (PH_sh + iso_factor * PH_iso + PI) / B_tot
            )

            lam_EH = beta * (EC / N_E)
            lam_HE = beta * (HC / N_H)

            hai_sh = lam_HP_sh * PS_sh * dt

            # 핵심 비교 지점
            move_HA = sigma * PH_sh * dt

            outS_sh = mu_S * PS_sh * dt
            outH_sh = mu_HAI * PH_sh * dt
            outH_iso = mu_HAI * PH_iso * dt
            outI = mu_I * PI * dt

            leaving = outS_sh + outH_sh + outH_iso + outI
            total_P = PS_sh + PH_sh + PH_iso + PI

            AS_tot = max(0.0, C_total - (total_P - leaving))
            AS_sh = AS_tot

            PS_sh += AS_sh - outS_sh - hai_sh
            PH_sh += hai_sh - outH_sh - move_HA
            PH_iso += move_HA - outH_iso
            PI += -outI

            PS_sh = np.clip(PS_sh, 0, C_sh)
            PH_sh = np.clip(PH_sh, 0, C_sh)
            PH_iso = np.clip(PH_iso, 0, C_iso)
            PI = np.clip(PI, 0, C_total)

            new_H = (lam_PH + lam_EH) * (N_H - HC) * dt
            HC = (HC + new_H) * (1 - p_wash)
            HC = np.clip(HC, 0, N_H)

            EC += lam_HE * (N_E - EC)*dt
            EC = np.clip(EC, 0, N_E)

            NewHAI_day[t] += hai_sh

        if t < T - 1:
            P_S_sh[t+1] = PS_sh
            P_HAI_sh[t+1] = PH_sh
            P_HAI_iso[t+1] = PH_iso
            P_I[t+1] = PI
            H_C[t+1] = HC
            Env_C[t+1] = EC

    monthly = (
        pd.DataFrame({"date": days, "NewHAI": NewHAI_day})
        .groupby(lambda i: days[i].to_period("M"))["NewHAI"]
        .sum()
        .reset_index()
        .rename(columns={"index": "month", "NewHAI": "NewHAI_month"})
    )

    # 위 groupby lambda가 불편하면 안전하게 다시 구성
    df_day = pd.DataFrame({
        "date": days,
        "NewHAI": NewHAI_day
    })

    monthly = (
        df_day.groupby(df_day["date"].dt.to_period("M"))["NewHAI"]
        .sum()
        .reset_index()
        .rename(columns={"date": "month", "NewHAI": "NewHAI_month"})
    )

    monthly["cum_NewHAI"] = monthly["NewHAI_month"].cumsum()

    comp_df = pd.DataFrame({
        "date": days,
        "P_S_sh": P_S_sh,
        "P_HAI_sh": P_HAI_sh,
        "P_HAI_iso": P_HAI_iso,
        "P_I": P_I,
        "H_C": H_C,
        "Env_C": Env_C,
    }).set_index("date")

    return days, NewHAI_day, monthly, comp_df
# %%
# %% run comparison

beta = 3.374
init_env = 9
tau0 = 140

days1, daily1, monthly1, comp1 = simulate_theta_iso_compare(
    beta=beta,
    init_env=init_env,
    tau0=tau0,
    iso_mode="sm_original"
)

days2, daily2, monthly2, comp2 = simulate_theta_iso_compare(
    beta=beta,
    init_env=init_env,
    tau0=tau0,
    iso_mode="abm_mean"
)

m1 = monthly1.copy()
m2 = monthly2.copy()

m1["month"] = pd.to_datetime(m1["month"].astype(str))
m2["month"] = pd.to_datetime(m2["month"].astype(str))

# =========================
# Monthly comparison
# =========================

plt.figure(figsize=(10, 5.5))

plt.plot(
    m1["month"],
    m1["NewHAI_month"],
    "o-",
    linewidth=2,
    label="SM original: sigma=1/14"
)

plt.plot(
    m2["month"],
    m2["NewHAI_month"],
    "s-",
    linewidth=2,
    label="ABM mean delay: sigma=1/7"
)

plt.xlabel("Month", fontsize=14)
plt.ylabel("Monthly HAI", fontsize=14)
plt.title("Effect of isolation-time approximation: monthly HAI", fontsize=16)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# Cumulative comparison
# =========================

plt.figure(figsize=(10, 5.5))

plt.plot(
    m1["month"],
    m1["cum_NewHAI"],
    "o-",
    linewidth=2,
    label="SM original: sigma=1/14"
)

plt.plot(
    m2["month"],
    m2["cum_NewHAI"],
    "s-",
    linewidth=2,
    label="ABM mean delay: sigma=1/7"
)

plt.xlabel("Month", fontsize=14)
plt.ylabel("Cumulative HAI", fontsize=14)
plt.title("Effect of isolation-time approximation: cumulative HAI", fontsize=16)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%
