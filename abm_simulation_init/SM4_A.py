
# %%누적








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


# %% ================== Step4: 누적 Poisson MLE + 95% CI ==================

# --- 설정 ---
abm_csv  = "../result/interv_prob_transmission_summary_A240_0.01-0.08.csv"   # ABM 요약 csv 경로
init_env = 2
tau0     = 40
theta_min, theta_max = 0.5,6       # θ 탐색 구간
beta_list = [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07]


# --- ABM CSV 읽고 mean/std 벡터 만들기 ---
df_abm = pd.read_csv(abm_csv)
df_abm["mean_vec"] = df_abm["mean"].apply(lambda s: np.array(ast.literal_eval(s), dtype=float))
df_abm["std_vec"]  = df_abm["std"].apply(lambda s: np.array(ast.literal_eval(s), dtype=float))

# 월 index (2017-01부터 n개월)
n_months = len(df_abm["mean_vec"].iloc[0])
start_month = "2017-01"
months = pd.period_range(start_month, periods=n_months, freq="M").to_timestamp()

def model_monthly_and_cum(theta, init_env=init_env, tau0=tau0):
    days, daily_inc, monthly_df, comp_df = simulate_theta(theta, init_env, tau0)
    mdf = monthly_df.copy()
    mdf["month"] = pd.to_datetime(mdf["month"].astype(str))
    mdf = mdf.set_index("month")
    monthly = np.array([mdf["NewHAI_month"].get(m, 0.0) for m in months])
    cum     = np.cumsum(monthly)
    return monthly, cum

def negloglik_theta_cum_poisson(theta, cum_obs, init_env, tau0):
    _, cum_model = model_monthly_and_cum(theta, init_env=init_env, tau0=tau0)
    lam = np.clip(cum_model, 1e-12, None)
    y   = np.asarray(cum_obs, dtype=float)
    # -log L(θ) = Σ [ λ_t(θ) - y_t log λ_t(θ) ] + const   (const는 θ와 무관하여 생략)
    nll = np.sum(lam - y * np.log(lam))
    return nll

# --- NEW: 프로파일 우도 95% CI ---
def ci95_profile_theta(cum_obs, init_env, tau0, theta_hat, nll_hat,
                       bounds=(1, 6.0), grid_n=200):
    """
    -2 log Λ(θ) ~ χ²_1(0.95)=3.84  ⇒  NLL(θ) = NLL(θ̂) + 1.92
    """
    thr = nll_hat + 1.92
    a, b = bounds
    grid = np.linspace(a, b, grid_n)
    vals = np.array([negloglik_theta_cum_poisson(th, cum_obs, init_env, tau0) for th in grid])
    g = vals - thr

    # θ̂ 위치
    i_hat = np.searchsorted(grid, theta_hat)

    # 왼쪽 경계
    left = a
    for i in range(i_hat, 0, -1):
        if g[i-1] > 0 and g[i] <= 0:
            left = brentq(lambda x: negloglik_theta_cum_poisson(x, cum_obs, init_env, tau0) - thr,
                          grid[i-1], grid[i])
            break

    # 오른쪽 경계
    right = b
    for i in range(i_hat, len(grid)-1):
        if g[i] <= 0 and g[i+1] > 0:
            right = brentq(lambda x: negloglik_theta_cum_poisson(x, cum_obs, init_env, tau0) - thr,
                           grid[i], grid[i+1])
            break

    return float(left), float(right)

def fit_theta_cum_poisson_for_one(beta_abm, y_mean):
    """
    (변경) θ̂, NLL_min과 함께 95% CI [θ_low, θ_high]까지 반환
    """
    cum_obs = np.cumsum(y_mean)
    theta_grid = np.linspace(theta_min, theta_max, 300)
    vals = np.array([negloglik_theta_cum_poisson(th, cum_obs, init_env, tau0) for th in theta_grid])

    idx = vals.argmin()
    theta_hat = float(theta_grid[idx])
    nll_min   = float(vals[idx])

    # --- NEW: 95% CI (프로파일) ---
    theta_low, theta_high = ci95_profile_theta(
        cum_obs, init_env, tau0, theta_hat, nll_min,
        bounds=(theta_min, theta_max), grid_n=200
    )
    return theta_hat, theta_low, theta_high, nll_min
# %%
# --- 루프 실행: 각 β_ABM별 θ̂, CI 출력/저장 ---
results = []
for b in beta_list:
    sub = df_abm.loc[np.isclose(df_abm["beta"], b)]
    if sub.empty:
        print(f"[경고] beta={b} 인 행이 CSV에 없음, 스킵")
        continue

    row = sub.iloc[0]
    beta_abm = float(row["beta"])
    y_mean   = row["mean_vec"]

    theta_hat, theta_low, theta_high, nll_min = fit_theta_cum_poisson_for_one(beta_abm, y_mean)
    print(f"=== CUM-Poisson MLE for beta_ABM = {beta_abm:0.3f} ===")
    print(f"  theta_hat = {theta_hat:.4f}, 95% CI = [{theta_low:.4f}, {theta_high:.4f}], NLL_cum_min = {nll_min:.2f}")

    results.append({
        "beta_abm": beta_abm,
        "theta_hat": theta_hat,
        "theta_low": theta_low,
        "theta_high": theta_high,
        "neg_loglik_cum_min": nll_min,
        "init_env": init_env,
        "tau0": tau0,
    })
# %%
# 저장
df_res = pd.DataFrame(results)
os.makedirs("sm_fit", exist_ok=True)
out_csv = "sm_fit/theta_pairs_subset_cumPoisson_A240.csv"
df_res.to_csv(out_csv, index=False, encoding="utf-8")
print("\n저장 완료 →", out_csv)
print(df_res.head())

# %%






# %% 비교 (누적 버전)
theta_hat = 5.4523 # 피팅 결과

days, daily_inc_hat, monthly_hat_df, comp_hat_df = simulate_theta(
    theta_hat,
    init_env,
    tau0
)

# 1) ABM 쪽: beta=0.10 행 뽑기  (지금은 0.100으로 되어 있음)
beta_test = 0.07
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
csv_path = Path("sm_fit/theta_pairs_subset_cumPoisson_A240.csv")
out_png  = Path("sm_fit/step4_betaabm_to_thetasm_plotA240.png")

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










# %% ===================== imports =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from matplotlib.colors import Normalize

# (선택) 3D surface
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# %% ===================== load data =====================
# 너 파일 경로로 바꿔줘
csv_path = "sm_fit/theta_pairs_subset_cumPoisson_A9140.csv"
df = pd.read_csv(csv_path)

# 필요한 컬럼 확인
need_cols = ["beta_abm", "theta_hat", "init_env", "tau0"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV에 컬럼이 부족함: {missing}\n현재 컬럼: {df.columns.tolist()}")

# 혹시 문자열로 읽혔으면 numeric으로
for c in need_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=need_cols).copy()

# %% ===================== helper: heatmap for each tau0 =====================
def plot_heatmap_for_tau(df, tau0, grid_n_beta=80, grid_n_init=80, method="linear"):
    sub = df[df["tau0"] == tau0].copy()
    if sub.empty:
        print(f"[skip] tau0={tau0}: 데이터 없음")
        return

    # 축: beta, init_env
    xb = sub["beta_abm"].values
    yi = sub["init_env"].values
    zi = sub["theta_hat"].values

    # grid
    xb_lin = np.linspace(xb.min(), xb.max(), grid_n_beta)
    yi_lin = np.linspace(yi.min(), yi.max(), grid_n_init)
    XB, YI = np.meshgrid(xb_lin, yi_lin)

    # interpolate θ on grid
    Z = griddata(points=np.c_[xb, yi], values=zi, xi=(XB, YI), method=method)

    # 그림
    plt.figure(figsize=(7, 5))
    # NaN은 흰색으로 보이니, 데이터가 sparse면 method='nearest'도 추천
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[xb_lin.min(), xb_lin.max(), yi_lin.min(), yi_lin.max()],
    )
    plt.colorbar(im, label="theta_hat")
    plt.scatter(xb, yi, s=40, edgecolor="k")
    plt.xlabel("beta_abm")
    plt.ylabel("init_env")
    plt.title(f"theta_hat surface (heatmap) at tau0={tau0}  | interp={method}")
    plt.tight_layout()
    plt.show()

# %% ===================== helper: 3D surface for each tau0 (optional) =====================
def plot_3d_surface_for_tau(df, tau0, grid_n_beta=60, grid_n_init=60, method="linear"):
    sub = df[df["tau0"] == tau0].copy()
    if sub.empty:
        print(f"[skip] tau0={tau0}: 데이터 없음")
        return

    xb = sub["beta_abm"].values
    yi = sub["init_env"].values
    zi = sub["theta_hat"].values

    xb_lin = np.linspace(xb.min(), xb.max(), grid_n_beta)
    yi_lin = np.linspace(yi.min(), yi.max(), grid_n_init)
    XB, YI = np.meshgrid(xb_lin, yi_lin)
    Z = griddata(points=np.c_[xb, yi], values=zi, xi=(XB, YI), method=method)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # surface
    ax.plot_surface(XB, YI, Z, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.85)
    # points
    ax.scatter(xb, yi, zi, s=35, edgecolor="k")
    ax.set_xlabel("beta_abm")
    ax.set_ylabel("init_env")
    ax.set_zlabel("theta_hat")
    ax.set_title(f"3D surface at tau0={tau0} | interp={method}")
    plt.tight_layout()
    plt.show()

# %% ===================== 1) 교수님 스타일: tau0별 단면(heatmap) =====================
taus = sorted(df["tau0"].unique().tolist())
print("available tau0:", taus)

# tau0=70,140이 있으면 우선 그려줌
for t in [70, 140]:
    if t in taus:
        plot_heatmap_for_tau(df, t, method="linear")
    else:
        print(f"tau0={t} 없음. 있는 것 중에서 골라야 함: {taus}")

# (필요하면 3D도)
# for t in [70, 140]:
#     if t in taus:
#         plot_3d_surface_for_tau(df, t, method="linear")

# %% ===================== 2) 교수님 코멘트 핵심: 이니셜도 연속축으로 '쌓기' (collapse 시도) =====================
# 아이디어: E0 = init_env * w(tau0)
# w(tau0)를 어떻게 둘지 정답은 없고, "겹쳐지는지"를 보고 고르는 방식이 현실적임.
# 가장 단순: w(tau0)=1  (그냥 init_env만 쓰기)
# 그 다음: w(tau0)=tau0 / tau_ref  (예: 140 기준)
tau_ref = 140.0

df2 = df.copy()

# 옵션 A: 그냥 init_env
df2["E0_A"] = df2["init_env"]

# 옵션 B: tau0 비례 가중
df2["E0_B"] = df2["init_env"] * (df2["tau0"] / tau_ref)

# (원하면) 옵션 C: "청소까지 남은 시간" 기반은 너 모델 정의 따라 바뀜 -> 여기선 예시만
# df2["E0_C"] = df2["init_env"] * (1.0 + 0.0*(df2["tau0"]/tau_ref))

def plot_collapse(df2, E0_col, title):
    plt.figure(figsize=(7,5))
    # tau0별로 색을 다르게 찍어서 "겹치나" 확인
    for t, sub in df2.groupby("tau0"):
        plt.scatter(sub["beta_abm"], sub[E0_col], s=35, label=f"tau0={int(t)}", alpha=0.85)

    plt.xlabel("beta_abm")
    plt.ylabel(E0_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# (β, E0) 평면에 점들이 tau0별로 잘 겹치면 -> 이니셜을 1개 축약변수로 줄일 가능성↑
plot_collapse(df2, "E0_A", "Collapse check: (beta_abm, E0=init_env)")
plot_collapse(df2, "E0_B", "Collapse check: (beta_abm, E0=init_env * (tau0/140))")

# %% ===================== 3) 최종 그림: (beta, E0) -> theta surface (collapsed) =====================
def plot_theta_surface_collapsed(df2, E0_col, grid_n_beta=80, grid_n_E0=80, method="linear"):
    xb = df2["beta_abm"].values
    e0 = df2[E0_col].values
    th = df2["theta_hat"].values

    xb_lin = np.linspace(xb.min(), xb.max(), grid_n_beta)
    e0_lin = np.linspace(e0.min(), e0.max(), grid_n_E0)
    XB, E0 = np.meshgrid(xb_lin, e0_lin)

    Z = griddata(points=np.c_[xb, e0], values=th, xi=(XB, E0), method=method)

    plt.figure(figsize=(7,5))
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[xb_lin.min(), xb_lin.max(), e0_lin.min(), e0_lin.max()],
    )
    plt.colorbar(im, label="theta_hat")
    plt.scatter(xb, e0, s=35, edgecolor="k")
    plt.xlabel("beta_abm")
    plt.ylabel(E0_col)
    plt.title(f"Collapsed surface: theta_hat = f(beta_abm, {E0_col}) | interp={method}")
    plt.tight_layout()
    plt.show()

plot_theta_surface_collapsed(df2, "E0_A", method="linear")
plot_theta_surface_collapsed(df2, "E0_B", method="linear")


# %%
