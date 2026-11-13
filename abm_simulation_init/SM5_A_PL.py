
# %% 1. 기본 세팅

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar, brentq
from scipy.special import gammaln
from scipy.stats import chi2


# =========================================================
# Period A: January 2017 - July 2018
# =========================================================

monthly_PI = pd.Series({
    "2017-01": 0, "2017-02": 0, "2017-03": 1,
    "2017-04": 0, "2017-05": 1, "2017-06": 0,
    "2017-07": 0, "2017-08": 0, "2017-09": 2,
    "2017-10": 0, "2017-11": 2, "2017-12": 0,
    "2018-01": 0, "2018-02": 0, "2018-03": 0,
    "2018-04": 0, "2018-05": 1, "2018-06": 0,
    "2018-07": 0
})

PI_dates = [
    "2017-03-01",
    "2017-05-30",
    "2017-09-17",
    "2017-09-29",
    "2017-11-14",
    "2017-11-20",
    "2018-05-25"
]


def _make_AI_from_dates(pi_dates, days):

    T = len(days)
    A = np.zeros(T)

    idx_map = {
        d: i for i, d in enumerate(days)
    }

    for d in pi_dates:

        ts = pd.to_datetime(d)
        idx = idx_map.get(ts, None)

        if idx is not None:
            A[idx] += 1.0

    return A


# %% 2. Surrogate model

def simulate_theta(
    beta,
    init_env,
    tau0,
    monthly_PI=monthly_PI,
    pi_dates=PI_dates
):

    # -----------------------------------------------------
    # Parameters
    # -----------------------------------------------------

    C_total = 30
    C_iso = 30
    C_sh = C_total

    N_H = 19
    N_E = 30

    mu_S = 1 / 7
    mu_HAI = 1 / 14
    mu_I = 1 / 7

    p_wash = 0.90

    contacts_per_day = 108
    dt = 1.0 / contacts_per_day

    deep_clean_period = 180
    iso_factor = 0.75

    isol_time = 7.0
    sigma = 1.0 / isol_time

    # -----------------------------------------------------
    # Time axis
    # -----------------------------------------------------

    start = pd.Period(
        monthly_PI.index.min(),
        freq="M"
    ).to_timestamp(how="start")

    end = pd.Period(
        monthly_PI.index.max(),
        freq="M"
    ).to_timestamp(how="end")

    days = pd.date_range(
        start,
        end,
        freq="D"
    )

    T = len(days)

    # -----------------------------------------------------
    # Imported colonized admission
    # -----------------------------------------------------

    A_I_day = _make_AI_from_dates(
        pi_dates,
        days
    )

    # -----------------------------------------------------
    # State variables
    # -----------------------------------------------------

    P_S_sh = np.zeros(T)
    P_HAI_sh = np.zeros(T)
    P_HAI_iso = np.zeros(T)
    P_I = np.zeros(T)

    H_C = np.zeros(T)
    Env_C = np.zeros(T)

    NewHAI_day = np.zeros(T)

    # -----------------------------------------------------
    # Initial conditions
    # -----------------------------------------------------

    P_S_sh[0] = C_total - 1

    P_HAI_sh[0] = 0
    P_HAI_iso[0] = 0

    P_I[0] = 1

    Env_C[0] = init_env

    # -----------------------------------------------------
    # Simulation
    # -----------------------------------------------------

    for t in range(T):

        # Periodic deep cleaning
        if (
            t > 0
            and (t + tau0) % deep_clean_period == 0
        ):

            xx = Env_C[t]
            Env_C[t] = 0.1 * xx

        PS_sh = P_S_sh[t]
        PH_sh = P_HAI_sh[t]

        PH_iso = P_HAI_iso[t]

        PI = P_I[t]
        HC = H_C[t]
        EC = Env_C[t]

        # -------------------------------------------------
        # Imported colonized patients
        # -------------------------------------------------

        inc = A_I_day[t]

        if inc > 0:

            total_P = (
                PS_sh
                + PH_sh
                + PH_iso
                + PI
            )

            stay_free = max(
                0.0,
                C_total - total_P
            )

            inc_eff = min(
                inc,
                stay_free
            )

            taken = min(
                PS_sh,
                inc_eff
            )

            PS_sh -= taken
            PI += taken

        # -------------------------------------------------
        # Contact-level updates
        # -------------------------------------------------

        for _ in range(contacts_per_day):

            B_tot = max(
                PS_sh + PH_sh + PH_iso + PI,
                1e-9
            )

            # HCW -> susceptible patients
            lam_HP_sh = beta * (HC / N_H)

            # Patients -> HCW
            lam_PH = beta * (
                (
                    PH_sh
                    + iso_factor * PH_iso
                    + PI
                ) / B_tot
            )

            # Environment -> HCW
            lam_EH = beta * (EC / N_E)

            # HCW -> environment
            lam_HE = beta * (HC / N_H)

            # New HAI
            hai_sh = (
                lam_HP_sh
                * PS_sh
                * dt
            )

            # Isolation transition
            move_HA = (
                sigma
                * PH_sh
                * dt
            )

            # Discharge
            outS_sh = (
                mu_S
                * PS_sh
                * dt
            )

            outH_sh = (
                mu_HAI
                * PH_sh
                * dt
            )

            outH_iso = (
                mu_HAI
                * PH_iso
                * dt
            )

            outI = (
                mu_I
                * PI
                * dt
            )

            leaving = (
                outS_sh
                + outH_sh
                + outH_iso
                + outI
            )

            total_P = (
                PS_sh
                + PH_sh
                + PH_iso
                + PI
            )

            # Susceptible admission
            AS_tot = max(
                0.0,
                C_total - (total_P - leaving)
            )

            AS_sh = AS_tot

            # Shared patients
            PS_sh += (
                AS_sh
                - outS_sh
                - hai_sh
            )

            PH_sh += (
                hai_sh
                - outH_sh
                - move_HA
            )

            # Isolated HAI
            PH_iso += (
                move_HA
                - outH_iso
            )

            # Imported patients
            PI += 0.0 - outI

            # Clip patient states
            PS_sh = np.clip(
                PS_sh, 0, C_sh
            )

            PH_sh = np.clip(
                PH_sh, 0, C_sh
            )

            PH_iso = np.clip(
                PH_iso, 0, C_iso
            )

            PI = np.clip(
                PI, 0, C_total
            )

            # HCW contamination
            new_H = (
                (lam_PH + lam_EH)
                * (N_H - HC)
                * dt
            )

            HC = (
                HC + new_H
            ) * (1 - p_wash)

            HC = np.clip(
                HC, 0, N_H
            )

            # Environmental contamination
            EC += (
                lam_HE
                * (N_E - EC)
                * dt
            )

            EC = np.clip(
                EC, 0, N_E
            )

            # Daily HAI incidence
            NewHAI_day[t] += hai_sh

        # -------------------------------------------------
        # Save next-day states
        # -------------------------------------------------

        if t < T - 1:

            P_S_sh[t + 1] = PS_sh
            P_HAI_sh[t + 1] = PH_sh

            P_HAI_iso[t + 1] = PH_iso

            P_I[t + 1] = PI

            H_C[t + 1] = HC
            Env_C[t + 1] = EC

    # -----------------------------------------------------
    # Monthly incidence
    # -----------------------------------------------------

    df = pd.DataFrame({
        "date": days,
        "NewHAI": NewHAI_day
    })

    monthly = (
        df.groupby(
            df["date"].dt.to_period("M")
        )["NewHAI"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "date": "month",
                "NewHAI": "NewHAI_month"
            }
        )
    )

    monthly["cum_NewHAI"] = (
        monthly["NewHAI_month"].cumsum()
    )

    # -----------------------------------------------------
    # Compartment dataframe
    # -----------------------------------------------------

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
        "Env_C": Env_C

    }).set_index("date")

    return (
        days,
        NewHAI_day,
        monthly,
        comp_df
    )


# %% 3. Observed data and Poisson likelihood

# Period A monthly HAI observations

y_month = np.array([
    5, 2, 0, 2, 1, 1, 2, 2, 6,
    1, 1, 0, 2, 1, 1, 2, 5, 2, 2
], dtype=float)

y_cum = np.cumsum(y_month)


def cum_from_theta(
    theta,
    init_env,
    tau0
):

    days, daily_inc, monthly_df, comp_df = (
        simulate_theta(
            theta,
            init_env,
            tau0
        )
    )

    lam_month = monthly_df[
        "NewHAI_month"
    ].to_numpy(dtype=float)

    lam_cum = np.cumsum(
        lam_month
    )

    if len(lam_cum) != len(y_cum):

        raise ValueError(
            "Observed and simulated "
            "periods do not match."
        )

    return lam_cum


def loglik_cum_poisson(
    theta,
    init_env,
    tau0
):

    mu = np.clip(
        cum_from_theta(
            theta,
            init_env,
            tau0
        ),
        1e-12,
        None
    )

    y = y_cum

    loglik = np.sum(
        y * np.log(mu)
        - mu
        - gammaln(y + 1)
    )

    return float(loglik)


# %% 4. Step 5: MLE and profile likelihood

# Period A settings
init_env = 9
tau0 = 140

bounds = (3.2,3.5)

# Start with 101.
# Increase to 201 for a smoother plot.
grid_n = 101


# ---------------------------------------------------------
# Negative log-likelihood
# ---------------------------------------------------------

def nll(beta):

    return -loglik_cum_poisson(
        beta,
        init_env,
        tau0
    )


# ---------------------------------------------------------
# MLE
# ---------------------------------------------------------

result = minimize_scalar(
    nll,
    bounds=bounds,
    method="bounded",
    options={
        "xatol": 1e-6,
        "maxiter": 800
    }
)

if not result.success:
    raise RuntimeError(
        result.message
    )

beta_hat = float(result.x)

nll_min = float(result.fun)
ll_hat = -nll_min


# ---------------------------------------------------------
# 95% profile likelihood threshold
# ---------------------------------------------------------

# LR(beta) = 2 * (ll_hat - ll(beta))
#
# 95% CI:
# LR(beta) <= chi2.ppf(0.95, df=1)

LR_threshold = chi2.ppf(
    0.95,
    df=1
)

NLL_threshold = (
    nll_min
    + LR_threshold / 2
)


def root_func(beta):

    return (
        nll(beta)
        - NLL_threshold
    )


lo, hi = bounds

# Do not mistake the search bounds for CI endpoints.
if (
    root_func(lo) <= 0
    or root_func(hi) <= 0
):

    raise ValueError(
        "95% CI is outside the search bounds. "
        "Expand bounds and rerun."
    )


# ---------------------------------------------------------
# Find CI endpoints
# ---------------------------------------------------------

beta_low = brentq(
    root_func,
    lo,
    beta_hat,
    xtol=1e-8
)

beta_high = brentq(
    root_func,
    beta_hat,
    hi,
    xtol=1e-8
)


# ---------------------------------------------------------
# Profile likelihood grid
# ---------------------------------------------------------

beta_grid = np.linspace(
    lo,
    hi,
    grid_n
)

ll_grid = np.array([
    loglik_cum_poisson(
        beta,
        init_env,
        tau0
    )
    for beta in beta_grid
])

# Include the exact MLE
beta_grid = np.append(
    beta_grid,
    beta_hat
)

ll_grid = np.append(
    ll_grid,
    ll_hat
)

idx = np.argsort(
    beta_grid
)

beta_grid = beta_grid[idx]
ll_grid = ll_grid[idx]


# ---------------------------------------------------------
# Likelihood ratio
# ---------------------------------------------------------

LR_grid = 2 * (
    ll_hat - ll_grid
)


# ---------------------------------------------------------
# AIC and BIC
# ---------------------------------------------------------

k = 1
n = len(y_cum)

AIC = (
    2 * k
    - 2 * ll_hat
)

BIC = (
    k * np.log(n)
    - 2 * ll_hat
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("=" * 60)

print("Period A - Step 5")

print("=" * 60)

print(
    f"beta_SM MLE = {beta_hat:.6f}"
)

print(
    f"95% CI = "
    f"({beta_low:.6f}, {beta_high:.6f})"
)

print(
    f"logLik = {ll_hat:.6f}"
)

print(
    f"AIC = {AIC:.4f}"
)

print(
    f"BIC = {BIC:.4f}"
)





# %% 5. Profile log-likelihood plot

fig, ax = plt.subplots(
    figsize=(9, 6)
)

# Profile curve
ax.plot(
    beta_grid,
    ll_grid,
    color="tab:blue",
    linewidth=2.5,
    label="Profile log-likelihood"
)

# MLE point
ax.scatter(
    beta_hat,
    ll_hat,
    color="black",
    s=65,
    zorder=5,
    label="MLE"
)

# 95% threshold
ax.axhline(
    -NLL_threshold,
    color="tab:red",
    linestyle="--",
    linewidth=1.8,
    label="95% likelihood threshold"
)

# CI limits
ax.axvline(
    beta_low,
    color="gray",
    linestyle=":",
    linewidth=1.8
)

ax.axvline(
    beta_high,
    color="gray",
    linestyle=":",
    linewidth=1.8
)

# CI shading
ax.axvspan(
    beta_low,
    beta_high,
    color="tab:blue",
    alpha=0.10,
    label="95% CI"
)

ax.set_xlabel(
    r"$\beta_{\mathrm{SM}}$",
    fontsize=18
)

ax.set_ylabel(
    "Log-likelihood",
    fontsize=18
)

ax.tick_params(
    axis="both",
    labelsize=18
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.legend(
    fontsize=13,
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "periodA_betaSM_profile_likelihood.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% 6. Likelihood ratio plot

fig, ax = plt.subplots(
    figsize=(9, 6)
)

# Likelihood ratio curve
ax.plot(
    beta_grid,
    LR_grid,
    color="tab:blue",
    linewidth=2.5,
    label="Likelihood ratio"
)

# 95% threshold
ax.axhline(
    LR_threshold,
    color="tab:red",
    linestyle="--",
    linewidth=1.8,
    label="95% threshold"
)

# MLE point
ax.scatter(
    beta_hat,
    0,
    color="black",
    s=65,
    zorder=5,
    label="MLE"
)

# CI limits
ax.axvline(
    beta_low,
    color="gray",
    linestyle=":",
    linewidth=1.8
)

ax.axvline(
    beta_high,
    color="gray",
    linestyle=":",
    linewidth=1.8
)

# CI shading
ax.axvspan(
    beta_low,
    beta_high,
    color="tab:blue",
    alpha=0.10,
    label="95% CI"
)

ax.set_xlabel(
    r"$\beta_{\mathrm{SM}}$",
    fontsize=18
)

ax.set_ylabel(
    r"Negative Log-likelihood",
    fontsize=18
)

ax.tick_params(
    axis="both",
    labelsize=18
)

ax.set_ylim(
    bottom=-0.15
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3
)

ax.legend(
    fontsize=13,
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "periodA_betaSM_likelihood_ratio.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% 7. Observed vs surrogate plot

lam_cum_hat = cum_from_theta(
    beta_hat,
    init_env,
    tau0
)

obs_cum = y_cum

n = len(lam_cum_hat)
x = np.arange(n)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

# Observed data
ax.plot(
    x,
    obs_cum,
    "o-",
    color="black",
    label="Observed data"
)

# Surrogate prediction
ax.plot(
    x,
    lam_cum_hat,
    "s-",
    color="orange",
    label="Surrogate model"
)

# Date labels
major_idx = np.arange(
    0,
    n,
    6
)

minor_idx = np.arange(
    0,
    n,
    3
)

date_labels = pd.date_range(
    start="2017-01-01",
    periods=n,
    freq="MS"
)

date_str = [
    d.strftime("%Y-%m-%d")
    for d in date_labels
]

ax.set_xticks(
    major_idx
)

ax.set_xticks(
    minor_idx,
    minor=True
)

ax.set_xticklabels(
    [
        date_str[i]
        for i in major_idx
    ],
    fontsize=18
)

ax.set_ylabel(
    "Cumulative HAI",
    fontsize=18
)

ax.tick_params(
    axis="both",
    labelsize=18
)

ax.grid(
    True,
    which="major",
    axis="x",
    linestyle="-"
)

ax.grid(
    True,
    which="minor",
    axis="x",
    linestyle="--"
)

ax.grid(
    True,
    axis="y",
    linestyle="-"
)

ax.legend(
    fontsize=17,
    frameon=False
)

plt.tight_layout()

plt.savefig(
    "periodA_observed_surrogate_fit.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% 8. Save CSV files

# Profile likelihood values
df_profile = pd.DataFrame({

    "beta_SM": beta_grid,

    "logLik": ll_grid,

    "LR": LR_grid

})

df_profile.to_csv(
    "periodA_betaSM_profile.csv",
    index=False
)


# Summary
df_summary = pd.DataFrame([{

    "beta_SM_hat": beta_hat,

    "CI_lower_95": beta_low,

    "CI_upper_95": beta_high,

    "logLik": ll_hat,

    "AIC": AIC,

    "BIC": BIC,

    "init_env": init_env,

    "tau0": tau0

}])

df_summary.to_csv(
    "periodA_betaSM_profile_summary.csv",
    index=False
)

print("\nAll results saved.")