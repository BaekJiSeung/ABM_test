# %% Step6: B version, infer ABM beta from SM beta

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

def infer_abm_beta_from_sm(
    sm_hat,
    sm_low=None,
    sm_high=None,
    pairs_csv="sm_fit/theta_pairs_subset_cumGaussian_B260.csv"
):
    """
    Step6:
    B period에서 만든 (beta_ABM, theta_SM) mapping을 이용해서
    observed data에서 얻은 beta_SM(theta_SM)을 beta_ABM으로 역보간.
    """

    # 1) mapping pair 로드
    df_raw = pd.read_csv(pairs_csv)

    # theta column 이름 유연 처리
    if "beta_sm_hat" in df_raw.columns:
        col_sm = "beta_sm_hat"
    elif "theta_hat" in df_raw.columns:
        col_sm = "theta_hat"
    else:
        raise ValueError("CSV에 beta_sm_hat 또는 theta_hat 열이 없습니다.")

    if "beta_abm" not in df_raw.columns:
        raise ValueError("CSV에 beta_abm 열이 없습니다.")

    df = (
        df_raw[["beta_abm", col_sm]]
        .dropna()
        .rename(columns={col_sm: "beta_sm_hat"})
        .sort_values("beta_abm")
        .reset_index(drop=True)
    )

    x_abm = df["beta_abm"].to_numpy(float)
    y_sm  = df["beta_sm_hat"].to_numpy(float)

    # 2) ABM beta -> SM theta 보간
    f_abm2sm = PchipInterpolator(x_abm, y_sm)

    abm_grid = np.linspace(x_abm.min(), x_abm.max(), 2001)
    sm_grid  = f_abm2sm(abm_grid)

    # 3) SM -> ABM 역보간 준비
    # sm_grid가 감소 방향이면 뒤집기
    if sm_grid[-1] < sm_grid[0]:
        sm_grid  = sm_grid[::-1]
        abm_grid = abm_grid[::-1]

    # 혹시 non-monotone 구간이 있거나 같은 값이 있으면 정렬 후 unique 처리
    order = np.argsort(sm_grid)
    sm_sorted  = sm_grid[order]
    abm_sorted = abm_grid[order]

    sm_unique, idx_unique = np.unique(sm_sorted, return_index=True)
    abm_unique = abm_sorted[idx_unique]

    g_sm2abm = PchipInterpolator(sm_unique, abm_unique)

    # 4) 범위 확인 및 clipping
    sm_min, sm_max = float(sm_unique.min()), float(sm_unique.max())

    sm_hat_c = np.clip(sm_hat, sm_min, sm_max)
    abm_hat = float(g_sm2abm(sm_hat_c))

    if sm_low is None or sm_high is None:
        return {
            "beta_abm_hat": abm_hat,
            "beta_abm_low": None,
            "beta_abm_high": None,
            "range_used": (sm_min, sm_max),
            "was_clipped": sm_hat != sm_hat_c
        }

    sm_lo_c = np.clip(sm_low, sm_min, sm_max)
    sm_hi_c = np.clip(sm_high, sm_min, sm_max)

    abm_low  = float(g_sm2abm(sm_lo_c))
    abm_high = float(g_sm2abm(sm_hi_c))

    if abm_low > abm_high:
        abm_low, abm_high = abm_high, abm_low

    return {
        "beta_abm_hat": abm_hat,
        "beta_abm_low": abm_low,
        "beta_abm_high": abm_high,
        "range_used": (sm_min, sm_max),
        "was_clipped": (sm_hat != sm_hat_c) or (sm_low != sm_lo_c) or (sm_high != sm_hi_c)
    }


# =========================
# B observed SM result
# =========================

res_B = infer_abm_beta_from_sm(
    sm_hat=3.2991,
    sm_low=3.2781,
    sm_high=3.3200,
    pairs_csv="sm_fit/theta_pairs_subset_cumGaussian_B260.csv"
)

print(res_B)
# %%
