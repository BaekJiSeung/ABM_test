"""Anchor-constrained, both-nondecreasing handwashing sweep (case 14..17).

The p_ABM=0.90 case is forced to beta_SM=3.387, p_SM=0.90.
Other cases are selected from existing 200x70 grid CSVs to minimize
total extra NLL, subject to beta_SM and p_SM nondecreasing with p_ABM.

Use ORIGINAL surrogate (hc <- (hc+new_hc)*(1-p_wash)), not wash108.
No large surrogate grid recalculation; only the off-grid anchor NLL is evaluated.
"""
import argparse
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ANCHOR_BETA_SM = 3.387
ANCHOR_P_SM = 0.90
GRID_CASES = (14, 15, 16, 17)
P_ABM = {14: 0.80, 15: 0.90, 16: 0.95, 17: 0.99}
ABM_BETA = 0.03847


def load_surrogate(path: Path):
    spec = importlib.util.spec_from_file_location("cre_surrogate_for_anchor", str(path.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "simulate_theta"):
        raise AttributeError("Model lacks simulate_theta")
    return module


def read_vec(v):
    return np.asarray(ast.literal_eval(v) if isinstance(v, str) else v, dtype=float)


def anchor_nll(model_script: Path, handwash_csv: Path):
    df = pd.read_csv(handwash_csv)
    mask = (
        np.isclose(pd.to_numeric(df["beta"], errors="coerce"), ABM_BETA, atol=1e-7, rtol=0)
        & np.isclose(pd.to_numeric(df["handwash"], errors="coerce"), 0.90, atol=1e-7, rtol=0)
    )
    subset = df.loc[mask]
    if len(subset) != 1:
        raise ValueError(f"Expected exactly one handwash p=0.90 row, found {len(subset)}")
    y = np.cumsum(read_vec(subset.iloc[0]["mean"]))
    sd = np.maximum(np.sqrt(np.cumsum(read_vec(subset.iloc[0]["std"])**2)), 1e-6)
    model = load_surrogate(model_script)
    pred = np.asarray(model.simulate_theta(ANCHOR_BETA_SM, 9, 140, ANCHOR_P_SM), float)
    if y.shape != (19,) or sd.shape != (19,) or pred.shape != (19,):
        raise ValueError("Expected 19 monthly cumulative HAI observations/predictions")
    return float(0.5*np.sum(np.log(2*np.pi*sd**2) + ((y-pred)/sd)**2))


def read_grid(grid_dir: Path, cid: int):
    path = grid_dir / f"grid_case_{cid:02d}.csv"
    df = pd.read_csv(path)
    required = {"beta_idx", "p_idx", "beta_sm", "p_sm", "nll"}
    if not required.issubset(df.columns) or len(df) != 14000:
        raise ValueError(f"{path} must contain 14,000 indexed grid rows")
    df = df.sort_values(["beta_idx", "p_idx"]).reset_index(drop=True)
    bi = df.beta_idx.to_numpy(int)
    pi = df.p_idx.to_numpy(int)
    if not (np.array_equal(bi, np.repeat(np.arange(200), 70))
            and np.array_equal(pi, np.tile(np.arange(70), 200))):
        raise ValueError(f"{path} has unexpected grid indices")
    return df


def prefix_best(a):
    """Inclusive 2D prefix min values + flat argmins, O(n_beta*n_p)."""
    nr, nc = a.shape
    val = np.full((nr, nc), np.inf)
    arg = np.full((nr, nc), -1, dtype=int)
    for i in range(nr):
        for j in range(nc):
            best = a[i, j]
            idx = i*nc+j if np.isfinite(best) else -1
            if i and val[i-1, j] < best:
                best, idx = val[i-1, j], arg[i-1, j]
            if j and val[i, j-1] < best:
                best, idx = val[i, j-1], arg[i, j-1]
            val[i, j], arg[i, j] = best, idx
    return val, arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", type=Path, default=Path("sm_fit/mapping_2D_17cases_GRID200x70/grids"))
    ap.add_argument("--model-script", type=Path, default=Path("SM4_A_hand_R2.py"))
    ap.add_argument("--handwash-csv", type=Path, default=Path("../result/interv_prob_transmission_summary_A9140_0.03847-0.03847_handwashALL.csv"))
    ap.add_argument("--output", type=Path, default=Path("handwash_anchor090_both_monotone.csv"))
    args = ap.parse_args()

    grids = {i: read_grid(args.grid_dir, i) for i in GRID_CASES}
    base_beta = grids[14].beta_sm.to_numpy().reshape(200,70)[:,0]
    base_p = grids[14].p_sm.to_numpy().reshape(200,70)[0,:]
    for i, df in grids.items():
        if not (np.allclose(df.beta_sm.to_numpy().reshape(200,70)[:,0], base_beta)
                and np.allclose(df.p_sm.to_numpy().reshape(200,70)[0,:], base_p)):
            raise ValueError(f"Grid axes disagree at case {i}")

    surfaces = {i: grids[i].nll.to_numpy(float).reshape(200,70) for i in GRID_CASES}
    baselines = {i: float(surfaces[i].min()) for i in GRID_CASES}

    # Both dimensions must be nondecreasing as ABM handwashing p goes up.
    below = (base_beta[:,None] <= ANCHOR_BETA_SM+1e-12) & (base_p[None,:] <= ANCHOR_P_SM+1e-12)
    above = (base_beta[:,None] >= ANCHOR_BETA_SM-1e-12) & (base_p[None,:] >= ANCHOR_P_SM-1e-12)

    d14 = np.where(below, surfaces[14]-baselines[14], np.inf)
    if not np.isfinite(d14).any():
        raise ValueError("No grid point below anchor for p_ABM=0.80")
    ij14 = np.unravel_index(np.argmin(d14), d14.shape)

    d16 = np.where(above, surfaces[16]-baselines[16], np.inf)
    d17 = np.where(above, surfaces[17]-baselines[17], np.inf)
    pv, pa = prefix_best(d16)
    total17 = d17 + pv
    if not np.isfinite(total17).any():
        raise ValueError("No monotone path for cases 16 and 17")
    ij17 = np.unravel_index(np.argmin(total17), total17.shape)
    k16 = int(pa[ij17])
    ij16 = (k16 // 70, k16 % 70)

    anchor_value = anchor_nll(args.model_script, args.handwash_csv)
    chosen = {14: ij14, 16: ij16, 17: ij17}
    rows = []
    for cid in GRID_CASES:
        if cid == 15:
            b, p, ll = ANCHOR_BETA_SM, ANCHOR_P_SM, anchor_value
        else:
            ij = chosen[cid]
            b, p, ll = float(base_beta[ij[0]]), float(base_p[ij[1]]), float(surfaces[cid][ij])
        rows.append(dict(case_id=cid, beta_abm=ABM_BETA, p_abm=P_ABM[cid],
                         beta_sm=b, p_sm=p, nll=ll, best_grid_nll=baselines[cid],
                         delta_nll=ll-baselines[cid], anchor=(cid == 15)))
    out = pd.DataFrame(rows)
    if np.any(np.diff(out.beta_sm.to_numpy()) < -1e-12) or np.any(np.diff(out.p_sm.to_numpy()) < -1e-12):
        raise AssertionError("Monotonicity check failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("\nHANDWASH SWEEP | p_ABM=0.90 anchor | both nondecreasing")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nMax ΔNLL={out.delta_nll.max():.6f}; Sum ΔNLL={out.delta_nll.sum():.6f}")
    print(f"Saved: {args.output.resolve()}")
    print("NOTE: Anchor is an imposed choice, not the optimum for the handwash sweep.")

if __name__ == "__main__":
    main()
