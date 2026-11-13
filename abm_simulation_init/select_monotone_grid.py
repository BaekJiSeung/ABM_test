"""
Select monotone mappings from precomputed exhaustive 2D NLL grids.

Input: grid_case_01.csv ... grid_case_17.csv (14,000 rows per file)
Output: four exact monotone mapping solutions, each with two objectives:
  - minsum: minimize total increase in NLL relative to per-case minimum.
  - minimax: minimize largest per-case NLL increase; break ties by total increase.

No surrogate simulations, no optimizers, no modifications to input files.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

CASES = {
    "beta_sweep": list(range(1, 14)),
    "handwash_sweep": list(range(14, 18)),
}


def load_case(path, expected_beta=200, expected_p=70):
    df = pd.read_csv(path)
    required = {"beta_idx", "p_idx", "beta_sm", "p_sm", "nll"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: missing columns: {sorted(required - set(df.columns))}")
    if not np.all(np.isfinite(df[["beta_sm", "p_sm", "nll"]].to_numpy())):
        raise ValueError(f"{path}: non-finite data")
    if len(df) != expected_beta * expected_p:
        raise ValueError(f"{path}: found {len(df)} rows, expected {expected_beta*expected_p}")
    if df.duplicated(["beta_idx", "p_idx"]).any():
        raise ValueError(f"{path}: duplicate grid indices")
    if df.beta_idx.nunique() != expected_beta or df.p_idx.nunique() != expected_p:
        raise ValueError(f"{path}: unexpected grid shape")
    df = df.copy()
    df["beta_idx"] = df["beta_idx"].astype(int)
    df["p_idx"] = df["p_idx"].astype(int)
    return df


def reduce_by_axis(df, axis):
    """For each axis index, retain the other-axis setting with minimum NLL."""
    key = "beta_idx" if axis == "beta_sm" else "p_idx"
    ranked = df.sort_values(["nll", "beta_idx", "p_idx"], kind="stable")
    best = ranked.drop_duplicates(key).sort_values(key).reset_index(drop=True)
    return best


def dp_solve(reduced, axis, threshold=np.inf, strict=False):
    """Minimize SUM(deltaNLL) under monotonicity and optional per-case delta limit.

    reduced is a list of (n_axis x columns) DataFrames, indexed 0..m-1
    """
    n = len(reduced)
    size = len(reduced[0])
    costs = np.stack([
        d["nll"].to_numpy(float) - float(d.attrs["baseline_nll"])
        for d in reduced
    ])
    costs = np.maximum(costs, 0.0)  # floating precision only
    costs[costs > threshold + 1e-12] = np.inf
    if not np.any(np.isfinite(costs[0])):
        return None

    dp = costs[0].copy()
    pointers = np.full((n, size), -1, dtype=int)
    for i in range(1, n):
        prev = dp
        # Prefix minimum with argmin: O(axis_size) per case
        prefix_val = np.full(size, np.inf)
        prefix_arg = np.full(size, -1, dtype=int)
        best_val, best_idx = np.inf, -1
        for j in range(size):
            if prev[j] < best_val:
                best_val, best_idx = prev[j], j
            prefix_val[j], prefix_arg[j] = best_val, best_idx
        next_dp = np.full(size, np.inf)
        for j in range(size):
            prior = j - 1 if strict else j
            if prior >= 0 and np.isfinite(prefix_val[prior]) and np.isfinite(costs[i, j]):
                next_dp[j] = prefix_val[prior] + costs[i, j]
                pointers[i, j] = prefix_arg[prior]
        dp = next_dp

    end = int(np.argmin(dp))
    if not np.isfinite(dp[end]):
        return None
    indices = [end]
    for i in range(n - 1, 0, -1):
        end = int(pointers[i, end])
        indices.append(end)
    indices.reverse()
    return indices, float(np.min(dp))


def choose(reduced, axis, objective="minsum", strict=False):
    if objective == "minsum":
        ans = dp_solve(reduced, axis, strict=strict)
        if ans is None:
            raise RuntimeError("No feasible monotone sequence")
        idx, total = ans
        threshold = np.inf
    else:
        # The smallest feasible maximum per-case delta is a member of this set.
        deltas = np.concatenate([
            np.maximum(d.nll.to_numpy(float) - d.attrs["baseline_nll"], 0.0)
            for d in reduced
        ])
        unique = np.unique(deltas)
        lo, hi = 0, len(unique) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if dp_solve(reduced, axis, threshold=float(unique[mid]), strict=strict) is None:
                lo = mid + 1
            else:
                hi = mid
        threshold = float(unique[lo])
        ans = dp_solve(reduced, axis, threshold=threshold, strict=strict)
        if ans is None:
            raise RuntimeError("No feasible minimax sequence")
        idx, total = ans

    selected = []
    for i, j in enumerate(idx):
        row = reduced[i].iloc[j].to_dict()
        row["case_id"] = int(reduced[i].attrs["case_id"])
        row["best_nll"] = float(reduced[i].attrs["baseline_nll"])
        row["delta_nll"] = max(0.0, float(row["nll"] - row["best_nll"]))
        selected.append(row)
    result = pd.DataFrame(selected)
    result["axis"] = axis
    result["objective"] = objective
    result["monotonicity"] = "strict_increase" if strict else "nondecreasing"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", type=Path,
                        default=Path("sm_fit/mapping_2D_17cases_GRID200x70/grids"),
                        help="Directory holding grid_case_01.csv ... grid_case_17.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true",
                        help="Require strictly increasing axis, not nondecreasing")
    args = parser.parse_args()
    grid_dir = args.grid_dir
    outdir = args.output_dir or grid_dir.parent / "monotone_selection"

    paths = {i: grid_dir / f"grid_case_{i:02d}.csv" for i in range(1, 18)}
    missing = [p for p in paths.values() if not p.is_file()]
    if missing:
        print(f"ERROR: {len(missing)} grid files not found in {grid_dir.resolve()}")
        for p in missing:
            print(f"  Missing: {p.name}")
        print("Place all 17 grid_case_XX.csv files in the same grids folder, ")
        print("or specify its location with --grid-dir PATH")
        return 1

    print("Loading all 17 grids (no simulation)...", flush=True)
    grids = {i: load_case(paths[i]) for i in paths}
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for sweep, ids in CASES.items():
        for axis in ("beta_sm", "p_sm"):
            reduced = []
            for case_id in ids:
                d = reduce_by_axis(grids[case_id], axis)
                d.attrs["baseline_nll"] = float(grids[case_id].nll.min())
                d.attrs["case_id"] = case_id
                reduced.append(d)
            for obj in ("minsum", "minimax"):
                sol = choose(reduced, axis, obj, strict=args.strict)
                name = f"{sweep}_{axis}_{obj}.csv"
                sol.to_csv(outdir / name, index=False, encoding="utf-8-sig")
                deltas = sol.delta_nll.to_numpy()
                s = dict(sweep=sweep, axis=axis, objective=obj,
                         strict=args.strict, n_cases=len(ids),
                         sum_delta_nll=float(deltas.sum()),
                         max_delta_nll=float(deltas.max()),
                         mean_delta_nll=float(deltas.mean()),
                         file=name)
                summary.append(s)
                print(f"\n[{sweep} / {axis} / {obj}] "
                      f"sum_delta={s['sum_delta_nll']:.6f}, "
                      f"max_delta={s['max_delta_nll']:.6f}")
                print(sol[["case_id", "beta_sm", "p_sm", "nll", "delta_nll"]]
                      .to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False,
                                 encoding="utf-8-sig")
    print(f"\nFinished. Results: {outdir.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
