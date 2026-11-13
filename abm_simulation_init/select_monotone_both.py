"""Find grid mappings where beta_SM AND p_SM both increase together.

Uses precomputed grid_case_01.csv .. grid_case_17.csv (200 x 70).
Each sweep (cases 1..13, 14..17) is optimized separately.

Outputs exact grid-constrained minsum and minimax solutions and CSV summaries.
No surrogate evaluations or L-BFGS-B steps.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SWEEPS = {"beta_sweep": range(1, 14), "handwash_sweep": range(14, 18)}
REQUIRED = {"beta_idx", "p_idx", "beta_sm", "p_sm", "nll"}


def load_grids(directory: Path):
    files = [directory / f"grid_case_{i:02d}.csv" for i in range(1, 18)]
    absent = [str(p) for p in files if not p.is_file()]
    if absent:
        raise FileNotFoundError("Missing grid CSVs:\n" + "\n".join(absent))
    raw, costs, baselines = {}, {}, {}
    master_beta = master_p = None
    for case_id, path in enumerate(files, 1):
        df = pd.read_csv(path)
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing {missing}")
        if len(df) != 14000 or df.duplicated(["beta_idx", "p_idx"]).any():
            raise ValueError(f"{path.name}: expected 14,000 unique grid points")
        df = df.sort_values(["beta_idx", "p_idx"]).reset_index(drop=True)
        bi = df.beta_idx.to_numpy(dtype=int)
        pi = df.p_idx.to_numpy(dtype=int)
        if not (np.array_equal(bi, np.repeat(np.arange(200), 70))
                and np.array_equal(pi, np.tile(np.arange(70), 200))):
            raise ValueError(f"{path.name}: unexpected grid indices")
        for col in ("beta_sm", "p_sm", "nll"):
            if not np.isfinite(df[col].to_numpy(dtype=float)).all():
                raise ValueError(f"{path.name}: invalid {col}")
        beta_vals = df.beta_sm.to_numpy().reshape(200, 70)[:, 0]
        p_vals = df.p_sm.to_numpy().reshape(200, 70)[0, :]
        if not (np.all(np.diff(beta_vals) > 0) and np.all(np.diff(p_vals) > 0)):
            raise ValueError(f"{path.name}: Grid axes not increasing")
        if master_beta is None:
            master_beta, master_p = beta_vals, p_vals
        elif not (np.allclose(beta_vals, master_beta, rtol=0, atol=1e-9)
                  and np.allclose(p_vals, master_p, rtol=0, atol=1e-9)):
            raise ValueError(f"{path.name}: different grid axes")
        nll = df.nll.to_numpy(float).reshape(200, 70)
        base = float(nll.min())
        raw[case_id] = nll
        baselines[case_id] = base
        costs[case_id] = np.maximum(0, nll - base)
    return raw, costs, baselines, master_beta, master_p


def feasible(costs, threshold, strict=False):
    """True if a complete path exists under maximum per-case delta threshold."""
    reachable = costs[0] <= threshold + 1e-12
    for c in costs[1:]:
        # prefix OR over previous beta and p indices
        prefix = np.logical_or.accumulate(
            np.logical_or.accumulate(reachable, axis=0), axis=1
        )
        if strict:
            predecessors = np.zeros_like(prefix)
            predecessors[1:, 1:] = prefix[:-1, :-1]
        else:
            predecessors = prefix
        reachable = predecessors & (c <= threshold + 1e-12)
        if not reachable.any():
            return False
    return bool(reachable.any())


def prefix_min_arg(arr, strict=False):
    """For every (i,j), find min arr[k,l], k<=i,l<=j (strict: k<i,l<j).

    Returns best predecessor costs and flattened indices. O(200*70).
    """
    rows, cols = arr.shape
    val = np.full_like(arr, np.inf, dtype=float)
    arg = np.full((rows, cols), -1, dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            best = arr[i, j]
            best_arg = i * cols + j if np.isfinite(best) else -1
            if i and val[i - 1, j] < best:
                best, best_arg = val[i - 1, j], arg[i - 1, j]
            if j and val[i, j - 1] < best:
                best, best_arg = val[i, j - 1], arg[i, j - 1]
            val[i, j] = best
            arg[i, j] = best_arg
    if strict:
        shifted_val = np.full_like(val, np.inf)
        shifted_arg = np.full_like(arg, -1)
        shifted_val[1:, 1:] = val[:-1, :-1]
        shifted_arg[1:, 1:] = arg[:-1, :-1]
        return shifted_val, shifted_arg
    return val, arg


def solve_minsum(costs, threshold=np.inf, strict=False):
    """Exact min-sum dynamic programming on the product-order 2D grid."""
    dp = np.where(costs[0] <= threshold + 1e-12, costs[0], np.inf)
    back = []
    for c in costs[1:]:
        prev_best, prev_arg = prefix_min_arg(dp, strict=strict)
        dp = np.where(c <= threshold + 1e-12, prev_best + c, np.inf)
        back.append(prev_arg)
    if not np.isfinite(dp).any():
        return None
    idx = int(np.argmin(dp))
    inds = [idx]
    cols = dp.shape[1]
    for pointers in reversed(back):
        idx = int(pointers.flat[idx])
        if idx < 0:
            raise RuntimeError("Broken DP backtracking")
        inds.append(idx)
    inds.reverse()
    return [(int(i // cols), int(i % cols)) for i in inds]


def choose_minimax(costs, strict=False):
    # The exact minimum attainable max delta is one of the grid delta values.
    values = np.unique(np.concatenate([c.ravel() for c in costs]))
    lo, hi = 0, len(values) - 1
    if not feasible(costs, float(values[-1]), strict=strict):
        raise ValueError("No feasible path; strict constraint may be too strong")
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(costs, float(values[mid]), strict=strict):
            hi = mid
        else:
            lo = mid + 1
    t = float(values[lo])
    # Of paths meeting the optimal max delta, minimize total delta.
    path = solve_minsum(costs, threshold=t, strict=strict)
    return path, t


def format_solution(case_ids, path, raw, costs, baselines, beta_axis, p_axis,
                    sweep, objective, strict):
    rows = []
    for case_id, (i, j) in zip(case_ids, path):
        rows.append({
            "case_id": case_id,
            "source": sweep,
            "beta_abm": float(0.01 + (case_id - 1)*0.005) if case_id <= 13 else .03847,
            "p_abm": .90 if case_id <= 13 else {14:.80, 15:.90, 16:.95, 17:.99}[case_id],
            "beta_idx": i,
            "p_idx": j,
            "beta_sm": float(beta_axis[i]),
            "p_sm": float(p_axis[j]),
            "nll": float(raw[case_id][i, j]),
            "best_nll": float(baselines[case_id]),
            "delta_nll": float(costs[case_id][i, j]),
            "objective": objective,
            "monotonicity": "both_strict" if strict else "both_nondecreasing",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path,
        default=Path("sm_fit/mapping_2D_17cases_GRID200x70/grids"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true",
        help="Require beta_SM and p_SM both strictly increase (default: nondecrease)")
    args = parser.parse_args()
    outdir = args.output_dir or args.grid_dir.parent / "monotone_both_selection"
    print(f"Loading 17 grid files from {args.grid_dir.resolve()} ...", flush=True)
    raw, costs, bases, beta_axis, p_axis = load_grids(args.grid_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for sweep, r in SWEEPS.items():
        ids = list(r)
        current = [costs[i] for i in ids]
        for objective in ("minsum", "minimax"):
            if objective == "minsum":
                path = solve_minsum(current, strict=args.strict)
                threshold = np.inf
            else:
                path, threshold = choose_minimax(current, strict=args.strict)
            if path is None:
                raise RuntimeError(f"No feasible solution for {sweep}/{objective}")
            sol = format_solution(ids, path, raw, costs, bases, beta_axis, p_axis,
                                  sweep, objective, args.strict)
            delta = sol.delta_nll.to_numpy()
            name = f"{sweep}_both_{objective}.csv"
            sol.to_csv(outdir / name, index=False, encoding="utf-8-sig")
            summary.append({
                "sweep": sweep,
                "axis": "both_beta_sm_and_p_sm",
                "objective": objective,
                "strict": args.strict,
                "n_cases": len(ids),
                "sum_delta_nll": float(delta.sum()),
                "max_delta_nll": float(delta.max()),
                "mean_delta_nll": float(delta.mean()),
                "file": name,
            })
            assert np.all(np.diff(sol.beta_sm) >= (1e-10 if args.strict else -1e-10))
            assert np.all(np.diff(sol.p_sm) >= (1e-10 if args.strict else -1e-10))
            print(f"\n[{sweep} | BOTH | {objective}]", flush=True)
            print(f"  sum ΔNLL: {delta.sum():.6f}  max ΔNLL: {delta.max():.6f}", flush=True)
            print(sol[["case_id", "beta_abm", "p_abm", "beta_sm", "p_sm", "nll", "delta_nll"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.6f}"), flush=True)
    pd.DataFrame(summary).to_csv(outdir / "summary_both.csv", index=False,
                                 encoding="utf-8-sig")
    print(f"\nCompleted. Files in {outdir.resolve()}")


if __name__ == "__main__":
    main()
