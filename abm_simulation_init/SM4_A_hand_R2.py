# Period A: (beta_ABM, p_ABM) -> (beta_SM, p_SM)
# 17 ABM cases, 17 workers; 220 grid points + 3 separated local fits per case.
# Variant: 108 handwashing opportunities PER DAY across all 19 HCWs,
# independent of numerical integration steps.
import ast
import os
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize

BETA_CSV = '../result/interv_prob_transmission_summary_A9140_0.01-0.07.csv'
HANDWASH_CSV = '../result/interv_prob_transmission_summary_A9140_0.03847-0.03847_handwashALL.csv'
N_JOBS = 17
INIT_ENV, TAU0 = 9, 140
# New handwashing dynamics yield a much smaller beta_SM scale.
# Pilot at p=.90: beta=.20 -> ~10 cumulative HAI; beta=.30 -> ~113.
# Use a broader low-beta range rather than reusing old beta bounds.
BETA_BOUNDS, P_BOUNDS = (0.01, 2.0), (0.5, 1.0)
BETA_GRID = np.linspace(*BETA_BOUNDS, 20)
P_GRID = np.linspace(*P_BOUNDS, 11)
N_STARTS = 3
MAXITER, MAXFUN, FTOL = 100, 3000, 1e-8
MIN_BETA_DISTANCE, MIN_P_DISTANCE = 0.5, 0.03

# Surrogate numerical time resolution (steps/day)
STEPS_PER_DAY = 108
# Total handwashing opportunities per day across ALL 19 HCWs (assumption)
WASH_OPPORTUNITIES_PER_DAY = 108.0


MONTHLY_PI = pd.Series({
    '2017-01': 0, '2017-02': 0, '2017-03': 1,
    '2017-04': 0, '2017-05': 1, '2017-06': 0,
    '2017-07': 0, '2017-08': 0, '2017-09': 2,
    '2017-10': 0, '2017-11': 2, '2017-12': 0,
    '2018-01': 0, '2018-02': 0, '2018-03': 0,
    '2018-04': 0, '2018-05': 1, '2018-06': 0,
    '2018-07': 0,
})
PI_DATES = [
    '2017-03-01', '2017-05-30', '2017-09-17',
    '2017-09-29', '2017-11-14', '2017-11-20',
    '2018-05-25',
]


def make_imported_days(days):
    daily = np.zeros(len(days))
    indices = {d: i for i, d in enumerate(days)}
    for date in PI_DATES:
        index = indices.get(pd.Timestamp(date))
        if index is not None:
            daily[index] += 1.0
    return daily


def simulate_theta(beta, init_env, tau0, p_wash):
    """Same Period A dynamics as previous run; returns 19 monthly cumulative HAIs."""
    capacity, n_h, n_e = 30, 19, 30
    mu_s, mu_hai, mu_i = 1 / 7, 1 / 14, 1 / 7
    dt = 1.0 / STEPS_PER_DAY
    if not (0.0 <= p_wash <= 1.0):
        raise ValueError("p_wash must lie in [0, 1]")

    # At one wash opportunity a uniformly selected HCW clears contamination
    # with probability p_wash. Expected survival of aggregate HCW contamination
    # per opportunity = (1 - p_wash/n_h).
    #
    # In one numerical step, the expected number of opportunities is
    # WASH_OPPORTUNITIES_PER_DAY / STEPS_PER_DAY.
    # This fixed-opportunity-frequency approximation makes the aggregate
    # daily wash effect independent of STEPS_PER_DAY.
    wash_survival_per_step = (
        1.0 - p_wash / n_h
    ) ** (WASH_OPPORTUNITIES_PER_DAY / STEPS_PER_DAY)
    clean_period, cleaning_eff = 180, 0.90
    iso_factor, sigma = 0.75, 1 / 7

    start = pd.Period(MONTHLY_PI.index.min(), freq='M').to_timestamp(how='start')
    end = pd.Period(MONTHLY_PI.index.max(), freq='M').to_timestamp(how='end')
    days = pd.date_range(start, end, freq='D')
    imported = make_imported_days(days)

    ps, ph, ph_iso, pi = float(capacity - 1), 0.0, 0.0, 1.0
    hc, ec = 0.0, float(init_env)
    daily_hai = np.zeros(len(days))

    for t in range(len(days)):
        if t > 0 and (t + tau0) % clean_period == 0:
            ec *= (1 - cleaning_eff)

        inc = imported[t]
        if inc > 0:
            # Imported patient replaces a susceptible patient, same as prior code.
            admitted = min(ps, inc)
            ps -= admitted
            pi += admitted

        for _ in range(STEPS_PER_DAY):
            total = max(ps + ph + ph_iso + pi, 1e-9)
            lam_hp = beta * (hc / n_h)
            lam_ph = beta * ((ph + iso_factor * ph_iso + pi) / total)
            lam_eh = beta * (ec / n_e)
            lam_he = beta * (hc / n_h)

            new_hai = lam_hp * ps * dt
            move_h = sigma * ph * dt
            move_i = sigma * pi * dt
            out_s = mu_s * ps * dt
            out_h = mu_hai * ph * dt
            out_h_iso = mu_hai * ph_iso * dt
            out_i = mu_i * pi * dt
            leaving = out_s + out_h + out_h_iso + out_i
            new_s = max(0.0, capacity - ((ps + ph + ph_iso + pi) - leaving))

            ps = np.clip(ps + new_s - out_s - new_hai, 0, capacity)
            ph = np.clip(ph + new_hai - out_h - move_h, 0, capacity)
            ph_iso = np.clip(ph_iso + move_h + move_i - out_h_iso, 0, capacity)
            pi = np.clip(pi - out_i - move_i, 0, capacity)

            new_hc = (lam_ph + lam_eh) * (n_h - hc) * dt
            hc = np.clip(
                (hc + new_hc) * wash_survival_per_step, 0.0, n_h
            )
            ec = np.clip(ec + lam_he * (n_e - ec) * dt, 0, n_e)
            daily_hai[t] += new_hai

    monthly_hai = (pd.Series(daily_hai, index=days)
                   .groupby(days.to_period('M')).sum().to_numpy(dtype=float))
    return np.cumsum(monthly_hai)


def parse_vector(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=float)


def build_cases():
    beta_df = pd.read_csv(BETA_CSV)
    hw_df = pd.read_csv(HANDWASH_CSV)

    # Explicitly select the 13 + 4 requested ABM parameter settings.
    # The CSVs may contain additional rows from other experiments.
    beta_targets = np.linspace(0.01, 0.07, 13)
    handwash_targets = [0.80, 0.90, 0.95, 0.99]
    atol = 1e-7

    cases = []

    def find_one(df, conditions, description):
        mask = np.ones(len(df), dtype=bool)
        for column, target in conditions.items():
            values = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
            mask &= np.isclose(values, target, atol=atol, rtol=0)
        matched = df.loc[mask]
        if matched.empty:
            raise ValueError(f'Missing ABM condition in CSV: {description}')
        if len(matched) > 1:
            print(f'[WARNING] {description}: {len(matched)} duplicate matches; using first row.', flush=True)
        return matched.iloc[0]

    for beta in beta_targets:
        row = find_one(beta_df, {'beta': float(beta)}, f'beta sweep beta={beta:.5f}')
        cases.append(dict(source='beta_sweep', beta_abm=float(beta),
                          p_abm=0.90, y_mean=parse_vector(row['mean']),
                          y_std=parse_vector(row['std'])))

    for p in handwash_targets:
        row = find_one(hw_df, {'beta': 0.03847, 'handwash': p},
                       f'handwash sweep beta=0.03847 p={p:.2f}')
        cases.append(dict(source='handwash_sweep', beta_abm=0.03847,
                          p_abm=float(p), y_mean=parse_vector(row['mean']),
                          y_std=parse_vector(row['std'])))

    print(f'Input CSV rows: beta={len(beta_df)}, handwash={len(hw_df)}', flush=True)
    print('Selected: beta=13, handwash=4, total=17', flush=True)

    if len(cases) != 17:
        raise ValueError(f'Expected 17 cases, found {len(cases)}.')
    for c in cases:
        if len(c['y_mean']) != 19 or len(c['y_std']) != 19:
            raise ValueError(f'Expected 19 monthly values: {c["source"]} {c["beta_abm"]} {c["p_abm"]}')
        if (not np.all(np.isfinite(c['y_mean'])) or
                not np.all(np.isfinite(c['y_std'])) or np.any(c['y_std'] < 0)):
            raise ValueError(f'Invalid mean/std: {c}')
    return cases


def optimize_one_case(case_id, case, output_dir):
    start = time.perf_counter()
    beta_abm, p_abm, source = case['beta_abm'], case['p_abm'], case['source']
    target = np.cumsum(case['y_mean'])
    sd = np.maximum(np.sqrt(np.cumsum(case['y_std'] ** 2)), 1e-6)
    n_eval = 0

    def nll_2d(x):
        nonlocal n_eval
        n_eval += 1
        pred = simulate_theta(float(x[0]), INIT_ENV, TAU0, float(x[1]))
        if len(pred) != len(target):
            raise ValueError('Model and observation month counts differ')
        return float(0.5 * np.sum(np.log(2 * np.pi * sd ** 2) + ((target - pred) / sd) ** 2))

    grid_start = time.perf_counter()
    grid_data = []
    for b in BETA_GRID:
        for p in P_GRID:
            grid_data.append((float(b), float(p), nll_2d((b, p))))
    df_grid = pd.DataFrame(grid_data, columns=['beta_sm', 'p_sm', 'nll'])
    df_grid.to_csv(os.path.join(output_dir, 'grids', f'grid_case_{case_id:02d}.csv'), index=False)
    grid_time = (time.perf_counter() - grid_start) / 60
    df_sorted = df_grid.sort_values('nll').reset_index(drop=True)
    grid_best = df_sorted.iloc[0]

    starts = []
    for row in df_sorted.itertuples(index=False):
        x = np.array([row.beta_sm, row.p_sm], dtype=float)
        if all(abs(x[0] - s[0]) >= MIN_BETA_DISTANCE or
               abs(x[1] - s[1]) >= MIN_P_DISTANCE for s in starts):
            starts.append(x)
        if len(starts) == N_STARTS:
            break
    if len(starts) != N_STARTS:
        raise RuntimeError(f'Only {len(starts)} distinct starts found')

    local_start = time.perf_counter()
    candidates = []
    for idx, x0 in enumerate(starts, start=1):
        before = n_eval
        t0 = time.perf_counter()
        res = minimize(nll_2d, x0, method='L-BFGS-B',
                       bounds=[BETA_BOUNDS, P_BOUNDS],
                       options=dict(maxiter=MAXITER, maxfun=MAXFUN, ftol=FTOL))
        item = dict(case_id=case_id, source=source, beta_abm=beta_abm, p_abm=p_abm,
                    candidate=idx, initial_beta_sm=float(x0[0]), initial_p_sm=float(x0[1]),
                    beta_sm=float(res.x[0]), p_sm=float(res.x[1]), nll=float(res.fun),
                    success=bool(res.success), message=str(res.message),
                    iterations=int(res.nit), evaluations=int(n_eval - before),
                    elapsed_min=(time.perf_counter() - t0) / 60)
        candidates.append(item)
        # Save after EVERY local fit, so partial candidate results survive interruption.
        pd.DataFrame(candidates).to_csv(
            os.path.join(output_dir, 'cases', f'case_{case_id:02d}_local_details.csv'),
            index=False)
        print(f'[CASE {case_id:02d}/17 LOCAL {idx}/3] beta_SM={res.x[0]:.6f} '
              f'p_SM={res.x[1]:.6f} NLL={res.fun:.6f} '
              f'success={res.success} iter={res.nit} eval={n_eval-before}', flush=True)

    local_time = (time.perf_counter() - local_start) / 60
    # Choose lowest finite NLL; preserve nonconverged solutions with their success flag.
    eligible = [r for r in candidates if np.isfinite(r['nll'])]
    best_local = min(eligible, key=lambda r: r['nll']) if eligible else None
    if best_local is not None and best_local['nll'] < float(grid_best['nll']):
        selected = f"Local {best_local['candidate']}"
        best_beta, best_p, best_nll = best_local['beta_sm'], best_local['p_sm'], best_local['nll']
        best_success = best_local['success']
    else:
        selected = 'Coarse Grid'
        best_beta, best_p, best_nll = (float(grid_best['beta_sm']),
                                      float(grid_best['p_sm']), float(grid_best['nll']))
        best_success = False

    result = dict(case_id=case_id, source=source, beta_abm=beta_abm, p_abm=p_abm,
                  best_beta_sm=best_beta, best_p_sm=best_p, best_nll=best_nll,
                  selected=selected, best_success=best_success,
                  grid_nll=float(grid_best['nll']), grid_time_min=grid_time,
                  local_time_min=local_time, total_evaluations=n_eval,
                  elapsed_min=(time.perf_counter() - start) / 60)
    # 17 rows, with 3 FULL candidate groups on each row.
    for item in candidates:
        prefix = f"local{item['candidate']}_"
        for field in ('initial_beta_sm', 'initial_p_sm', 'beta_sm', 'p_sm',
                      'nll', 'success', 'message', 'iterations', 'evaluations', 'elapsed_min'):
            result[prefix + field] = item[field]
    pd.DataFrame([result]).to_csv(
        os.path.join(output_dir, 'cases', f'case_{case_id:02d}_result.csv'), index=False)
    print(f'[DONE {case_id:02d}/17] ABM=({beta_abm:.5f}, {p_abm:.2f}) '
          f'best=({best_beta:.6f}, {best_p:.6f}), NLL={best_nll:.6f} '
          f'({selected}, success={best_success})', flush=True)
    return result, candidates


def safe_optimize(case_id, case, output_dir):
    try:
        result, candidates = optimize_one_case(case_id, case, output_dir)
        return dict(ok=True, result=result, candidates=candidates)
    except Exception:
        err = traceback.format_exc()
        with open(os.path.join(output_dir, 'cases', f'case_{case_id:02d}_ERROR.txt'),
                  'w', encoding='utf-8') as f:
            f.write(err)
        print(f'[ERROR {case_id:02d}/17] {err}', flush=True)
        return dict(ok=False, case_id=case_id, error=err)


def main():
    cases = build_cases()
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('sm_fit', f'mapping_2D_17cases_wash108_100iter_{run_id}')
    os.makedirs(os.path.join(output_dir, 'cases'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'grids'), exist_ok=True)
    print('17-case mapping: maxiter=100, maxfun=3000, ftol=1e-8', flush=True)
    print(f'STEPS_PER_DAY={STEPS_PER_DAY}; '
          f'WASH_OPPORTUNITIES_PER_DAY={WASH_OPPORTUNITIES_PER_DAY}; '
          'HCW wash selection = uniform', flush=True)
    print('Output folder:', os.path.abspath(output_dir), flush=True)
    for i, c in enumerate(cases, 1):
        print(f'{i:02d}: {c["source"]}: beta_ABM={c["beta_abm"]:.5f}, '
              f'p_ABM={c["p_abm"]:.2f}', flush=True)

    wall_start = time.perf_counter()
    jobs = Parallel(n_jobs=N_JOBS, backend='loky', verbose=10, pre_dispatch='n_jobs')(
        delayed(safe_optimize)(i, case, output_dir)
        for i, case in enumerate(cases, 1)
    )
    wall_time = (time.perf_counter() - wall_start) / 60
    completed = [x for x in jobs if x['ok']]
    failures = [x['case_id'] for x in jobs if not x['ok']]

    df_wide = pd.DataFrame([x['result'] for x in completed])
    df_long = pd.DataFrame([row for x in completed for row in x['candidates']])
    if not df_wide.empty:
        df_wide = df_wide.sort_values('case_id').reset_index(drop=True)
    if not df_long.empty:
        df_long = df_long.sort_values(['case_id', 'candidate']).reset_index(drop=True)

    wide_path = os.path.join(output_dir, 'mapping_2D_17cases_all3_wide.csv')
    long_path = os.path.join(output_dir, 'mapping_2D_17cases_all3_long.csv')
    df_wide.to_csv(wide_path, index=False, encoding='utf-8-sig')
    df_long.to_csv(long_path, index=False, encoding='utf-8-sig')
    pd.DataFrame([dict(total_cases=17, completed=len(completed), failed=len(failures),
                       failed_ids=','.join(map(str, failures)), n_jobs=N_JOBS,
                       grid_points=220, local_starts=N_STARTS, maxiter=MAXITER,
                       maxfun=MAXFUN, ftol=FTOL, steps_per_day=STEPS_PER_DAY,
                       wash_opportunities_per_day=WASH_OPPORTUNITIES_PER_DAY,
                       wash_allocation='uniform_over_19_hcws', wall_time_min=wall_time)]).to_csv(
        os.path.join(output_dir, 'run_summary.csv'), index=False)

    print('\n========== ALL 3 LOCAL ESTIMATES (51 rows) ==========', flush=True)
    if not df_long.empty:
        cols = ['case_id', 'beta_abm', 'p_abm', 'candidate', 'beta_sm', 'p_sm',
                'nll', 'success', 'iterations', 'evaluations']
        print(df_long[cols].to_string(index=False), flush=True)
    print('\n========== ONE ROW PER ABM CASE (17 rows) ==========', flush=True)
    if not df_wide.empty:
        cols = ['case_id', 'beta_abm', 'p_abm']
        for i in range(1, 4):
            cols += [f'local{i}_beta_sm', f'local{i}_p_sm', f'local{i}_nll',
                     f'local{i}_success', f'local{i}_iterations']
        cols += ['selected', 'best_beta_sm', 'best_p_sm', 'best_nll', 'best_success']
        with pd.option_context('display.width', 320, 'display.max_columns', None):
            print(df_wide[cols].to_string(index=False), flush=True)
    print(f'\nFinished {len(completed)}/17, failed={failures}; wall time={wall_time:.2f} min')
    print('WIDE:', wide_path)
    print('LONG:', long_path)
    print('Individual CSVs:', os.path.join(output_dir, 'cases'))
    print('Grid CSVs:', os.path.join(output_dir, 'grids'))


if __name__ == '__main__':
    main()
