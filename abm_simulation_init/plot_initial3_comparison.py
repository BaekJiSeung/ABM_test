"""Compare THREE INITIAL starting points (not optimized results) for ABM beta=0.04, p=0.90.

Place this script next to SM4_A_hand_R2.py and run:
    python plot_initial3_comparison.py

It reads `initial_beta_sm` and `initial_p_sm` from the original
case_07_local_details.csv, and never calls an optimizer.
"""
from pathlib import Path
import ast
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'SM4_A_hand_R2.py'
ABM_CSV = (ROOT / '../result/interv_prob_transmission_summary_A9140_0.01-0.07.csv').resolve()
OUTPUT = ROOT / 'initial3_comparison'
CASE_ID = 7
BETA_ABM = 0.040
P_ABM = 0.90
INIT_ENV, TAU0 = 9, 140


def read_details():
    # Pick original local-optimization results folder, not the dense Grid folder.
    candidates = list((ROOT / 'sm_fit').glob('mapping_2D_17cases_*/cases/case_07_local_details.csv'))
    candidates = [x for x in candidates if 'GRID200x70' not in str(x)]
    if not candidates:
        raise FileNotFoundError(
            'case_07_local_details.csv not found. Expected under ' 
            'sm_fit/mapping_2D_17cases_100iter_*/cases/. ' 
            'This file stores the actual INITIAL points (initial_beta_sm, initial_p_sm).'
        )
    details_path = max(candidates, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(details_path)
    needed = {'candidate', 'initial_beta_sm', 'initial_p_sm'}
    if not needed.issubset(df.columns):
        raise ValueError(f'{details_path} missing columns: {sorted(needed - set(df.columns))}')
    df = df.sort_values('candidate').drop_duplicates('candidate')
    df = df[df.candidate.isin([1, 2, 3])]
    if len(df) != 3:
        raise ValueError(f'Expected 3 initial points for case 07; found {len(df)} in {details_path}')
    return df, details_path


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Cannot find Surrogate model: {MODEL_PATH}')
    spec = importlib.util.spec_from_file_location('user_cre_surrogate', MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_array(value):
    return np.asarray(ast.literal_eval(value) if isinstance(value, str) else value, dtype=float)


def abm_data():
    data = pd.read_csv(ABM_CSV)
    matches = data[np.isclose(pd.to_numeric(data['beta'], errors='coerce'), BETA_ABM, atol=1e-7, rtol=0)]
    if len(matches) != 1:
        raise ValueError(f'Expected one ABM row for beta={BETA_ABM}; got {len(matches)}')
    row = matches.iloc[0]
    monthly_mean = parse_array(row['mean'])
    monthly_std = parse_array(row['std'])
    if len(monthly_mean) != 19 or len(monthly_std) != 19:
        raise ValueError('Expected 19 monthly values from ABM data')
    cumulative = np.cumsum(monthly_mean)
    cumulative_sd = np.sqrt(np.cumsum(monthly_std ** 2))
    return cumulative, cumulative_sd


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    details, source = read_details()
    model = load_model()
    abm, abm_sd = abm_data()
    months = np.arange(1, len(abm) + 1)
    predictions = pd.DataFrame({
        'month': months,
        'abm_mean_cumulative': abm,
        'abm_mean_minus_1sd': abm - abm_sd,
        'abm_mean_plus_1sd': abm + abm_sd,
    })

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.fill_between(months, abm - abm_sd, abm + abm_sd, alpha=0.16, label='ABM mean ±1 SD')
    ax.plot(months, abm, 'o-', lw=2, label='ABM mean', color='black', markersize=3)
    rows = []
    for row in details.itertuples(index=False):
        beta = float(row.initial_beta_sm)
        p = float(row.initial_p_sm)
        prediction = np.asarray(model.simulate_theta(beta, INIT_ENV, TAU0, p), dtype=float)
        if len(prediction) != len(abm):
            raise ValueError('Surrogate prediction has unexpected number of months')
        idx = int(row.candidate)
        predictions[f'initial_{idx}'] = prediction
        sd = np.maximum(abm_sd, 1e-6)
        nll = 0.5 * np.sum(np.log(2 * np.pi * sd**2) + ((abm - prediction) / sd)**2)
        rmse = np.sqrt(np.mean((abm - prediction)**2))
        rows.append({'initial_point': idx, 'initial_beta_sm': beta,
                     'initial_p_sm': p, 'nll': nll, 'rmse': rmse,
                     'final_cumulative_hai': prediction[-1]})
        ax.plot(months, prediction, lw=1.9, label=f'Initial {idx}: β={beta:.3f}, p={p:.3f}')

    ax.set_xlabel('Month')
    ax.set_ylabel('Cumulative HAI incidence')
    ax.set_title('Surrogate predictions at three INITIAL points\nABM: beta=0.040, handwashing=0.90')
    ax.set_xticks(months)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'initial3_HAI_comparison.png', dpi=300)
    fig.savefig(OUTPUT / 'initial3_HAI_comparison.pdf')
    plt.close(fig)

    predictions.to_csv(OUTPUT / 'initial3_HAI_predictions.csv', index=False, encoding='utf-8-sig')
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / 'initial3_HAI_metrics.csv', index=False, encoding='utf-8-sig')
    print('INITIAL POINT source:', source)
    print(metrics.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('Saved to:', OUTPUT)


if __name__ == '__main__':
    main()
