"""Single-case, parallel 2D fitting for Period A / wash108 surrogate.

Copy beside your MODIFIED SM4_A_hand_R2.py and run:
    python SM4_A_hand_R2_case07_fast.py --jobs -1

The script imports simulate_theta from the model file and does NOT change its
physical equations. It performs a 20x11 grid, chooses 3 separated starts,
optimizes 3 pairs independently, and plots all 3 actual local-fit curves.
"""
from __future__ import annotations
import os
# Avoid over-subscription: one thread per parallel worker.
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import ast
import argparse
import importlib.util
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed

ABM_BETA, ABM_P = 0.040, 0.90


def load_model(model_file):
    spec = importlib.util.spec_from_file_location('cre_wash108_case07', str(Path(model_file).resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {model_file}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'simulate_theta'):
        raise AttributeError('simulate_theta missing from model script')
    if not hasattr(mod, 'WASH_OPPORTUNITIES_PER_DAY'):
        raise ValueError('Please point to the modified wash108 version of SM4_A_hand_R2.py')
    return mod


def read_vec(x):
    return np.asarray(ast.literal_eval(x) if isinstance(x,str) else x, dtype=float)


def evaluate_grid_row(beta, p_values, target, sd, model_file):
    mod = load_model(model_file)
    logterm = np.log(2*np.pi*sd**2)
    result = []
    for p in p_values:
        pred = np.asarray(mod.simulate_theta(float(beta),mod.INIT_ENV,mod.TAU0,float(p)),dtype=float)
        nll = float(.5*np.sum(logterm+((target-pred)/sd)**2))
        result.append([float(beta),float(p),nll])
    return result


def optimize_local(rank, x0, target, sd, model_file, maxiter, maxfun, ftol, bounds):
    mod = load_model(model_file)
    logterm = np.log(2*np.pi*sd**2)
    calls = 0
    def f(x):
        nonlocal calls
        calls += 1
        pred = np.asarray(mod.simulate_theta(float(x[0]),mod.INIT_ENV,mod.TAU0,float(x[1])),dtype=float)
        return float(.5*np.sum(logterm+((target-pred)/sd)**2))
    start = time.perf_counter()
    result = minimize(f,np.asarray(x0,dtype=float),method='L-BFGS-B',bounds=bounds,
                      options=dict(maxiter=maxiter,maxfun=maxfun,ftol=ftol))
    pred = np.asarray(mod.simulate_theta(float(result.x[0]),mod.INIT_ENV,mod.TAU0,float(result.x[1])),dtype=float)
    return dict(candidate=rank, initial_beta_sm=float(x0[0]),initial_p_sm=float(x0[1]),
                beta_sm=float(result.x[0]),p_sm=float(result.x[1]),nll=float(result.fun),
                success=bool(result.success),iterations=int(result.nit),evaluations=calls,
                elapsed_sec=time.perf_counter()-start,message=str(result.message),prediction=pred)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-script',type=Path,default=Path('SM4_A_hand_R2.py'))
    p.add_argument('--abm-csv',type=Path,default=Path('../result/interv_prob_transmission_summary_A9140_0.01-0.07.csv'))
    p.add_argument('--jobs',type=int,default=-1,help='-1 means use all logical CPUs')
    p.add_argument('--out-dir',type=Path,default=None)
    p.add_argument('--beta-grid',type=int,default=20)
    p.add_argument('--p-grid',type=int,default=11)
    p.add_argument('--maxiter',type=int,default=100)
    args=p.parse_args()
    if args.beta_grid < 3 or args.p_grid < 3: p.error('Grid sizes must both be >= 3')
    mod=load_model(args.model_script)
    model_file=args.model_script.resolve()
    df=pd.read_csv(args.abm_csv)
    mask=np.isclose(pd.to_numeric(df['beta'],errors='coerce'),ABM_BETA,atol=1e-7,rtol=0)
    if 'handwash' in df:
        mask &= np.isclose(pd.to_numeric(df['handwash'],errors='coerce'),ABM_P,atol=1e-7,rtol=0)
    sub=df.loc[mask]
    if len(sub)!=1: raise ValueError(f'Expected exactly one ABM row beta=.040, p=.90; found {len(sub)}')
    mean,std=read_vec(sub.iloc[0]['mean']),read_vec(sub.iloc[0]['std'])
    if mean.shape!=(19,) or std.shape!=(19,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std<0):
        raise ValueError('ABM mean/std must be 19 finite monthly values; std nonnegative')
    target=np.cumsum(mean)
    sd=np.maximum(np.sqrt(np.cumsum(std**2)),1e-6)
    out=args.out_dir or Path('sm_fit')/('case07_wash108_fast_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=True)
    beta_grid=np.linspace(*mod.BETA_BOUNDS,args.beta_grid)
    p_grid=np.linspace(*mod.P_BOUNDS,args.p_grid)
    print(f'Case 07 only | ABM beta={ABM_BETA}, p={ABM_P}',flush=True)
    print(f'Model: {model_file} | steps/day={mod.STEPS_PER_DAY} | washes/day={mod.WASH_OPPORTUNITIES_PER_DAY}',flush=True)
    print(f'Grid {len(beta_grid)}x{len(p_grid)}={len(beta_grid)*len(p_grid)}, jobs={args.jobs}',flush=True)
    start=time.perf_counter()
    rows=Parallel(n_jobs=args.jobs,backend='loky',verbose=5)(
        delayed(evaluate_grid_row)(b,p_grid,target,sd,model_file) for b in beta_grid)
    grid=pd.DataFrame([r for chunk in rows for r in chunk],columns=['beta_sm','p_sm','nll'])
    grid.to_csv(out/'case07_grid.csv',index=False)
    ordered=grid.sort_values('nll').reset_index(drop=True)
    # Starts along the ridge should be separated to test whether 3 optima agree.
    starts=[]
    min_beta_sep=max(.05,(mod.BETA_BOUNDS[1]-mod.BETA_BOUNDS[0])*.05)
    min_p_sep=.04
    for row in ordered.itertuples(index=False):
        x=np.array([row.beta_sm,row.p_sm])
        if all(abs(x[0]-v[0])>=min_beta_sep or abs(x[1]-v[1])>=min_p_sep for v in starts):
            starts.append(x)
        if len(starts)==3:break
    if len(starts)!=3: raise RuntimeError('Unable to select 3 distinct grid starting points')
    print('Three local starts:',starts,flush=True)
    maxfun=getattr(mod,'MAXFUN',3000)
    ftol=getattr(mod,'FTOL',1e-8)
    bounds=[tuple(mod.BETA_BOUNDS),tuple(mod.P_BOUNDS)]
    local=Parallel(n_jobs=min(3,abs(args.jobs) if args.jobs>0 else 3),backend='loky',verbose=5)(
        delayed(optimize_local)(i,x,target,sd,model_file,args.maxiter,maxfun,ftol,bounds)
        for i,x in enumerate(starts,1))
    predictions={'month':np.arange(1,20),'abm_cumulative_mean':target,
                 'abm_minus_1sd':target-sd,'abm_plus_1sd':target+sd}
    metrics=[]
    for item in local:
        item=dict(item)
        curve=item.pop('prediction')
        name=f"local_{item['candidate']}"
        predictions[name]=curve
        item['rmse']=float(np.sqrt(np.mean((target-curve)**2)))
        item['final_cumulative_hai']=float(curve[-1])
        item['effective_kappa_old_form']=item['beta_sm']**2*(1-item['p_sm'])/max(item['p_sm'],1e-15)
        metrics.append(item)
    pd.DataFrame(predictions).to_csv(out/'case07_local3_predictions.csv',index=False)
    metrics_df=pd.DataFrame(metrics)
    metrics_df.to_csv(out/'case07_local3_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(10.5,6.4))
    xx=np.arange(1,20)
    ax.fill_between(xx,target-sd,target+sd,color='0.65',alpha=.25,label='ABM mean ± cumulative SD')
    ax.plot(xx,target,'ko-',lw=2.4,ms=4,label='ABM mean')
    for item in metrics:
        i=int(item['candidate'])
        ax.plot(xx,predictions[f'local_{i}'],lw=2.0,marker=['s','^','D'][i-1],markersize=3,
                linestyle=['-','--','-.'][i-1],
                label=f"Local {i}: β={item['beta_sm']:.5f}, p={item['p_sm']:.5f}" +
                      ('' if item['success'] else ' (not converged)'))
    ax.set(title='Case 07: Three Local Fits vs ABM (wash108 surrogate)',xlabel='Month',
           ylabel='Cumulative HAI incidence',xticks=xx)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out/'case07_local3_comparison.png',dpi=250,bbox_inches='tight')
    fig.savefig(out/'case07_local3_comparison.pdf',bbox_inches='tight')
    plt.close(fig)
    # Additional plot shows distinctions even if curves nearly overlap.
    fig,ax=plt.subplots(figsize=(10,5.5))
    for item in metrics:
        i=int(item['candidate'])
        ax.plot(xx,predictions[f'local_{i}']-target,marker='.',lw=2,
                label=f'Local {i} (NLL={item["nll"]:.5f})')
    ax.axhline(0,color='black',lw=1)
    ax.set(title='Case 07: Surrogate - ABM cumulative residuals',xlabel='Month',
           ylabel='Cumulative HAI difference',xticks=xx)
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(out/'case07_local3_residuals.png',dpi=250,bbox_inches='tight')
    plt.close(fig)
    print('\n=== ACTUAL 3 LOCAL FITS ===',flush=True)
    print(metrics_df[['candidate','initial_beta_sm','initial_p_sm','beta_sm','p_sm','nll',
                      'success','iterations','evaluations','rmse']].to_string(index=False),flush=True)
    print(f'Elapsed: {(time.perf_counter()-start)/60:.2f} minutes | Outputs: {out.resolve()}',flush=True)

if __name__=='__main__':main()
