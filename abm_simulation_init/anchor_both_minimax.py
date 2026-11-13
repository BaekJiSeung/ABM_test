"""Anchor-constrained, both-monotone minimax selection for beta sweep.

Example (run from directory containing SM4_A_hand_R2.py):
 python anchor_both_minimax.py --grid-dir sm_fit/mapping_2D_17cases_GRID200x70/grids \
    --beta-csv ../result/interv_prob_transmission_summary_A9140_0.01-0.07.csv \
    --model-script SM4_A_hand_R2.py

Case 07 (beta_ABM=.040, p_ABM=.90) is EXACTLY fixed to
(beta_SM=3.387, p_SM=.90), although those are off the original grid.
All 13 cases must be coordinate-wise nondecreasing in beta_SM and p_SM.
Objective: minimize max_i [NLL_i - min_grid_NLL_i]; break ties by sum.
"""
from __future__ import annotations
import argparse
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ANCHOR_CASE = 7
ANCHOR_BETA = .040
ANCHOR_SM_BETA = 3.387
ANCHOR_SM_P = .90
N_CASES = 13


def load_model(path):
    spec = importlib.util.spec_from_file_location('cre_surrogate_model', str(path.resolve()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def anchor_nll(model, beta_csv):
    df = pd.read_csv(beta_csv)
    subset = df[np.isclose(pd.to_numeric(df['beta'], errors='coerce'), ANCHOR_BETA, rtol=0, atol=1e-7)]
    if len(subset) != 1:
        raise ValueError(f'Expected exactly one ABM row at beta={ANCHOR_BETA}; found {len(subset)}')
    row = subset.iloc[0]
    parse = lambda v: np.asarray(ast.literal_eval(v) if isinstance(v, str) else v, dtype=float)
    mean = parse(row['mean'])
    std = parse(row['std'])
    if len(mean) != 19 or len(std) != 19:
        raise ValueError('Expected 19 monthly ABM means and standard deviations')
    target = np.cumsum(mean)
    sd = np.maximum(np.sqrt(np.cumsum(std**2)), 1e-6)
    pred = np.asarray(model.simulate_theta(ANCHOR_SM_BETA, 9, 140, ANCHOR_SM_P), dtype=float)
    if len(pred) != 19:
        raise ValueError('Surrogate must return 19 cumulative monthly HAI values')
    return float(.5*np.sum(np.log(2*np.pi*sd**2)+((target-pred)/sd)**2))


def prefix_or(a):
    return np.logical_or.accumulate(np.logical_or.accumulate(a, axis=0), axis=1)


def prefix_min_arg(a):
    """Prefix minima and predecessor indices, inclusive both coordinates."""
    nr, nc = a.shape
    v = np.full((nr,nc),np.inf)
    arg = np.full((nr,nc),-1,dtype=np.int32)
    for i in range(nr):
        for j in range(nc):
            best = a[i,j]; idx = i*nc+j if np.isfinite(best) else -1
            if i and v[i-1,j] < best:
                best,idx=v[i-1,j],arg[i-1,j]
            if j and v[i,j-1] < best:
                best,idx=v[i,j-1],arg[i,j-1]
            v[i,j],arg[i,j]=best,idx
    return v,arg


def feasible(costs,t):
    reach = costs[0] <= t+1e-12
    for c in costs[1:]:
        reach = prefix_or(reach) & (c<=t+1e-12)
        if not reach.any(): return False
    return bool(reach.any())


def best_path(costs,t):
    dp = np.where(costs[0]<=t+1e-12,costs[0],np.inf)
    pointers=[]
    for c in costs[1:]:
        pv,pa=prefix_min_arg(dp)
        dp=np.where(c<=t+1e-12,pv+c,np.inf)
        pointers.append(pa)
    if not np.isfinite(dp).any(): raise ValueError('No feasible path')
    idx=int(np.argmin(dp)); inds=[idx]
    nc=dp.shape[1]
    for pa in reversed(pointers):
        idx=int(pa.flat[idx]); inds.append(idx)
    return [(i//nc,i%nc) for i in reversed(inds)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-dir',type=Path,required=True)
    parser.add_argument('--beta-csv',type=Path,required=True)
    parser.add_argument('--model-script',type=Path,default=Path('SM4_A_hand_R2.py'))
    parser.add_argument('--output',type=Path,default=Path('beta_sweep_anchor_both_minimax.csv'))
    args=parser.parse_args()
    frames=[]
    for case in range(1,N_CASES+1):
        path=args.grid_dir/f'grid_case_{case:02d}.csv'
        df=pd.read_csv(path)
        if len(df)!=14000 or not {'beta_idx','p_idx','beta_sm','p_sm','nll'}.issubset(df.columns):
            raise ValueError(f'{path}: expected 14,000 points with standard columns')
        df=df.sort_values(['beta_idx','p_idx']).reset_index(drop=True)
        if not (np.array_equal(df.beta_idx, np.repeat(np.arange(200),70)) and
                np.array_equal(df.p_idx,np.tile(np.arange(70),200))):
            raise ValueError(f'{path}: unexpected grid order/index')
        frames.append(df)
    base_beta=frames[0].beta_sm.to_numpy().reshape(200,70)[:,0]
    base_p=frames[0].p_sm.to_numpy().reshape(200,70)[0,:]
    if not (np.all(np.diff(base_beta)>0) and np.all(np.diff(base_p)>0)):
        raise ValueError('Axes must be strictly increasing')
    for df in frames[1:]:
        if not (np.allclose(df.beta_sm.to_numpy().reshape(200,70)[:,0],base_beta) and
                np.allclose(df.p_sm.to_numpy().reshape(200,70)[0,:],base_p)):
            raise ValueError('Grid axes mismatch across cases')
    # Insert exact anchor values as NEW axes. Other cases cannot use off-grid values.
    beta_axis=np.sort(np.unique(np.r_[base_beta,ANCHOR_SM_BETA]))
    p_axis=np.sort(np.unique(np.r_[base_p,ANCHOR_SM_P]))
    bi=np.searchsorted(beta_axis,base_beta)
    pi=np.searchsorted(p_axis,base_p)
    ai=int(np.searchsorted(beta_axis,ANCHOR_SM_BETA))
    aj=int(np.searchsorted(p_axis,ANCHOR_SM_P))
    baseline=[float(df.nll.min()) for df in frames]
    nll_anchor=anchor_nll(load_model(args.model_script),args.beta_csv)
    raw=[]; costs=[]
    for i,df in enumerate(frames):
        z=np.full((len(beta_axis),len(p_axis)),np.inf)
        if i != ANCHOR_CASE-1:
            z[np.ix_(bi,pi)]=df.nll.to_numpy(float).reshape(200,70)
        else:
            z[ai,aj]=nll_anchor
        raw.append(z)
        costs.append(np.maximum(z-baseline[i],0))
    vals=np.unique(np.concatenate([x[np.isfinite(x)] for x in costs]))
    lo,hi=0,len(vals)-1
    if not feasible(costs,float(vals[-1])):
        raise ValueError('No monotone path through anchor')
    while lo<hi:
        mid=(lo+hi)//2
        if feasible(costs,float(vals[mid])): hi=mid
        else: lo=mid+1
    threshold=float(vals[lo])
    path=best_path(costs,threshold)
    rows=[]
    for k,(i,j) in enumerate(path):
        rows.append({'case_id':k+1,'beta_abm':round(.01+.005*k,5),'p_abm':.90,
                     'beta_sm':beta_axis[i],'p_sm':p_axis[j],
                     'nll':raw[k][i,j],'best_grid_nll':baseline[k],
                     'delta_nll':costs[k][i,j],'anchor':k+1==ANCHOR_CASE})
    result=pd.DataFrame(rows)
    if not (np.all(np.diff(result.beta_sm)>=-1e-12) and np.all(np.diff(result.p_sm)>=-1e-12)):
        raise AssertionError('Monotonicity violation')
    if not (abs(result.iloc[6].beta_sm-ANCHOR_SM_BETA)<1e-12 and abs(result.iloc[6].p_sm-ANCHOR_SM_P)<1e-12):
        raise AssertionError('Anchor violation')
    args.output.parent.mkdir(exist_ok=True,parents=True)
    result.to_csv(args.output,index=False,encoding='utf-8-sig')
    print(result.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print(f'Anchor NLL: {nll_anchor:.6f}; baseline Grid NLL: {baseline[6]:.6f}')
    print(f'Max ΔNLL: {result.delta_nll.max():.6f}; Total ΔNLL: {result.delta_nll.sum():.6f}')
    print(f'Wrote {args.output.resolve()}')

if __name__=='__main__':
    main()
