"""Posthoc calibration/evaluation comparisons from public scalar records.

No model, fit, random draw or patient input is used. These unadjusted
diagnostics neither prove exchangeability nor identify a procedural cause.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def compare(calibration, evaluation, scale=1.0):
    a, b = np.asarray(calibration, float), np.asarray(evaluation, float)
    if any(x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() for x in (a, b)):
        raise ValueError('two complete finite samples of length at least two required')
    n, m = len(a), len(b)
    sa, sb = np.sort(a), np.sort(b)
    # Independent pair-count reconstruction, including half of every tie.
    left, right = np.searchsorted(sa, b, side='left'), np.searchsorted(sa, b, side='right')
    u = float(np.sum((left + right) / 2))
    mw = stats.mannwhitneyu(b, a, alternative='greater', method='asymptotic', use_continuity=True)
    assert u == mw.statistic
    grid, tie_counts = np.unique(np.r_[a, b], return_counts=True)
    variance = m*n/12 * (m+n+1 - np.sum(tie_counts**3-tie_counts)/((m+n)*(m+n-1)))
    p_u = float(stats.norm.sf((u-m*n/2-.5)/math.sqrt(variance))) if variance > 0 else 1.0
    assert abs(p_u-mw.pvalue) < 1e-12
    ca = np.searchsorted(sa, grid, side='right') / n
    cb = np.searchsorted(sb, grid, side='right') / m
    distance = float(np.max(np.abs(ca-cb)))
    ks = stats.ks_2samp(a, b, alternative='two-sided', method='asymp')
    assert abs(distance-ks.statistic) < 1e-14
    shift = float((b.mean()-a.mean())*scale)
    se = float(math.sqrt(a.var(ddof=1)/n+b.var(ddof=1)/m)*abs(scale))
    return {'calibration_n':n, 'evaluation_n':m, 'MW_eval_greater_U':u,
            'MW_eval_greater_p':float(mw.pvalue), 'KS_D':distance, 'KS_two_sided_asymptotic_p':float(ks.pvalue),
            'calibration_mean':float(a.mean()*scale), 'evaluation_mean':float(b.mean()*scale),
            'mean_shift':shift, 'independent_cohort_SE':se,
            'calibration_SD':float(a.std(ddof=1)*abs(scale)), 'evaluation_SD':float(b.std(ddof=1)*abs(scale)),
            'shift_in_pooled_SD':float((b.mean()-a.mean())/math.sqrt(((n-1)*a.var(ddof=1)+(m-1)*b.var(ddof=1))/(n+m-2))) if variance > 0 else 0.0,
            'U_pair_count_verified':True, 'KS_ECDF_verified':True}


def sign_tail(positive, total):
    """Binomial(., .5) reference; unequal-size skewed means need not have equal signs."""
    return sum(math.comb(total, k) for k in range(positive, total+1))/2**total


def read_records(path):
    with Path(path).open(newline='') as stream:
        yield from csv.DictReader(stream)


def diagnose(data):
    data = Path(data)
    groups, seen = {}, set()
    for row in read_records(data/'pools.csv'):
        n,pool,law,purpose,index = [int(row[k]) for k in ('N','pool','law','purpose','index')]
        identity = (n,pool,law,purpose,index)
        if identity in seen: raise ValueError('duplicate pool identity')
        seen.add(identity)
        if law == 0: groups.setdefault((n,pool,purpose), []).append((index,float(row['statistic'])))
    expected={(n,pool,law,purpose,index) for n in (1557,6274) for pool in range(20)
              for law,purpose,count in ((0,0,999),(0,1,500),(7,1,200)) for index in range(count)}
    assert seen == expected
    scale=1000/(2*np.log(2)); pools=[]; combined=[]
    for n in (1557,6274):
        all_a,all_b,shifts=[],[],[]
        for pool in range(20):
            a=np.array([v for _,v in sorted(groups[(n,pool,0)])])
            b=np.array([v for _,v in sorted(groups[(n,pool,1)])])
            row=compare(a,b,scale);pools.append({'N':n,'pool':pool,**row})
            all_a.extend(a);all_b.extend(b);shifts.append(row['mean_shift'])
        row=compare(all_a,all_b,scale);positive=sum(x>0 for x in shifts)
        row.update(N=n,positive_mean_shift_pools=positive,pools=20,
                   pool_mean_shift=float(np.mean(shifts)),pool_difference_SD=float(np.std(shifts,ddof=1)),
                   pool_difference_SE=float(np.std(shifts,ddof=1)/np.sqrt(20)),
                   equal_probability_sign_reference_tail=sign_tail(positive,20),
                   sign_reference_assumption='Independent pool differences have positive probability .5; this is not implied by equal distributions with unequal sample counts.')
        assert abs(row['mean_shift']-row['pool_mean_shift']) < 1e-12
        combined.append(row)
    base={};seen_lrt=set()
    for row in read_records(data/'lrt.csv'):
        n,law,purpose,index=[int(row[k]) for k in ('N','law','purpose','index')];method=row['method']
        identity=(n,law,purpose,index,method);assert identity not in seen_lrt;seen_lrt.add(identity)
        if law==0:base.setdefault((n,method,purpose),[]).append((index,float(row['statistic'])))
    methods=sorted({key[1] for key in base});assert len(methods)==2 and len(seen_lrt)==8796
    confirmation=[]
    for n in (1557,6274):
        for method in methods:
            values=[]
            for purpose,count in ((0,999),(1,1000)):
                rows=sorted(base[(n,method,purpose)]);assert [x[0] for x in rows]==list(range(count))
                values.append([x[1] for x in rows])
            residual='LRT' not in method
            confirmation.append({'N':n,'method':method,'units':'bits per 1000' if residual else 'raw likelihood ratio',
                                 **compare(*values,scale if residual else 1)})
    return {'scope':'Posthoc unadjusted comparisons prompted by the referee; not a test of algorithmic causation or proof of exchangeability.',
            'pool_statistic_units':'bits per 1000, raw D multiplied by 1000/(2 ln 2)',
            'pool_records_verified':len(seen),'confirmation_records_verified':len(seen_lrt),
            'pool_combined':combined,'per_pool':pools,'confirmation':confirmation,
            'fits':0,'draws':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=diagnose(args.data)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'pool_combined':result['pool_combined'],'confirmation':result['confirmation']}))


if __name__=='__main__': main()
