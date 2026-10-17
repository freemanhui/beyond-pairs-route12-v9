"""Explicit-input benchmark entry points; no private paths or biological assets.

The numerical core is the unchanged v11 implementation. The default demo is
small software evidence, not a calibration or power study.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from . import core as c

SCOPE = {'patient_data': False, 'input_provenance': 'built-in artificial law',
         'surrogate_is_MSK_reproduction': False,
         'calibration': 'specified known null law, not cohort-refitted bootstrap'}

def run_scope(custom=False, counts_only=False):
    if counts_only:
        return {'patient_data': 'unknown', 'input_provenance': 'external counts, unknown',
                'calibration': 'not performed', 'known_reference_law': False}
    if custom:
        return {'patient_data': 'unknown', 'input_provenance': 'external laws, unknown',
                'calibration': SCOPE['calibration']}
    return dict(SCOPE)

def law_binding(q,p,path=None):
    digest=lambda a:hashlib.sha256(np.asarray(a,dtype='<f8').tobytes(order='C')).hexdigest()
    return {'dimension': len(q).bit_length()-1, 'states': len(q),
            'state_order': 'least-significant-bit-first binary enumeration',
            'canonical_encoding': 'C-order little-endian float64',
            'q_sha256': digest(q), 'p_sha256': digest(p),
            'input_file_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None}
SETTINGS = [(cut, count) for cut in (1e-8,1e-10,1e-12) for count in (3,5,10)]
SEED_NAMESPACE = (13, 0)
DEMO_DEFAULTS = [('N',1557),('pools',2),('calibration',99),('null',20),
                 ('alternative',20),('maximum-cohorts',278),('seed',20261017)]

def calibration_resolution(calibration, alpha=.05):
    if type(calibration) is not int or calibration < 1:
        raise ValueError('positive integer calibration count required')
    minimum=1/(calibration+1)
    return {'calibration_count':calibration,'minimum_add_one_p':minimum,'alpha':alpha,
            'rejection_attainable':minimum<=alpha,
            'scope':'Artificial software demonstration; small evaluation rosters do not qualify calibration or power.'}

def surrogate():
    """Ten binary variables: uniform Q, P=Q(1+0.3 product of first 3 spins)."""
    states, _, _, _ = c.designs(10)
    q = np.full(1024, 1/1024)
    p = q*(1+.3*np.prod(2*states[:,:3]-1,axis=1))
    return q, p

def validate_law(value):
    value = np.asarray(value,dtype=float)
    if (value.ndim!=1 or len(value)<8 or len(value)>4096
            or len(value)&(len(value)-1) or not np.isfinite(value).all()
            or np.any(value<=0) or abs(value.sum()-1)>1e-12):
        raise ValueError('positive normalized finite law on 3 to 12 binary variables required')
    return value

def load_laws(path):
    with np.load(path,allow_pickle=False) as z:
        q,p=validate_law(z['q']),validate_law(z['p'])
    if q.shape!=p.shape:raise ValueError('q and p must have the same state order and support')
    # Verify that the declared null is pairwise, without fitting it.
    _,g,_,_=c.designs(len(q).bit_length()-1)
    design=np.column_stack([np.ones(len(q)),g]);logq=np.log(q)
    residual=logq-design@np.linalg.lstsq(design,logq,rcond=None)[0]
    if np.max(np.abs(residual))>1e-8:raise ValueError('declared q is not a pairwise exponential law')
    return q,p

def validate_counts(value):
    value=np.asarray(value)
    if (value.ndim!=2 or value.shape[0]!=5 or value.shape[1]<8 or value.shape[1]>4096
            or value.shape[1]&(value.shape[1]-1) or not np.isfinite(value).all()
            or np.any(value<0) or np.any(value!=np.floor(value))
            or np.any(value.sum(axis=1)<=1)):
        raise ValueError('five nonnegative integer count tables with at least two observations per fold required')
    return value.astype(np.int64)

def validate_roster(n,pools,calibration,null,alternative,maximum):
    if any(type(x) is not int or x<1 for x in [n,pools,calibration,null,alternative,maximum]) or n<20:
        raise ValueError('positive integer roster and N>=20 required')
    total=pools*(calibration+null+alternative)
    if total>maximum:raise ValueError('roster exceeds explicit --maximum-cohorts; no automatic expansion')
    return total

def create_output(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False);return path

def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def endpoint_status(parameters,attempted,accepted,error):
    return {'attempted':int(attempted),'accepted':int(accepted),'error':error,
            'finite_parameter_rows':int(np.isfinite(parameters).all(axis=1).sum())}

def analyse_cohort(tables,out,q=None,extended=True,scope=None):
    """Counts-only D/LRT, optionally supplemented by known-Q diagnostics."""
    tables=validate_counts(tables);out=create_output(out)
    params,diag,nfit,nok,error=c.fit_training(tables)
    th2,th3,ldiag,lrt,raw,la,lk,lerror=c.pooled_lrt(tables.sum(axis=0))
    np.savez_compressed(out/'endpoints.npz',counts=tables,fold_parameters=params,fold_diagnostics=diag,
                        pooled_pair=th2,pooled_triple=th3,pooled_diagnostics=ldiag)
    report={'fold_fits':endpoint_status(params,nfit,nok,error),
            'LRT_fits':{'attempted':la,'accepted':lk,'error':lerror},
            'fit_attempts':nfit+la,'fit_accepted':nok+lk,'statistics':{},
            'scope': dict(scope or run_scope(counts_only=True), calibration='not performed in this cohort record',
                          known_reference_law=q is not None)}
    if error is None:
        value,folds,masks,ranks=c.fully_learned(tables,params,check=True)
        report.update(fold_D=folds.tolist(),mask_sizes=masks.tolist(),ranks=ranks.tolist())
        report['statistics']['residual_D']=value
        if extended:
            for cut,minimum in SETTINGS:
                v,_,_,_=c.fully_learned(tables,params,cutoff=cut,minimum=minimum,check=True)
                report['statistics'][f'eig={cut:g},count={minimum}']=v
    if lerror is None:
        report['statistics']['LRT']=float(lrt);report['raw_LRT']=float(raw)
        if q is not None and extended:
            q=validate_law(q);_,g,_,_=c.designs(len(q).bit_length()-1)
            qhat,_=c.law(th2,g)
            values,masks=c.oracle_factorial(tables.sum(axis=0),q,qhat)
            report['statistics'].update(zip(c.ORACLE_METHODS,map(float,values)))
            report['factorial_mask_sizes']=masks.tolist()
    report['complete']=error is None and lerror is None
    put(out/'result.json',report)
    if not report['complete']:raise RuntimeError('incomplete numerical endpoint retained; see result.json; no case dropped')
    return report

def demo(args):
    total=validate_roster(args.N,args.pools,args.calibration,args.null,args.alternative,args.maximum_cohorts)
    q,p=load_laws(args.laws) if args.laws else surrogate()
    out=create_output(args.output);start=time.monotonic();records=[];fits=0
    scope=run_scope(custom=bool(args.laws))
    put(out/'registration.json',{'N':args.N,'pools':args.pools,'calibration':args.calibration,'null':args.null,
        'alternative':args.alternative,'cohorts':total,'seed':args.seed,'pair_triple_fit_bound':total*7,
        'law_source':'user-supplied positive laws' if args.laws else 'public uniform/parity surrogate',
        'law_binding':law_binding(q,p,args.laws), 'scope':scope,
        'seed_namespace':list(SEED_NAMESPACE), 'calibration_resolution':calibration_resolution(args.calibration)})
    for pool in range(args.pools):
        for law_id,purpose,count in [(0,0,args.calibration),(0,1,args.null),(1,1,args.alternative)]:
            for index in range(count):
                identity=[args.N,pool,law_id,purpose,index]
                seed=[args.seed,*SEED_NAMESPACE,*identity]
                rng=np.random.default_rng(np.random.SeedSequence(seed))
                tables=np.array([rng.multinomial(int(n),q if law_id==0 else p) for n in c.fold_sizes(args.N)])
                result=analyse_cohort(tables,out/('cohort-'+'-'.join(map(str,identity))),q,scope=scope)
                fits+=result['fit_attempts'];records.append(dict(identity=identity,seed=seed,statistics=result['statistics']))
                print(json.dumps({'completed':len(records),'cohorts':total,'fits':fits}),flush=True)
    rows=[]
    for pool in range(args.pools):
        local=[r for r in records if r['identity'][1]==pool]
        for method in records[0]['statistics']:
            cal=[r['statistics'][method] for r in local if r['identity'][3]==0]
            for row in local:
                identity=row['identity'];v=row['statistics'][method]
                decision=int(c.rank_decisions(cal,[v])[0]) if identity[3]==1 else ''
                rows.append(dict(N=identity[0],pool=pool,law=identity[2],purpose=identity[3],index=identity[4],method=method,statistic=v,reject=decision))
    with (out/'decisions.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # The public parity law is through-third-order with exactly uniform pair projection.
    if not args.laws:
        _,g,_,full=c.designs(10)
        a2=c.sandwich(p,q,g);a3=c.sandwich(p,p,full)
        for obj in [a2,a3]:obj.pop('H');obj.pop('J')
        put(out/'exact-surrogate.json',dict(L3_bits=float(p@np.log2(p/q)),pair_trace=a2,triple_trace=a3))
    put(out/'summary.json',{'passed':True,'cohorts':total,'new_artificial_pair_triple_fits':fits,
                           'scientific_MSK_cohorts':'unknown provenance' if args.laws else 0,
                           'elapsed_s':time.monotonic()-start,'scope':scope})

def build_parser():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo');d.add_argument('--output',required=True);d.add_argument('--laws')
    for name,default in DEMO_DEFAULTS:
        d.add_argument('--'+name,type=int,default=default)
    one=sub.add_parser('cohort');one.add_argument('--counts',required=True);one.add_argument('--output',required=True)
    return parser

def main():
    args=build_parser().parse_args()
    if args.command=='demo':demo(args)
    else:
        values=np.load(args.counts,allow_pickle=False)
        analyse_cohort(values,args.output,extended=False)

if __name__=='__main__':main()
