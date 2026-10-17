"""Single-invocation, retained-endpoint referee comparisons and calibration pools."""
import argparse
import json
import os
from pathlib import Path
import time

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
from beyond_pairs.route12_benchmarks import core as c
from beyond_pairs.route12_v9.inputs import Inputs
from route12_benchmarks_runtime import R,A,B,bind,write,verify,storage,elapsed


class Blocks:
    def __init__(self,out,fields):
        self.out=out;out.mkdir();self.fields=fields;self.pending=[];self.blocks=[];self.total=0
        self.journal=(out/'journal.jsonl').open('x')
    def add(self,arrays,note):
        self.pending.append(arrays);self.total+=1
        self.journal.write(json.dumps(note,allow_nan=False)+'\n')
        if len(self.pending)==100:self.flush()
    def flush(self):
        if not self.pending:return
        path=self.out/f'block-{len(self.blocks)+1:04d}.npz'
        np.savez_compressed(path,**{key:np.asarray([r[i] for r in self.pending]) for i,key in enumerate(self.fields)})
        self.blocks.append({'binding':bind(path),'count':len(self.pending)})
        self.journal.flush();os.fsync(self.journal.fileno());self.pending=[]
        print(json.dumps({'completed':self.total,'blocks':len(self.blocks),'phase':self.out.name}),flush=True)
    def close(self):
        self.flush();self.journal.close()
        write(self.out/'index.json',{'count':self.total,'fields':self.fields,'blocks':self.blocks})


def law_pair():
    with np.load(A/'staged-inputs/exact-laws-v1.npz',allow_pickle=False) as z:
        return z['law_0'].copy(),z['law_7'].copy()


def rate_table(ids,values,names,expected_calibration):
    ids=np.asarray(ids);values=np.asarray(values);answer=[]
    for n in [1557,6274]:
        selection=ids[:,0]==n;cal=selection&(ids[:,1]==0)&(ids[:,2]==0)
        assert cal.sum()==expected_calibration
        for j,name in enumerate(names):
            calibrated=bool(np.isfinite(values[cal,j]).all())
            for law_id in sorted(set(ids[selection&(ids[:,2]==1),1])):
                mask=selection&(ids[:,1]==law_id)&(ids[:,2]==1);complete=mask&np.isfinite(values[:,j])
                row={'N':n,'method':name,'law':int(law_id),'planned':int(mask.sum()),'finite':int(complete.sum()),'calibration_complete':calibrated}
                if calibrated:
                    decisions=c.rank_decisions(values[cal,j],values[complete,j]);count=int(decisions.sum())
                    row.update(rejections=count,undecided=int(mask.sum()-complete.sum()),
                               rate=count/int(mask.sum()) if complete.sum()==mask.sum() else None,
                               conditional_CP95=c.cp(count,int(mask.sum())) if complete.sum()==mask.sum() else None,
                               threshold=float(np.sort(values[cal,j])[expected_calibration-int(np.floor(.05*(expected_calibration+1)))]))
                answer.append(row)
    return answer


def oracle():
    inputs=Inputs();q,p=law_pair();_,g,t,_=c.designs(10);prepared=c.residual(q,g,t)
    out=B/'matched-oracle';blocks=Blocks(out,['identity','theta2','diagnostics','statistics','mask_sizes','accepted'])
    identities=[];statistics=[];failures=[];attempts=0
    for row in inputs.catalogue['cohorts']:
        identity=row['identity'];tables=inputs.cohort(identity);counts=tables.sum(axis=0)
        theta=np.full(55,np.nan);diag=np.full(4,np.nan);values=np.full(10,np.nan);masks=np.full(3,-1);error=None;attempts+=1
        try:
            fitted=c._projection(counts/counts.sum(),g,None);theta=fitted['theta'];diag=np.array(list(fitted['diagnostics'].values()))
            values,masks=c.oracle_factorial(counts,q,fitted['law'],prepared=prepared)
        except Exception as exc:
            if hasattr(exc,'partial') and 'theta' in exc.partial:theta=exc.partial['theta']
            error=str(exc);failures.append({'identity':identity,'error':error})
        blocks.add([identity,theta,diag,values,masks,error is None],{'identity':identity,'accepted':error is None,'error':error,'fit_attempts':1})
        identities.append(identity);statistics.append(values)
    blocks.close();write(out/'summary.json',{'planned':4000,'completed':len(identities),'new_pair_fit_attempts':attempts,
          'failures':failures,'methods':list(c.ORACLE_METHODS),'rates':rate_table(identities,statistics,c.ORACLE_METHODS,400),
          'scope':'Same pooled historical counts; fixed full/known-Q masks and a separately calibrated shared adaptive mask. Learned pooled pair fit is new. Not a unique causal allocation of power.'})


def lrt():
    pilot=json.loads((B/'cost-pilot/results.json').read_text());assert pilot['projection']['LRT_cost_pass']
    source=json.loads((A/'confirmation/invocation-01/index.json').read_text())['records']
    blocks=Blocks(B/'lrt',['identity','theta2','theta3','diagnostics','statistics','raw_lrt','attempted','accepted'])
    identities=[];statistics=[];failures=[];attempted=accepted=0
    for row in source:
        verify(row['binding'])
        with np.load(R/row['binding']['path'],allow_pickle=False) as z:counts=z['counts'].sum(axis=0)
        th2,th3,diag,value,raw,nfit,nok,error=c.pooled_lrt(counts)
        pair=[value,row['statistics'][3]];attempted+=nfit;accepted+=nok
        if error:failures.append({'identity':row['identity'],'error':error})
        blocks.add([row['identity'],th2,th3,diag,pair,raw,nfit,nok],{'identity':row['identity'],'error':error,'attempted':nfit,'accepted':nok})
        identities.append(row['identity']);statistics.append(pair)
    blocks.close();ids=np.array(identities);v=np.array(statistics);paired=[]
    for n in [1557,6274]:
        cal=(ids[:,0]==n)&(ids[:,1]==0)&(ids[:,2]==0)
        for law in [0,7]:
            ev=(ids[:,0]==n)&(ids[:,1]==law)&(ids[:,2]==1)
            if np.isfinite(v[cal]).all() and np.isfinite(v[ev]).all():
                paired.append({'N':n,'law':law,**c.paired(c.rank_decisions(v[cal,0],v[ev,0]),c.rank_decisions(v[cal,1],v[ev,1]))})
    write(B/'lrt/summary.json',{'planned':4398,'completed':len(ids),'new_fit_attempts':attempted,'accepted_fits':accepted,
          'failures':failures,'rates':rate_table(ids,v,['known-law LRT','fully learned D'],999),'paired_LRT_minus_D':paired,
          'scope':'Pooled unfloored pair/triple likelihood fits, known-law rank calibration, identical saved calibration/evaluation counts; retrospective matched comparison.'})


def pools():
    pilot=json.loads((B/'cost-pilot/results.json').read_text());assert pilot['projection']['pool_cost_pass']
    reg=json.loads((B/'registration-v11.json').read_text());ledger=json.loads((B/'phase-ledger-v11.json').read_text())
    # Full roster admission before the first scientific draw, with review/presentation reserves.
    projected=pilot['projection']['pool_seconds']
    assert ledger['new_compute_charged_s']+projected+1200<=reg['new_compute_cap_s']
    assert elapsed(reg)+projected+2400+3600+120<=reg['whole_programme_cap_s']
    assert storage()+pilot['projection']['pool_whole_bytes']+pilot['projection']['other_reserved_bytes']<=reg['resources']['new_whole_storage_cap_bytes']
    q,p=law_pair();blocks=Blocks(B/'pools',['identity','counts','parameters','diagnostics','statistic','fold_statistics','mask_sizes','ranks','attempted','accepted'])
    summaries=[];attempts=accepted=0;failures=[]
    for n in [1557,6274]:
        sizes=c.fold_sizes(n)
        for pool in range(20):
            stats={};pool_failures=[]
            for law_id,purpose,count in [(0,0,999),(0,1,500),(7,1,200)]:
                stats[(law_id,purpose)]=[]
                for index in range(count):
                    identity=[n,pool,law_id,purpose,index];seed=[20261016,11,3,*identity]
                    rng=np.random.default_rng(np.random.SeedSequence(seed));law=q if law_id==0 else p
                    tables=np.array([rng.multinomial(int(m),law) for m in sizes],dtype=np.uint16)
                    params,diag,nfit,nok,error=c.fit_training(tables);attempts+=nfit;accepted+=nok
                    value=np.nan;folds=np.full(5,np.nan);masks=ranks=np.full(5,-1)
                    if error is None:
                        try:value,folds,masks,ranks=c.fully_learned(tables,params,check=True)
                        except Exception as exc:error=str(exc)
                    if error:
                        item={'identity':identity,'error':error};pool_failures.append(item);failures.append(item)
                    stats[(law_id,purpose)].append(value)
                    blocks.add([identity,tables,params,diag,value,folds,masks,ranks,nfit,nok],{'identity':identity,'error':error,'attempted':nfit,'accepted':nok})
            result={'N':n,'pool':pool,'failures':pool_failures,'calibration_complete':bool(np.isfinite(stats[(0,0)]).all())}
            if not pool_failures:
                result.update(null_rejections=int(c.rank_decisions(stats[(0,0)],stats[(0,1)]).sum()),
                              alternative_rejections=int(c.rank_decisions(stats[(0,0)],stats[(7,1)]).sum()),
                              threshold=float(np.sort(stats[(0,0)])[949]))
            summaries.append(result);write(B/'pools'/f'pool-{n}-{pool:02d}.json',result)
    blocks.close();aggregates=[]
    for n in [1557,6274]:
        selected=[r for r in summaries if r['N']==n]
        if all(not r['failures'] for r in selected):
            aggregates.append({'N':n,'null':c.pool_summary([r['null_rejections'] for r in selected],[500]*20),
                               'alternative':c.pool_summary([r['alternative_rejections'] for r in selected],[200]*20)})
    write(B/'pools/summary.json',{'planned':67960,'completed':blocks.total,'new_pair_fit_attempts':attempts,'accepted_pair_fits':accepted,
          'failed_cohorts':failures,'pools':summaries,'aggregates':aggregates,
          'scope':'Forty independent calibration/evaluation pools. Equal-pool marginal estimates with cluster-level Monte Carlo uncertainty, not pooled-binomial or composite-null guarantees.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['oracle','lrt','pools']);a=p.parse_args()
    {'oracle':oracle,'lrt':lrt,'pools':pools}[a.phase]()
