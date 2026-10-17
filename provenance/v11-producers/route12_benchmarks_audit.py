"""Independent endpoint, direct-contraction, identity and decision review; no fits."""
import argparse,csv,gzip,hashlib,json,os,subprocess
from functools import lru_cache
from itertools import combinations
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from scipy.stats import beta,t as student
from route12_benchmarks_runtime import R,A,B,bind,write,verify

MAX={'moment':0.,'dual':0.,'statistic':0.,'normalization':0.}

@lru_cache(maxsize=2)
def design(d):
    x=np.array([[(i>>j)&1 for j in range(d)] for i in range(2**d)],float)
    f=np.column_stack([np.prod(x[:,list(indices)],axis=1) for order in [1,2,3] for indices in combinations(range(d),order)])
    return f[:,:d*(d+1)//2],f[:,d*(d+1)//2:],f

def sizes(n):
    exact=np.array([1250,1195,1298,1268,1263],dtype=np.int64)*n
    base=exact//6274;order=sorted(range(5),key=lambda j:(-(exact[j]%6274),j))
    for j in order[:n-int(base.sum())]:base[j]+=1
    return base

def endpoint(theta,f,target):
    energy=np.ascontiguousarray(f)@theta;top=float(max(energy));logz=top+np.log(np.sum(np.exp(energy-top)))
    logp=energy-logz;p=np.exp(logp)
    errors=[float(np.max(np.abs((target-p)@f))),float(abs((logz-theta@(target@f)+p@logp)/np.log(2))),float(abs(p.sum()-1))]
    for k,e in zip(['moment','dual','normalization'],errors):MAX[k]=max(MAX[k],e)
    assert np.isfinite(theta).all() and all(e<=tol for e,tol in zip(errors,[1e-9,1e-7,1e-12]))
    return p,logp

def weight(cov,cut=1e-10):
    if cov.shape==(0,0):return cov,0
    values,vectors=np.linalg.eigh(.5*(cov+cov.T));keep=values>cut*values[-1]
    assert values[-1]>0 and values[0]>=-max(1e-12,values[-1]*1e-8)
    result=(vectors[:,keep]/values[keep])@vectors[:,keep].T
    return result,int(keep.sum())

def residual(q,g,t):
    # The registered computational map forms the nuisance inverse via F F^T.
    # Division by eigenvalues is algebraically equivalent but can materially
    # alter downstream nearly singular cross-weight statistics. Preserve this
    # exact map contract; the U contraction below is independently written.
    gc=g-q@g;tc=t-q@t;cov=(gc*q[:,None]).T@gc
    values,vectors=np.linalg.eigh((cov+cov.T)/2);keep=values>1e-10*values[-1]
    assert values[-1]>0 and values[0]>=-max(1e-12,values[-1]*1e-8)
    factor=np.ascontiguousarray(vectors[:,keep]/np.sqrt(values[keep]))
    inverse=factor@factor.T
    return np.ascontiguousarray(tc-gc@(((tc*q[:,None]).T@gc)@inverse).T)

def contraction(counts,r,w):
    n=counts.sum()
    if not r.shape[1]:return 0.
    total=counts@r;diagonal=np.einsum('ij,ij->i',r@w,r)
    return float((total@w@total-counts@diagonal)/(n*(n-1)))

def close(actual,expected,tol=1e-8):
    error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    MAX['statistic']=max(MAX['statistic'],error);assert error<=tol,(error,tol)

def d_statistics(tables,parameters,settings=((1e-10,5),)):
    g,t,_=design(tables.shape[1].bit_length()-1);total=tables.sum(axis=0);n=total.sum();fs=tables.sum(axis=1)
    values=np.zeros((len(settings),5));masks=np.zeros_like(values,dtype=int);ranks=masks.copy()
    for k,held in enumerate(tables):
        train=total-held;q,_=endpoint(parameters[k],g,train/train.sum());r=residual(q,g,t)
        for j,(cut,minimum) in enumerate(settings):
            mask=train.sum()*(q@t)>=minimum;rr=np.ascontiguousarray(r[:,mask]);w,rank=weight((rr*q[:,None]).T@rr,cut)
            values[j,k]=contraction(held,rr,w);masks[j,k]=mask.sum();ranks[j,k]=rank
    return values@fs/n,values,masks,ranks

def oracle_values(counts,theta,q0):
    g,t,_=design(len(counts).bit_length()-1)
    reconstructed,_=endpoint(theta,g,counts/counts.sum())
    # The fitted endpoint contract stores the solver's normalized weights.
    # Reproduce that declared representation; exp(energy-logZ) is only an
    # approximately equal law and can select a different mask at count 5.
    energy=g@theta;scaled=np.exp(energy-energy.max());qh=scaled/scaled.sum()
    assert np.max(np.abs(qh-reconstructed))<=1e-12
    ro=residual(q0,g,t);rl=residual(qh,g,t);values=[];maskcounts=[]
    for mask in [np.ones(120,bool),counts.sum()*(q0@t)>=5]:
        a,b=np.ascontiguousarray(ro[:,mask]),np.ascontiguousarray(rl[:,mask]);wa,_=weight((a*q0[:,None]).T@a);wb,_=weight((b*qh[:,None]).T@b)
        for rr in [a,b]:
            for w in [wa,wb]:values.append(contraction(counts,rr,w))
        maskcounts.append(int(mask.sum()))
    mask=counts.sum()*(qh@t)>=5
    for q,r in [(q0,ro),(qh,rl)]:
        rr=np.ascontiguousarray(r[:,mask]);w,_=weight((rr*q[:,None]).T@rr);values.append(contraction(counts,rr,w))
    return np.array(values),maskcounts+[int(mask.sum())]

def decision(cal,ev):
    cal=np.asarray(cal);out=[]
    for start in range(0,len(ev),100):
        count=(cal[:,None]>=np.array(ev[start:start+100])[None,:]).sum(axis=0)
        out.extend(20*(1+count)<=len(cal)+1)
    return np.asarray(out,bool)

def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

def check_rates(summary,ids,values):
    ids=np.array(ids);values=np.array(values);names=list(dict.fromkeys(row['method'] for row in summary['rates']))
    for row in summary['rates']:
        j=names.index(row['method']);cal=(ids[:,0]==row['N'])&(ids[:,1]==0)&(ids[:,2]==0)
        ev=(ids[:,0]==row['N'])&(ids[:,1]==row['law'])&(ids[:,2]==1)
        complete=np.isfinite(values[cal,j]).all();finite=ev&np.isfinite(values[:,j])
        assert row['planned']==int(ev.sum()) and row['finite']==int(finite.sum()) and row['calibration_complete']==bool(complete)
        if complete:
            k=int(decision(values[cal,j],values[finite,j]).sum());assert row['rejections']==k and row['undecided']==int(ev.sum()-finite.sum())
            rank=len(values[cal,j])-int(np.floor(.05*(len(values[cal,j])+1)))
            close(row['threshold'],np.sort(values[cal,j])[rank],0.)
            if finite.sum()==ev.sum():close(row['conditional_CP95'],interval(k,int(ev.sum())),1e-12)
            else:assert row['conditional_CP95'] is None and row['rate'] is None

def block_rows(area):
    idx=json.loads((area/'index.json').read_text());seen=0
    for block in idx['blocks']:
        verify(block['binding'])
        with np.load(R/block['binding']['path'],allow_pickle=False) as item:
            for i in range(block['count']):yield {k:item[k][i].copy() for k in idx['fields']};seen+=1
    assert seen==idx['count']

def pilot():
    old=json.loads((B/'cost-pilot/results.json').read_text());g,t,f=design(10)
    for r in old['records']:
        verify(r['record'])
        with np.load(R/r['record']['path'],allow_pickle=False) as z:
            tables=z['counts'];counts=tables.sum(axis=0);p2,log2=endpoint(z['theta2'],g,counts/counts.sum());p3,log3=endpoint(z['theta3'],f,counts/counts.sum())
            close(max(0.,2*counts@(log3-log2)),r['LRT']);a,b,m,k=d_statistics(tables,z['parameters']);close(a[0],r['D']);close(b[0],z['fold_statistics'])
    write(B/'audit-artificial-known-v11.json',{'passed':True,'artificial_cohorts':8,'pair_triple_vectors':56,'new_fits':0,'maxima':MAX})

def run(resume=False):
    assert json.loads((B/'audit-artificial-known-v11.json').read_text())['passed']
    assert json.loads((B/'contract-correction-cases/summary.json').read_text())['passed']
    out=B/'independent-review'
    if resume:
        scope=json.loads((B/'audit-continuation-scope-v11.json').read_text());verify(scope['inherited_trace'])
        assert sorted(x.name for x in out.iterdir())==['traces.json']
    else:out.mkdir()
    counts={};identities={}
    with np.load(A/'staged-inputs/exact-laws-v1.npz',allow_pickle=False) as z:
        q0=z['law_0'].copy();p=z['law_7'].copy();p3=z['source_P3'].copy();theta2=z['q_theta'].copy();theta3=z['source_P3_theta'].copy()
    g,t,f=design(10)
    if not resume:
        qcheck,_=endpoint(theta2,g,p);p3check,_=endpoint(theta3,f,p)
        close(qcheck,q0,1e-12);close(p3check,p3,1e-12)
        traces=json.loads((B/'traces/results.json').read_text())
        for name,q,feat in [('pair',q0,g),('triple',p3,f)]:
            h=(feat-q@feat).T@(q[:,None]*(feat-q@feat));j=(feat-p@feat).T@(p[:,None]*(feat-p@feat))
            trace=float(np.trace(np.linalg.inv(h)@j));close(trace,traces[name]['trace'],1e-7)
            with np.load(B/'traces/covariances.npz',allow_pickle=False) as saved:
                close(h,saved[name+'_H'],1e-12);close(j,saved[name+'_J'],1e-12)
        write(out/'traces.json',{'passed':True,'new_fits':0,'pair_trace':traces['pair']['trace'],'triple_trace':traces['triple']['trace']})
    # Independent reconstruction of the original historical count tables from bound batch files.
    cat=json.loads((A/'input-catalogue-v1.json').read_text());catalog={tuple(r['identity']):r for r in cat['cohorts']}
    bindings={Path(r['path']).name:r for r in cat['closed_binding_inputs']};batchname=None;batch=None
    ids=[];values=[];oracle_accepted=0
    for row in block_rows(B/'matched-oracle'):
        identity=tuple(map(int,row['identity']));source=catalog[identity];name=Path(source['batch']).name
        if name!=batchname:
            path=A/'staged-inputs'/name;binding=bindings[name];raw=path.read_bytes();assert len(raw)==binding['bytes'] and hashlib.sha256(raw).hexdigest()==binding['sha256']
            with np.load(path,allow_pickle=False) as z:batch={k:z[k].copy() for k in ['counts','metadata']}
            batchname=name
        index=source['row'];assert tuple(batch['metadata'][index])==identity
        if row['accepted']:
            oracle_accepted+=1
            v,m=oracle_values(batch['counts'][index].sum(axis=0),row['theta2'],q0)
            if np.max(np.abs(v-row['statistics']))>1e-8 or not np.array_equal(m,row['mask_sizes']):
                write(out/'failure-context.json',{'identity':list(identity),'reconstructed_statistics':v.tolist(),'saved_statistics':row['statistics'].tolist(),'reconstructed_masks':list(m),'saved_masks':row['mask_sizes'].tolist()})
            close(v,row['statistics']);assert np.array_equal(m,row['mask_sizes'])
        ids.append(identity);values.append(row['statistics'])
    assert len(set(ids))==len(ids)==4000
    oracle_summary=json.loads((B/'matched-oracle/summary.json').read_text());assert oracle_accepted==4000-len(oracle_summary['failures'])
    check_rates(oracle_summary,ids,values);counts['matched_oracle']=len(ids)
    write(out/'matched-oracle.json',{'passed':True,'cohorts':len(ids),'maxima':MAX.copy()})
    confirm=json.loads((A/'confirmation/invocation-01/index.json').read_text())['records'];confirm={tuple(x['identity']):x for x in confirm}
    ids=[];values=[];accepted=0
    for row in block_rows(B/'lrt'):
        identity=tuple(map(int,row['identity']));source=confirm[identity];verify(source['binding'])
        with np.load(R/source['binding']['path'],allow_pickle=False) as z:counts0=z['counts'].sum(axis=0)
        emp=counts0/counts0.sum()
        if row['accepted']>=1:_,l2=endpoint(row['theta2'],g,emp);accepted+=1
        if row['accepted']==2:
            _,l3=endpoint(row['theta3'],f,emp);raw=2*counts0@(l3-l2);close(raw,row['raw_lrt']);close(max(0.,raw),row['statistics'][0]);accepted+=1
        close(row['statistics'][1],source['statistics'][3],0.)
        ids.append(identity);values.append(row['statistics'])
    assert len(set(ids))==len(ids)==4398;summary=json.loads((B/'lrt/summary.json').read_text());assert accepted==summary['accepted_fits'];check_rates(summary,ids,values)
    counts['LRT']=len(ids);write(out/'lrt.json',{'passed':True,'cohorts':len(ids),'accepted_endpoints':accepted,'maxima':MAX.copy()})
    if (B/'pools/summary.json').exists():
        pool_stats={};pool_ids=set();pairs=0
        for i,row in enumerate(block_rows(B/'pools')):
            n,pool,law,purpose,index=map(int,row['identity']);identity=(n,pool,law,purpose,index);assert identity not in pool_ids;pool_ids.add(identity)
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261016,11,3,*identity])))
            expected=np.array([rng.multinomial(int(m),q0 if law==0 else p) for m in sizes(n)])
            assert np.array_equal(expected,row['counts']);pairs+=int(row['accepted'])
            if row['accepted']==5 and np.isfinite(row['statistic']):
                a,b,m,k=d_statistics(expected,row['parameters']);close(a[0],row['statistic']);close(b[0],row['fold_statistics']);assert np.array_equal(m[0],row['mask_sizes']) and np.array_equal(k[0],row['ranks'])
            elif row['accepted']:
                for k in range(int(row['accepted'])):
                    train=expected.sum(axis=0)-expected[k];endpoint(row['parameters'][k],g,train/train.sum())
            pool_stats.setdefault((n,pool,law,purpose),[]).append((index,float(row['statistic'])))
            if (i+1)%1000==0:print(json.dumps({'audit_pools_completed':i+1}),flush=True)
        assert len(pool_ids)==67960
        summary=json.loads((B/'pools/summary.json').read_text());assert pairs==summary['accepted_pair_fits'];observations={}
        for row in summary['pools']:
            n,pool=row['N'],row['pool'];arrays={}
            for law,purpose,size in [(0,0,999),(0,1,500),(7,1,200)]:
                selected=sorted(pool_stats[(n,pool,law,purpose)]);assert [x[0] for x in selected]==list(range(size));arrays[(law,purpose)]=np.array([x[1] for x in selected])
            if not row['failures']:
                null=int(decision(arrays[(0,0)],arrays[(0,1)]).sum());alt=int(decision(arrays[(0,0)],arrays[(7,1)]).sum())
                close(row['threshold'],np.sort(arrays[(0,0)])[949],0.)
                assert null==row['null_rejections'] and alt==row['alternative_rejections'];observations.setdefault(n,[]).append((null,alt))
        for agg in summary['aggregates']:
            for j,(kind,denom) in enumerate([('null',500),('alternative',200)]):
                ar=np.array(observations[agg['N']])[:,j]/denom;result=agg[kind];se=ar.std(ddof=1)/np.sqrt(20)
                close([ar.mean(),ar.std(ddof=1),se],[result['equal_pool_mean'],result['between_pool_SD'],result['MCSE']],1e-12)
                margin=student.ppf(.975,19)*se;close(result['approximate_t_interval'],[max(0,ar.mean()-margin),min(1,ar.mean()+margin)],1e-12)
                radius=np.sqrt(np.log(40)/40)
                close(result['Hoeffding_pool_interval'],[max(0,ar.mean()-radius),min(1,ar.mean()+radius)],1e-12)
                for value,bounds in zip(np.array(observations[agg['N']])[:,j],result['conditional_intervals']):close(bounds,interval(int(value),denom),1e-12)
        counts['new_pool_cohorts']=len(pool_ids);write(out/'pools.json',{'passed':True,'cohorts':len(pool_ids),'accepted_pair_endpoints':pairs,'maxima':MAX.copy()})
    if (B/'sensitivity/summary.json').exists():
        settings=[(cut,minimum) for cut in [1e-8,1e-10,1e-12] for minimum in [3,5,10]];ids=[];values=[]
        for row in block_rows(B/'sensitivity'):
            identity=tuple(map(int,row['identity']));source=confirm[identity]
            finite=np.isfinite(row['statistics'])
            if finite.any():
                with np.load(R/source['binding']['path'],allow_pickle=False) as z:a,b,m,k=d_statistics(z['counts'],z['parameters'],[s for s,keep in zip(settings,finite) if keep])
                close(a,row['statistics'][finite]);close(b,row['fold_statistics'][finite]);assert np.array_equal(m,row['mask_sizes'][finite]) and np.array_equal(k,row['ranks'][finite])
            ids.append(identity);values.append(row['statistics'])
        assert len(set(ids))==4398;check_rates(json.loads((B/'sensitivity/summary.json').read_text()),ids,values)
        counts['sensitivity_cohorts']=4398;write(out/'sensitivity.json',{'passed':True,'cohorts':4398,'settings':9,'maxima':MAX.copy()})
    extra_review(out,counts)
    for row in json.loads((B/'registration-v11.json').read_text())['bindings']:verify(row)
    write(out/'summary.json',{'passed':True,'counts':counts,'maxima':MAX,'new_optimizer_calls':0,'inherited_trace_check':bind(out/'traces.json') if resume else None,'initial_failed_invocation_preserved':resume,'maxima_scope':'Resumed cohort/endpoint checks; first covariance-check maxima were not recorded and are not reconstructed.' if resume else 'Full review','diagnosis':bind(B/'audit-diagnosis/result.json'),'numerical_representation_scope':'Reconstruction follows the producer probability normalization and factor-product nuisance map, with independent explicit U contractions. Near-singular cross-weight sensitivity to algebraically equivalent map arithmetic is retained, not claimed absent.','scope':'All completed new outputs independently reconstructed; the successful pre-stop trace check is inherited exactly when resumed. Missing or failed phases remain explicitly unqualified.'})


def extra_review(out,counts):
    if (B/'coefficient-recovery/summary.json').exists():
        cat=json.loads((B/'coefficient-inputs-v11.json').read_text())
        with np.load(B/'coefficient-recovery/saved-coefficients.npz',allow_pickle=False) as z:ids=z['identity'].copy();params=z['fold_coefficients'].copy();truth=z['truth'].copy()
        with (B/'traces/triple-coefficients.csv').open() as stream:truth_rows=list(csv.DictReader(stream))
        coefficient_key=next(k for k in truth_rows[0] if 'coefficient' in k)
        close(truth,[float(r[coefficient_key]) for r in truth_rows],0.)
        with np.load(A/'staged-inputs/exact-laws-v1.npz',allow_pickle=False) as z:close(truth,z['source_P3_theta'][55:],0.)
        _,_,f=design(10)
        for i,r in enumerate(cat['records']):
            raw=subprocess.check_output(['git','show',cat['commit']+':'+r['path']],cwd=R);assert hashlib.sha256(raw).hexdigest()==r['sha256'];item=json.loads(gzip.decompress(raw));tables=np.array(item['table_counts'])
            assert list(ids[i])==[item['sample_size'],item['draw']]
            for k,fold in enumerate(item['reading']['folds']):
                theta=np.array(fold['projections'][2]['theta']);endpoint(theta,f,(tables.sum(axis=0)-tables[k])/(tables.sum()-tables[k].sum()));close(theta[55:],params[i,k],0.)
        with (B/'coefficient-recovery/recovery.csv').open() as stream:rows=list(csv.DictReader(stream))
        for i,row in enumerate(rows):
            n=int(row['N']);j=i%120;values=params[ids[:,0]==n,:,j]@sizes(n)/n
            close([float(row['truth']),float(row['cohort_mean']),float(row['between_cohort_SD']),float(row['RMSE'])],
                  [truth[j],values.mean(),values.std(ddof=1),np.sqrt(np.mean((values-truth[j])**2))],1e-10)
            close([float(row['fold_min']),float(row['fold_max'])],[params[ids[:,0]==n,:,j].min(),params[ids[:,0]==n,:,j].max()],0.)
        summary=json.loads((B/'coefficient-recovery/summary.json').read_text())
        close(summary['maximum_absolute_fold_coefficient'],np.abs(params).max(),0.)
        for item in summary['N_summaries']:
            selected=[r for r in rows if int(r['N'])==item['N']]
            close([item['median_coefficient_SD'],item['median_RMSE']],
                  [np.median([float(r['between_cohort_SD']) for r in selected]),np.median([float(r['RMSE']) for r in selected])],1e-12)
        counts['retained_coefficient_endpoints']=625;write(out/'coefficient-recovery.json',{'passed':True,'endpoints':625,'summary_rows':600})
    if (B/'hidden-coordinates/summary.json').exists():
        summary=json.loads((B/'hidden-coordinates/summary.json').read_text());old=json.loads((R/'experiments/2026-10-16-route12-v1/mixtures/results.json').read_text())
        models={(r['variables'],r['model_name']):r for r in old['models']};done=0
        entropy=lambda p:-float(p@np.log2(p));KL=lambda p,q:float(p@(np.log2(p)-np.log2(q)))
        for row in summary['records']:
            if row['status']!='passed':continue
            d,name,z=row['variables'],row['model'],row['coordinate'];prefix=f'{d}-{name}-z{z}';g,t,f=design(d)
            model=models[(d,name)]
            with np.load(R/'experiments/2026-10-16-route12-v1/mixtures'/model['parameters_file']['path'],allow_pickle=False) as a:
                wh=a['aligned_weights'];ch=a['aligned_components'];h=a['hidden'];pooled=a['law'];weights=np.array([wh[h[:,z]==v].sum() for v in [0,1]])
                components=np.array([wh[h[:,z]==v]@ch[h[:,z]==v]/weights[v] for v in [0,1]])
            parts=[];qparts=[];within=within_total=0.
            for k in [0,1]:
                with np.load(B/'hidden-coordinates'/f'{prefix}-part{k}.npz',allow_pickle=False) as a:
                    close(a['input_law'],components[k],1e-12);q,_=endpoint(a['theta2'],g,components[k]);p3,_=endpoint(a['theta3'],f,components[k]);close(q,a['P2'],1e-12);close(p3,a['P3'],1e-12)
                    within+=weights[k]*KL(p3,q);within_total+=weights[k]*KL(components[k],q);qparts.append(q)
            qbar=weights@np.array(qparts)
            with np.load(B/'hidden-coordinates'/f'{prefix}-mixture.npz',allow_pickle=False) as a:
                close(a['input_law'],qbar,1e-12);q,_=endpoint(a['theta2'],g,qbar);p3,_=endpoint(a['theta3'],f,qbar);mixlayer=KL(p3,q)
            ip=sum(w*KL(comp,pooled) for w,comp in zip(weights,components));iq=sum(w*KL(comp,qbar) for w,comp in zip(weights,qparts))
            close([within,mixlayer,row['pooled_L3_bits']-within-mixlayer,iq-ip],
                  [row['within_L3_bits'],row['pair_mixture_L3_bits'],row['L3_nonadditivity_bits'],row['MI_correction_bits']],1e-7)
            done+=1
        counts['hidden_coordinate_cases']=done;write(out/'hidden-coordinates.json',{'passed':True,'accepted_cases':done,'planned':48})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','run','resume']);a=p.parse_args();{'pilot':pilot,'run':run,'resume':lambda:run(True)}[a.phase]()
