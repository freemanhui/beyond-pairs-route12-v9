"""Prespecified threshold/coordinate sensitivities and saved coefficient recovery."""
import argparse,csv,gzip,hashlib,json,os,subprocess
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from beyond_pairs.route12_benchmarks import core as c
from beyond_pairs.route12.representation import traced_layers
from beyond_pairs.information_layers import kl_bits
from route12_benchmarks_runtime import R,A,B,bind,write,verify
from route12_benchmarks_studies import Blocks,rate_table


def sensitivity():
    source=json.loads((A/'confirmation/invocation-01/index.json').read_text())['records']
    settings=[(cut,minimum) for cut in [1e-8,1e-10,1e-12] for minimum in [3,5,10]]
    names=[f'eig={cut:g},count={minimum}' for cut,minimum in settings]
    out=B/'sensitivity';blocks=Blocks(out,['identity','statistics','fold_statistics','mask_sizes','ranks'])
    ids=[];statistics=[];primary_error=0.;failures=[]
    for row in source:
        verify(row['binding'])
        with np.load(R/row['binding']['path'],allow_pickle=False) as z:tables=z['counts'].copy();parameters=z['parameters'].copy()
        results=[];errors=[]
        for j,(cut,minimum) in enumerate(settings):
            try:result=c.fully_learned(tables,parameters,cutoff=cut,minimum=minimum,check=True)
            except Exception as exc:
                result=(np.nan,np.full(5,np.nan),np.full(5,-1),np.full(5,-1))
                item={'identity':row['identity'],'setting':j,'error':str(exc)};failures.append(item);errors.append(item)
            results.append(result)
        values=np.array([r[0] for r in results])
        if np.isfinite(values[4]):
            primary_error=max(primary_error,abs(values[4]-row['statistics'][3]));assert primary_error<=1e-8
        blocks.add([row['identity'],values,np.array([r[1] for r in results]),np.array([r[2] for r in results]),np.array([r[3] for r in results])],{'identity':row['identity'],'settings':9,'new_fits':0,'errors':errors})
        ids.append(row['identity']);statistics.append(values)
    blocks.close();write(out/'summary.json',{'cohorts':len(ids),'settings':names,'rates':rate_table(ids,statistics,names,999),
        'failures':failures,'maximum_original_primary_discrepancy':primary_error,'new_fits':0,'new_draws':0,'scope':'Every setting calibrated separately; original primary cutoff/mask unchanged. Failed numerical settings remain undecided; no silent removal.'})


def grouped(weights,components,hidden,coordinate):
    mass=np.array([weights[hidden[:,coordinate]==k].sum() for k in [0,1]])
    laws=np.array([weights[hidden[:,coordinate]==k]@components[hidden[:,coordinate]==k]/mass[k] for k in [0,1]])
    assert np.max(np.abs(mass@laws-weights@components))<1e-12
    return mass,laws


def hidden():
    if not json.loads((B/'secondary-known-v11.json').read_text())['passed']:raise ValueError('supplemental known controls absent')
    out=B/'hidden-coordinates';out.mkdir();catalogue=json.loads((B/'secondary-inputs-v11.json').read_text())
    old=json.loads((R/catalogue['mixtures_results']['path']).read_text());verify(catalogue['mixtures_results'])
    models={(m['variables'],m['model_name']):m for m in old['models']};rows=[];attempted=0
    for mix in old['mixtures']:
        d=mix['variables'];name=mix['model_name'];model=models[(d,name)];binding=model['parameters_file']
        path=R/'experiments/2026-10-16-route12-v1/mixtures'/binding['path'];verify({**binding,'path':str(path.relative_to(R))})
        with np.load(path,allow_pickle=False) as z:
            weights=z['aligned_weights'].copy();components=z['aligned_components'].copy();hidden=z['hidden'].copy();pooled=z['law'].copy()
        selected=model['partial_group_source_axis'];assert components.shape[0]==32
        for coordinate in range(5):
            if coordinate==selected:continue
            w,parts=grouped(weights,components,hidden,coordinate);assert np.max(np.abs(w@parts-pooled))<1e-12
            traces=[];error=None;prefix=f'{d}-{name}-z{coordinate}'
            try:
                for k,law in enumerate(parts):
                    attempted+=2;layer=traced_layers(law);traces.append(layer)
                    np.savez_compressed(out/f'{prefix}-part{k}.npz',input_law=layer['input_law'],P2=layer['laws']['P2'],P3=layer['laws']['P3'],theta2=layer['parameters']['P2'],theta3=layer['parameters']['P3'])
                qparts=np.array([x['laws']['P2'] for x in traces]);qbar=w@qparts;attempted+=2;mixture=traced_layers(qbar)
                np.savez_compressed(out/f'{prefix}-mixture.npz',weights=w,components=parts,input_law=qbar,P2=mixture['laws']['P2'],P3=mixture['laws']['P3'],theta2=mixture['parameters']['P2'],theta3=mixture['parameters']['P3'])
                within=sum(wi*x['layers_bits']['third_order'] for wi,x in zip(w,traces));within_total=sum(wi*x['layers_bits']['total_above_pairs'] for wi,x in zip(w,traces))
                ip=sum(wi*kl_bits(pi,pooled) for wi,pi in zip(w,parts));iq=sum(wi*kl_bits(qi,qbar) for wi,qi in zip(w,qparts))
                m=mix['metrics'];correction=m['pooled_L3_bits']-within-mixture['layers_bits']['third_order']
                identity=m['pooled_total_bits']-within_total-mixture['layers_bits']['total_above_pairs']-iq+ip
                assert abs(identity)<1e-7
                row={'variables':d,'model':name,'coordinate':coordinate,'pooled_L3_bits':m['pooled_L3_bits'],
                     'within_L3_bits':float(within),'pair_mixture_L3_bits':mixture['layers_bits']['third_order'],
                     'L3_nonadditivity_bits':float(correction),'MI_correction_bits':float(iq-ip),'total_identity_error':float(abs(identity)),
                     'weights':w.tolist(),'status':'passed'}
            except Exception as exc:
                error=str(exc);row={'variables':d,'model':name,'coordinate':coordinate,'status':'failed','error':error}
                if hasattr(exc,'partial'):
                    arrays={k:v for k,v in exc.partial.items() if isinstance(v,np.ndarray)}
                    if arrays:np.savez_compressed(out/f'{prefix}-failure.npz',**arrays)
            rows.append(row);write(out/f'{prefix}.json',row)
            print(json.dumps({'case':len(rows),'status':row['status']}),flush=True)
    write(out/'summary.json',{'coordinate_cases':48,'records':rows,'projection_position_attempts_upper_bound':attempted,
        'actual_call_count_is_at_most_position_bound':True,'new_latent_or_patient_fits':0,'scope':'All four nonselected aligned hidden coordinates; signed representation diagnostics, not identified biological shares.'})


def cohort_average(theta,sizes):
    return np.asarray(sizes)@np.asarray(theta)/sum(sizes)


def coefficients():
    catalogue=json.loads((B/'coefficient-inputs-v11.json').read_text());out=B/'coefficient-recovery';out.mkdir()
    with np.load(A/'staged-inputs/exact-laws-v1.npz',allow_pickle=False) as z:truth=z['source_P3_theta'][55:].copy()
    arrays=[];ids=[];maxmoment=maxdual=0.
    _,_,_,features=c.designs(10)
    for row in catalogue['records']:
        raw=subprocess.check_output(['git','show',catalogue['commit']+':'+row['path']],cwd=R)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        item=json.loads(gzip.decompress(raw));n=item['sample_size'];draw=item['draw'];tables=np.array(item['table_counts']);params=[];sizes=[]
        assert item['seed']==[20261240,n,draw]
        for k,fold in enumerate(item['reading']['folds']):
            projection=fold['projections'][2];theta=np.array(projection['theta']);target=(tables.sum(axis=0)-tables[k])/(tables.sum()-tables[k].sum())
            law,loglaw=c.law(theta,features);moment=float(np.max(np.abs((target-law)@features)))
            energy=features@theta;top=energy.max();dual=(top+np.log(np.exp(energy-top).sum())-theta@(target@features))/np.log(2)
            dualerror=abs(dual+law@loglaw/np.log(2));maxmoment=max(maxmoment,moment);maxdual=max(maxdual,dualerror)
            assert moment<=1e-9 and dualerror<=1e-7 and np.max(np.abs(law-np.array(projection['law'])))<=1e-12
            params.append(theta[55:]);sizes.append(tables[k].sum())
        ids.append([n,draw]);arrays.append(params)
    endpoints=np.array(arrays);ids=np.array(ids);np.savez_compressed(out/'saved-coefficients.npz',identity=ids,fold_coefficients=endpoints,truth=truth)
    labels=c.triple_labels(['TP53','KRAS','EGFR','STK11','KEAP1','RBM10','SMARCA4','ARID1A','SETD2','PIK3CA']);rows=[]
    for n in [1000,2000,4000,8000,16000]:
        selected=endpoints[ids[:,0]==n];means=np.array([cohort_average(x,c.fold_sizes(n)) for x in selected])
        for j,label in enumerate(labels):
            rows.append({'N':n,'triple':label,'truth':float(truth[j]),'cohort_mean':float(means[:,j].mean()),'between_cohort_SD':float(means[:,j].std(ddof=1)),
                         'RMSE':float(np.sqrt(np.mean((means[:,j]-truth[j])**2))),'fold_min':float(selected[:,:,j].min()),'fold_max':float(selected[:,:,j].max())})
    with (out/'recovery.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'summary.json',{'saved_cohorts':125,'saved_P3_endpoints':625,'triples':120,'new_fits':0,'new_draws':0,
        'maximum_moment_error':maxmoment,'maximum_dual_error_bits':maxdual,'maximum_absolute_fold_coefficient':float(np.max(np.abs(endpoints))),
        'N_summaries':[{'N':n,'median_coefficient_SD':float(np.median([r['between_cohort_SD'] for r in rows if r['N']==n])),
                       'median_RMSE':float(np.median([r['RMSE'] for r in rows if r['N']==n]))} for n in [1000,2000,4000,8000,16000]],
        'scope':'0/1 natural coefficients in nats; within-cohort held-size-weighted training coefficients, SD across25 independent cohorts. Fold extrema retain boundary sensitivity. Not additive information or clinical effects.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['sensitivity','hidden','coefficients']);a=p.parse_args()
    {'sensitivity':sensitivity,'hidden':hidden,'coefficients':coefficients}[a.phase]()
