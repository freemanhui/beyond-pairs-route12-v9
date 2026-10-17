"""Known-gate acceptance, fixed-law targets and a disjoint artificial cost pilot."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zlib

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
from beyond_pairs.route12_benchmarks import core as c
from route12_benchmarks_runtime import R,A,B,bind,write,verify,storage


def known():
    xml=B/'known-cases-v11.xml';root=ET.parse(xml).getroot();cases=root.findall('.//testcase')
    assert len(cases)==24 and not root.findall('.//failure') and not root.findall('.//error') and not root.findall('.//skipped')
    receipt=B/'development-owner/invocation-01/receipt.json';r=json.loads(receipt.read_text())
    assert r['exit']==0 and r['reaped'] and r['owned_group_absent'] and r['resource_stop'] is None
    write(B/'known-gate-v11.json',{'passed':True,'cases':24,'case_names':[x.attrib['name'] for x in cases],
          'test_receipt':bind(xml),'owner':bind(receipt),'scope':'Accept the single completed new artificial test run; zero repeated controls or study-law inputs.'})
    print(json.dumps({'new_known_cases':24,'passed':True,'repeated_tests':0}))


def laws():
    reg=json.loads((B/'registration-v11.json').read_text())
    p=A/'staged-inputs/exact-laws-v1.npz';verify(next(r for r in reg['bindings'] if r['path']==str(p.relative_to(R))))
    with np.load(p,allow_pickle=False) as item:return {k:item[k].copy() for k in ['law_0','law_7','source_P3','q_theta','source_P3_theta']}


def traces():
    out=B/'traces';out.mkdir();saved=laws();p=saved['law_7'];q=saved['law_0'];p3=saved['source_P3']
    _,g,_,full=c.designs(10)
    for target,theta,features in [(q,saved['q_theta'],g),(p3,saved['source_P3_theta'],full)]:
        law,_=c.law(theta,features)
        assert np.max(np.abs(law-target))<1e-12
        assert np.max(np.abs((p-target)@features))<1e-9
    pair=c.sandwich(p,q,g);third=c.sandwich(p,p3,full)
    np.savez_compressed(out/'covariances.npz',pair_H=pair.pop('H'),pair_J=pair.pop('J'),triple_H=third.pop('H'),triple_J=third.pop('J'))
    old=json.loads((A/'referee-round2-v9/saved-summary-v9.json').read_text());rows=[]
    for row in old['sweep']:
        coefficient=1000/(2*np.log(2))*np.mean(1/np.array(row['train_sizes']))
        trace_reference=34.08171109-(third['trace']-pair['trace'])*coefficient
        rows.append({**row,'pair_trace':pair['trace'],'triple_trace':third['trace'],'trace_difference':third['trace']-pair['trace'],
                     'trace_reference':float(trace_reference),'regular_to_trace_shift':float(trace_reference-row['regular_heuristic']),
                     'observed_minus_trace_reference':float(row['observed_display_mean']-trace_reference)})
    genes=['TP53','KRAS','EGFR','STK11','KEAP1','RBM10','SMARCA4','ARID1A','SETD2','PIK3CA']
    with (out/'triple-coefficients.csv').open('x') as stream:
        writer=csv.writer(stream);writer.writerow(['canonical_index','triple','coefficient_nats','basis'])
        for i,(label,value) in enumerate(zip(c.triple_labels(genes),saved['source_P3_theta'][55:])):writer.writerow([i,label,float(value),'0/1 monomial'])
    write(out/'results.json',{'pair':pair,'triple':third,'trace_difference':third['trace']-pair['trace'],'sweep':rows,
         'coefficient_vector':bind(out/'triple-coefficients.csv'),'basis':'Canonical 0/1 monomials, natural parameters in nats; no additive information shares or patient effects.',
         'new_fits':0,'new_draws':0,'interpretation':'Leading fixed-family regular-misspecification reference only. Remaining finite-N discrepancy is not a uniquely identified floor/boundary effect.',
         'recovery_variability':'Not calculated here; requires inventory of comparable retained full-P3 endpoints.'})
    print(json.dumps({'pair_trace':pair['trace'],'triple_trace':third['trace'],'trace_difference':third['trace']-pair['trace'],'coefficients':120}))


def pilot():
    out=B/'cost-pilot';out.mkdir();_,g,t,full=c.designs(10)
    theta=np.zeros(175);theta[:10]=[-1.2,-1.5,-2.,-2.2,-2.4,-2.5,-2.7,-2.8,-3.,-3.1];theta[10:18]=[.2,-.15,.1,.2,-.1,.1,.2,-.2]
    q,_=c.law(theta,full);theta[55]=1.2;theta[70]=-.6;p,_=c.law(theta,full)
    records=[]
    for n in [1557,6274]:
        for law_id,law in [(0,q),(1,p)]:
            for index in range(2):
                seed=[20261016,11,99,n,law_id,index];rng=np.random.default_rng(np.random.SeedSequence(seed))
                tables=np.array([rng.multinomial(int(m),law) for m in c.fold_sizes(n)],dtype=np.uint16)
                start=time.monotonic();params,diag,attempted,accepted,error=c.fit_training(tables)
                if error:raise ValueError('artificial pair pilot failed: '+error)
                value,folds,masks,ranks=c.fully_learned(tables,params,check=True);d_wall=time.monotonic()-start
                start=time.monotonic();lrt=c.pooled_lrt(tables.sum(axis=0));lrt_wall=time.monotonic()-start
                if lrt[-1]:raise ValueError('artificial LRT pilot failed: '+lrt[-1])
                packed=len(zlib.compress(tables.tobytes()))+len(zlib.compress(params.tobytes()))+512
                identity=f'{n}-{law_id}-{index}'
                np.savez_compressed(out/(identity+'.npz'),counts=tables,parameters=params,fold_statistics=folds,mask_sizes=masks,ranks=ranks,pair_diagnostics=diag,theta2=lrt[0],theta3=lrt[1],lrt_diagnostics=lrt[2])
                records.append({'identity':[n,law_id,index],'seed':seed,'D_wall_s':d_wall,'LRT_wall_s':lrt_wall,'packed_D_raw_byte_bound':packed,
                                'D':value,'LRT':lrt[3],'artificial_pair_fits':6,'artificial_triple_fits':1,'record':bind(out/(identity+'.npz'))})
    reg=json.loads((B/'registration-v11.json').read_text());used=storage()
    maximum_d=max(r['D_wall_s'] for r in records);maximum_lrt=max(r['LRT_wall_s'] for r in records);maximum_bytes=max(r['packed_D_raw_byte_bound'] for r in records)
    pooltime=2*67960*maximum_d;lrttime=2*4398*maximum_lrt;poolbytes=2*3*67960*maximum_bytes
    reserve=256*1024**2+reg['resources']['closing_reserve_bytes']
    write(out/'results.json',{'records':records,'artificial_only':True,'new_scientific_identities':0,'new_artificial_fits':56,
          'max_D_wall_s':maximum_d,'max_LRT_wall_s':maximum_lrt,'max_packed_D_raw_byte_bound':maximum_bytes,
          'projection':{'LRT_seconds':lrttime,'pool_seconds':pooltime,'pool_whole_bytes':poolbytes,'other_reserved_bytes':reserve,
                        'used_bytes':used,'pool_cost_pass':pooltime<=7200 and used+poolbytes+reserve<=reg['resources']['new_whole_storage_cap_bytes'],
                        'LRT_cost_pass':lrttime<=3600},
          'storage_convention':'Counts and all five theta endpoints retained in compressed blocks; per-case bound sums separately compressed arrays plus512 bytes for statistics, identities, diagnostics and amortized block metadata, times3 custody copies, times2 safety.'})
    print(json.dumps({'LRT_projected_s':lrttime,'pool_projected_s':pooltime,'pool_projected_whole_bytes':poolbytes,'pool_cost_pass':pooltime<=7200 and used+poolbytes+reserve<=reg['resources']['new_whole_storage_cap_bytes']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['known','traces','pilot']);args=parser.parse_args()
    {'known':known,'traces':traces,'pilot':pilot}[args.phase]()
