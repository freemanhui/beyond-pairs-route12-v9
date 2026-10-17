"""Reaggregate published scalar records and redraw figures; no fitting or draws."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from . import core as c

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def aggregate(records, pooled=False):
    groups=defaultdict(list)
    for r in records:groups[(int(r['N']),int(r['pool']) if pooled else 0,r.get('method','residual_D'))].append(r)
    answer=[];checked=0
    for (n,pool,method),rows in groups.items():
        cal=np.array([float(r['statistic']) for r in rows if int(r['purpose'])==0])
        for law in sorted({int(r['law']) for r in rows if int(r['purpose'])==1}):
            ev=[r for r in rows if int(r['purpose'])==1 and int(r['law'])==law]
            values=np.array([float(r['statistic']) for r in ev]);decision=c.rank_decisions(cal,values)
            if not np.array_equal(decision,np.array([int(r['reject']) for r in ev],bool)):
                raise ValueError('CSV decisions disagree with add-one recalculation')
            numerator=len(cal)+1-np.searchsorted(np.sort(cal),values,side='left')
            if any(str(int(k))!=r['rank_numerator'] for k,r in zip(numerator,ev)):
                raise ValueError('CSV rank numerator mismatch')
            count=int(decision.sum());checked+=len(ev)
            cautions=sorted({r.get('numerical_caution','') for r in rows if r.get('numerical_caution','')})
            answer.append(dict(N=n,pool=pool,method=method,law=law,rejections=count,total=len(ev),
                               rate=count/len(ev),CP95=c.cp(count,len(ev)),numerical_caution=cautions,threshold=float(np.sort(cal)[len(cal)-int(np.floor(.05*(len(cal)+1)))])))
    return answer,checked

def reaggregate(data):
    tables={};checked=0
    for name in ['lrt','matched-oracle','sensitivity','pools']:
        rows=read(data/(name+'.csv'));tables[name],count=aggregate(rows,name=='pools');checked+=count
    lrt=read(data/'lrt.csv');pairs=[]
    for n in [1557,6274]:
        for law in [0,7]:
            rows=[r for r in lrt if int(r['N'])==n and int(r['law'])==law and int(r['purpose'])==1]
            by=defaultdict(dict)
            for r in rows:by[int(r['index'])][r['method']]=int(r['reject'])
            aa=[by[i]['known-law LRT'] for i in sorted(by)];bb=[by[i]['fully learned D'] for i in sorted(by)]
            pairs.append(dict(N=n,law=law,**c.paired(aa,bb)))
    tables['paired_LRT_minus_D']=pairs;tables['pool_aggregates']=[]
    for n in [1557,6274]:
        for law in [0,7]:
            rows=sorted((r for r in tables['pools'] if r['N']==n and r['law']==law),key=lambda r:r['pool'])
            tables['pool_aggregates'].append(dict(N=n,law=law,**c.pool_summary([r['rejections'] for r in rows],[r['total'] for r in rows])))
    for name in ['triple-coefficients-rounded','hidden-coordinates','cutoff-comparisons','sandwich-sweep']:
        tables[name]=read(data/(name+'.csv'))
    for name in ['sandwich-family-summary','rank-reference-diagnostics']:
        tables[name]=json.loads((data/(name+'.json')).read_text())
    tables['verified_evaluation_decisions']=checked
    tables['scope']='Reaggregation of released scalar records, not regeneration of biological fits, patient validation, or independent external replication.'
    return tables

def figures(tables,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(6.5,5.8))
    for ax,law,title in [(axes[0,0],0,'A  Null rejection'),(axes[0,1],7,'B  Frozen-law detection')]:
        for offset,method,color,marker,label in [(-.06,'known-law LRT','#D55E00','s','Pooled LRT'),(.06,'fully learned D','#0072B2','o','Five-fold residual')]:
            rows=[next(r for r in tables['lrt'] if r['N']==n and r['law']==law and r['method']==method) for n in [1557,6274]]
            y=np.array([r['rate'] for r in rows]);bounds=np.array([r['CP95'] for r in rows])
            ax.errorbar(np.arange(2)+offset,y,yerr=[y-bounds[:,0],bounds[:,1]-y],fmt=marker,capsize=3,color=color,label=label)
        ax.set_xticks([0,1],['1,557','6,274']);ax.set_xlim(-.35,1.35);ax.set_xlabel('Total observations');ax.set_title(title)
        ax.set_ylim((0,.1) if law==0 else (.75,1.015));ax.set_ylabel('Fraction')
        ax.axhline(.05 if law==0 else .8,color='.65',ls=':',lw=1)
    for ax,law,title in [(axes[1,0],0,'C  Mask threshold: null'),(axes[1,1],7,'D  Mask threshold: detection')]:
        for n,color,marker in [(1557,'#0072B2','o'),(6274,'#009E73','^')]:
            rows=[next(r for r in tables['sensitivity'] if r['N']==n and r['law']==law and r['method']==f'eig=1e-10,count={m}') for m in [3,5,10]]
            ax.plot([3,5,10],[r['rate'] for r in rows],marker=marker,color=color,label=f'N={n:,}')
        ax.set_xticks([3,5,10]);ax.set_xlabel('Minimum expected triple count');ax.set_title(title);ax.set_ylabel('Fraction')
        ax.set_ylim((0,.1) if law==0 else (.75,1.015));ax.axhline(.05 if law==0 else .8,color='.65',ls=':',lw=1)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False,loc='upper left',fontsize=10)
    axes[1,0].legend(frameon=False,loc='upper right',fontsize=10)
    fig.subplots_adjust(left=.105,right=.985,bottom=.1,top=.94,wspace=.35,hspace=.52)
    for ext in ['pdf','png']:fig.savefig(out/('figure2-lrt-mask.'+ext),dpi=180)
    plt.close(fig)
    rows=tables['sandwich-sweep'];ns=[int(r['N']) for r in rows];fig,axes=plt.subplots(1,2,figsize=(6.5,3.5))
    val=lambda key:np.array([float(r[key]) for r in rows])
    axes[0].errorbar(ns,val('observed_display_mean'),yerr=val('draw_SD'),fmt='o-',capsize=3,color='#0072B2',label='Observed mean')
    axes[0].plot(ns,val('regular_heuristic'),ls=':',color='.45',lw=2,label='Nominal reference')
    axes[0].plot(ns,val('trace_reference'),ls='--',color='#D55E00',label='Sandwich reference')
    axes[0].axhline(34.08171109,color='#009E73',ls='-.',label='Fixed information')
    axes[0].set(title='A  Prediction',ylabel='Bits per 1,000',ylim=(-290,70))
    axes[1].errorbar(ns,val('observed_minus_trace_reference'),yerr=val('MCSE'),fmt='o-',capsize=3,color='#0072B2')
    axes[1].set(title='B  Gap (bits per 1,000)',ylabel='Observed minus reference',ylim=(-180,15))
    for ax in axes:
        ax.axhline(0,color='.6',lw=.8);ax.set_xscale('log',base=2);ax.set_xticks(ns,['1k','2k','4k','8k','16k']);ax.set_xlabel('Total observations');ax.spines[['top','right']].set_visible(False)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.01))
    fig.subplots_adjust(left=.13,right=.98,bottom=.34,top=.87,wspace=.5)
    for ext in ['pdf','png']:fig.savefig(out/('figure1-prediction.'+ext),dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--figures',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tables=reaggregate(args.data)
    with (args.output/'tables.json').open('x') as f:json.dump(tables,f,indent=2,allow_nan=False);f.write('\n')
    if args.figures:figures(tables,args.output)
    print(json.dumps({'verified_evaluation_decisions':tables['verified_evaluation_decisions'],'scope':tables['scope']}))

if __name__=='__main__':main()
