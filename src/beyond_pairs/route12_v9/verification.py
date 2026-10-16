"""Independent canonical feature/endpoint/direct-contraction implementation.

No producer numerics, replay or optimizer import. It uses the registered same
canonical eigenspace convention, with independently written reconstruction.
"""
from itertools import combinations
import numpy as np


def design(variables):
    states=np.array([[(i>>j)&1 for j in range(variables)]
                     for i in range(2**variables)],float)
    columns=[]
    for degree in (1,2,3):
        for subset in combinations(range(variables),degree):
            column=np.ones(len(states))
            for j in subset:
                column*=states[:,j]
            columns.append(column)
    full=np.ascontiguousarray(np.stack(columns,axis=1),dtype=np.float64)
    pair=variables*(variables+1)//2
    return np.ascontiguousarray(full[:,:pair]),np.ascontiguousarray(full[:,pair:])


def inverse(covariance):
    if not covariance.size:
        return np.empty((0,0)),0
    c=np.ascontiguousarray((covariance+covariance.T)*.5)
    eigenvalues,eigenvectors=np.linalg.eigh(c)
    if eigenvalues[-1]<=0 or eigenvalues[0]<-max(1e-12,eigenvalues[-1]*1e-8):
        raise ValueError('independent inverse support check failed')
    selected=eigenvalues>eigenvalues[-1]*1e-10
    root=np.ascontiguousarray(eigenvectors[:,selected]/np.sqrt(eigenvalues[selected]))
    product=root@root.T
    return np.ascontiguousarray((product+product.T)*.5),int(selected.sum())


def residual(law,g,t):
    gc=np.ascontiguousarray(g-law@g)
    tc=np.ascontiguousarray(t-law@t)
    nuisance,_=inverse(np.ascontiguousarray((gc*law[:,None]).T@gc))
    coefficient=((tc*law[:,None]).T@gc)@nuisance
    return np.ascontiguousarray(tc-gc@coefficient.T)


def verify(tables,p,q0,result,*,prepared=None):
    if prepared is None:
        g,t=design(len(p).bit_length()-1)
        sizes=tables.sum(axis=1)
        mask=(tables.sum()-sizes.max())*(q0@t)>=5
        r0=np.ascontiguousarray(residual(q0,g,t)[:,mask])
        w0,rank0=inverse(np.ascontiguousarray((r0*q0[:,None]).T@r0))
        prepared=(g,t,mask,r0,w0,rank0)
    g,t,fixed,r0,w0,rank0=prepared
    sizes=tables.sum(axis=1);total=tables.sum(axis=0)
    calculated=np.zeros((5,4));truth=np.zeros((5,4));maxmoment=0.;maxdual=0.
    for k,held in enumerate(tables):
        theta=result['parameters'][k]
        energy=g@theta;top=energy.max();exponent=np.exp(energy-top)
        logz=top+np.log(exponent.sum());q=np.exp(energy-logz)
        empirical=(total-held)/(total.sum()-held.sum())
        moment=float(np.max(np.abs((empirical-q)@g)))
        dual=(logz-theta@(empirical@g))/np.log(2)
        entropy=-float(q@(energy-logz)/np.log(2))
        maxmoment=max(maxmoment,moment);maxdual=max(maxdual,abs(dual-entropy))
        if moment>1e-9 or abs(dual-entropy)>1e-7 or abs(q.sum()-1)>1e-12:
            raise ValueError('independent parameter moment/dual/normalization failed')
        r=residual(q,g,t)
        rf=np.ascontiguousarray(r[:,fixed]);wf,rankf=inverse(np.ascontiguousarray((rf*q[:,None]).T@rf))
        learned=(total.sum()-held.sum())*(q@t)>=5
        rl=np.ascontiguousarray(r[:,learned]);wl,rankl=inverse(np.ascontiguousarray((rl*q[:,None]).T@rl))
        for j,(matrix,weight,mask,rank) in enumerate([(r0,w0,fixed,rank0),(rf,w0,fixed,rank0),(rf,wf,fixed,rankf),(rl,wl,learned,rankl)]):
            if not np.array_equal(result['method_masks'][k,j],mask) or result['ranks'][k,j]!=rank:
                raise ValueError('independent mask/rank reconstruction failed')
            summed=held@matrix
            diagonal=np.sum((matrix@weight)*matrix,axis=1)
            calculated[k,j]=(summed@weight@summed-held@diagonal)/(sizes[k]*(sizes[k]-1))
            mean=(p@matrix).astype(np.longdouble)
            truth[k,j]=mean@weight.astype(np.longdouble)@mean
    aggregate=sizes@calculated/sizes.sum();target=sizes@truth/sizes.sum()
    errors=[float(np.max(np.abs(calculated-result['fold_statistics']))),
            float(np.max(np.abs(truth-result['fold_targets']))),
            float(np.max(np.abs(aggregate-result['statistics']))),
            float(np.max(np.abs(target-result['conditional_targets'])))]
    return {'passed':max(errors)<=1e-8,'errors':errors,'maximum_numeric_error':max(errors),
            'parameter_vectors':5,'max_moment':maxmoment,'max_dual_error_bits':maxdual,
            'new_optimizer_calls':0}


def prepare_reference(tables,q0):
    g,t=design(len(q0).bit_length()-1)
    sizes=tables.sum(axis=1)
    mask=(tables.sum()-sizes.max())*(q0@t)>=5
    r0=np.ascontiguousarray(residual(q0,g,t)[:,mask])
    w0,rank0=inverse(np.ascontiguousarray((r0*q0[:,None]).T@r0))
    return g,t,mask,r0,w0,rank0
