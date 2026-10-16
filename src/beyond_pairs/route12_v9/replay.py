"""Reconstruct new maps from unchanged fitted endpoints; no optimizer imports."""
from functools import lru_cache
import numpy as np
from . import numerics as nu

METHODS=('A_oracle','B_estimated_residual_fixed_weight_mask',
         'C_estimated_weight_fixed_mask','D_estimated_weight_and_mask')


@lru_cache(maxsize=2)
def designs(variables):
    _, full=nu.features(variables)
    pair=variables+variables*(variables-1)//2
    return nu.canonical(full[:,:pair]),nu.canonical(full[:,pair:])


def reference_map(q,minimum_train):
    variables=len(q).bit_length()-1
    g,t=designs(variables)
    residual,diagnostic=nu.residual(q,g,t)
    mask=minimum_train*(q@t)>=5
    selected=nu.canonical(residual[:,mask])
    inv=nu.psd_inverse(nu.canonical((selected*q[:,None]).T@selected))
    return {'r':selected,'inverse':inv,'mask':mask,'nuisance':diagnostic}


def pair_diagnostics(train,theta,g):
    q,logq=nu.law_from_theta(theta,g)
    empirical=train/train.sum()
    energy=g@theta
    top=energy.max()
    dual=(top+np.log(np.exp(energy-top).sum())-theta@(empirical@g))/np.log(2)
    entropy=-float(q@logq/np.log(2))
    diagnostic=np.array([np.max(np.abs((empirical-q)@g)),abs(dual-entropy),abs(q.sum()-1)])
    if (not np.isfinite(diagnostic).all() or diagnostic[0]>1e-9
            or diagnostic[1]>1e-7 or diagnostic[2]>1e-12):
        raise ValueError('unchanged P2 moment/dual/normalization gate failed')
    return q,diagnostic


def analyse(tables,p,q0,parameters,*,prepared_reference=None):
    tables=np.asarray(tables)
    variables=len(p).bit_length()-1
    g,t=designs(variables)
    if tables.shape!=(5,len(p)) or parameters.shape!=(5,g.shape[1]):
        raise ValueError('complete five-fold table/parameter shapes required')
    if not np.isfinite(parameters).all() or np.any(tables<0) or not np.equal(tables,np.floor(tables)).all():
        raise ValueError('complete finite parameters and nonnegative integer counts required')
    total=tables.sum(axis=0)
    sizes=tables.sum(axis=1)
    if np.any(sizes<=1) or np.max(np.abs((p-q0)@g))>1e-9:
        raise ValueError('held sizes or known pair moments invalid')
    ref=prepared_reference or reference_map(q0,int(total.sum()-sizes.max()))
    statistics=np.zeros((5,4));targets=np.zeros((5,4))
    masks=np.zeros((5,4,t.shape[1]),bool);ranks=np.zeros((5,4),int)
    diagnostics=[];pair_errors=[];max_contraction=0.
    for k,held in enumerate(tables):
        train=total-held
        q,pair_error=pair_diagnostics(train,parameters[k],g)
        pair_errors.append(pair_error)
        residual,nuisance=nu.residual(q,g,t)
        fixed=nu.canonical(residual[:,ref['mask']])
        fixed_inverse=nu.psd_inverse(nu.canonical((fixed*q[:,None]).T@fixed))
        mask=train.sum()*(q@t)>=5
        learned=nu.canonical(residual[:,mask])
        learned_inverse=nu.psd_inverse(nu.canonical((learned*q[:,None]).T@learned))
        maps=[(ref['r'],ref['inverse'],ref['mask']),
              (fixed,ref['inverse'],ref['mask']),
              (fixed,fixed_inverse,ref['mask']),
              (learned,learned_inverse,mask)]
        errors=[]
        for j,(r,inverse,selected) in enumerate(maps):
            statistics[k,j],targets[k,j],error=nu.kernel_values(held,p,r,inverse)
            max_contraction=max(max_contraction,*error.values())
            masks[k,j]=selected;ranks[k,j]=inverse['rank']
            errors.append(error)
        diagnostics.append({'fold':k,'nuisance':nuisance,
            'fixed':fixed_inverse['diagnostics'],'learned':learned_inverse['diagnostics'],
            'contractions':errors})
    weights=sizes/sizes.sum()
    return {'status':'passed' if max_contraction<=1e-8 else 'failed_contraction_gate',
        'parameters':parameters,'fold_statistics':statistics,'fold_targets':targets,
        'statistics':weights@statistics,'conditional_targets':weights@targets,
        'method_masks':masks,'ranks':ranks,'pair_errors':np.array(pair_errors),
        'maximum_contraction_error':max_contraction,'map_diagnostics':diagnostics,
        'held_sizes':sizes,'methods':METHODS,'new_optimizer_calls':0,
        'target_kind':'training_conditional_quadratic','full_entropy_qualification':False}
