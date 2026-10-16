"""Unfloored reference scores and variance/range-sensitive bound diagnostics."""
import numpy as np
from scipy.special import logsumexp
from .contracts import counts

REFERENCES = {'known_exact_artificial', 'numerically_projected_known_law'}


def positive_law(value):
    p = np.asarray(value, float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p<=0).any() or abs(p.sum()-1)>1e-12:
        raise ValueError('positive normalised finite law required')
    return p


def kl_bits(p, q):
    p, q = positive_law(p), positive_law(q)
    return float(p@(np.log(p)-np.log(q))/np.log(2))


def oracle_reference_gain(p, q, log_r, *, reference_kind):
    if reference_kind not in REFERENCES:
        raise ValueError('specified known pairwise reference required')
    p, q = positive_law(p), positive_law(q)
    log_r = np.asarray(log_r, float)
    if log_r.shape != p.shape or not np.isfinite(log_r).all() or abs(logsumexp(log_r))>1e-10:
        raise ValueError('normalised finite unfloored log law required')
    return {'GQ_bits':float(p@(log_r-np.log(q))/np.log(2)),
            'reference_kind':reference_kind,'floor_kind':'none',
            'finite_sample_certificate':False}


def feature_span_error(h, features):
    h, features = np.asarray(h,float), np.asarray(features,float)
    design = np.column_stack([np.ones(len(h)),features])
    beta = np.linalg.lstsq(design,h,rcond=None)[0]
    error = float(np.max(np.abs(design@beta-h)))
    if not np.isfinite(error) or error>1e-8:
        raise ValueError('log score is outside the declared exponential-family span')
    return error


def empirical_bernstein_lower_bound(held_counts, h, q, *, delta, reference_kind,
                                   span_error_allowance, reference_error_allowance):
    if reference_kind not in REFERENCES or not 0<delta<1:
        raise ValueError('known reference and valid allocated delta required')
    c, h, q = counts(held_counts), np.asarray(h,float), positive_law(q)
    n = int(c.sum())
    if n<2 or h.shape!=c.shape or q.shape!=c.shape or not np.isfinite(h).all():
        raise ValueError('finite held counts and score over the entire support required')
    if min(span_error_allowance,reference_error_allowance)<0:
        raise ValueError('nonnegative numerical allowances required')
    mean = float(c@h/n)
    sample_variance = float(c@((h-mean)**2)/(n-1))
    width = float(np.ptp(h))
    log_factor = np.log(2/delta)
    variance_penalty = float(np.sqrt(2*sample_variance*log_factor/n))
    range_penalty = float(7*width*log_factor/(3*(n-1)))
    normalizer = float(logsumexp(np.log(q)+h))
    lower = (mean-variance_penalty-range_penalty-normalizer)/np.log(2)-span_error_allowance-reference_error_allowance
    return {'n':n,'delta':delta,'mean_nats':mean,'sample_variance_nats2':sample_variance,
        'range_nats':width,'variance_penalty_bits':variance_penalty/np.log(2),
        'range_penalty_bits':range_penalty/np.log(2),'log_normalizer_nats':normalizer,
        'unclipped_bits':float(lower),'lower_bound_bits':float(max(0.,lower)),
        'span_error_allowance_bits':span_error_allowance,
        'reference_error_allowance_bits':reference_error_allowance,
        'reference_kind':reference_kind,
        'rigorous_machine_reference_certificate':reference_kind=='known_exact_artificial',
        'scope':'Numerical known-reference illustration unless exact reference/error enclosure is established.'}
