"""Finite-state benchmark targets and statistics for the v11 registration."""
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.stats import beta, t as student_t

from beyond_pairs.measurement_score import fold_sizes
from beyond_pairs.route12.masked_projection import _projection
from beyond_pairs.route12_v9 import numerics as frozen


@lru_cache(maxsize=4)
def designs(d):
    states, full = frozen.features(d)
    pair = d * (d + 1) // 2
    return states, full[:, :pair], full[:, pair:], full


def law(theta, features):
    return frozen.law_from_theta(theta, features)


def covariance(p, features):
    centered = features - p @ features
    return (centered * p[:, None]).T @ centered


def sandwich(p, projection, features):
    """Full, unregularized trace; refuse an unidentified inverse."""
    h = covariance(projection, features)
    j = covariance(p, features)
    return sandwich_matrices(h, j)


def sandwich_matrices(h, j):
    values, vectors = np.linalg.eigh((h + h.T) / 2)
    if values[0] <= 0:
        raise ValueError('singular sandwich Hessian: no qualified full inverse')
    solved = np.linalg.solve(h, j)
    trace = float(np.trace(solved))
    independent = float(np.sum(np.diag(vectors.T @ j @ vectors) / values))
    residual = float(np.max(np.abs(h @ solved - j)))
    if residual > 1e-9 or abs(trace-independent) > 1e-7 * max(1., abs(trace)):
        raise ValueError('independent sandwich contraction/residual failed')
    return {'trace': trace, 'eigen_trace': independent, 'solve_residual': residual,
            'minimum_eigenvalue': float(values[0]),
            'condition': float(values[-1]/values[0]), 'dimension': h.shape[0],
            'H': h, 'J': j}


def inverse_factor(matrix, cutoff=1e-10):
    matrix = np.asarray(matrix, float)
    if matrix.shape == (0, 0):
        return matrix.copy(), 0
    if np.max(np.abs(matrix-matrix.T)) > max(1e-12, np.max(np.abs(matrix))*1e-12):
        raise ValueError('asymmetric covariance')
    values, vectors = np.linalg.eigh((matrix+matrix.T)/2)
    if values[-1] <= 0 or values[0] < -max(1e-12, values[-1]*1e-8):
        raise ValueError('indefinite covariance')
    keep = values > cutoff*values[-1]
    return np.ascontiguousarray(vectors[:, keep]/np.sqrt(values[keep])), int(keep.sum())


def residual(q, g, triples):
    gc, tc = g-q@g, triples-q@triples
    f, _ = inverse_factor((gc*q[:, None]).T@gc)
    cross = (tc*q[:, None]).T@gc
    return np.ascontiguousarray(tc-gc@(cross@(f@f.T)).T)


def map_components(q, g, triples, mask, cutoff=1e-10, prepared_residual=None):
    r = residual(q, g, triples) if prepared_residual is None else prepared_residual
    selected = np.ascontiguousarray(r[:, mask])
    factor, rank = inverse_factor((selected*q[:, None]).T@selected, cutoff)
    return selected, factor, rank


def u_value(counts, r, factor, *, check=False):
    counts = np.asarray(counts, float)
    n = float(counts.sum())
    if n <= 1 or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('nonnegative counts with at least two observations required')
    if r.shape[1] == 0:
        return 0.
    z = r@factor
    total = counts@z
    result = float((total@total-counts@np.sum(z*z, axis=1))/(n*(n-1)))
    if check:
        w = factor@factor.T
        summed = counts@r
        alternate = float((summed@w@summed-counts@np.sum((r@w)*r, axis=1))/(n*(n-1)))
        if abs(result-alternate) > 1e-8:
            raise ValueError('U contraction gate failed')
    return result


def accepted_pair(train, theta, g):
    q, logq = law(theta, g)
    empirical = train/train.sum()
    energy = g@theta
    top = energy.max()
    dual = (top+np.log(np.exp(energy-top).sum())-theta@(empirical@g))/np.log(2)
    diagnostics = np.array([np.max(np.abs((empirical-q)@g)),
                            abs(dual+q@logq/np.log(2)), abs(q.sum()-1)])
    if not np.isfinite(diagnostics).all() or np.any(diagnostics > [1e-9, 1e-7, 1e-12]):
        raise ValueError('pair endpoint moment/dual/normalization gate')
    return q, diagnostics


def fully_learned(tables, parameters, *, cutoff=1e-10, minimum=5, check=False):
    d = tables.shape[1].bit_length()-1
    _, g, triples, _ = designs(d)
    total = tables.sum(axis=0)
    sizes = tables.sum(axis=1)
    values, mask_sizes, ranks = [], [], []
    for k, held in enumerate(tables):
        train = total-held
        q, _ = accepted_pair(train, parameters[k], g)
        mask = train.sum()*(q@triples) >= minimum
        r, f, rank = map_components(q, g, triples, mask, cutoff)
        values.append(u_value(held, r, f, check=check))
        mask_sizes.append(int(mask.sum())); ranks.append(rank)
    return float(np.asarray(values)@(sizes/sizes.sum())), np.array(values), np.array(mask_sizes), np.array(ranks)


def fit_training(tables):
    _, g, _, _ = designs(tables.shape[1].bit_length()-1)
    parameters = np.full((5, g.shape[1]), np.nan)
    diagnostics = np.full((5, 4), np.nan)
    total = tables.sum(axis=0)
    for k, held in enumerate(tables):
        empirical = (total-held)/(total.sum()-held.sum())
        try:
            result = _projection(empirical, g, None)
        except Exception as exc:
            if hasattr(exc, 'partial') and 'theta' in exc.partial:
                parameters[k] = exc.partial['theta']
            return parameters, diagnostics, k+1, k, str(exc)
        parameters[k] = result['theta']
        diagnostics[k] = list(result['diagnostics'].values())
    return parameters, diagnostics, 5, 5, None


def lrt_from_parameters(counts, theta2, theta3):
    _, g, _, full = designs(len(counts).bit_length()-1)
    _, log2 = law(theta2, g); _, log3 = law(theta3, full)
    raw = float(2*np.asarray(counts)@(log3-log2))
    if raw < -2*counts.sum()*2e-7:
        raise ValueError('negative nested likelihood beyond registered roundoff tolerance')
    return max(0., raw), raw


def pooled_lrt(counts):
    _, g, _, full = designs(len(counts).bit_length()-1)
    empirical = counts/counts.sum()
    theta2 = np.full(g.shape[1], np.nan); theta3 = np.full(full.shape[1], np.nan)
    diagnostics = np.full((2, 4), np.nan)
    try:
        p2 = _projection(empirical, g, None)
        theta2 = p2['theta']; diagnostics[0] = list(p2['diagnostics'].values())
    except Exception as exc:
        if hasattr(exc, 'partial') and 'theta' in exc.partial: theta2 = exc.partial['theta']
        return theta2, theta3, diagnostics, np.nan, np.nan, 1, 0, str(exc)
    try:
        p3 = _projection(empirical, full, None, start=np.r_[theta2, np.zeros(full.shape[1]-g.shape[1])])
        theta3 = p3['theta']; diagnostics[1] = list(p3['diagnostics'].values())
        value, raw = lrt_from_parameters(counts, theta2, theta3)
    except Exception as exc:
        if hasattr(exc, 'partial') and 'theta' in exc.partial: theta3 = exc.partial['theta']
        return theta2, theta3, diagnostics, np.nan, np.nan, 2, 1, str(exc)
    return theta2, theta3, diagnostics, value, raw, 2, 2, None


ORACLE_METHODS = tuple(f'{mask}_{r}_{w}' for mask in ('full', 'fixed')
                       for r in ('oracleR', 'learnedR') for w in ('oracleW', 'learnedW')) + ('adaptive_oracle', 'adaptive_learned')


def oracle_factorial(counts, q0, qhat, *, prepared=None):
    _, g, triples, _ = designs(len(counts).bit_length()-1)
    ro = residual(q0, g, triples) if prepared is None else prepared
    rl = residual(qhat, g, triples)
    n = counts.sum(); values = []; masks = []
    for mask in [np.ones(triples.shape[1], bool), n*(q0@triples)>=5]:
        oo, fo, _ = map_components(q0, g, triples, mask, prepared_residual=ro)
        ll, fl, _ = map_components(qhat, g, triples, mask, prepared_residual=rl)
        for r in [oo, ll]:
            for f in [fo, fl]: values.append(u_value(counts, r, f, check=True))
        masks.append(int(mask.sum()))
    mask = n*(qhat@triples)>=5
    for q, r in [(q0, ro), (qhat, rl)]:
        rr, f, _ = map_components(q, g, triples, mask, prepared_residual=r)
        values.append(u_value(counts, rr, f, check=True))
    return np.array(values), np.array(masks+[int(mask.sum())])


def rank_decisions(calibration, evaluations, alpha=.05):
    cal = np.sort(np.asarray(calibration, float)); values = np.asarray(evaluations, float)
    if not len(cal) or not np.isfinite(cal).all() or not np.isfinite(values).all():
        raise ValueError('complete finite calibration and evaluations required')
    positions = np.searchsorted(cal, values, side='left')
    numerators = len(cal)+1-positions
    return numerators <= int(np.floor(alpha*(len(cal)+1)+1e-12))


def cp(count, total):
    return [float(beta.ppf(.025, count, total-count+1)) if count else 0.,
            float(beta.ppf(.975, count+1, total-count)) if count < total else 1.]


def paired(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    delta = a.astype(float)-b.astype(float)
    return {'A_only': int(np.sum(a & ~b)), 'B_only': int(np.sum(b & ~a)),
            'both': int(np.sum(a & b)), 'neither': int(np.sum(~a & ~b)),
            'mean_difference': float(delta.mean()),
            'paired_SE': float(delta.std(ddof=1)/np.sqrt(len(delta)))}


def pool_summary(counts, denominators):
    rates = np.asarray(counts)/np.asarray(denominators)
    if len(rates) < 2: raise ValueError('multiple independent pools required')
    mean = float(rates.mean()); sd = float(rates.std(ddof=1)); se = sd/np.sqrt(len(rates))
    margin = float(student_t.ppf(.975, len(rates)-1)*se)
    bound = float(np.sqrt(np.log(40)/(2*len(rates))))
    return {'pools':len(rates), 'equal_pool_mean':mean, 'between_pool_SD':sd, 'MCSE':se,
            'approximate_t_interval':[max(0.,mean-margin),min(1.,mean+margin)],
            'Hoeffding_pool_interval':[max(0.,mean-bound),min(1.,mean+bound)],
            'counts':list(map(int,counts)), 'denominators':list(map(int,denominators)),
            'conditional_intervals':[cp(int(k),int(n)) for k,n in zip(counts,denominators)]}


def triple_labels(genes):
    return ['/'.join(genes[i] for i in c) for c in combinations(range(len(genes)),3)]
