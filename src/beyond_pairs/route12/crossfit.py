"""Cross-fitted quadratic targets, without an entropy or patient-CI claim."""
from hashlib import sha256

import numpy as np

from beyond_pairs import measurement_score as ms
from beyond_pairs import order_ladder as ol
from .numerics import checked_maxent, ProjectionFailure


def _counts(values):
    c = np.asarray(values, dtype=float)
    if (c.ndim != 1 or not np.isfinite(c).all() or (c < 0).any()
            or not np.equal(c, np.floor(c)).all()):
        raise ValueError('finite nonnegative integer observation counts required')
    return c


def counts_digest(counts):
    return sha256(_counts(counts).astype('<i8').tobytes()).hexdigest()


def distinct_pair_mean(counts, residual, weight):
    c = _counts(counts)
    n = float(c.sum())
    r, w = np.asarray(residual, float), np.asarray(weight, float)
    if (n <= 1 or r.ndim != 2 or r.shape[0] != len(c)
            or w.shape != (r.shape[1], r.shape[1])
            or not np.isfinite(r).all() or not np.isfinite(w).all()
            or not np.allclose(w, w.T, rtol=0, atol=1e-12)):
        raise ValueError('two held observations and a finite conformable symmetric kernel required')
    total = c @ r
    self_terms = np.einsum('si,ij,sj->s', r, w, r)
    return float((total @ w @ total - c @ self_terms) / (n*(n-1)))


def fit_training_map(train_counts, *, minimum_expected=5, mask_policy='learned',
                     weight_policy='estimated', reference=None, options=None,
                     cached_pair=None):
    c = _counts(train_counts)
    variables = len(c).bit_length()-1
    if len(c) != 2**variables or not 3 <= variables <= 12 or c.sum() <= 1:
        raise ValueError('nonempty enumerated binary training counts required')
    if not np.isfinite(minimum_expected) or minimum_expected < 0:
        raise ValueError('expected-count threshold must be nonnegative and finite')
    opts = {} if options is None else dict(options)
    if opts.keys()-{'inverse_relative_cutoff'}:
        raise ValueError('unregistered map option')
    relative = opts.get('inverse_relative_cutoff', 1e-10)
    if not np.isfinite(relative) or not 0 < relative <= 1e-10:
        raise ValueError('registered positive inverse cutoff required')
    _, g, t = ms.features(variables)
    digest = counts_digest(c)
    if cached_pair is None:
        attempt = checked_maxent(c/c.sum(), g)
        q, theta = attempt['law'], attempt['theta']
        diagnostics = np.array([attempt['moment_error'],
                                abs(attempt['dual_entropy_bits']-ol.entropy_bits(q)),
                                abs(float(q.sum())-1)])
        if diagnostics[0] > 1e-9 or diagnostics[1] > 1e-7 or diagnostics[2] > 1e-12:
            raise ProjectionFailure('pair projection numerical gate failed', {**attempt, 'accepted': False})
        origin = 'new_pair_zero_v1'
    else:
        if (cached_pair.get('train_digest') != digest
                or cached_pair.get('start_convention') != 'pair_zero_v1'):
            raise ValueError('incompatible cached pair input/start convention')
        q, theta = np.asarray(cached_pair['law'], float), np.asarray(cached_pair['theta'], float)
        if (q.shape != c.shape or theta.shape != (g.shape[1],)
                or not np.isfinite(theta).all()):
            raise ValueError('incompatible cached pair dimensions')
        energy = g @ theta
        if not np.isfinite(energy).all():
            raise ValueError('cached pair has nonfinite energy')
        expected = np.exp(energy-energy.max()); expected /= expected.sum()
        p = c/c.sum()
        if (not np.isfinite(q).all() or not np.isfinite(expected).all() or (q <= 0).any()
                or np.max(np.abs(q-expected)) > 1e-12
                or np.max(np.abs((p-q) @ g)) > 1e-9):
            raise ValueError('cached pair reconstruction/moment gate failed')
        diagnostics = np.array([np.max(np.abs((p-q) @ g)),
                                abs(float((np.log(np.exp(energy-energy.max()).sum())+energy.max()-theta@(p@g))/np.log(2))
                                    + float(q @ np.log2(q))), abs(q.sum()-1)])
        if not np.isfinite(diagnostics).all() or diagnostics[1] > 1e-7:
            raise ValueError('cached pair entropy/dual gate failed')
        origin = 'compatible_saved_pair_zero_v1'
    mg, mt = q@g, q@t
    cg, ct = g-mg, t-mt
    inverse, nuisance_rank, _ = ms.inverse_psd((cg*q[:, None]).T@cg)
    b = ((ct*q[:, None]).T@cg) @ inverse
    full_residual = ct-cg@b.T
    if mask_policy == 'learned':
        mask = c.sum()*mt >= minimum_expected
    elif mask_policy == 'fixed' and reference is not None:
        mask = np.asarray(reference['mask'])
        if mask.dtype != bool or mask.shape != (t.shape[1],):
            raise ValueError('fixed reference mask has wrong shape/type')
    elif mask_policy == 'full':
        mask = np.ones(t.shape[1], bool)
    else:
        raise ValueError('registered mask policy/reference required')
    residual = full_residual[:, mask]
    covariance = (residual*q[:, None]).T@residual
    if not mask.any():
        weight, rank, eig = np.empty((0, 0)), 0, np.array([0.])
    elif weight_policy == 'estimated':
        weight, rank, eig = ms.inverse_psd(covariance, relative)
    elif weight_policy == 'reference' and reference is not None:
        weight = np.asarray(reference['weight'], float)
        if (weight.shape != covariance.shape or not np.isfinite(weight).all()
                or not np.allclose(weight, weight.T, rtol=0, atol=1e-12)):
            raise ValueError('fixed reference weight has wrong shape/symmetry')
        eig = np.linalg.eigvalsh(weight)
        if eig[0] < -1e-10:
            raise ValueError('reference weight must be positive semidefinite')
        rank = int(np.sum(eig > max(1e-12, eig[-1]*relative)))
    else:
        raise ValueError('registered weight policy/reference required')
    return {'status': 'passed' if mask.any() else 'no_supported_coordinates',
            'residual': residual, 'weight': weight, 'mask': mask.copy(), 'rank': rank,
            'nuisance_rank': nuisance_rank, 'theta': theta, 'law': q,
            'pair_diagnostics': diagnostics, 'pair_origin': origin,
            'train_size': int(c.sum()), 'train_digest': digest,
            'minimum_eigenvalue': float(eig[0]), 'maximum_eigenvalue': float(eig[-1]),
            'mask_policy': mask_policy, 'weight_policy': weight_policy,
            'target': 'training_conditional_quadratic', 'population_confidence': False}


def aggregate_folds(values, held_sizes):
    values, sizes = np.asarray(values, float), np.asarray(held_sizes, float)
    if (values.ndim != 1 or values.shape != sizes.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(sizes).all()
            or (sizes <= 1).any()):
        raise ValueError('all finite folds and at least two held observations per fold required')
    return float(values @ sizes/sizes.sum())


def crossfit_measurement(fold_counts, *, reference=None, policies=None,
                         cached_pairs=None, minimum_expected=5):
    tables = np.asarray(fold_counts)
    if tables.ndim != 2 or tables.shape[0] != 5:
        raise ValueError('five fold tables required')
    for row in tables:
        _counts(row)
    total, folds = tables.sum(axis=0), []
    kwargs = {} if policies is None else dict(policies)
    for k, held in enumerate(tables):
        mapping = fit_training_map(total-held, minimum_expected=minimum_expected,
                                   reference=reference, **kwargs,
                                   cached_pair=None if cached_pairs is None else cached_pairs[k])
        reading = distinct_pair_mean(held, mapping['residual'], mapping['weight'])
        folds.append({**mapping, 'fold': k, 'held_size': int(held.sum()),
                      'per_observation': reading})
    value = aggregate_folds([f['per_observation'] for f in folds], [f['held_size'] for f in folds])
    return {'folds': folds, 'per_observation': value,
            'quadratic_bits_per_1000': 1000*value/(2*np.log(2)),
            'aggregation': 'held_size_weighted', 'population_confidence': False}
