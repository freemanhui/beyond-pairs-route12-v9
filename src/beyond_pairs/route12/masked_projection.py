"""Exact masked projections, plug-in targets and known-reference dual bounds."""
import numpy as np
from scipy.special import logsumexp

from beyond_pairs import order_ladder as ol
from beyond_pairs.information_layers import kl_bits
from .crossfit import _counts
from .numerics import checked_maxent, ProjectionFailure


def _law(values):
    p = np.asarray(values, float)
    n = len(p).bit_length()-1 if p.ndim == 1 else 0
    if (p.ndim != 1 or len(p) != 2**n or not 3 <= n <= 12
            or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0):
        raise ValueError('finite nonnegative binary law/counts required')
    return p/p.sum(), n


def _features(variables, mask):
    _, full = ol.state_features(variables, 3)
    pair = variables+variables*(variables-1)//2
    selected = np.asarray(mask)
    if selected.dtype != bool or selected.shape != (full.shape[1]-pair,):
        raise ValueError('one boolean mask entry per ordered triple required')
    return full[:, :pair], full, np.r_[np.arange(pair), pair+np.flatnonzero(selected)]


def _projection(p, features, options, *, start=None, cached=None):
    opts = {'solver_tolerance': 1e-10, 'moment_limit': 1e-9, 'entropy_limit_bits': 1e-7}
    if options is not None:
        if set(options)-set(opts):
            raise ValueError('unknown projection option')
        opts.update(options)
    if (any(not np.isfinite(x) or x <= 0 for x in opts.values())
            or opts['moment_limit'] > 1e-9 or opts['entropy_limit_bits'] > 1e-7
            or opts['solver_tolerance'] > 1e-10):
        raise ValueError('registered positive tolerances cannot be loosened')
    if cached is None:
        attempt = checked_maxent(p, features, start=start, tolerance=opts['solver_tolerance'])
        law, theta, dual = attempt['law'], attempt['theta'], attempt['dual_entropy_bits']
    else:
        law, theta = np.asarray(cached['law'], float), np.asarray(cached['theta'], float)
        if law.shape != p.shape or theta.shape != (features.shape[1],):
            raise ValueError('cached projection shape mismatch')
        dual = (logsumexp(features@theta)-theta@(p@features))/np.log(2)
    if not np.isfinite(theta).all() or not np.isfinite(law).all() or (law < 0).any():
        raise ValueError('nonfinite or negative projection')
    reconstruction = np.exp(features@theta-logsumexp(features@theta))
    diagnostics = {'moment_error': float(np.max(np.abs((p-law)@features))),
                   'entropy_dual_error_bits': abs(float(dual)-ol.entropy_bits(law)),
                   'normalisation_error': abs(float(law.sum())-1),
                   'reconstruction_error': float(np.max(np.abs(law-reconstruction)))}
    if (diagnostics['moment_error'] > opts['moment_limit']
            or diagnostics['entropy_dual_error_bits'] > opts['entropy_limit_bits']
            or diagnostics['normalisation_error'] > 1e-12
            or diagnostics['reconstruction_error'] > 1e-12):
        raise ProjectionFailure('projection moment/dual/normalisation/reconstruction gate failed',
                                {'theta': theta, 'law': law, 'diagnostics': diagnostics, 'accepted': False})
    return {'law': law, 'theta': theta, 'entropy_bits': ol.entropy_bits(law),
            'dual_entropy_bits': float(dual), 'diagnostics': diagnostics}


def masked_projection(law_or_counts, mask, *, start_pair=None, options=None):
    p, variables = _law(law_or_counts)
    g, full, columns = _features(variables, mask)
    start = np.zeros(len(columns))
    if start_pair is not None:
        start[:g.shape[1]] = start_pair
    result = _projection(p, full[:, columns], options, start=start)
    return {**result, 'mask': np.asarray(mask).copy(), 'columns': columns,
            'target': 'plugin_masked_projection', 'population_lower_confidence_bound': False}


def masked_truth(known_law, mask, *, cached_pair=None, cached_full=None, options=None):
    p, variables = _law(known_law)
    if abs(np.asarray(known_law).sum()-1) > 1e-12:
        raise ValueError('known truth must be a normalised law, not counts')
    g, full, columns = _features(variables, mask)
    pair = _projection(p, g, options, cached=cached_pair)
    if len(columns) == g.shape[1]:
        masked = pair
    else:
        start = np.zeros(len(columns)); start[:g.shape[1]] = pair['theta']
        masked = _projection(p, full[:, columns], options, start=start)
    if len(columns) == full.shape[1] and cached_full is None:
        complete = masked
    else:
        start = np.zeros(full.shape[1]); start[:g.shape[1]] = pair['theta']
        try:
            complete = _projection(p, full, options, start=start, cached=cached_full)
        except ProjectionFailure as error:
            raise ProjectionFailure(str(error), {**error.partial, 'accepted_masked': masked,
                'failed_projection_kind': 'cached_full' if cached_full is not None else 'full'}) from error
    layer = pair['entropy_bits']-masked['entropy_bits']
    full_layer = pair['entropy_bits']-complete['entropy_bits']
    remainder = kl_bits(complete['law'], masked['law'])
    identity = max(abs(layer-kl_bits(masked['law'], pair['law'])),
                   abs(full_layer-layer-remainder))
    if layer < -1e-7 or remainder < -1e-7 or identity > 1e-7:
        raise ProjectionFailure('masked projection nesting/KL identity gate failed',
            {'theta': masked['theta'], 'law': masked['law'], 'accepted_masked': masked,
             'failed_projection_kind': 'masked_identity', 'accepted': False})
    return {'masked_layer_bits': layer, 'full_L3_bits': full_layer,
            'remainder_bits': remainder, 'identity_error_bits': identity,
            'pair': pair, 'masked': masked, 'full': complete,
            'target': 'exact_specified_law_masked_information',
            'population_lower_confidence_bound': False}


def heldout_dual_score(held_counts, h, reference_law):
    counts = _counts(held_counts)
    q, _ = _law(reference_law)
    h = np.asarray(h, float)
    if (counts.sum() <= 0 or h.shape != counts.shape or q.shape != counts.shape
            or not np.isfinite(h).all() or (q <= 0).any()
            or abs(np.asarray(reference_law).sum()-1) > 1e-12):
        raise ValueError('held counts, finite h and positive normalised reference required')
    return float((counts@h/counts.sum()-logsumexp(np.log(q)+h))/np.log(2))


def known_reference_lower_bound(held_counts, h, reference_law, *, alpha,
                                reference_kind, folds=1, mask=None):
    if reference_kind != 'known_exact' or not 0 < alpha < 1 or folds < 1 or int(folds) != folds:
        raise ValueError('known exact reference and valid fold/alpha allocation required')
    q, variables = _law(reference_law)
    _, full = ol.state_features(variables, 3)
    if mask is not None:
        _, full, columns = _features(variables, mask)
        full = full[:, columns]
    design = np.column_stack([np.ones(len(q)), full])
    h = np.asarray(h, float)
    if h.shape != q.shape or not np.isfinite(h).all():
        raise ValueError('finite h over the whole support required')
    beta, *_ = np.linalg.lstsq(design, h, rcond=None)
    residual = float(np.max(np.abs(design@beta-h)))
    if residual > 1e-8:
        raise ValueError('h lies outside the declared third-order feature span')
    counts = _counts(held_counts)
    score = heldout_dual_score(counts, h, q)
    span = float(np.ptp(h))
    penalty = span*np.sqrt(np.log(folds/alpha)/(2*counts.sum()))/np.log(2)
    # The exact feature-span witness differs by at most residual in each state.
    # Mean and log-normaliser are each sup-norm Lipschitz: charge both errors.
    span_allowance = 2*residual/np.log(2)
    lower = score-penalty-span_allowance
    return {'unclipped_bits': lower, 'lower_bound_bits': max(0., lower),
            'range_nats': span, 'penalty_bits': float(penalty), 'held_size': int(counts.sum()),
            'span_residual_nats': residual, 'span_allowance_bits': span_allowance,
            'alpha_per_fold': alpha/folds, 'reference_kind': reference_kind,
            'target': 'full_L3_of_known_reference_generating_law',
            'scope': 'Valid only when the specified reference is exactly P2(P), h is trained without this held fold, and the registered iid/support assumptions hold.'}
