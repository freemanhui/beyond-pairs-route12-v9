"""Exact information layers of an enumerated law, not estimates from patients.

The projections use the existing moment-matching solver without a probability
floor. Entropies and KL divergences are in bits; model parameters are in nats.
Diagnostics retain numerical errors rather than clipping negative differences.
See docs/research-directive-2026-10-09.md, sections 3.1, 3.6 and D1.
"""
import numpy as np
from scipy.special import logsumexp

from beyond_pairs import order_ladder as ol
from beyond_pairs.hidden_layer import Units


def _probabilities(values, name):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f'{name} must be a finite nonnegative probability vector')
    if not np.isclose(p.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f'{name} must sum to one; counts are not accepted')
    return p


def kl_bits(p, q):
    """D(p || q), with infinite divergence for a missing reference support cell."""
    p, q = _probabilities(p, 'p'), _probabilities(q, 'q')
    if p.shape != q.shape:
        raise ValueError('p and q must have the same state space')
    present = p > 0
    if (q[present] == 0).any():
        return float('inf')
    return float(p[present] @ (np.log2(p[present]) - np.log2(q[present])))


def information_layers(p, tolerance=1e-10):
    """P1/P2/P3, Lambda2, L3, L and L>3 for at least three binary variables.

    Boundary laws are allowed. Their projections may be limiting family members;
    the reported moment and entropy/KL errors must still meet the run's criteria.
    No finite-sample correction or residual-data test is performed here.
    """
    p = _probabilities(p, 'p')
    variables = len(p).bit_length() - 1
    if variables < 3 or len(p) != 2**variables:
        raise ValueError('enumerate all states of at least three binary variables')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive and finite')
    laws, parameters, moments, dual_errors = {}, {}, {}, {}
    previous = np.zeros(0)
    for order in (1, 2, 3):
        _, features = ol.state_features(variables, order)
        start = np.zeros(features.shape[1])
        start[:len(previous)] = previous
        q, previous, dual_entropy = ol.maxent(p, features, start, tolerance=tolerance)
        key = f'P{order}'
        laws[key], parameters[key] = q, previous.copy()
        moments[key] = float(np.abs((p - q) @ features).max())
        dual_errors[key] = abs(dual_entropy - ol.entropy_bits(q))
    entropy = {'P': ol.entropy_bits(p), **{k: ol.entropy_bits(q) for k, q in laws.items()}}
    layers = {
        'pairwise': entropy['P1'] - entropy['P2'],
        'third_order': entropy['P2'] - entropy['P3'],
        'total_above_pairs': entropy['P2'] - entropy['P'],
        'above_order_three': entropy['P3'] - entropy['P'],
    }
    divergence = {
        'pairwise': kl_bits(laws['P2'], laws['P1']),
        'third_order': kl_bits(laws['P3'], laws['P2']),
        'total_above_pairs': kl_bits(p, laws['P2']),
        'above_order_three': kl_bits(p, laws['P3']),
    }
    identity_errors = {k: abs(layers[k] - divergence[k]) for k in layers}
    identity_errors['telescoping'] = abs(layers['total_above_pairs'] - layers['third_order'] - layers['above_order_three'])
    identity_errors['pythagorean_P_P3_P2'] = abs(divergence['total_above_pairs'] - divergence['third_order'] - divergence['above_order_three'])
    identity_errors['pythagorean_P_P2_P1'] = abs(kl_bits(p, laws['P1']) - divergence['total_above_pairs'] - divergence['pairwise'])
    denominator = layers['pairwise']
    return {
        'variables': variables, 'laws': laws, 'parameters': parameters,
        'entropy_bits': entropy, 'layers_bits': layers, 'divergences_bits': divergence,
        'shares': {
            'third_order_over_pairwise': layers['third_order']/denominator if denominator > 10*tolerance else None,
            'total_over_pairwise': layers['total_above_pairs']/denominator if denominator > 10*tolerance else None,
        },
        'diagnostics': {
            'moment_error': moments, 'entropy_dual_error_bits': dual_errors,
            'identity_error_bits': identity_errors,
            'input_zero_states': int((p == 0).sum()),
            'projection_minimum_probability': {k: float(q.min()) for k, q in laws.items()},
            'probability_floor_applied': False,
        },
    }


def unit_components(model, theta):
    """Hidden configurations, their probabilities, and P(x | h) for Units.

    Supports both RBMs and visible-pairs-plus-units. Configuration weights include
    the visible partition functions; they are not independent sigmoid(b) weights.
    """
    if not isinstance(model, Units):
        raise TypeError('unit_components requires a Units model')
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (model.size,) or not np.isfinite(theta).all():
        raise ValueError('theta must contain every finite model parameter')
    hidden = ((np.arange(2**model.units)[:, None] >> np.arange(model.units)) & 1).astype(float)
    linear, bias, weight = model.parts(theta)
    energy = (model.linear @ linear)[None, :] + (hidden @ weight) @ model.states.T
    normalizer = logsumexp(energy, axis=1)
    log_weight = hidden @ bias + normalizer
    mixture_weights = np.exp(log_weight - logsumexp(log_weight))
    return hidden, mixture_weights, np.exp(energy - normalizer[:, None])


def mixture_information(weights, components, reference):
    """Verify sum pi D(Qc||R) = D(P||R) + I(X;C) in bits.

    Zero-weight components are ignored. R must cover the mixture support, since
    subtracting two infinite divergences would not be a numerical identity check.
    """
    weights, reference = _probabilities(weights, 'weights'), _probabilities(reference, 'reference')
    components = np.asarray(components, dtype=np.float64)
    if components.shape != (len(weights), len(reference)):
        raise ValueError('components must have one probability row per weight')
    for row in components:
        _probabilities(row, 'component')
    mixture = weights @ components
    if ((mixture > 0) & (reference == 0)).any():
        raise ValueError('reference must cover the mixture support')
    present = weights > 0
    average = float(sum(w*kl_bits(q, reference) for w, q in zip(weights[present], components[present])))
    mutual = float(sum(w*kl_bits(q, mixture) for w, q in zip(weights[present], components[present])))
    marginal = kl_bits(mixture, reference)
    entropy_mutual = ol.entropy_bits(mixture) - sum(w*ol.entropy_bits(q) for w, q in zip(weights, components))
    return {
        'mixture': mixture,
        'average_component_divergence_bits': average,
        'mixture_divergence_bits': marginal,
        'mutual_information_bits': mutual,
        'compensation_error_bits': abs(average - marginal - mutual),
        'mutual_information_entropy_error_bits': abs(mutual - entropy_mutual),
    }
