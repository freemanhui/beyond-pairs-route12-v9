"""The historical floored held-out ladder, with retained numerical diagnostics.

This estimates prediction gains in finite samples. They are not the unfloored
information layers of a population law or a test of latent dimension.
"""
import numpy as np

from . import order_ladder as ol


class LadderFailure(RuntimeError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


def projection_chain(p, top=4, *, solver_tolerance=1e-10,
                     moment_limit=1e-9, entropy_limit_bits=1e-7):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('law must be a finite nonnegative vector')
    variables = len(p).bit_length() - 1
    if len(p) != 2**variables or not 1 <= top <= variables or abs(p.sum() - 1) > 1e-12:
        raise ValueError('law must enumerate normalised binary states and a valid top order')
    records, previous = [], np.zeros(0)
    for order in range(1, top + 1):
        _, features = ol.state_features(variables, order)
        start = np.zeros(features.shape[1]); start[:len(previous)] = previous
        try:
            q, theta, dual_entropy = ol.maxent(p, features, start, tolerance=solver_tolerance)
        except Exception as error:
            raise LadderFailure('projection order ' + str(order) + ': ' + str(error),
                                {'failed_order': order, 'completed_projections': records}) from error
        diagnostics = {'moment_error': float(np.max(np.abs((p - q) @ features))),
                       'entropy_dual_error_bits': abs(float(dual_entropy) - ol.entropy_bits(q)),
                       'normalisation_error': abs(float(q.sum()) - 1),
                       'zero_probability_states': int((q == 0).sum()),
                       'finite_law_and_parameters': bool(np.isfinite(q).all() and np.isfinite(theta).all())}
        passed = (diagnostics['finite_law_and_parameters'] and not (q < 0).any()
                  and diagnostics['normalisation_error'] <= 1e-12
                  and diagnostics['moment_error'] <= moment_limit
                  and diagnostics['entropy_dual_error_bits'] <= entropy_limit_bits)
        records.append({'order': order, 'law': q.tolist(), 'theta': theta.tolist(),
                        'entropy_bits': ol.entropy_bits(q), 'diagnostics': diagnostics,
                        'numerical_gate_passed': bool(passed)})
        if not passed:
            raise LadderFailure('projection diagnostics failed at order ' + str(order),
                                {'failed_order': order, 'completed_projections': records})
        previous = theta
    return records


def score_law(law, training_size, held_counts):
    law, held = np.asarray(law, dtype=float), np.asarray(held_counts, dtype=float)
    if (law.ndim != 1 or law.shape != held.shape or len(law) == 0
            or not np.isfinite(law).all() or not np.isfinite(held).all()
            or (law < 0).any() or (held < 0).any() or abs(law.sum() - 1) > 1e-12
            or held.sum() <= 0 or not np.isfinite(training_size) or training_size < 1):
        raise ValueError('normalised law, nonempty counts and at least one training observation required')
    floor = 1.0 / training_size
    scored = (1 - floor) * law + floor / len(law)
    return float(held @ np.log2(scored) / held.sum())


def folded_ladder(tables, top=4, **projection_options):
    tables = np.asarray(tables, dtype=float)
    if (tables.ndim != 2 or tables.shape[0] != 5 or not np.isfinite(tables).all()
            or (tables < 0).any() or (tables.sum(axis=1) <= 0).any()):
        raise ValueError('five nonempty finite nonnegative fold tables required')
    total, folds, scores = tables.sum(axis=0), [], []
    for fold, held in enumerate(tables):
        train = total - held
        try:
            projections = projection_chain(train / train.sum(), top=top, **projection_options)
        except LadderFailure as error:
            raise LadderFailure('fold ' + str(fold) + ': ' + str(error),
                                {'failed_fold': fold, 'completed_folds': folds,
                                 'failed_projection': error.partial}) from error
        reading = [score_law(row['law'], float(train.sum()), held) for row in projections]
        scores.append(reading)
        folds.append({'outer_fold': fold, 'train_size': float(train.sum()),
                      'test_size': float(held.sum()), 'projections': projections,
                      'scores_bits_per_observation': reading,
                      'scoring_floor': 1.0 / float(train.sum())})
    scores = np.asarray(scores)
    gains = 1000 * np.diff(scores, axis=1)
    return {'folds': folds, 'mean_scores_bits_per_observation': scores.mean(axis=0).tolist(),
            'gains_by_fold_bits_per_1000': gains.tolist(),
            'mean_gains_bits_per_1000': (np.diff(scores, axis=1).mean(axis=0) * 1000).tolist(),
            'fold_scatter_bits_per_1000': gains.std(axis=0, ddof=1).tolist(),
            'positive_folds': (gains > 0).sum(axis=0).tolist(),
            'top_minus_pair_bits_per_1000': float(1000 * (scores[:, -1] - scores[:, 1]).mean()),
            'aggregation': 'Plain mean of five held-fold mean scores, matching the historical ladder.',
            'scope': 'Descriptive floored prediction increments, not unfloored layers, a support test or latent dimension.'}


def known_laws():
    states, pair = ol.state_features(4, 2)
    theta = np.array([-.8, -.2, .4, -.5, .6, -.3, .2, .1, -.4, .5])
    energy = pair @ theta
    weights = np.exp(energy - energy.max())
    yield 'pairwise', weights / weights.sum()
    spin = 2 * states - 1
    yield 'pure_triple', (1 + .6 * spin[:, :3].prod(axis=1)) / 16
    yield 'pure_four_way', (1 + .6 * spin.prod(axis=1)) / 16


def known_expected_scores(name, p, fold_sizes):
    """Closed population projections independent of numerical moment matching."""
    states, _ = ol.state_features(4, 1)
    if name == 'pairwise':
        marginals = p @ states
        independent = np.prod(np.where(states == 1, marginals, 1 - marginals), axis=1)
        laws = [independent, p, p, p]
    elif name == 'pure_triple':
        laws = [np.full(16, 1 / 16), np.full(16, 1 / 16), p, p]
    elif name == 'pure_four_way':
        laws = [np.full(16, 1 / 16)] * 3 + [p]
    else:
        raise ValueError('unregistered known law')
    return np.array([[score_law(q, sum(fold_sizes) - n, n * p) for q in laws]
                     for n in fold_sizes])
