"""Quadratic third-order residual measurements; no entropy estimator claim.

The U identity is unbiased only for a fixed residual map and weight. A common
fitted pairwise law makes it a different, empirically evaluated procedure.
"""
from itertools import combinations

import numpy as np
from scipy.special import logsumexp

from . import order_ladder as ol


class MeasurementFailure(RuntimeError):
    pass


def features(variables):
    states, all_features = ol.state_features(variables, 3)
    nuisance = variables + variables * (variables - 1) // 2
    return states, all_features[:, :nuisance], all_features[:, nuisance:]


def inverse_psd(matrix, relative=1e-10):
    matrix = (matrix + matrix.T) / 2
    eigenvalues, vectors = np.linalg.eigh(matrix)
    largest = float(eigenvalues[-1])
    if largest <= 0 or eigenvalues[0] < -max(1e-12, largest * 1e-8):
        raise MeasurementFailure('covariance has no positive support or is indefinite')
    keep = eigenvalues > relative * largest
    inverse = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
    return inverse, int(keep.sum()), eigenvalues


def residual_map(q, g, t, n, minimum_expected=5):
    q = np.asarray(q, float)
    mg, mt = q @ g, q @ t
    cg, ct = g - mg, t - mt
    cgg = (cg * q[:, None]).T @ cg
    ctg = (ct * q[:, None]).T @ cg
    ginv, grank, _ = inverse_psd(cgg)
    b = ctg @ ginv
    residual = ct - cg @ b.T
    v = (residual * q[:, None]).T @ residual
    mask = (n * mt >= minimum_expected)
    if not mask.any():
        raise MeasurementFailure('no triple reaches the fixed expected-count cutoff')
    selected = np.flatnonzero(mask)
    covariance = v[np.ix_(selected, selected)]
    w, rank, eig = inverse_psd(covariance)
    return {'residual': residual[:, selected], 'weight': w,
            'variance': np.diag(covariance), 'mask': mask, 'rank': rank,
            'nuisance_rank': grank, 'minimum_eigenvalue': float(eig[0]),
            'maximum_eigenvalue': float(eig[-1]), 'mean_triples': mt}


def quadratic_readings(counts, mapping):
    counts = np.asarray(counts, float)
    n = float(counts.sum())
    if n <= 1 or (counts < 0).any() or not np.isfinite(counts).all():
        raise ValueError('at least two finite nonnegative observations required')
    r, w = mapping['residual'], mapping['weight']
    mean = counts @ r / n
    score = float(n * mean @ w @ mean)
    individual = np.einsum('ij,jk,ik->i', r, w, r)
    self_sum = float(counts @ individual)
    u = float((n * score - self_sum) / (n - 1))
    diagonal = n * mean**2 / mapping['variance']
    return np.array([u, score, float(diagonal.sum()), float(diagonal.max())]), self_sum


def pair_fit(counts, g):
    p = np.asarray(counts, float) / np.sum(counts)
    q, theta, dual = ol.maxent(p, g, tolerance=1e-10)
    diagnostics = np.array([float(np.max(np.abs((p - q) @ g))),
                            abs(float(dual) - ol.entropy_bits(q)),
                            abs(float(q.sum()) - 1)])
    if (not np.isfinite(q).all() or not np.isfinite(theta).all()
            or diagnostics[0] > 1e-9 or diagnostics[1] > 1e-7
            or diagnostics[2] > 1e-12):
        raise MeasurementFailure('pairwise projection diagnostics failed')
    return q, theta, diagnostics


def fixed_pair_tilt(q, g, t, direction, amplitude):
    """Change triple coefficients while holding the specified pair means fixed."""
    target = q @ g
    base = np.log(q)
    offset = amplitude * (t @ direction)
    theta = np.zeros(g.shape[1])
    def dual(x):
        energy = base + offset + g @ x
        z = logsumexp(energy)
        return float(z - x @ target), np.exp(energy - z)
    value, p = dual(theta)
    for _ in range(200):
        mean = p @ g
        gradient = mean - target
        if np.max(np.abs(gradient)) < 1e-10:
            moment_error = float(np.max(np.abs((p - q) @ g)))
            entropy_dual = (-value + float(p @ (base + offset))) / np.log(2)
            dual_error = abs(entropy_dual + ol.entropy_bits(p))
            if moment_error > 1e-9 or dual_error > 1e-7:
                raise MeasurementFailure('plant moment/entropy gate failed')
            kl = float(p @ (np.log(p) - np.log(q)) / np.log(2))
            return p, theta, kl, np.array([moment_error, dual_error])
        centred = g - mean
        hessian = (centred * p[:, None]).T @ centred
        inv, _, _ = inverse_psd(hessian, 1e-13)
        step = -inv @ gradient
        slope, scale = float(gradient @ step), 1.0
        trial, trial_p = dual(theta + step)
        while trial > value + 1e-4 * scale * slope and -scale * slope > 1e-13:
            scale /= 2
            trial, trial_p = dual(theta + scale * step)
        theta, value, p = theta + scale * step, trial, trial_p
    raise MeasurementFailure('fixed-pair plant did not converge')


def fold_sizes(n):
    original = np.array([1250, 1195, 1298, 1268, 1263], dtype=np.int64)
    sizes = n * original // original.sum()
    remainder = n * original % original.sum()
    order = sorted(range(5), key=lambda i: (-int(remainder[i]), i))
    for i in order[:int(n - sizes.sum())]:
        sizes[i] += 1
    return sizes


def triple_names(variables):
    return list(combinations(range(variables), 3))
