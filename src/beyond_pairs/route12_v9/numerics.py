"""Canonical finite-state arrays and diagnosed symmetric eigenspace kernels."""
from itertools import combinations
import numpy as np


def canonical(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError('finite float64 array required')
    return array


def features(variables):
    if not isinstance(variables, int) or not 3 <= variables <= 12:
        raise ValueError('registered binary feature size required')
    states = (np.arange(2**variables)[:, None] >> np.arange(variables)) & 1
    columns = [np.prod(states[:, c], axis=1) for degree in (1,2,3)
               for c in combinations(range(variables), degree)]
    return states, canonical(np.column_stack(columns))


def law_from_theta(theta, design):
    energy = canonical(design) @ canonical(theta)
    top = energy.max()
    logz = top+np.log(np.exp(energy-top).sum())
    return np.exp(energy-logz), energy-logz


def covariance(q, design):
    q, design = canonical(q), canonical(design)
    centered = canonical(design-q@design)
    return canonical((centered*q[:, None]).T@centered)


def psd_inverse(matrix, relative=1e-10):
    raw = canonical(matrix)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or relative != 1e-10:
        raise ValueError('square covariance and fixed inverse cutoff required')
    if not raw.size:
        return {'weight': raw.copy(), 'factor': raw.copy(), 'rank': 0,
                'eigenvalues': np.empty(0), 'diagnostics': {'ridge':0, 'retained_condition':0., 'raw_asymmetry':0., 'product_asymmetry':0.}}
    asym = float(np.max(np.abs(raw-raw.T)))
    scale = float(np.max(np.abs(raw)))
    if asym > max(1e-12, scale*1e-12):
        raise ValueError('material covariance asymmetry refused')
    symmetric = canonical((raw+raw.T)/2)
    values, vectors = np.linalg.eigh(symmetric)
    if values[-1] <= 0 or values[0] < -max(1e-12,values[-1]*1e-8):
        raise ValueError('no positive support or materially indefinite covariance')
    keep = values > relative*values[-1]
    basis, spectrum = vectors[:,keep], values[keep]
    factor = canonical(basis/np.sqrt(spectrum))
    product = canonical(factor@factor.T)
    division_product = canonical((basis/spectrum)@basis.T)
    product_asym = float(np.max(np.abs(product-product.T)))
    weight = canonical((product+product.T)/2)
    return {'weight':weight, 'factor':factor, 'rank':int(keep.sum()),
            'eigenvalues':values, 'basis':canonical(basis), 'retained_eigenvalues':spectrum,
            'diagnostics':{'raw_asymmetry':asym, 'raw_scale':scale,
                'product_asymmetry':product_asym, 'minimum_eigenvalue':float(values[0]),
                'division_product_asymmetry':float(np.max(np.abs(division_product-division_product.T))),
                'maximum_eigenvalue':float(values[-1]),
                'minimum_retained_eigenvalue':float(spectrum[0]),
                'retained_condition':float(spectrum[-1]/spectrum[0]), 'ridge':0}}


def residual(q, g, t):
    q, g, t = canonical(q), canonical(g), canonical(t)
    gc, tc = canonical(g-q@g), canonical(t-q@t)
    inverse = psd_inverse(canonical((gc*q[:,None]).T@gc))
    cross = canonical((tc*q[:,None]).T@gc)
    coefficient = canonical(cross@inverse['weight'])
    return canonical(tc-gc@coefficient.T), inverse['diagnostics']


def kernel_values(counts, p, r, inverse):
    c, p, r = canonical(counts), canonical(p), canonical(r)
    n = float(c.sum())
    if n <= 1:
        raise ValueError('two held observations required')
    if not r.shape[1]:
        return 0., 0., {'target_contraction_error':0., 'statistic_contraction_error':0.}
    z = canonical(r@inverse['factor'])
    total = c@z
    diagonal = np.sum(z*z, axis=1)
    reading = float((total@total-c@diagonal)/(n*(n-1)))
    mean = p@r
    projected = mean@inverse['factor']
    target = float(projected@projected)
    ld_mean = mean.astype(np.longdouble)
    direct_target = float(ld_mean@inverse['weight'].astype(np.longdouble)@ld_mean)
    # The second U contraction uses W explicitly and an independent self-term contraction.
    summed = c@r
    explicit_diagonal = np.sum((r@inverse['weight'])*r, axis=1)
    direct_stat = float((summed@inverse['weight']@summed-c@explicit_diagonal)/(n*(n-1)))
    return reading, target, {'target_contraction_error':abs(target-direct_target),
        'statistic_contraction_error':abs(reading-direct_stat)}
