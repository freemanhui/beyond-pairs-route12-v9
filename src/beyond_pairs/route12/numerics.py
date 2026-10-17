"""The frozen Newton algorithm with explicit successful/failed attempt endpoints.

The original solver remains unchanged. This isolated variant adds input checks
and a retained endpoint on failure, required by the new study's audit contract.
"""
import numpy as np


class ProjectionFailure(RuntimeError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


def checked_maxent(p, features, start=None, *, tolerance=1e-10, iterations=500):
    p, features = np.asarray(p, float), np.asarray(features, float)
    if (p.ndim != 1 or features.ndim != 2 or features.shape[0] != len(p)
            or not features.shape[1] or not np.isfinite(p).all()
            or not np.isfinite(features).all() or (p < 0).any()
            or abs(p.sum()-1) > 1e-12 or not 0 < tolerance <= 1e-10
            or not isinstance(iterations, int) or not 1 <= iterations <= 500):
        raise ValueError('normalised law, finite features and registered solver limits required')
    theta = np.zeros(features.shape[1]) if start is None else np.array(start, float)
    if theta.shape != (features.shape[1],) or not np.isfinite(theta).all():
        raise ValueError('finite conformable start vector required')
    target = p@features

    def dual(vector):
        energy = features@vector
        top = energy.max()
        weight = np.exp(energy-top)
        normalizer = weight.sum()
        return float(top+np.log(normalizer)-vector@target), weight/normalizer

    value, law = dual(theta)
    completed = 0

    def endpoint(accepted):
        return {'theta': theta.copy(), 'law': law.copy(),
                'dual_entropy_bits': float(value/np.log(2)),
                'moment_error': float(np.max(np.abs(law@features-target))),
                'iterations': completed, 'accepted': accepted,
                'start_convention': 'pair_zero_v1' if start is None else 'supplied_vector_v1'}

    try:
        for _ in range(iterations):
            mean = law@features
            gradient = mean-target
            if np.abs(gradient).max() < tolerance:
                return endpoint(True)
            hessian = (features*law[:, None]).T@features-np.outer(mean, mean)
            eigenvalue, vector = np.linalg.eigh(hessian)
            keep = eigenvalue > 1e-13*eigenvalue[-1]
            if not keep.any():
                raise ArithmeticError('no positive Hessian support for nonzero gradient')
            step = -(vector[:, keep]/eigenvalue[keep])@(vector[:, keep].T@gradient)
            slope, scale = float(gradient@step), 1.
            trial, trial_law = dual(theta+step)
            while trial > value+1e-4*scale*slope and -scale*slope > 1e-13:
                scale /= 2
                trial, trial_law = dual(theta+scale*step)
            theta, value, law = theta+scale*step, trial, trial_law
            completed += 1
            if not np.isfinite(theta).all() or not np.isfinite(law).all() or not np.isfinite(value):
                raise ArithmeticError('nonfinite Newton endpoint')
    except (ArithmeticError, np.linalg.LinAlgError) as error:
        raise ProjectionFailure(str(error), endpoint(False)) from error
    raise ProjectionFailure('maximum-entropy fit did not converge within limit', endpoint(False))
