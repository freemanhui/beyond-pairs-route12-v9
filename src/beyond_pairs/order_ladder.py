"""Information beyond pairs, order by order, for a few binary variables.

For a law P of n binary variables, P_k is the maximum-entropy law that shares every
margin of up to k variables with P: P_1 is the independent law, P_2 the pairwise law
and P_n is P itself. Entropy falls along this ladder. The fall from P_(k-1) to P_k is
the information that first appears at order k, and the fall from P_k to P is what the
margins of k variables leave out (Amari 2001; Schneidman, Still, Berry and Bialek
2003). All 2**n states are enumerated, so every number is exact up to the tolerance
of the solver.

A law with empty states lies on the boundary of these models, where some parameters
are infinite. Newton's method on the dual still converges there: each step divides
the mass on a state that must be empty by about e.
"""
from itertools import combinations

import numpy as np


def state_features(variables, order):
    """All x in {0, 1}^variables, and the products of up to `order` of their coordinates."""
    states = ((np.arange(2**variables)[:, None] >> np.arange(variables)) & 1).astype(np.float64)
    columns = [states[:, list(subset)].prod(axis=1) for size in range(1, order + 1)
               for subset in combinations(range(variables), size)]
    return states, np.stack(columns, axis=1)


def entropy_bits(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p*np.log2(p)).sum())


def maxent(p, features, start=None, tolerance=1e-10, iterations=500):
    """The maximum-entropy law with the feature means of p: the law, its parameters, its entropy in bits."""
    target = p @ features
    theta = np.zeros(features.shape[1]) if start is None else np.array(start, dtype=np.float64)

    def dual(theta):
        energy = features @ theta
        top = energy.max()
        weight = np.exp(energy - top)
        total = weight.sum()
        return top + np.log(total) - theta @ target, weight/total

    value, law = dual(theta)
    for _ in range(iterations):
        mean = law @ features
        gradient = mean - target
        if np.abs(gradient).max() < tolerance:
            return law, theta, float(value/np.log(2))
        hessian = (features*law[:, None]).T @ features - np.outer(mean, mean)
        eigenvalue, vector = np.linalg.eigh(hessian)
        keep = eigenvalue > 1e-13*eigenvalue[-1]       # the rest is rounding, or parameters that change nothing
        step = -(vector[:, keep]/eigenvalue[keep]) @ (vector[:, keep].T @ gradient)
        slope, scale = gradient @ step, 1.0
        trial, trial_law = dual(theta + step)
        while trial > value + 1e-4*scale*slope and -scale*slope > 1e-13:   # below that, rounding hides the decrease
            scale /= 2
            trial, trial_law = dual(theta + scale*step)
        theta, value, law = theta + scale*step, trial, trial_law
    raise RuntimeError('the maximum-entropy fit did not converge')


def ladder(p, variables, top):
    """Entropy in bits of P_1 ... P_top, and the laws themselves."""
    entropies, laws, theta = [], [], np.zeros(0)
    for order in range(1, top + 1):
        _, features = state_features(variables, order)
        start = np.zeros(features.shape[1])
        start[:len(theta)] = theta
        law, theta, bits = maxent(p, features, start)
        entropies.append(bits)
        laws.append(law)
    return np.array(entropies), laws


def fixed_by_margins(p, variables, order):
    """True if every margin of order + 1 variables lacks an even and an odd pattern.

    Two laws with the same margins of `order` variables differ, on any order + 1
    variables, by a multiple of the parity sign. An empty even pattern and an empty
    odd pattern force that multiple to zero. A pattern that is empty on some
    variables stays empty, in both parities, when one more variable is added, so the
    same holds for every larger set. p is then the only law with its margins of
    `order` variables, and it has no information above that order. The condition is
    sufficient, not necessary.
    """
    present = ((np.flatnonzero(np.asarray(p) > 0)[:, None] >> np.arange(variables)) & 1)
    half = 2**order
    for subset in combinations(range(variables), order + 1):
        patterns = np.unique(present[:, list(subset)], axis=0)
        odd = int((patterns.sum(axis=1) % 2).sum())
        if odd == half or len(patterns) - odd == half:
            return False
    return True


def held_out_bits(train, test, variables, top):
    """Bits per held-out observation under P_1 ... P_top fitted to the training counts.

    Each fitted law is mixed with the uniform law at the weight of one training
    observation, so that a state the fit leaves empty costs a finite amount.
    """
    train, test = np.asarray(train, dtype=np.float64), np.asarray(test, dtype=np.float64)
    _, laws = ladder(train/train.sum(), variables, top)
    floor = 1.0/train.sum()
    return np.array([test @ np.log2((1 - floor)*law + floor/len(law)) for law in laws])/test.sum()
