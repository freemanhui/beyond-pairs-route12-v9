"""Laws of a few binary variables that get their dependence from hidden variables.

Four families, each computed exactly over all 2**n states of the visible variables.

  hidden units       the margin of a law of the variables and k hidden binary units in
                     which every unit is coupled to every variable and to nothing else
                     (a restricted Boltzmann machine). With one unit it is a mixture of
                     two independent laws.
  pairs and units    the same with a pairwise term between the visible variables.
  hidden axes        every variable is set with a probability that follows a logistic
                     curve along d hidden real axes, independently given the position;
                     positions are standard normal.
  pairs and an axis  a pairwise law whose single terms shift along one hidden axis.

Summed over the hidden variables, each family holds terms of every order between the
visible variables, with far fewer parameters than a law with free triple terms. Whether
such a law predicts held-out observations as well as the pairwise law or the law with
triple terms is what `scripts/hidden_variable_ladder.py` asks.

All fits are maximum likelihood by a quasi-Newton search from several starts, with every
parameter kept between -BOUND and BOUND. The likelihood of these families has several
maxima, and a search that starts at random often ends in one where a hidden variable is
tied to a single variable. The starts therefore come in three kinds, in turn: hidden
weights drawn at random; hidden weights along the main directions in which the
variables vary together; and hidden weights that move every variable the same way. The
best of the starts is kept.
"""
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from beyond_pairs import order_ladder as ol

BOUND = 12.0


def _softplus(values):
    return np.logaddexp(0.0, values)


def _logit(share, floor=1e-3):
    share = np.clip(share, floor, 1 - floor)
    return np.log(share/(1 - share))


def _hidden_start(states, shares, rng, number, hidden):
    """Starting weights of `hidden` hidden variables on the variables, one row each; the kind of start goes by its number."""
    variables = states.shape[1]
    kind = number % 3
    if kind == 0:
        return rng.normal(0.0, 1.0, size=(hidden, variables))
    if kind == 1:                                        # the main directions of joint variation, with a drawn length and sign
        mean = shares @ states
        _, vector = np.linalg.eigh((states*shares[:, None]).T @ states - np.outer(mean, mean))
        rows = vector[:, ::-1].T[:hidden]
        rows = np.r_[rows, rng.normal(0.0, 1.0, size=(hidden - len(rows), variables))] if len(rows) < hidden else rows
        return rows*np.sqrt(variables)*rng.uniform(0.5, 2.0, size=(hidden, 1))*rng.choice([-1.0, 1.0], size=(hidden, 1))
    out = rng.normal(0.0, 1.0, size=(hidden, variables))  # the first hidden variable moves every variable the same way
    out[0] = rng.choice([-1.0, 1.0])*np.abs(rng.normal(1.5, 0.5, size=variables))
    return out


class Units:
    """k hidden binary units, each coupled to every variable; with `pairs`, a pairwise term between the variables too."""

    def __init__(self, variables, units, pairs=False):
        self.variables, self.units, self.pairs = variables, units, pairs
        self.states, both = ol.state_features(variables, 2)
        self.linear = both if pairs else self.states
        self.size = self.linear.shape[1] + units*(1 + variables)

    def parts(self, theta):
        cut = self.linear.shape[1]
        return theta[:cut], theta[cut:cut + self.units], theta[cut + self.units:].reshape(self.units, self.variables)

    def _weights(self, theta):
        linear, bias, weight = self.parts(theta)
        drive = bias + self.states @ weight.T
        return self.linear @ linear + _softplus(drive).sum(axis=1), drive

    def log_law(self, theta):
        weights, _ = self._weights(theta)
        return weights - logsumexp(weights)

    def value(self, theta, shares):
        """Minus the mean log-likelihood of observations with the given shares of the states, and its gradient."""
        weights, drive = self._weights(theta)
        log_law = weights - logsumexp(weights)
        rest = shares - np.exp(log_law)
        active = expit(drive)
        gradient = np.r_[rest @ self.linear, rest @ active, ((active*rest[:, None]).T @ self.states).ravel()]
        return -float(shares @ log_law), -gradient

    def start(self, shares, rng, linear=None, number=0):
        first = np.zeros(self.linear.shape[1])
        first[:self.variables] = _logit(shares @ self.states)
        if linear is not None and number % 3 == 0:        # the given first terms go with the hidden weights drawn at random
            first[:len(linear)] = linear
        weight = _hidden_start(self.states, shares, rng, number, self.units)
        bias = rng.normal(0.0, 1.0, size=self.units) - weight @ (shares @ self.states)      # each unit on in about half of the observations
        return np.r_[first, bias, weight.ravel()]

    def posterior(self, theta):
        """For every state, the probability that each hidden unit is on."""
        return expit(self._weights(theta)[1])


def _normal_nodes(axes, per_axis):
    """Positions and weights that integrate a function of `axes` standard normal variables."""
    place, weight = np.polynomial.hermite_e.hermegauss(per_axis)
    weight = weight/weight.sum()
    if axes == 1:
        return place[:, None], weight
    if axes == 2:
        first, second = np.meshgrid(place, place, indexing='ij')
        return np.c_[first.ravel(), second.ravel()], np.outer(weight, weight).ravel()
    raise ValueError('one or two hidden axes')


class Axes:
    """Every variable follows a logistic curve along d hidden axes; the variables are independent given the position."""

    def __init__(self, variables, axes, per_axis=None):
        self.variables, self.axes = variables, axes
        self.states, _ = ol.state_features(variables, 1)
        self.place, weight = _normal_nodes(axes, per_axis or (41 if axes == 1 else 21))
        self.log_weight = np.log(weight)
        self.size = variables*(1 + axes)

    def parts(self, theta):
        return theta[:self.variables], theta[self.variables:].reshape(self.variables, self.axes)

    def _given(self, theta):
        level, slope = self.parts(theta)
        logit = level + self.place @ slope.T                                      # nodes x variables
        return self.states @ logit.T - _softplus(logit).sum(axis=1), logit         # states x nodes: log p(x | position)

    def log_law(self, theta):
        return logsumexp(self._given(theta)[0] + self.log_weight, axis=1)

    def value(self, theta, shares):
        given, logit = self._given(theta)
        joint = given + self.log_weight
        log_law = logsumexp(joint, axis=1)
        held = np.exp(joint - log_law[:, None])*shares[:, None]                  # share of the observations in a state and at a node
        rest = held.T @ self.states - held.sum(axis=0)[:, None]*expit(logit)     # nodes x variables
        return -float(shares @ log_law), -np.r_[rest.sum(axis=0), (rest.T @ self.place).ravel()]

    def start(self, shares, rng, linear=None, number=0):
        return np.r_[_logit(shares @ self.states), _hidden_start(self.states, shares, rng, number, self.axes).T.ravel()]

    def posterior(self, theta):
        """For every state, the mean position on each hidden axis."""
        joint = self._given(theta)[0] + self.log_weight
        return np.exp(joint - logsumexp(joint, axis=1)[:, None]) @ self.place


class PairsAxis:
    """A pairwise law whose single terms shift along one hidden axis with standard normal positions."""

    def __init__(self, variables, per_axis=41):
        self.variables = variables
        self.states, self.linear = ol.state_features(variables, 2)
        self.place, weight = _normal_nodes(1, per_axis)
        self.place = self.place[:, 0]
        self.log_weight = np.log(weight)
        self.size = self.linear.shape[1] + variables

    def parts(self, theta):
        cut = self.linear.shape[1]
        return theta[:cut], theta[cut:]

    def _given(self, theta):
        linear, shift = self.parts(theta)
        energy = (self.linear @ linear)[:, None] + np.outer(self.states @ shift, self.place)       # states x nodes
        return energy - logsumexp(energy, axis=0)

    def log_law(self, theta):
        return logsumexp(self._given(theta) + self.log_weight, axis=1)

    def value(self, theta, shares):
        given = self._given(theta)
        joint = given + self.log_weight
        log_law = logsumexp(joint, axis=1)
        held = np.exp(joint - log_law[:, None])*shares[:, None]
        apart = held - np.exp(given)*held.sum(axis=0)                             # observed less expected, state by node
        return -float(shares @ log_law), -np.r_[apart.sum(axis=1) @ self.linear, (apart @ self.place) @ self.states]

    def start(self, shares, rng, linear=None, number=0):
        first = np.zeros(self.linear.shape[1])
        first[:self.variables] = _logit(shares @ self.states)
        if linear is not None and number % 3 == 0:
            first[:len(linear)] = linear
        return np.r_[first, _hidden_start(self.states, shares, rng, number, 1)[0]]

    def posterior(self, theta):
        """For every state, the mean position on the hidden axis."""
        joint = self._given(theta) + self.log_weight
        return (np.exp(joint - logsumexp(joint, axis=1)[:, None]) @ self.place)[:, None]


class Exponential:
    """A law of the exponential family with the given features: the independent, pairwise and triple-term laws of the ladder."""

    def __init__(self, features):
        self.features = features
        self.size = features.shape[1]

    def log_law(self, theta):
        energy = self.features @ theta
        return energy - logsumexp(energy)

    def value(self, theta, shares):
        log_law = self.log_law(theta)
        return -float(shares @ log_law), -((shares - np.exp(log_law)) @ self.features)


def _search(family, theta, free, shares):
    """One search over the free parameters from the given ones: the value reached, and the parameters."""
    def value(part):
        full = theta.copy()
        full[free] = part
        found, gradient = family.value(full, shares)
        return found, gradient[free]

    found = minimize(value, theta[free], jac=True, method='L-BFGS-B', bounds=[(-BOUND, BOUND)]*int(free.sum()),
                     options={'maxiter': 3000, 'maxfun': 6000, 'ftol': 1e-13, 'gtol': 1e-8})
    out = theta.copy()
    out[free] = found.x
    return float(found.fun), out


def refit(family, theta, free, counts):
    """The parameters again, with only the free ones moved, at maximum likelihood on the counts; the others stay as given.

    When the biases of hidden units are free the likelihood has several maxima: a unit can be on in few of the observations
    or in most of them, and a search from the given parameters often ends in the wrong one. The search then begins from the
    given parameters, from parameters in which every unit is on in few or in most of the observations (every combination,
    at two distances), and from the fit with all units switched off, which a law with units can only improve on. The best
    end is kept.
    """
    shares = np.asarray(counts, dtype=np.float64)
    shares = shares/shares.sum()
    theta, free = np.array(theta, dtype=np.float64), np.asarray(free, dtype=bool)
    starts = [theta]
    if isinstance(family, Units) and free[family.linear.shape[1]:family.linear.shape[1] + family.units].all():
        biases = np.arange(family.linear.shape[1], family.linear.shape[1] + family.units)
        centre = -(family.parts(theta)[2] @ (shares @ family.states))       # where a unit is on in half of the observations
        for distance in (3.0, 8.0):
            for signs in product((-distance, distance), repeat=family.units):
                start = theta.copy()
                start[biases] = np.clip(centre + np.array(signs), -BOUND, BOUND)
                starts.append(start)
        off, rest = theta.copy(), free.copy()
        off[biases], rest[biases] = -BOUND, False
        starts.append(_search(family, off, rest, shares)[1])
    return min((_search(family, start, free, shares) for start in starts), key=lambda found: found[0])[1]


def fit(family, counts, starts, seed, linear=None):
    """The best of several maximum-likelihood fits: the law, its parameters, and its log-likelihood per observation in bits.

    `linear` gives the first terms of the starts whose hidden weights are drawn at random
    (the terms of a fitted pairwise law, for the families that hold one); the other
    starts begin from the single terms alone.
    """
    shares = np.asarray(counts, dtype=np.float64)
    shares = shares/shares.sum()
    best = None
    for number in range(starts):
        rng = np.random.default_rng([seed, number])
        found = minimize(family.value, family.start(shares, rng, linear, number), args=(shares,), jac=True, method='L-BFGS-B',
                         bounds=[(-BOUND, BOUND)]*family.size, options={'maxiter': 3000, 'maxfun': 6000})
        if best is None or found.fun < best.fun:
            best = found
    return np.exp(family.log_law(best.x)), best.x, -float(best.fun)/np.log(2)


def held_out_bits(law, train_total, test):
    """Bits per held-out observation under a fitted law, mixed with the uniform law as in `order_ladder.held_out_bits`."""
    test = np.asarray(test, dtype=np.float64)
    floor = 1.0/train_total
    return float(test @ np.log2((1 - floor)*law + floor/len(law))/test.sum())
