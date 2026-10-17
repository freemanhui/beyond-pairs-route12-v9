"""Declared hidden-label convention and fully traced exact-law diagnostics."""
from hashlib import sha256
from itertools import permutations, product

import numpy as np

from beyond_pairs import order_ladder as ol
from beyond_pairs.information_layers import kl_bits
from .masked_projection import _law, _projection
from .numerics import ProjectionFailure


def align_hidden_labels(source_weights, target_weights):
    source, target = np.asarray(source_weights, float), np.asarray(target_weights, float)
    if (source.ndim != 2 or source.shape != target.shape or not 1 <= len(source) <= 5
            or source.shape[1] < 3 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError('matching finite one-to-five hidden coupling rows required')
    candidates = []
    for perm in permutations(range(len(source))):
        for flips in product([0, 1], repeat=len(source)):
            aligned = (1-2*np.array(flips))[:, None]*target[list(perm)]
            cost = float(np.sum((source-aligned)**2))
            candidates.append((cost, perm, flips))
    minimum = min(x[0] for x in candidates)
    tied = [x for x in candidates if x[0] <= minimum+1e-12]
    cost, perm, flips = min(tied, key=lambda x:(x[1], x[2]))
    hidden = ((np.arange(2**len(source))[:, None] >> np.arange(len(source))) & 1)
    target_hidden = np.empty_like(hidden)
    target_hidden[:, list(perm)] = hidden ^ np.array(flips)
    index = target_hidden @ (2**np.arange(len(source)))
    return {'target_unit_permutation': list(perm), 'target_unit_complements': list(flips),
            'target_row_index': index, 'squared_coupling_distance': cost,
            'minimum_squared_coupling_distance': minimum, 'ties_within_1e_12': len(tied),
            'candidate_count': len(candidates),
            'convention': 'minimum coupling distance, lexicographic permutation/complement tie break; no layer/score outcome used'}


def traced_layers(values):
    p, variables = _law(values)
    if (p <= 0).any() or abs(np.asarray(values).sum()-1) > 1e-12:
        raise ValueError('positive normalised exact-law target required')
    _, g = ol.state_features(variables, 2)
    _, full_features = ol.state_features(variables, 3)
    pair = _projection(p, g, None)
    start = np.zeros(full_features.shape[1]);start[:len(pair['theta'])] = pair['theta']
    try:
        full = _projection(p, full_features, None, start=start)
    except ProjectionFailure as error:
        raise ProjectionFailure(str(error), {**error.partial, 'accepted_pair': pair,
                                             'input_law': p}) from error
    l3 = pair['entropy_bits']-full['entropy_bits']
    total = pair['entropy_bits']-ol.entropy_bits(p)
    if max(abs(l3-kl_bits(full['law'], pair['law'])), abs(total-kl_bits(p, pair['law']))) > 1e-7:
        raise ProjectionFailure('traced exact-layer KL/entropy identity failed',
                                {'accepted_pair': pair, 'accepted_full': full, 'input_law': p})
    return {'input_law': p, 'laws': {'P2': pair['law'], 'P3': full['law']},
            'parameters': {'P2': pair['theta'], 'P3': full['theta']},
            'layers_bits': {'third_order': l3, 'total_above_pairs': total,
                            'above_order_three': total-l3},
            'diagnostics': {'P2': pair['diagnostics'], 'P3': full['diagnostics']}}


def traced_mixture(weights, component_laws, *, cache=None):
    weights, components = np.asarray(weights, float), np.asarray(component_laws, float)
    if (weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all()
            or (weights <= 0).any() or abs(weights.sum()-1) > 1e-12
            or components.ndim != 2 or len(components) != len(weights)):
        raise ValueError('positive normalised weights and matching component laws required')
    weights = weights/weights.sum()
    cache = {} if cache is None else cache
    def get(value):
        canonical, _ = _law(value)
        key = sha256(canonical.astype('<f8').tobytes()).hexdigest()
        if key not in cache:
            cache[key] = traced_layers(value)
        return key, cache[key]
    part_keys, parts = [], []
    for row in components:
        key, result = get(row);part_keys.append(key);parts.append(result)
    p = weights@np.array([x['input_law'] for x in parts])
    qz = np.array([x['laws']['P2'] for x in parts]);qbar = weights@qz
    pooled_key, pooled = get(p);mixture_key, mixture = get(qbar)
    if np.max(np.abs(pooled['laws']['P2']-mixture['laws']['P2'])) > 1e-8:
        raise ValueError('common pooled/P2-mixture projection failed')
    ip = float(sum(w*kl_bits(part['input_law'], pooled['input_law']) for w, part in zip(weights, parts)))
    iq = float(sum(w*kl_bits(q, mixture['input_law']) for w, q in zip(weights, qz)))
    total = pooled['layers_bits']['total_above_pairs']
    within = float(sum(w*x['layers_bits']['total_above_pairs'] for w,x in zip(weights,parts)))
    mixture_total = mixture['layers_bits']['total_above_pairs']
    l3 = pooled['layers_bits']['third_order']
    within_l3 = float(sum(w*x['layers_bits']['third_order'] for w,x in zip(weights,parts)))
    mixture_l3 = mixture['layers_bits']['third_order']
    error = abs(total-within-mixture_total-iq+ip)
    if error > 1e-7:
        raise ValueError('signed total-information identity failed')
    metrics = {'pooled_L3_bits': l3, 'within_L3_bits': within_l3,
        'pair_mixture_L3_bits': mixture_l3, 'L3_nonadditivity_bits': l3-within_l3-mixture_l3,
        'pooled_total_bits': total, 'within_total_bits': within,
        'pair_mixture_total_bits': mixture_total, 'MI_P_bits': ip, 'MI_Q_bits': iq,
        'MI_correction_bits': iq-ip, 'total_identity_error_bits': error,
        'pooled_above_three_bits': total-l3, 'within_above_three_bits': within-within_l3,
        'pair_mixture_above_three_bits': mixture_total-mixture_l3}
    return {'weights': weights, 'part_keys': part_keys, 'pooled_key': pooled_key,
            'pair_mixture_key': mixture_key, 'metrics': metrics, 'parts': parts,
            'pooled': pooled, 'pair_mixture': mixture, 'causal_attribution': False}
