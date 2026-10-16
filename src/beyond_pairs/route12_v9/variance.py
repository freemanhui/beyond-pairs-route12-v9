"""Fixed oracle degenerate-kernel variance; no learned-map covariance theorem."""
import numpy as np


def oracle_split_variance(zeta2,fold_sizes):
    sizes=np.asarray(fold_sizes)
    if (sizes.ndim!=1 or not sizes.size or sizes.dtype.kind not in 'iu'
            or (sizes<2).any() or not np.isfinite(zeta2) or zeta2<0):
        raise ValueError('integer folds>=2 and finite nonnegative kernel moment required')
    n=int(sizes.sum())
    return {'pooled':float(2*zeta2/(n*(n-1))),
            'split':float(2*zeta2*np.sum(sizes/(sizes-1))/n**2),
            'structural_ratio':float((n-1)*np.sum(sizes/(sizes-1))/n),
            'scope':'Fixed same degenerate oracle kernel on disjoint iid folds only.'}
