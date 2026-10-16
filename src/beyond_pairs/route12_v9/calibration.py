"""Fixed-rank known-null decisions, without fitted-composite-null promotion."""
import numpy as np
from scipy.stats import beta


def known_null_mc_pvalue(observed, calibration, *, expected_B):
    values = np.asarray(calibration, dtype=np.float64)
    if (not isinstance(expected_B,int) or expected_B<1 or values.shape!=(expected_B,)
            or not np.isfinite(values).all() or not np.isfinite(observed)):
        raise ValueError('complete finite registered calibration roster required')
    return float((1+np.count_nonzero(values>=observed))/(expected_B+1))


def fixed_binomial(hits, complete, planned):
    if any(not isinstance(x,(int,np.integer)) for x in [hits,complete,planned]) or not 0<=hits<=complete<=planned or planned<1:
        raise ValueError('valid fixed planned integer accounting required')
    out={'rejections':int(hits),'complete':int(complete),'planned':int(planned),
         'qualified':complete==planned,'rate':None,'interval_95':None}
    if complete==planned:
        out['rate']=hits/planned
        out['interval_95']=[float(beta.ppf(.025,hits,planned-hits+1)) if hits else 0.,
            float(beta.ppf(.975,hits+1,planned-hits)) if hits<planned else 1.]
    return out


def calibration_pool_tail(B, order, *, continuous, iid):
    if not continuous or not iid or not 1<=order<=B:
        raise ValueError('continuous iid null and valid order required')
    return B+1-order,order
