"""Deterministic artificial arithmetic example: zero random draws or fits."""
import json
import numpy as np
from beyond_pairs.route12_v9 import numerics, replay, verification, information

def demonstrate():
    states,_=numerics.features(3)
    parity=np.prod(2*states-1,axis=1)
    p=(1+.4*parity)/8
    q=np.full(8,1/8)
    held=np.where(parity>0,7,3).astype(np.int64)
    tables=np.tile(held,(5,1))
    theta=np.zeros((5,6))  # Exact pairwise law: no optimizer is called.
    result=replay.analyse(tables,p,q,theta)
    review=verification.verify(tables,p,q,result)
    expected_raw=216/1560
    assert result['status']=='passed' and review['passed']
    assert np.max(np.abs(result['statistics']-expected_raw))<1e-10
    assert np.max(np.abs(result['conditional_targets']-.16))<1e-10
    score=information.oracle_reference_gain(p,q,np.log(p),reference_kind='known_exact_artificial')['GQ_bits']
    exact_information=1+.7*np.log2(.7)+.3*np.log2(.3)
    assert abs(score-exact_information)<1e-12
    return {'software_example':'deterministic_three_spin_parity','states':8,'folds':5,'observations_per_fold':40,
            'raw_U':result['statistics'].tolist(),'raw_conditional_quadratic_target':result['conditional_targets'].tolist(),
            'raw_U_exact_fraction':'216/1560','oracle_GQ_bits':score,
            'quadratic_is_full_entropy_information':False,'independent_arithmetic_pass':review['passed'],
            'new_optimizer_calls':0,'random_draws':0,'patient_or_fitted_law_assets':False}
if __name__=='__main__':print(json.dumps(demonstrate(),indent=2))
