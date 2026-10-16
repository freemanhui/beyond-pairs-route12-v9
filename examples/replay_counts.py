"""Reconstruct method statistics from user-supplied permitted counts and fits."""
import argparse,json
import numpy as np
from beyond_pairs.route12_v9 import replay,verification

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_npz',help='tables, p, q0 and parameters; allow_pickle is disabled')
    args=parser.parse_args()
    with np.load(args.input_npz,allow_pickle=False) as data:
        tables=data['tables'];p=data['p'];q0=data['q0'];theta=data['parameters']
    result=replay.analyse(tables,p,q0,theta)
    review=verification.verify(tables,p,q0,result)
    if result['status']!='passed' or not review['passed']:raise RuntimeError('numerical reconstruction failed')
    print(json.dumps({'methods':list(replay.METHODS),'raw_statistics':result['statistics'].tolist(),
                     'raw_conditional_targets':result['conditional_targets'].tolist(),
                     'bits_per_1000_statistics':(result['statistics']*1000/(2*np.log(2))).tolist(),
                     'bits_per_1000_conditional_targets':(result['conditional_targets']*1000/(2*np.log(2))).tolist(),
                     'not_full_L3':True,'review':review},indent=2))
