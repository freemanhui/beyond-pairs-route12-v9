"""24 new artificial boundary/identity cases; no frozen study law is read."""
import hashlib
from itertools import combinations
import numpy as np
import pytest

from beyond_pairs.route12_benchmarks import core as c
from beyond_pairs.route12_v9 import replay


def test_01_trace_diagonal():
    assert c.sandwich_matrices(np.diag([2.,3.]),np.diag([4.,9.]))['trace']==pytest.approx(5.)


def test_02_equal_law_dimension():
    _,g,_,_=c.designs(3);q,_=c.law(np.arange(6)/17,g)
    assert c.sandwich(q,q,g)['trace']==pytest.approx(6.)


def test_03_singular_refused():
    with pytest.raises(ValueError):c.sandwich_matrices(np.diag([1.,0.]),np.eye(2))


def test_04_reparameterization():
    _,g,_,_=c.designs(3);p=np.arange(1,9,dtype=float)/36;q=np.full(8,.125)
    transform=np.diag(np.arange(1,7));transform[0,2]=.4
    assert c.sandwich(p,q,g)['trace']==pytest.approx(c.sandwich(p,q,g@transform)['trace'],rel=1e-10)


def test_05_covariance_explicit():
    p=np.array([.2,.3,.5]);f=np.array([[2.,-1.],[0.,3.],[1.,1.]])
    mean=sum(p[i]*f[i] for i in range(3))
    expected=sum(p[i]*np.outer(f[i]-mean,f[i]-mean) for i in range(3))
    assert np.allclose(c.covariance(p,f),expected,rtol=0,atol=1e-14)


def test_06_pair_moments():
    _,g,_,_=c.designs(3);q,_=c.law(np.array([-.3,.1,.2,-.2,.1,.3]),g)
    result=c._projection(q,g,None)
    assert np.max(np.abs((q-result['law'])@g))<1e-9


def test_07_parity_lrt():
    states,_,_,_=c.designs(3);counts=np.where(states.sum(axis=1)%2==0,60,40)
    *_,value,raw,attempted,accepted,error=c.pooled_lrt(counts)
    expected=2*400*(.6*np.log(1.2)+.4*np.log(.8))
    assert error is None and attempted==accepted==2
    assert value==pytest.approx(expected,abs=1e-7)


def test_08_identical_lrt_zero():
    _,g,_,full=c.designs(3);theta=np.arange(6)/31
    value,raw=c.lrt_from_parameters(np.arange(1,9),theta,np.r_[theta,0.])
    assert abs(value)<1e-12 and abs(raw)<1e-12


def test_09_boundary_endpoint():
    counts=np.array([20,0,10,0,20,0,10,0]);result=c.pooled_lrt(counts)
    assert result[-1] is None
    assert np.all(result[2][:,0]<=1e-9) and np.all(result[2][:,1]<=1e-7)


def test_10_negative_lrt_refused():
    with pytest.raises(ValueError):c.lrt_from_parameters(np.array([100,0,0,0,0,0,0,0]),np.zeros(6),np.ones(7)*3)


def test_11_rank_999():
    assert c.rank_decisions(np.arange(999),[949,949.5,950]).tolist()==[False,True,True]


def test_12_rank_ties():
    assert not c.rank_decisions(np.ones(999),[1.])[0]


def test_13_rank_400():
    assert c.rank_decisions(np.arange(400),[380,380.5]).tolist()==[False,True]


def test_14_empty_mask_zero():
    assert c.u_value(np.ones(8),np.empty((8,0)),np.empty((0,0)))==0


def test_15_U_explicit_pairs():
    counts=np.array([2,1,1]);r=np.array([[1.,0.],[.2,2.],[-1.,.5]]);f=np.array([[1.,.1],[.2,.8]])
    z=np.repeat(r@f,counts,axis=0)
    expected=sum(z[i]@z[j] for i in range(4) for j in range(4) if i!=j)/12
    assert c.u_value(counts,r,f,check=True)==pytest.approx(expected)


def test_16_D_existing_definition():
    _,g,_,_=c.designs(3);q,_=c.law(np.arange(6)/30-.1,g)
    rng=np.random.default_rng(20261116)
    tables=np.array([rng.multinomial(int(n),q) for n in c.fold_sizes(400)])
    params,_,_,accepted,error=c.fit_training(tables)
    assert accepted==5 and error is None
    ours=c.fully_learned(tables,params,check=True)[0]
    old=replay.analyse(tables,q,q,params)['statistics'][3]
    assert ours==pytest.approx(old,abs=1e-10)


def test_17_fold_total():
    assert c.fold_sizes(1557).sum()==1557
    assert c.fold_sizes(6274).tolist()==[1250,1195,1298,1268,1263]


def test_18_namespace_independence():
    values=[np.random.default_rng(np.random.SeedSequence([20261016,11,3,1557,pool,0,purpose,0])).integers(2**63) for pool,purpose in [(0,0),(0,1),(1,0)]]
    assert len(set(values))==3


def test_19_paired_accounting():
    out=c.paired([1,1,0,0],[1,0,1,0]);assert [out[k] for k in ['both','A_only','B_only','neither']]==[1,1,1,1]
    assert out['mean_difference']==0


def test_20_pool_equal_weights():
    out=c.pool_summary([1,90],[10,100]);assert out['equal_pool_mean']==pytest.approx(.5)


def test_21_pool_uncertainty_clusters():
    out=c.pool_summary([0,100],[100,100]);assert out['pools']==2
    assert out['MCSE']==pytest.approx(.5)


def test_22_hash_refusal(tmp_path):
    p=tmp_path/'record';p.write_bytes(b'v11-before');expected=hashlib.sha256(p.read_bytes()).hexdigest();p.write_bytes(b'v11-after')
    assert hashlib.sha256(p.read_bytes()).hexdigest()!=expected


def test_23_partial_fit_retention(monkeypatch):
    from beyond_pairs.route12.numerics import ProjectionFailure
    def fail(*args,**kwargs):raise ProjectionFailure('new artificial failure',{'theta':np.arange(6,dtype=float)})
    monkeypatch.setattr(c,'_projection',fail)
    result=c.fit_training(np.ones((5,8),dtype=int))
    assert result[2:4]==(1,0) and np.array_equal(result[0][0],np.arange(6)) and np.isnan(result[0][1:]).all()


def test_24_labels():
    genes=['a','b','c','d'];assert c.triple_labels(genes)==['a/b/c','a/b/d','a/c/d','b/c/d']
    _,_,triples,_=c.designs(4)
    assert triples.shape[1]==len(list(combinations(genes,3)))
