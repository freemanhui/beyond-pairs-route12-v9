"""New default-demo/package boundaries; no scientific fit or random draw."""
import pytest
from beyond_pairs.route12_benchmarks import portable as p

def test_default_roster_has_fixed_278_case_bound():
    a=p.build_parser().parse_args(['demo','--output','unused'])
    assert (a.N,a.pools,a.calibration,a.null,a.alternative,a.maximum_cohorts)==(1557,2,99,20,20,278)
    assert p.validate_roster(a.N,a.pools,a.calibration,a.null,a.alternative,a.maximum_cohorts)==278

def test_default_calibration_can_reject():
    r=p.calibration_resolution(99)
    assert r['minimum_add_one_p']==.01 and r['rejection_attainable']

def test_small_explicit_calibration_is_honestly_labelled():
    r=p.calibration_resolution(3)
    assert r['minimum_add_one_p']==.25 and not r['rejection_attainable']

def test_unapproved_default_expansion_refused():
    with pytest.raises(ValueError):p.validate_roster(1557,2,100,20,20,278)

def test_demo_namespace_is_distinct_from_prior_demo_and_science():
    assert p.SEED_NAMESPACE==(13,0) and p.SEED_NAMESPACE not in [(12,0),(11,3)]

def test_demo_metadata_does_not_claim_power():
    assert 'do not qualify calibration or power' in p.calibration_resolution(99)['scope']
