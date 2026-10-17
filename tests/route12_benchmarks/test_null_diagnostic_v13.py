"""New arithmetic examples for the posthoc report diagnostic; no fits/draws."""
import math
import pytest
from beyond_pairs.route12_benchmarks.null_diagnostic import compare, sign_tail

@pytest.mark.parametrize('a,b,u',[([0,1],[2,3],4),([0,1],[1,2],3.5),([0,1],[0,1],2),([2,3],[0,1],0)])
def test_pair_count(a,b,u): assert compare(a,b)['MW_eval_greater_U']==u

@pytest.mark.parametrize('a,b,d',[([0,1],[2,3],1),([0,1],[0,1],0),([0,1],[1,2],.5)])
def test_ECDF(a,b,d): assert compare(a,b)['KS_D']==d

def test_mean_shift():
    r=compare([0,2],[1,3]);assert r['mean_shift']==1 and abs(r['independent_cohort_SE']-math.sqrt(2))<1e-14

def test_constant_SE():
    r=compare([1,1],[1,1]);assert r['mean_shift']==r['independent_cohort_SE']==0

@pytest.mark.parametrize('positive,expected',[(20,1/2**20),(10,sum(math.comb(20,k) for k in range(10,21))/2**20)])
def test_sign_reference(positive,expected): assert sign_tail(positive,20)==expected

def test_positive_unit_scaling():
    x=compare([0,2],[1,4]);y=compare([0,2],[1,4],10)
    assert y['mean_shift']==10*x['mean_shift'] and y['independent_cohort_SE']==10*x['independent_cohort_SE']
    assert y['MW_eval_greater_p']==x['MW_eval_greater_p'] and y['KS_D']==x['KS_D']
