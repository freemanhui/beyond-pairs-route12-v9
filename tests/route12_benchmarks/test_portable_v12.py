"""New portable-interface boundaries, distinct from the closed v11 known gate."""
import numpy as np
import pytest
from beyond_pairs.route12_benchmarks import portable as p

def test_public_surrogate_has_exact_pair_reference():
    q, alternative = p.surrogate()
    _, g, _, _ = p.c.designs(10)
    assert q.shape == alternative.shape == (1024,)
    assert np.all(q > 0) and np.all(alternative > 0)
    assert abs(alternative.sum()-1) < 1e-14
    assert np.max(np.abs((alternative-q)@g)) < 1e-14
    assert not np.array_equal(alternative,q)

@pytest.mark.parametrize('bad',[np.ones(7)/7, np.array([0.,.5,.5,0.]), np.full(8,np.nan), np.ones(8)])
def test_invalid_law_file_refused(tmp_path,bad):
    path=tmp_path/'bad.npz';np.savez(path,q=bad,p=bad)
    with pytest.raises(ValueError):p.load_laws(path)

def test_explicit_count_schema_rejects_noninteger():
    x=np.ones((5,8));x[0,0]=.5
    with pytest.raises(ValueError):p.validate_counts(x)

def test_existing_output_is_not_overwritten(tmp_path):
    out=tmp_path/'keep';out.mkdir();(out/'sentinel').write_text('retain')
    with pytest.raises(FileExistsError):p.create_output(out)
    assert (out/'sentinel').read_text()=='retain'

def test_counts_only_interface_requires_no_known_law():
    x=np.arange(40).reshape(5,8)+1
    assert np.array_equal(p.validate_counts(x),x)

def test_no_silent_case_expansion():
    with pytest.raises(ValueError):p.validate_roster(1557,20,999,500,200,12)
    assert p.validate_roster(1557,2,3,1,2,12)==12

def test_law_shape_mismatch_refused(tmp_path):
    path=tmp_path/'mismatch.npz';np.savez(path,q=np.ones(8)/8,p=np.ones(16)/16)
    with pytest.raises(ValueError):p.load_laws(path)

def test_release_scope_is_explicit():
    assert p.SCOPE['patient_data'] is False
    assert p.SCOPE['surrogate_is_MSK_reproduction'] is False

def test_saved_endpoint_summary_preserves_failure():
    result=p.endpoint_status(np.full((5,6),np.nan),2,1,'artificial failure')
    assert result['attempted']==2 and result['accepted']==1
    assert result['error']=='artificial failure'


def test_external_inputs_do_not_assert_patient_free_or_calibrated():
    s=p.run_scope(counts_only=True)
    assert s['patient_data']=='unknown' and s['calibration']=='not performed'
    assert p.run_scope(custom=True)['patient_data']=='unknown'
    assert p.run_scope()['patient_data'] is False

def test_canonical_law_binding_detects_changed_generator(tmp_path):
    q=np.full(8,1/8);r=q.copy();r[0]+=.01;r[1]-=.01
    f=tmp_path/'laws.npz';np.savez(f,q=q,p=r)
    a=p.law_binding(q,r,f);b=p.law_binding(q.astype('>f8'),r,f)
    assert a==b and a['q_sha256']!=a['p_sha256']
    assert a['input_file_sha256'] and a['dimension']==3
