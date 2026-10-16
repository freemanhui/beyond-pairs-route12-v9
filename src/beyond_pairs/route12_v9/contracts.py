"""Small provenance and input contracts for the registered v9 programme."""
import hashlib
import numpy as np


def counts(values):
    out = np.asarray(values)
    if (out.ndim != 1 or not np.isfinite(out).all() or (out < 0).any()
            or not np.equal(out, np.floor(out)).all()):
        raise ValueError('finite nonnegative integer counts required')
    return np.ascontiguousarray(out, dtype=np.int64)


def counts_digest(values):
    return hashlib.sha256(counts(values).astype('<i8').tobytes()).hexdigest()


def require_training_digest(values, expected):
    if counts_digest(values) != expected:
        raise ValueError('training-only input digest differs')


def require_procedure(actual, expected):
    if not isinstance(actual, str) or actual != expected:
        raise ValueError('procedure/source configuration differs')


def admit(*, free, floor, used, cap, reserve, elapsed, time_cap):
    if min(free, floor, used, cap, reserve, elapsed) < 0:
        raise ValueError('resource counters must be nonnegative')
    if free < floor+(cap-used) or used >= cap-reserve or elapsed >= time_cap:
        raise ValueError('finite storage/floor/time admission refused')
    return True
