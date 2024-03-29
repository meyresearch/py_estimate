################################################################################
#
#   test_dtram.py - testing the py_estimate dTRAM class
#
#   author: Christoph Wehmeyer <christoph.wehmeyer@fu-berlin.de>
#
################################################################################

from nose.tools import assert_raises, assert_true
from py_estimate.estimator import DTRAM
from pytram import ExpressionError, NotConvergedWarning
import numpy as np

def test_expression_error_None():
    """test DTRAM throws ExpressionError with None"""
    assert_raises(
        ExpressionError,
        DTRAM,
        np.ones(shape=(2, 3, 3), dtype=np.intc), None)

def test_expression_error_int():
    """test DTRAM throws ExpressionError with number"""
    assert_raises(
        ExpressionError,
        DTRAM,
        np.ones(shape=(2, 3, 3), dtype=np.intc), 5)

def test_expression_error_list():
    """test DTRAM throws ExpressionError with list"""
    assert_raises(
        ExpressionError,
        DTRAM,
        np.ones(shape=(2, 3, 3), dtype=np.intc), [1, 2])

def test_expression_error_dim():
    """test DTRAM throws ExpressionError with wrong dimension"""
    assert_raises(
        ExpressionError,
        DTRAM,
        np.ones(shape=(2, 3, 3), dtype=np.intc),
        np.ones(shape=(2, 2, 2), dtype=np.float64))

def test_expression_error_markov():
    """test DTRAM throws ExpressionError with wrong Markov state count"""
    assert_raises(
        ExpressionError,
        DTRAM,
        np.ones(shape=(2, 3, 3), dtype=np.intc),
        np.ones(shape=(2, 2), dtype=np.float64))

def test_expression_error_therm():
    """test DTRAM throws ExpressionError with wrong thermodynamic state count"""
    assert_raises(
        ExpressionError,
        DTRAM,
        np.ones(shape=(2, 3, 3), dtype=np.intc),
        np.ones(shape=(1, 3), dtype=np.float64))

def test_expression_error_int16():
    """test DTRAM throws ExpressionError with wrong dtype (int16)"""
    assert_raises(
        ExpressionError,
        DTRAM,
        np.ones(shape=(2, 3, 3), dtype=np.intc),
        np.ones(shape=(2, 3), dtype=np.int16))

def test_expression_error_float32():
    """test DTRAM throws ExpressionError with wrong dtype (float32)"""
    assert_raises(
        ExpressionError,
        DTRAM,
        np.ones(shape=(2, 3, 3), dtype=np.intc),
        np.ones(shape=(2, 3), dtype=np.float32))

def test_toy_model():
    """test DTRAM with toy model"""
    C_K_ij = np.array([
        [[2358, 29, 0], [29, 0, 32], [0, 32, 197518]],
        [[16818, 16763, 0], [16763, 0, 16510], [0, 16510, 16635]]], dtype=np.intc)
    b_K_i = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 8.0]], dtype=np.float64)
    dtram = DTRAM(C_K_ij, b_K_i)
    assert_raises(NotConvergedWarning, dtram.sc_iteration, maxiter=1, ftol=1.0E-80, verbose=False)
    dtram.sc_iteration(maxiter=200000, ftol=1.0E-15, verbose=True)
    pi = np.array([1.82026887e-02, 3.30458960e-04, 9.81466852e-01], dtype=np.float64)
    print(pi)
    print(dtram.pi_i)
    assert_true(np.max(np.abs(dtram.pi_i - pi)) < 1.0E-8)

















