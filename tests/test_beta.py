import pytest
from invtempbeta.beta import (fgammaL, W, newton_raphson, BetaCalc)
import numpy as np


def test_fgammaL():
    assert fgammaL(0.5)  == -1.0
    assert fgammaL(0.0)  ==  0.0
    assert fgammaL(-0.5) == -0.1715728752538099

def test_W():
    assert W(1, [0]) == 1
    assert np.array(W(0.0, np.array([1., -1.])) == [[1., 0.],[0., 1.]]).all()

def test_newton_raphson():
    # test small values
    assert newton_raphson(lambda x: 1./x, 0., h=1e-16) == -2e-16
    # test suppress warnings
    assert np.isnan(newton_raphson(lambda x: np.sqrt(x), -1.))

class TestBetaCalc():
    def test_init(self):
        with pytest.raises(ValueError) as err:
            B = BetaCalc(-1) 
        assert str(err.value) == "n must be > 0"
    def test_beta(self):
        B = BetaCalc(n=4)
        assert B.beta([-1.,0.,1.,-1.]) == 0.6114834025
        assert B.beta([-1.,1.,1.,-1.]) == 1.1365775542
        assert B.beta([1.,1.,1.,1.])  == -0.1823215568
        assert np.isnan(B.beta([0.,0.,0.,0.]))
        with pytest.raises(ValueError) as err:
            B.beta([1.,1.,1.,1.,1.])
        assert str(err.value) == "input length v != input n in BetaCalc instance"
        B = BetaCalc(n=4, x=0.5)
        assert B.beta([-1.,0.,0.,0.]) == -1.4601432212
        B = BetaCalc(n=4, x=0.)
        assert np.isnan(B.beta([-1.,0.,0.,0.]))