from invtempbeta.beta import (fgammaL, W, newton_raphson)
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