import numpy as np  
from beta import BetaCalc


def auto_cor(v: list) -> np.ndarray:
    """ cyclic autocorrelation of the vector v """
    if v:
        return np.fft.ifft(np.fft.fft(v[::-1])*np.fft.fft(v))[::-1].real / len(v)
    else:
        return np.array([])


def random_chain_generation(n, values, limit=1000, mult_attempt=10):
    """ 
    Random generation of chains consisting the values of the monomer repulsive energy
    from given set values=[e_1, e_2, ..,e_n] corresponding to different beta(values).
    limit is size of output set of chains and corresponding beta
    limit * mult_attepmt is number of attemptions to generate these sets
    """
    B = BetaCalc(n=n)
    attempt_limit = mult_attempt * limit
    err_counter = 0
    accept_counter = 0
    X, y = [], []
    possibilities = [1.0 / len(values)] * len(values)
    while True:
        v = np.random.choice(values, n, possibilities)
        b = B.beta(v)
        if np.isfinite(b) and b not in y:
            X.append(list(v))
            y.append(b)
            accept_counter += 1
            if accept_counter >= limit:
                return np.array(X), np.array(y)
        else:
            err_counter += 1
            if err_counter > attempt_limit:
                raise Exception("Attempt limit overflowed")


if __name__ == '__main__':
    X, y = random_chain_generation(n=16, values=[0.0,-1.0], limit = 2200)
    X = np.exp(X)
    y = np.exp(y)
    np.savetxt('X.txt', X)
    np.savetxt('y.txt', y)   
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')
