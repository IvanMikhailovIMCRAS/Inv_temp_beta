from random import shuffle
from itertools import combinations
from beta import *


def data_generation(n, value, limit=1000):
    B = BetaCalc(n=n)
    v0 = np.zeros(n,dtype=float)
    v = np.copy(v0)
    indexies = list(range(n))
    counter = 0
    X, y = [], []
    b = 0.0
    for m in range(1,n):
        shuffle(indexies)
        i0 = indexies[0]
        ym = []
        for i in combinations(indexies, m):
            if i[0] != i0: 
                continue
            id = [k-1 for k in i]
            v[:] = v0[:]
            v[id] = value
            b = B.beta(v)
            if b not in ym:
                X.append(list(v))
                ym.append(b)
                counter += 1
                if counter >= limit:
                    y += ym
                    return np.array(X), np.array(y).reshape((len(y),1)) 
        y += ym
    v[indexies] = value
    X.append(list(v))
    y.append(B.beta(v))
    return np.array(X), np.array(y).reshape((len(y),1)) 

def random_data_generation(n, values, limit=1000, mult_err=10):
    B = BetaCalc(n=n)
    err_limit = mult_err * limit
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
                    return np.array(X), np.array(y).reshape((len(y),1)) 
        else:
            err_counter += 1
            assert err_counter < err_limit
    return np.array(X), np.array(y).reshape((len(y),1)) 

if __name__ == '__main__':
    pass
