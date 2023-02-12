import numpy as np
from random import shuffle
from itertools import combinations
from beta import BetaCalc
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def auto_cor(v: list) -> np.ndarray:
    """ cyclic autocorrelation of the vector v """
    if v:
        return np.fft.ifft(np.fft.fft(v[::-1])*np.fft.fft(v))[::-1].real / len(v)
    else:
        return np.array([])


def random_chain_generation(n, values, limit=1000, mult_attempt=10):
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

def triada(X, y):
    assert len(X) == len(y)
    n = len(X) // 3 * 3
    indices = y[:n].argsort()
    i_set = 0
    ind = [[],[],[]]
    for i in indices:
        ind[i_set].append(i)
        if i_set == 2:
            i_set = 0
        else:
            i_set += 1
    return X[ind[0]], y[ind[0]], X[ind[1]], y[ind[1]], X[ind[2]], y[ind[2]]

def deviations(y, y_predict):
    mse = np.sqrt(mean_squared_error(y, y_predict))
    mae = mean_absolute_error(y, y_predict)
    return mse, mae


if __name__ == '__main__':
    # X, y = random_chain_generation(n=16, values=[0.0,-1.0], limit = 2200)
    # X = np.exp(X)
    # y = np.exp(y)
    # np.savetxt('X.txt', X)
    # np.savetxt('y.txt', y)

    
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')

    for i, x in enumerate(X):
        X[i] = auto_cor(list(x))



    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    # X = np.array([[1.,2.,3.],[0.1, 0.2, 0.3],[1.1, 2.2, 3.3],[1.11, 2.22, 3.33]])
    # y = np.array([1.0 + 2.0 * np.sum(x) for x in X])
    # X = np.array([[1.1,2.2,3.3],[0.11, 0.22, 0.33],[1.1, 2.2, 3.3],[2.2, 4.4, 3.3]])
    #print(y)
    lin_reg = LinearRegression()
    pf = PolynomialFeatures(degree=5)
    X_train = pf.fit_transform(X)
    model = lin_reg.fit(X, y)
    kf = KFold(n_splits=5, shuffle=True)
    from sklearn.neighbors import RadiusNeighborsRegressor
    neigh = RadiusNeighborsRegressor(radius=0.3)
    scores = cross_val_score(estimator=neigh, cv=kf, X=X, y=y, scoring='neg_mean_squared_log_error')
    print(np.sqrt(-scores.mean()), np.sqrt(-scores).std())

    print(np.log(y).mean())


    # from sklearn.cross_decomposition import PLSRegression
    # pls1 = PLSRegression(n_components=1)
    # #scores = cross_val_score(pls1, X_train, y, cv=5, scoring='r2')

    # from sklearn.svm import SVR
    # svr = SVR(kernel = 'poly')
    # from sklearn import linear_model
    # clf = linear_model.Lasso(alpha=0.03, tol=0.0001)
    # scores = cross_val_score(clf, X, y, cv=5, scoring='r2')

    # #print(scores)
    # print(scores.mean(), scores.std())

    



    
