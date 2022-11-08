# data simulation: data generating processes
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor


def simulate_linear_sparse(n=1000, theta=0, c=100, c_=5, pt1=0.5, ptreat=None, seed=None, panel=True):
    """
    [input]
    - n : sample size
    - theta : true ATT
    - c : number of control variables
    - c_: number of 'useful' control variables
    - pt1 : percentage of time period 1 sample
    - ptreat : rough propensity of getting treated
    - seed : seed for random number generation for replicability
    - panel: indicating if it is panel sample
    [return]
    - data frame of simulated data
    - for panel data: (X, D, Y0, Y1)
    - for repeated cross-section data: (X, D, T, Y)
    """
    if c_ > c:
        raise ValueError('c_ must not be larger than c')
    if pt1 > 1 or pt1 < 0:
        raise ValueError('pt1 must be between 0 and 1')
    if ptreat is not None and (ptreat > 1 or ptreat < 0):
        raise ValueError('ptreat must be between 0 and 1')
    np.random.seed(seed)
    # sample control variables from N(0,1)
    X = np.random.multivariate_normal([0] * c, np.identity(c), n)
    # construct control variable coefficient in PS model as (1, 1/2, ..., 1/c_, 0, 0, ..., 0)
    beta = [1 / k for k in range(1, c_ + 1)]
    beta.extend([0] * (c - c_))
    # true propensity score and treatment assignment based on PS
    fps = np.dot(X, beta)
    # logistic distribution with shifting
    if ptreat is not None:
        cutoff = np.percentile(fps, (1 - ptreat) * 100)
        fps -= cutoff
    U = np.random.random_sample(n)
    D = np.clip(np.sign(np.exp(fps) / (1 + np.exp(fps)) - U), 0, 1)

    # modified control variable coefficient for outcome calculation
    beta = [i + 0.1 for i in beta]
    df = pd.DataFrame({'X' + str(i): X[:, i] for i in range(c)})
    df['D'] = D
    # outcomes
    Y0 = np.dot(X, beta) + np.random.normal(0, 0.1, n)
    Y1 = np.dot(X, beta) + 0.1 + theta * D + np.random.normal(0, 0.1, n)
    if panel:
        df['Y0'] = Y0
        df['Y1'] = Y1
    else:
        # random assignment of time period
        T = np.random.choice([0, 1], n, p=[1 - pt1, pt1])
        # outcome
        # error term from N(0,0.1)
        Y = (1 - T) * Y0 + T * Y1
        df['T'] = T
        df['Y'] = Y
    return df


def simulate_linear_hetero(n=1000, theta=0, pt1=0.5, mis_ps=False, mis_or=False, hetero=True, seed=None, panel=True):
    """
    [input]
    - n : sample size
    - theta : true ATT
    - pt1 : percentage of time period 1 sample
    - mis_ps : indicating if PS model is misspecified
    - mis_or : indicating if OR model is misspecified
    - hetero : indicating if to include time-invariant unobserved heterogeneity
    - seed : seed for random number generation for replicability
    - panel: indicating if it is panel sample
    [return]
    - data frame of simulated data
    - for panel data: (X, D, Y0, Y1)
    - for repeated cross-section data: (X, D, T, Y)
    """
    np.random.seed(seed)
    # ‘latent' variables
    Z = np.random.multivariate_normal([0] * 4, np.identity(4), n)
    # observed variables
    X1 = np.exp(0.5 * Z[:, 0])
    X2 = 10 + Z[:, 1] / (1 + np.exp(Z[:, 0]))
    X3 = np.power(0.6 + Z[:, 0] * Z[:, 2] / 25, 3)
    X4 = (20 + Z[:, 1] + Z[:, 3]) ** 2
    X1 = (X1 - np.mean(X1)) / np.std(X1)
    X2 = (X2 - np.mean(X2)) / np.std(X2)
    X3 = (X3 - np.mean(X3)) / np.std(X3)
    X4 = (X4 - np.mean(X4)) / np.std(X4)

    W1_or, W2_or, W3_or, W4_or = X1, X2, X3, X4
    W1_ps, W2_ps, W3_ps, W4_ps = X1, X2, X3, X4

    # true ps/or models based on unobserved variables to mimic misspecification
    if mis_ps:
        W1_ps, W2_ps, W3_ps, W4_ps = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
    if mis_or:
        W1_or, W2_or, W3_or, W4_or = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]

    fps = 0.75 * (-W1_ps + 0.5 * W2_ps - 0.25 * W3_ps - 0.1 * W4_ps)
    freg = 210 + 27.4 * W1_or + 13.7 * (W2_or + W3_or + W4_or)
    U = np.random.random_sample(n)
    D = np.clip(np.sign(np.exp(fps) / (1 + np.exp(fps)) - U), 0, 1)
    v = np.random.normal(D * freg, 1, n)  # todo ML models cannot handle this heterogeneity
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'D': D})
    Y0 = freg + np.random.normal(0, 1, n)
    Y1 = 2 * freg + theta * D + np.random.normal(0, 1, n)
    if hetero:
        Y0 += v
        Y1 += v
    if panel:
        df['Y0'] = Y0
        df['Y1'] = Y1
    else:
        T = np.clip(np.sign(pt1 - np.random.random_sample(n)), 0, 1)
        Y = (1 - T) * Y0 + T * Y1
        df['T'] = T
        df['Y'] = Y
    return df


def simulate_nonlinear_nuisance(n=1000, theta=0, pt1=0.5, seed=None, panel=True):
    """
    [input]
    - n : sample size
    - theta : true ATT
    - pt1 : percentage of time period 1 sample
    - seed : seed for random number generation for replicability
    - panel: indicating if it is panel sample
    [return]
    - data frame of simulated data
    - for panel data: (X, D, Y0, Y1)
    - for repeated cross-section data: (X, D, T, Y)
    """
    np.random.seed(seed)
    Z = np.random.multivariate_normal([0] * 4, np.identity(4), n)
    X1 = Z[:, 0] * Z[:, 1] + Z[:, 0] ** 2
    X2 = (0.5 + Z[:, 2]) / np.exp(Z[:, 3])
    X3 = (Z[:, 0] + Z[:, 2]) ** 2
    X4 = Z[:, 1] * np.log(1 + np.exp(Z[:, 3]))
    X1 = (X1 - np.mean(X1)) / np.std(X1)
    X2 = (X2 - np.mean(X2)) / np.std(X2)
    X3 = (X3 - np.mean(X3)) / np.std(X3)
    X4 = (X4 - np.mean(X4)) / np.std(X4)

    freg = 0.5 * X1 + 1.2 * X2 - 2 * X3 + 5 * np.exp(X4)
    fps = 2.3 * np.exp(X1) + 8 * X2 + 1.5 * X3 - 5 * X4

    U = np.random.random_sample(n)
    D = np.clip(np.sign(1 / (1 + np.exp(-fps)) - U), 0, 1)
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'D': D})
    Y0 = freg + np.random.normal(0, 0.1)
    Y1 = freg + np.random.normal(0, 0.1) + theta * D
    if panel:
        df['Y0'] = Y0
        df['Y1'] = Y1
    else:
        T = np.clip(np.sign(pt1 - np.random.random_sample(n)), 0, 1)
        Y = (1 - T) * Y0 + T * Y1
        df['T'] = T
        df['Y'] = Y
    return df


def simulate_ml_nuisance(n=1000, theta=0, pt1=0.5, seed=None, panel=True):
    """
    [input]
    - n : sample size
    - theta : true ATT
    - pt1 : percentage of time period 1 sample
    - seed : seed for random number generation for replicability
    - panel: indicating if it is panel sample
    [return]
    - data frame of simulated data
    - for panel data: (X, D, Y0, Y1)
    - for repeated cross-section data: (X, D, T, Y)
    """
    np.random.seed(seed)
    X = np.random.multivariate_normal([0] * 4, np.identity(4), n)

    # currently 10 layers
    layers = (4, 4, 4, 4, 2, 2, 2, 2, 1, 1)
    model = MLPRegressor(hidden_layer_sizes=layers, activation='relu')
    model.out_activation_ = 'relu'
    model.n_layers_ = 10

    # random weights
    model.coefs_ = [np.reshape(np.random.normal(0, 1, 16), (4, 4)),
                    np.reshape(np.random.normal(0, 1, 16), (4, 4)),
                    np.reshape(np.random.normal(0, 1, 16), (4, 4)),
                    np.reshape(np.random.normal(0, 1, 16), (4, 4)),
                    np.reshape(np.random.normal(0, 1, 8), (4, 2)),
                    np.reshape(np.random.normal(0, 1, 4), (2, 2)),
                    np.reshape(np.random.normal(0, 1, 4), (2, 2)),
                    np.reshape(np.random.normal(0, 1, 4), (2, 2)),
                    np.reshape(np.random.normal(0, 1, 2), (2, 1)),
                    np.reshape(np.random.normal(0, 1, 1), (1, 1)),
                    np.reshape(np.random.normal(0, 1, 1), (1, 1))]

    # random biases
    model.intercepts_ = [np.random.normal(0, 1, 4),
                         np.random.normal(0, 1, 4),
                         np.random.normal(0, 1, 4),
                         np.random.normal(0, 1, 4),
                         np.random.normal(0, 1, 2),
                         np.random.normal(0, 1, 2),
                         np.random.normal(0, 1, 2),
                         np.random.normal(0, 1, 2),
                         np.random.normal(0, 1, 1),
                         np.random.normal(0, 1, 1),
                         np.random.normal(0, 1, 1)]

    freg = model.predict(X)

    beta = np.random.normal(0, 1, 5)
    fps = np.dot(X, beta[0:4]) + beta[4]
    U = np.random.random_sample(n)
    D = np.clip(np.sign(1 / (1 + np.exp(-fps)) - U), 0, 1)
    df = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'X3': X[:, 2], 'X4': X[:, 3], 'D': D})
    Y0 = freg + np.random.normal(0, 0.1)
    Y1 = freg + 0.5 + theta * D + np.random.normal(0, 0.1)
    if panel:
        df['Y0'] = Y0
        df['Y1'] = Y1
    else:
        T = np.clip(np.sign(pt1 - np.random.random_sample(n)), 0, 1)
        Y = (1 - T) * Y0 + T * Y1
        df['T'] = T
        df['Y'] = Y
    return df


def simulate_data(code: int, n=1000, theta=0, pt1=0.5, seed=None, panel=True):
    if code == 0:
        return simulate_linear_sparse(n=n, theta=theta, pt1=pt1, seed=seed, panel=panel)
    elif code == 1:
        return simulate_linear_hetero(n=n, theta=theta, pt1=pt1, seed=seed, panel=panel)
    elif code == 2:
        return simulate_nonlinear_nuisance(n=n, theta=theta, pt1=pt1, seed=seed, panel=panel)
    elif code == 3:
        return simulate_ml_nuisance(n=n, theta=theta, pt1=pt1, seed=seed, panel=panel)
    else:
        raise ValueError('Please choose code 0-3.')


if __name__ == "__main__":
    # testing
    for i in range(4):
        df = simulate_data(i, panel=True)
        print(df.head())
        df = simulate_data(i, panel=False)
        print(df.head())
