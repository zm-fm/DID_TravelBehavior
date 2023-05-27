# utility functions for simulation and empirical study

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, LassoCV, RidgeCV
from sklearn.svm import LinearSVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, KernelDensity
from sklearn.neural_network import MLPClassifier, MLPRegressor


def simulate_data(n=1000, theta=0, pt1=0.5, panel=True, hetero=True, unobs=True, mis_ps=False, mis_or=False):
    """
    [input]
    - n : int number of samples
    - theta : true ATT
    - pt1 : float between 0 and 1 proportion of post-intervention samples in repeated cross-section data
    - panel : boolean if it is panel
    - hetero : boolean if effect is heterogeneous
    - unobs : boolean if there is unobserved confounding
    - mis_ps : boolean if propensity score function is mis-specified
    - mis_or : boolean if outcome regression function is mis-specified
    [return]
    - df : Dataframe of simulated data sample
    """

    rng = np.random.default_rng()
    # dimension, mean, covariance
    d = 5
    means = [0] * d
    std = np.diag([1] * d)
    eig = np.random.uniform(0, 2, d)
    eig = eig * d / sum(eig)
    corr = stats.random_correlation.rvs(eig, random_state=rng)
    cov = std.dot(corr.dot(std))

    # correlated normally distributed covariates
    mvnorm = stats.multivariate_normal(mean=means, cov=cov)
    X = mvnorm.rvs(n, random_state=rng)

    # Gaussian copula
    norm = stats.norm()
    X_unif = norm.cdf(X)

    m1 = stats.gumbel_r()
    m2 = stats.beta(3, 2)

    # correlated non-normal distributed covariates
    X[:, 0] = m1.ppf(X_unif[:, 0])
    X[:, 1] = m2.ppf(X_unif[:, 1])
    # transfer beta to binary
    X[:, 1] = np.where(X[:, 1] > 0.5, 1, 0)
    X_ps, X_or = X, X

    # if mis-specified, use another set of independent normally distributed variables for PS or OR function
    if mis_ps:
        mvnorm = stats.multivariate_normal(mean=means, cov=std)
        X_ps = mvnorm.rvs(n, random_state=rng)
    if mis_or:
        mvnorm = stats.multivariate_normal(mean=means, cov=std)
        X_or = mvnorm.rvs(n, random_state=rng)

    # use kernel density as nonparametric treatment assignment process
    base = KernelDensity(kernel='gaussian', bandwidth=0.75)
    base.fit(X_ps)
    f_ps = np.exp(base.score_samples(X_ps))

    # randomize a kernel ridge regression as base outcome generating model
    base = KernelRidge(alpha=1.0)
    base.n_features_in_ = 5
    base.dual_coef_ = np.random.normal(0, 1, n) * 50  # todo
    base.X_fit_ = np.array(X_or)
    f_or = base.predict(X_or)

    if unobs:
        # X5 is unobserved
        df = pd.DataFrame({'X' + str(i): X[:, i] for i in range(d - 1)})
    else:
        df = pd.DataFrame({'X' + str(i): X[:, i] for i in range(d)})

    # treatment assignment
    U = np.random.random_sample(n)
    D = np.clip(np.sign(1 / (1 + np.exp(-f_ps)) - U), 0, 1)
    df['D'] = D
    if panel:
        # placeholder T variable for panel data
        T = np.ones(n)
    else:
        T = np.clip(np.sign(pt1 - np.random.random_sample(n)), 0, 1)

    # heterogeneous treatment effect
    if hetero:
        effect = (X[:, 0] + X[:, 2]) ** 2
        m = np.mean(effect[np.nonzero(D * T)])
        # to align the mean to theta
        effect[np.nonzero(D * T)] -= m - theta
        effect *= D
    else:
        effect = theta * D

    # outcomes
    Y0 = f_or + np.random.normal(0, 0.1, n)
    Y1 = 2 * f_or + np.random.normal(0, 0.1, n) + effect
    if panel:
        df['Y0'] = Y0
        df['Y1'] = Y1
        df['effect'] = effect
    else:
        Y = (1 - T) * Y0 + T * Y1
        df['T'] = T
        df['Y'] = Y
        df['effect'] = effect * T
    return df


def model_selection(model='Lasso', rand=None):
    """
    [input]
    - model : str name of models for propensity score estimation and control-group outcome
    - rand : seed for random number generation for replicability
    - **kwargs : optional custom hyperparameters for ML models (p.s. mostly for DT, RF, GB, KNN, SVM, and MLP)
    [return]
    - instances of chosen ML models for propensity score and outcome regression
    """
    # model selection
    if model == 'Linear':
        ps_model = LogisticRegression(penalty='none', random_state=rand)
        or_model = LinearRegression()
    elif model == 'Lasso':
        ps_model = LogisticRegressionCV(cv=5, random_state=rand, penalty="l1", solver="saga", n_jobs=4)
        or_model = LassoCV(cv=5, random_state=rand, n_jobs=4)
    elif model == 'Ridge':
        ps_model = LogisticRegressionCV(cv=5, random_state=rand, penalty="l2", solver="saga", n_jobs=4)
        or_model = RidgeCV(cv=5)
    elif model == 'GBM':
        ps_model = LGBMClassifier(random_state=rand, n_jobs=4)
        or_model = LGBMRegressor(random_state=rand, n_jobs=4)
    elif model == 'DT':
        ps_model = DecisionTreeClassifier(random_state=rand)
        or_model = DecisionTreeRegressor(random_state=rand)
    elif model == 'RF':
        # ps_model = RandomForestClassifier(max_depth=10, random_state=rand, n_jobs=4)
        # or_model = RandomForestRegressor(max_depth=10, random_state=rand, n_jobs=4)
        ps_model = RandomForestClassifier(random_state=rand, n_jobs=4)
        or_model = RandomForestRegressor(random_state=rand, n_jobs=4)
    elif model == 'SVM':
        ps_model = SVC(random_state=rand, probability=True)
        or_model = LinearSVR(random_state=rand)
    elif model == 'KNN':
        ps_model = KNeighborsClassifier(n_jobs=4)
        or_model = KNeighborsRegressor(n_jobs=4)
    elif model == 'MLP':
        ps_model = MLPClassifier(solver='adam', random_state=rand)
        or_model = MLPRegressor(solver='adam', random_state=rand)
    elif model == 'N.A.':
        ps_model, or_model = None, None
    else:
        raise ValueError('Please choose from the following model types: Lasso, Ridge, GBM, DT, RF, SVM, KNN, MLP, N.A.')
    return ps_model, or_model


def did_simulation(df: pd.DataFrame, ps_model, or_model, method, true_att=0, panel=True):
    """
    [input]
    - df : pd.DataFrame input data table
    - ps_model: instance of propensity score model
    - or_model: instance of outcome regression model
    - method: estimator used
    - true_att : true att used in DGP, need to be consistent with DGP parameter
    - panel : boolean indicating if it is panel
    [return]
    - result : tuple of ATT bias, coverage rate, and length of 95% confidence interval
    """
    x_cols = [c for c in df.columns if 'X' in c]
    d_col = 'D'
    if panel:
        y_cols = ['Y0', 'Y1']
        t_col = None
    else:
        y_cols = ['Y']
        t_col = 'T'
    # invoke benchmark method
    if ps_model is None or or_model is None:
        result = method(df, d_col, x_cols, y_cols, t_col, panel=panel)
    else:
        result = method(df, ps_model, or_model, d_col, x_cols, y_cols, t_col, panel=panel)
    return result.get('ATT') - true_att, \
           int(result.get('C.I.')[0] < true_att < result.get('C.I.')[1]), \
           np.abs(result.get('C.I.')[1] - result.get('C.I.')[0]), result.get('PS'), result.get('OR')


def parse_parameter(file):
    """
    [input]
    - file : file path to the hyperparameter tuning output
    [return]
    - result : dictionary of best hyperparameters from tuning
    """
    df = pd.read_csv(file)
    # based on absolute value of bias
    df['abs'] = df['Bias'].abs()
    df.sort_values(by='abs', inplace=True)
    # keep best ones by estimator-model combination
    df.drop_duplicates(subset=['Method', 'Model'], inplace=True)
    df['key'] = list(zip(df['Method'], df['Model']))
    df = df[['key', 'Param']]
    dct = df.set_index('key').T.to_dict('list')
    # format the parameters into usable structure
    for k, v in dct.items():
        v = v[0]
        v = v.replace('--', '-*').split('-')
        l = []
        for i in v:
            if '*' in i:
                i = i.replace('*', '-')
            if i == 'None':
                i = None
            elif i.lstrip('-').isdigit():
                i = int(i)
            elif i.lstrip('-').replace('.', '').isdigit():
                i = float(i)
            elif '(' in i:
                i = i.replace('(', '').replace(')', '').replace(' ', '').split(',')
                i = tuple([int(ii) for ii in i if ii.isdigit()])
            l.append(i)
        dct[k] = l
    return dct


def did(df: pd.DataFrame, ps_model, or_model, method, d_col: str, x_cols: list, y_cols: list, t_col=None, panel=True):
    """
    [input]
    - df : pd.DataFrame input data table
    - ps_model: instance of propensity score model
    - or_model: instance of outcome regression model
    - method: estimator used
    - d_col : str column name of treatment variable
    - x_cols : list column names of control variables
    - y_col : list column names of outcome(s), single-element for rcs
    - t_col : str column name of time variable, None for panel
    - panel : boolean indicating if it is panel
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """
    if ps_model is None or or_model is None:
        result = method(df, d_col, x_cols, y_cols, t_col, panel=panel)
    else:
        result = method(df, ps_model, or_model, d_col, x_cols, y_cols, t_col, panel=panel)
    return result
