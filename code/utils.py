# utility functions for estimators

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, LassoCV, RidgeCV
from sklearn.svm import LinearSVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


def model_selection(model='Lasso', rand=None, **kwargs):
    """
    [input]
    - model : str name of models for propensity score estimation and control-group outcome
    - rand : seed for random number generation for replicability
    - **kwargs : optional custom hyper-parameters for ML models (p.s. mostly for DT, RF, GB, KNN, SVM, and MLP)
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
        # ps_model = LGBMClassifier(max_depth=10, random_state=rand, n_jobs=4)
        # or_model = LGBMRegressor(max_depth=10, random_state=rand, n_jobs=4)
        ps_model = LGBMClassifier(random_state=rand, n_jobs=4)
        or_model = LGBMRegressor(random_state=rand, n_jobs=4)
    elif model == 'DT':
        # ps_model = DecisionTreeClassifier(max_depth=10, random_state=rand)
        # or_model = DecisionTreeRegressor(max_depth=10, random_state=rand)
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
        # ps_model = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
        # or_model = KNeighborsRegressor(n_neighbors=5, n_jobs=4)
        ps_model = KNeighborsClassifier(n_jobs=4)
        or_model = KNeighborsRegressor(n_jobs=4)
    elif model == 'MLP':
        # ps_model = MLPClassifier(hidden_layer_sizes=(2, 1, 1), solver='adam', random_state=rand)
        # or_model = MLPRegressor(hidden_layer_sizes=(2, 1, 1), solver='adam', random_state=rand)
        ps_model = MLPClassifier(solver='adam', random_state=rand)
        or_model = MLPRegressor(solver='adam', random_state=rand)
    else:
        raise ValueError('Please choose from the following model types: Lasso, Ridge, GBM, DT, RF, SVM, KNN, MLP')

    try:
        ps_model.set_params(**kwargs)
        or_model.set_params(**kwargs)
    except ValueError:
        raise ValueError('Illegal model parameter.')
    return ps_model, or_model
