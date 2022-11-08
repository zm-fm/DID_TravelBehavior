# DRDID for repeated cross-section data in Python
# reference: Sant’Anna, Pedro H. C., and Jun Zhao. 2020.
# “Doubly Robust Difference-in-Differences Estimators.”
# Journal of Econometrics 219(1):101–22. doi: 10.1016/j.jeconom.2020.06.003.


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import scipy.stats
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def drdid_panel(df: pd.DataFrame, ps_model, or_model, y0_col: str, y1_col: str, d_col: str, x_cols: list, eps=1e-5):
    """
    [input]
    - df : pd.DataFrame input data table
    - ps_model : instance of PS model
    - or_model : instance of OR model
    - y0_col : str column name of outcome at time 0
    - y1_col : str column name of outcome at time 1
    - d_col : str column name of treatment variable
    - x_cols : list column names of control variables
    - eps : threshold for propensity score trimming
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """
    # fit PS model and compute PS
    ps_model.fit(df[x_cols], df[d_col])

    ps_score = ps_model.score(df[x_cols], df[d_col])
    pscore = np.clip(ps_model.predict_proba(df[x_cols])[:, 1], eps, 1 - eps)

    # PS-based weight for weighted OLS below
    or_w = pscore / (1 - pscore)

    # ignore sampling weight in KNN and MLP
    if type(ps_model) in [KNeighborsClassifier, MLPClassifier]:
        or_model.fit(df[df[d_col] == 0][x_cols], df[df[d_col] == 0][y1_col] - df[df[d_col] == 0][y0_col])
        or_score = or_model.score(df[df[d_col] == 0][x_cols], df[df[d_col] == 0][y1_col] - df[df[d_col] == 0][y0_col])
    else:
        or_model.fit(df[df[d_col] == 0][x_cols], df[df[d_col] == 0][y1_col] - df[df[d_col] == 0][y0_col],
                     or_w[df[d_col] == 0])
        or_score = or_model.score(df[df[d_col] == 0][x_cols], df[df[d_col] == 0][y1_col] - df[df[d_col] == 0][y0_col],
                                  or_w[df[d_col] == 0])
    y_delta = or_model.predict(df[x_cols])

    # compute bias-reduced doubly robust DID estimators
    summand = (1 - (1 - df[d_col]) / (1 - pscore)) * (df[y1_col] - df[y0_col] - y_delta)
    att = np.mean(summand) / np.mean(df[d_col])

    # get the influence function to compute standard error
    inf_func = (summand - df[d_col] * att) / np.mean(df[d_col])
    se = np.std(inf_func) / np.sqrt(len(df))
    # t-test
    t = att / se
    p = scipy.stats.t.sf(abs(t), len(df) - 2) * 2
    t_crit = np.abs(scipy.stats.t.ppf((1 - 0.95) / 2, len(df) - 2))
    ci = (att - t_crit * se, att + t_crit * se)
    result = {'ATT': att, 'S.E.': se, 't-stat': t, 'p-value': p, 'C.I.': ci, 'PS': ps_score, 'OR': or_score}
    return result


def drdid_rcs(df: pd.DataFrame, ps_model, or_model, y_col: str, t_col: str, d_col: str, x_cols: list, eps=1e-5):
    """
    [input]
    - df : pd.DataFrame input data table
    - ps_model : instance of PS model
    - or_model : instance of OR model
    - y_col : str column name of outcome
    - t_col : str column name of time variable
    - d_col : str column name of treatment variable
    - x_cols : list column names of control variables
    - eps : threshold for propensity score trimming
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """

    # fit PS model and compute PS
    ps_model.fit(df[x_cols], df[d_col])

    ps_score = ps_model.score(df[x_cols], df[d_col])
    pscore = np.clip(ps_model.predict_proba(df[x_cols])[:, 1], eps, 1 - eps)

    # PS-based weight for weighted OLS below
    or_w = pscore / (1 - pscore)

    # compute OR predicted values for control group
    # using subset to fit, predict for whole sample
    or_score = 0
    index = (df[t_col] == 0) & (df[d_col] == 0)
    # ignore sampling weights in KNN and MLP
    if type(ps_model) in [KNeighborsClassifier, MLPClassifier]:
        or_model.fit(df[index][x_cols], df[index][y_col])
        or_score += or_model.score(df[index][x_cols], df[index][y_col])
    else:
        or_model.fit(df[index][x_cols], df[index][y_col], or_w[index])
        or_score += or_model.score(df[index][x_cols], df[index][y_col], or_w[index])
    y_con_pre = or_model.predict(df[x_cols])

    index = (df[t_col] == 1) & (df[d_col] == 0)
    # ignore sampling weights in KNN and MLP
    if type(ps_model) in [KNeighborsClassifier, MLPClassifier]:
        or_model.fit(df[index][x_cols], df[index][y_col])
        or_score += or_model.score(df[index][x_cols], df[index][y_col])
    else:
        or_model.fit(df[index][x_cols], df[index][y_col], or_w[index])
        or_score += or_model.score(df[index][x_cols], df[index][y_col], or_w[index])
    y_con_post = or_model.predict(df[x_cols])
    # combine control-group OR results
    y_con = y_con_pre * (1 - df[t_col]) + y_con_post * df[t_col]
    or_score /= 2

    # compute OR predicted values for treatment group with OLS, not weighted
    index = (df[t_col] == 0) & (df[d_col] == 1)
    or_model.fit(df[index][x_cols], df[index][y_col])
    y_treat_pre = or_model.predict(df[x_cols])
    index = (df[t_col] == 1) & (df[d_col] == 1)
    or_model.fit(df[index][x_cols], df[index][y_col])
    y_treat_post = or_model.predict(df[x_cols])

    # prepare weights for different components in estimator
    w_treat_pre = df[d_col] * (1 - df[t_col])
    w_treat_post = df[d_col] * df[t_col]
    w_con_pre = pscore * (1 - df[d_col]) * (1 - df[t_col]) / (1 - pscore)
    w_con_post = pscore * (1 - df[d_col]) * df[t_col] / (1 - pscore)

    w_d = df[d_col]
    w_dt1 = df[d_col] * df[t_col]
    w_dt0 = df[d_col] * (1 - df[t_col])

    # influence function elements & estimator components
    eta_treat_pre = w_treat_pre * (df[y_col] - y_con) / np.mean(w_treat_pre)
    eta_treat_post = w_treat_post * (df[y_col] - y_con) / np.mean(w_treat_post)
    eta_con_pre = w_con_pre * (df[y_col] - y_con) / np.mean(w_con_pre)
    eta_con_post = w_con_post * (df[y_col] - y_con) / np.mean(w_con_post)

    # added locally efficient DRDID elements & components
    eta_d_post = w_d * (y_treat_post - y_con_post) / np.mean(w_d)
    eta_dt1_post = w_dt1 * (y_treat_post - y_con_post) / np.mean(w_dt1)
    eta_d_pre = w_d * (y_treat_pre - y_con_pre) / np.mean(w_d)
    eta_dt0_pre = w_dt0 * (y_treat_pre - y_con_pre) / np.mean(w_dt0)

    att = (eta_treat_post - eta_treat_pre) - (eta_con_post - eta_con_pre) + \
          (eta_d_post - eta_dt1_post) - (eta_d_pre - eta_dt0_pre)
    att = np.mean(att)

    # influence function for S.E.
    # influence function for treatment group
    inf_treat_pre = eta_treat_pre - w_treat_pre * np.mean(eta_treat_pre) / np.mean(w_treat_pre)
    inf_treat_post = eta_treat_post - w_treat_post * np.mean(eta_treat_post) / np.mean(w_treat_post)
    inf_treat = inf_treat_post - inf_treat_pre
    # influence function for control group
    inf_con_pre = eta_con_pre - w_con_pre * np.mean(eta_con_pre) / np.mean(w_con_pre)
    inf_con_post = eta_con_post - w_con_post * np.mean(eta_con_post) / np.mean(w_con_post)
    inf_con = inf_con_post - inf_con_pre
    # influence function for added elements
    inf_eff = (eta_d_post - w_d * np.mean(eta_d_post) / np.mean(w_d)) - \
              (eta_dt1_post - w_dt1 * np.mean(eta_dt1_post) / np.mean(w_dt1)) - \
              (eta_d_pre - w_d * np.mean(eta_d_pre) / np.mean(w_d)) + \
              (eta_dt0_pre - w_dt0 * np.mean(eta_dt0_pre) / np.mean(w_dt0))
    # influence function for locally efficient DR estimator
    inf_func = inf_treat - inf_con + inf_eff
    se = np.std(inf_func) / np.sqrt(len(df))
    # t-test
    t = att / se
    p = scipy.stats.t.sf(abs(t), len(df) - 2) * 2
    t_crit = np.abs(scipy.stats.t.ppf((1 - 0.95) / 2, len(df) - 2))
    ci = (att - t_crit * se, att + t_crit * se)
    result = {'ATT': att, 'S.E.': se, 't-stat': t, 'p-value': p, 'C.I.': ci, 'PS': ps_score, 'OR': or_score}
    return result


def drdid(df: pd.DataFrame, ps_model, or_model, d_col: str, x_cols: list, y_col: list, t_col=None, eps=1e-5, panel=True):
    """
    [input]
    - df : pd.DataFrame input data table
    - ps_model : instance of PS model
    - or_model : instance of OR model
    - d_col : str column name of treatment variable
    - x_cols : list column names of control variables
    - y_col : list column names of outcome(s), single-element for rcs
    - t_col : str column name of time variable, None for panel
    - eps : threshold for propensity score trimming
    - panel : boolean indicating if it is panel
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """

    # wrapper function of panel and rcs models
    if panel and t_col is not None:
        raise Warning('T column should be None for panel model, ignored.')
    if panel and len(y_col) != 2:
        raise ValueError('Y1 and Y0 should be specified in Y column list for panel model.')
    if not panel and t_col is None:
        raise ValueError('T column should not be None for repeated cross-section model.')
    if not panel and len(y_col) != 1:
        raise ValueError('Y column list only needs one column name.')

    if panel:
        result = drdid_panel(df, ps_model, or_model, y_col[0], y_col[1], d_col, x_cols, eps)
    else:
        result = drdid_rcs(df, ps_model, or_model, y_col[0], t_col, d_col, x_cols, eps)
    return result
