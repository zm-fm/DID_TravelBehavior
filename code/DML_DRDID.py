# to combine DRDID and DMLDiD for treatment effect estimation
# References
# Chang, Neng Chieh. 2020.
# “Double/Debiased Machine Learning for Difference-in-Differences Models.”
# Econometrics Journal 23(2):177–91. doi: 10.1093/ectj/utaa001.
# Sant’Anna, Pedro H. C., and Jun Zhao. 2020.
# “Doubly Robust Difference-in-Differences Estimators.”
# Journal of Econometrics 219(1):101–22. doi: 10.1016/j.jeconom.2020.06.003.

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def dml_drdid_panel(df: pd.DataFrame, ps_model, or_model, y0_col: str, y1_col: str, d_col: str, x_cols: list,
                    eps=1e-5, n=1, rand=None):
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
    - n : number of rounds
    - rand : seed for random number generation for replicability
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """

    # define PS model and OR model
    # split data into 2 partitions for cross-fitting
    df_set = train_test_split(df, random_state=rand, test_size=0.5)
    K = 2
    att_list, var_list, ps_list, or_list = [], [], [], []
    for _ in range(n):
        for i in range(K):
            # 2 partitions 1 for model estimation other for value prediction
            k = 0 if i == 0 else 1
            c = 1 if i == 0 else 0

            ps_model.fit(df_set[c][x_cols], df_set[c][d_col])
            ps_score_ = ps_model.score(df_set[c][x_cols], df_set[c][d_col])

            pscore = np.clip(ps_model.predict_proba(df_set[k][x_cols])[:, 1], eps, 1 - eps)

            pscore_ = np.clip(ps_model.predict_proba(df_set[c][x_cols])[:, 1], eps, 1 - eps)
            # PS-based weight for weighted OR below
            or_w = pscore_ / (1 - pscore_)

            # ignore sampling weights in KNN and MLP
            if type(ps_model) in [KNeighborsClassifier, MLPClassifier]:
                or_model.fit(df_set[c][df_set[c][d_col] == 0][x_cols],
                             df_set[c][df_set[c][d_col] == 0][y1_col] - df_set[c][df_set[c][d_col] == 0][y0_col])
                or_score_ = or_model.score(df_set[c][df_set[c][d_col] == 0][x_cols],
                                           df_set[c][df_set[c][d_col] == 0][y1_col] - df_set[c][df_set[c][d_col] == 0][y0_col])
            else:
                or_model.fit(df_set[c][df_set[c][d_col] == 0][x_cols],
                             df_set[c][df_set[c][d_col] == 0][y1_col] - df_set[c][df_set[c][d_col] == 0][y0_col],
                             or_w[df_set[c][d_col] == 0])
                or_score_ = or_model.score(df_set[c][df_set[c][d_col] == 0][x_cols],
                                           df_set[c][df_set[c][d_col] == 0][y1_col] - df_set[c][df_set[c][d_col] == 0][y0_col],
                                           or_w[df_set[c][d_col] == 0])
            y_delta = or_model.predict(df_set[k][x_cols])

            # compute bias-reduced doubly robust DID estimators
            summand = (1 - (1 - df_set[k][d_col]) / (1 - pscore)) * (df_set[k][y1_col] - df_set[k][y0_col] - y_delta)
            att = np.mean(summand) / np.mean(df_set[k][d_col])
            att_list.append(att)
            ps_list.append(ps_score_)
            or_list.append(or_score_)
            # get the influence function to compute standard error
            inf_func = (summand - df_set[k][d_col] * att) / np.mean(df_set[k][d_col])
            var = np.var(inf_func)
            var_list.append(var)
    att = np.mean(att_list)
    ps_score = np.mean(ps_list)
    or_score = np.mean(or_list)
    se = np.sqrt(np.mean(var_list) / len(df))
    # t-test
    t = att / se
    p = scipy.stats.t.sf(abs(t), len(df) - 2) * 2
    t_crit = np.abs(scipy.stats.t.ppf((1 - 0.95) / 2, len(df) - 2))
    ci = (att - t_crit * se, att + t_crit * se)
    result = {'ATT': att, 'S.E.': se, 't-stat': t, 'p-value': p, 'C.I.': ci, 'PS': ps_score, 'OR': or_score}
    return result


def dml_drdid_rcs(df: pd.DataFrame, ps_model, or_model, y_col: str, t_col: str, d_col: str, x_cols: list,
                  eps=1e-5, n=1, rand=None):
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
    - n : number of rounds
    - rand : seed for random number generation for replicability
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """

    # split data into 2 partitions for cross-fitting
    df_set = train_test_split(df, random_state=rand, test_size=0.5)
    K = 2
    att_list, var_list, ps_list, or_list = [], [], [], []
    for _ in range(n):
        for i in range(K):
            # 2 partitions 1 for model estimation other for value prediction
            k = 0 if i == 0 else 1
            c = 1 if i == 0 else 0

            # fit PS model and compute PS
            ps_model.fit(df_set[c][x_cols], df_set[c][d_col])

            ps_score_ = ps_model.score(df_set[c][x_cols], df_set[c][d_col])
            pscore = np.clip(ps_model.predict_proba(df_set[k][x_cols])[:, 1], eps, 1 - eps)

            pscore_ = np.clip(ps_model.predict_proba(df_set[c][x_cols])[:, 1], eps, 1 - eps)
            # PS-based weight for weighted OR below
            or_w = pscore_ / (1 - pscore_)

            or_score_ = 0
            # compute OR predicted values for control group
            index = (df_set[c][t_col] == 0) & (df_set[c][d_col] == 0)
            # ignore sampling weights in KNN and MLP
            if type(ps_model) in [KNeighborsClassifier, MLPClassifier]:
                or_model.fit(df_set[c][index][x_cols], df_set[c][index][y_col])
                or_score_ += or_model.score(df_set[c][index][x_cols], df_set[c][index][y_col])
            else:
                or_model.fit(df_set[c][index][x_cols], df_set[c][index][y_col], or_w[index])
                or_score_ += or_model.score(df_set[c][index][x_cols], df_set[c][index][y_col], or_w[index])
            y_con_pre = or_model.predict(df_set[k][x_cols])

            index = (df_set[c][t_col] == 1) & (df_set[c][d_col] == 0)
            # ignore sampling weights in KNN and MLP
            if type(ps_model) in [KNeighborsClassifier, MLPClassifier]:
                or_model.fit(df_set[c][index][x_cols], df_set[c][index][y_col])
                or_score_ += or_model.score(df_set[c][index][x_cols], df_set[c][index][y_col])
            else:
                or_model.fit(df_set[c][index][x_cols], df_set[c][index][y_col], or_w[index])
                or_score_ += or_model.score(df_set[c][index][x_cols], df_set[c][index][y_col], or_w[index])
            or_score_ /= 2
            y_con_post = or_model.predict(df_set[k][x_cols])
            # combine control-group OR results
            y_con = y_con_pre * (1 - df_set[k][t_col]) + y_con_post * df_set[k][t_col]

            # compute OR predicted values for treatment group with OLS, not weighted
            index = (df_set[c][t_col] == 0) & (df_set[c][d_col] == 1)
            or_model.fit(df_set[c][index][x_cols], df_set[c][index][y_col])
            y_treat_pre = or_model.predict(df_set[k][x_cols])
            index = (df_set[c][t_col] == 1) & (df_set[c][d_col] == 1)
            or_model.fit(df_set[c][index][x_cols], df_set[c][index][y_col])
            y_treat_post = or_model.predict(df_set[k][x_cols])

            # prepare weights for different components in estimator
            w_treat_pre = df_set[k][d_col] * (1 - df_set[k][t_col])
            w_treat_post = df_set[k][d_col] * df_set[k][t_col]
            w_con_pre = pscore * (1 - df_set[k][d_col]) * (1 - df_set[k][t_col]) / (1 - pscore)
            w_con_post = pscore * (1 - df_set[k][d_col]) * df_set[k][t_col] / (1 - pscore)

            w_d = df_set[k][d_col]
            w_dt1 = df_set[k][d_col] * df_set[k][t_col]
            w_dt0 = df_set[k][d_col] * (1 - df_set[k][t_col])

            # influence function elements & estimator components
            eta_treat_pre = w_treat_pre * (df_set[k][y_col] - y_con) / np.mean(w_treat_pre)
            eta_treat_post = w_treat_post * (df_set[k][y_col] - y_con) / np.mean(w_treat_post)
            eta_con_pre = w_con_pre * (df_set[k][y_col] - y_con) / np.mean(w_con_pre)
            eta_con_post = w_con_post * (df_set[k][y_col] - y_con) / np.mean(w_con_post)

            # added locally efficient DRDID elements & components
            eta_d_post = w_d * (y_treat_post - y_con_post) / np.mean(w_d)
            eta_dt1_post = w_dt1 * (y_treat_post - y_con_post) / np.mean(w_dt1)
            eta_d_pre = w_d * (y_treat_pre - y_con_pre) / np.mean(w_d)
            eta_dt0_pre = w_dt0 * (y_treat_pre - y_con_pre) / np.mean(w_dt0)

            att = (eta_treat_post - eta_treat_pre) - (eta_con_post - eta_con_pre) + \
                  (eta_d_post - eta_dt1_post) - (eta_d_pre - eta_dt0_pre)
            att = np.mean(att)
            att_list.append(att)
            ps_list.append(ps_score_)
            or_list.append(or_score_)
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
            var = np.var(inf_func)
            var_list.append(var)
    att = np.mean(att_list)
    ps_score = np.mean(ps_list)
    or_score = np.mean(or_list)
    se = np.sqrt(np.mean(var_list) / len(df))
    # t-test
    t = att / se
    p = scipy.stats.t.sf(abs(t), len(df) - 2) * 2
    t_crit = np.abs(scipy.stats.t.ppf((1 - 0.95) / 2, len(df) - 2))
    ci = (att - t_crit * se, att + t_crit * se)
    result = {'ATT': att, 'S.E.': se, 't-stat': t, 'p-value': p, 'C.I.': ci, 'PS': ps_score, 'OR': or_score}
    return result


def dml_drdid(df: pd.DataFrame, ps_model, or_model, d_col: str, x_cols: list, y_col: list, t_col=None,
              eps=1e-5, n=1, rand=None, panel=True):
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
    - n : number of rounds
    - rand : seed for random number generation for replicability
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
        result = dml_drdid_panel(df, ps_model, or_model, y_col[0], y_col[1], d_col, x_cols, eps, n, rand)
    else:
        result = dml_drdid_rcs(df, ps_model, or_model, y_col[0], t_col, d_col, x_cols, eps, n, rand)
    return result
