# DML_DiD in python
# reference: Chang, Neng Chieh. 2020.
# “Double/Debiased Machine Learning for Difference-in-Differences Models.”
# Econometrics Journal 23(2):177–91. doi: 10.1093/ectj/utaa001.


from sklearn.model_selection import train_test_split
import scipy.stats
import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")


def dmldid_panel(df: pd.DataFrame, ps_model, or_model, y0_col: str, y1_col: str, d_col: str, x_cols: list,
                 eps=1e-5, n=1, rand=None, timing=False):
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
    - timing: boolean indicating whether to report time elapsed
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    - (optional) time consumed in seconds
    """
    start = time.time()
    # model selection
    # split data into 2 partitions for cross-fitting
    df_set = train_test_split(df, random_state=rand, test_size=0.5)
    K = 2
    theta_list, ps_list, or_list = [], [], []
    for _ in range(n):
        theta_pair, ps_pair, or_pair = [], [], []
        for i in range(K):
            # 2 partitions 1 for model estimation other for value prediction
            k = 0 if i == 0 else 1
            c = 1 if i == 0 else 0
            # propensity score model
            ps_model.fit(df_set[c][x_cols], df_set[c][d_col])

            ps_score_ = ps_model.score(df_set[c][x_cols], df_set[c][d_col])
            # score calculation and trimming
            ghat = np.clip(ps_model.predict_proba(df_set[k][x_cols])[:, 1], eps, 1 - eps)

            y0 = df_set[c].query(f"{d_col} < 1")[y0_col]
            y1 = df_set[c].query(f"{d_col} < 1")[y1_col]
            y_ = y1 - y0
            x = df_set[c].query(f"{d_col} < 1")[x_cols]

            or_model.fit(x, y_)
            or_score_ = or_model.score(x, y_)
            lhat = or_model.predict(df_set[k][x_cols])

            phat = df_set[c][d_col].mean()

            theta_ = (df_set[k][y1_col] - df_set[k][y0_col] - lhat) * (df_set[k][d_col] - ghat) /\
                     (1 - ghat) / phat
            theta_pair.append(theta_.mean())
            ps_pair.append(ps_score_)
            or_pair.append(or_score_)
        theta_list.append(np.mean(theta_pair))
        ps_list.append(np.mean(ps_pair))
        or_list.append(np.mean(or_pair))

    # to compute standard error
    sd_list = []
    for _ in range(n):
        sd_pair = []
        for i in range(K):
            # 2 partitions 1 for model estimation other for value prediction
            k = 0 if i == 0 else 1
            c = 1 if i == 0 else 0
            # propensity score model
            ps_model.fit(df_set[c][x_cols], df_set[c][d_col])
            # score calculation and trimming
            ghat = np.clip(ps_model.predict_proba(df_set[k][x_cols])[:, 1], eps, 1 - eps)

            y0 = df_set[c].query(f"{d_col} < 1")[y0_col]
            y1 = df_set[c].query(f"{d_col} < 1")[y1_col]
            y_ = y1 - y0
            x = df_set[c].query(f"{d_col} < 1")[x_cols]

            or_model.fit(x, y_)
            lhat = or_model.predict(df_set[k][x_cols])

            phat = df_set[c][d_col].mean()

            G = - np.mean(theta_list) / phat

            s = (df_set[k][y1_col] - df_set[k][y0_col]) / phat * (df_set[k][d_col] - ghat) / (1 - ghat) - \
                (df_set[k][d_col] - ghat) / phat / (1 - ghat) * lhat - np.mean(theta_list) + \
                G * (df_set[k][d_col] - phat)

            sd_pair.append((s ** 2).mean())
        sd_list.append(np.mean(sd_pair))

    att = np.mean(theta_list)
    ps_score = np.mean(ps_list)
    or_score = np.mean(or_list)
    se = np.sqrt(np.mean(sd_list) / len(df))
    # t-test
    t = att / se
    p = scipy.stats.t.sf(abs(t), len(df) - 2) * 2
    t_crit = np.abs(scipy.stats.t.ppf((1 - 0.95) / 2, len(df) - 2))
    ci = (att - t_crit * se, att + t_crit * se)
    result = {'ATT': att, 'S.E.': se, 't-stat': t, 'p-value': p, 'C.I.': ci, 'PS': ps_score, 'OR': or_score}
    if timing:
        result['time'] = time.time() - start
    # return results
    return result


def dmldid_rcs(df: pd.DataFrame, ps_model, or_model, y_col: str, t_col: str, d_col: str, x_cols: list,
               eps=1e-5, n=1, rand=None, timing=False):
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
    - timing: boolean indicating whether to report time elapsed
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    - (optional) time consumed in seconds
    """
    start = time.time()
    # model selection
    # split data into 2 partitions for cross-fitting
    df_set = train_test_split(df, random_state=rand, test_size=0.5)
    K = 2
    theta_list, ps_list, or_list = [], [], []
    for _ in range(n):
        theta_pair, ps_pair, or_pair = [], [], []
        for i in range(K):
            # 2 partitions 1 for model estimation other for value prediction
            k = 0 if i == 0 else 1
            c = 1 if i == 0 else 0
            # propensity score model
            ps_model.fit(df_set[c][x_cols], df_set[c][d_col])

            ps_score_ = ps_model.score(df_set[c][x_cols], df_set[c][d_col])
            # score calculation and trimming
            ghat = np.clip(ps_model.predict_proba(df_set[k][x_cols])[:, 1], eps, 1 - eps)
            # mean value for time period
            lambda_ = df_set[c][t_col].mean()

            # setup for control-group outcome model
            y = df_set[c].query(f"{d_col} < 1")[y_col]
            t = df_set[c].query(f"{d_col} < 1")[t_col]
            y_ = (t - lambda_) * y
            x = df_set[c].query(f"{d_col} < 1")[x_cols]

            # control-group outcome change regression model
            or_model.fit(x, y_)
            or_score_ = or_model.score(x, y_)
            lhat = or_model.predict(df_set[k][x_cols])

            phat = df_set[c][d_col].mean()

            # ATT, each row is ATT for a single sample
            theta_ = ((df_set[k][t_col] - lambda_) * df_set[k][y_col] - lhat) * (df_set[k][d_col] - ghat) / \
                     (1 - ghat) / (lambda_ * (1 - lambda_)) / phat

            theta_pair.append(theta_.mean())
            ps_pair.append(ps_score_)
            or_pair.append(or_score_)
        theta_list.append(np.mean(theta_pair))
        ps_list.append(np.mean(ps_pair))
        or_list.append(np.mean(or_pair))

    # to compute standard error
    sd_list = []
    for _ in range(n):
        sd_pair = []
        for i in range(K):
            # first half same as ATT part
            k = 0 if i == 0 else 1
            c = 1 if i == 0 else 0

            ps_model.fit(df_set[c][x_cols], df_set[c][d_col])
            ghat = np.clip(ps_model.predict_proba(df_set[k][x_cols])[:, 1], eps, 1 - eps)
            lambda_ = df_set[c][t_col].mean()

            y = df_set[c].query(f"{d_col} < 1")[y_col]
            t = df_set[c].query(f"{d_col} < 1")[t_col]
            y_ = (t - lambda_) * y
            x = df_set[c].query(f"{d_col} < 1")[x_cols]

            or_model.fit(x, y_)
            lhat = or_model.predict(df_set[k][x_cols])

            phat = df_set[c][d_col].mean()

            # standard deviation estimation
            G = - (1 - 2 * lambda_) * np.mean(theta_list) / (lambda_ * (1 - lambda_)) - \
                (df_set[k][y_col] * (df_set[k][d_col] - ghat) / (1 - ghat) / (lambda_ * (1 - lambda_)) / phat).mean()
            s = ((df_set[k][t_col] - lambda_) * df_set[k][y_col] - lhat) * (df_set[k][d_col] - ghat) / \
                (1 - ghat) / phat / (lambda_ * (1 - lambda_)) - \
                df_set[k][d_col] * np.mean(theta_list) / phat + G * (df_set[k][t_col] - lambda_)

            # variance
            sd_pair.append((s ** 2).mean())
        sd_list.append(np.mean(sd_pair))

    att = np.mean(theta_list)
    ps_score = np.mean(ps_list)
    or_score = np.mean(or_list)
    se = np.sqrt(np.mean(sd_list) / len(df))
    # t-test
    t = att / se
    p = scipy.stats.t.sf(abs(t), len(df) - 2) * 2
    # get 95% confidence interval analytically
    t_crit = np.abs(scipy.stats.t.ppf((1 - 0.95) / 2, len(df) - 2))
    ci = (att - t_crit * se, att + t_crit * se)
    result = {'ATT': att, 'S.E.': se, 't-stat': t, 'p-value': p, 'C.I.': ci, 'PS': ps_score, 'OR': or_score}
    if timing:
        result['time'] = time.time() - start
    return result


def dmldid(df: pd.DataFrame, ps_model, or_model, d_col: str, x_cols: list, y_col: list, t_col=None,
           eps=1e-5, n=1, rand=None, timing=False, panel=True):
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
    - timing: boolean indicating whether to report time elapsed
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
        result = dmldid_panel(df, ps_model, or_model, y_col[0], y_col[1], d_col, x_cols, eps, n, rand, timing)
    else:
        result = dmldid_rcs(df, ps_model, or_model, y_col[0], t_col, d_col, x_cols, eps, n, rand, timing)
    return result
