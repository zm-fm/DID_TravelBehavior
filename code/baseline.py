# to implement baseline DiD estimators
# two-way fixed effect regression
# semi-parametric estimator of Abadie (2005)

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LassoCV
import pandas as pd
import statsmodels.api as sm
import scipy.stats


def twfe(df: pd.DataFrame, d_col: str, x_cols: list, y_cols: list, t_col=None, panel=True):
    """
    [input]
    - df : pd.DataFrame input data table
    - d_col : str column name of treatment variable
    - x_cols : list column names of control variables
    - y_cols : list column names of outcome(s), single-element for rcs
    - t_col : str column name of time variable, None for panel
    - panel : boolean indicating if it is panel
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """
    if panel and t_col is not None:
        raise ValueError('T column should be None for panel model.')
    if panel and len(y_cols) != 2:
        raise ValueError('Y1 and Y0 should be specified in Y column list for panel model.')
    if not panel and t_col is None:
        raise ValueError('T column should not be None for repeated cross-section model.')
    if not panel and len(y_cols) != 1:
        raise ValueError('Y column list only needs one column name.')
    # change panel (wide) to long form
    if panel:
        if y_cols != ['Y0', 'Y1']:
            df.rename(columns=dict(zip(y_cols, ['Y0', 'Y1'])), inplace=True)
        df['id'] = df.index
        df = pd.wide_to_long(df, 'Y', i='id', j='T')
        df.reset_index(inplace=True)
        t_col = 'T'
        y_cols = ['Y']
    # interaction term of D and T
    df['DT'] = df[t_col] * df[d_col]

    # construct all Xs, including D, T, and DT
    x_ = x_cols.copy()
    x_.append(t_col)
    x_.append(d_col)
    x_.append('DT')

    # add constant term
    X = sm.add_constant(df[x_])

    # OLS regression
    model = sm.OLS(df[y_cols[0]], X)
    output = model.fit()

    att = output.params['DT']
    se = output.bse['DT']
    t = att / se
    p = output.pvalues['DT']
    ci = (output.conf_int().loc['DT'][0], output.conf_int().loc['DT'][1])
    result = {'ATT': att, 'S.E.': se, 't-stat': t, 'p-value': p, 'C.I.': ci}
    return result


def semi_did_panel(df: pd.DataFrame, y0_col: str, y1_col: str, d_col: str, x_cols: list, eps=1e-5, rand=None):
    """
    [input]
    - df : pd.DataFrame input data table
    - y0_col : str column name of outcome at time 0
    - y1_col : str column name of outcome at time 1
    - d_col : str column name of treatment variable
    - x_cols : list column names of control variables
    - eps : threshold for propensity score trimming
    - rand : seed for random number generation for replicability
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """

    # PS model
    ps_model = LogisticRegressionCV(cv=5, random_state=rand, penalty="l1", solver="saga", n_jobs=4)
    ps_model.fit(df[x_cols], df[d_col])
    ghat = np.clip(ps_model.predict_proba(df[x_cols])[:, 1], eps, 1 - eps)
    phat = df[d_col].mean()

    # ATT estimate
    att = ((df[y1_col] - df[y0_col]) / phat * (df[d_col] - ghat) / (1 - ghat)).mean()

    y0 = df[df[d_col] == 0][y0_col]
    y1 = df[df[d_col] == 0][y1_col]
    y = y1 - y0
    x = df[df[d_col] == 0][x_cols]

    # OR model, only for S.E. estimation
    l_model = LassoCV(cv=5, random_state=rand, n_jobs=4)
    l_model.fit(x, y)
    lhat = l_model.predict(df[x_cols])

    G = - att / phat
    var = (((df[y1_col] - df[y0_col]) / phat * (df[d_col] - ghat) / (1 - ghat) -
            (df[d_col] - ghat) / phat / (1 - ghat) * lhat - att + G * (df[d_col] - phat)) ** 2).mean()
    se = np.sqrt(var / len(df))
    t = att / se
    p = scipy.stats.t.sf(abs(t), len(df) - 2) * 2
    t_crit = np.abs(scipy.stats.t.ppf((1 - 0.95) / 2, len(df) - 2))
    ci = (att - t_crit * se, att + t_crit * se)
    result = {'ATT': att, 'S.E.': se, 't-stat': t, 'p-value': p, 'C.I.': ci}
    return result


def semi_did_rcs(df: pd.DataFrame, y_col: str, t_col: str, d_col: str, x_cols: list, eps=1e-5, rand=None):
    """
    [input]
    - df : pd.DataFrame input data table
    - y_col : str column name of outcome
    - t_col : str column name of time variable
    - d_col : str column name of treatment variable
    - x_cols : list column names of control variables
    - eps : threshold for propensity score trimming
    - rand : seed for random number generation for replicability
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """

    # PS model
    ps_model = LogisticRegressionCV(cv=5, random_state=rand, penalty="l1", solver="saga", n_jobs=4)
    ps_model.fit(df[x_cols], df[d_col])
    ghat = np.clip(ps_model.predict_proba(df[x_cols])[:, 1], eps, 1 - eps)
    phat = df[d_col].mean()
    lambda_ = df[t_col].mean()

    phi = (df[t_col] - lambda_) * (df[d_col] - ghat) / lambda_ / (1 - lambda_) / ghat / (1 - ghat)

    # ATT estimate
    att = (df[y_col] * phi * ghat / phat).mean()

    y = (df[df[d_col] == 0][t_col] - lambda_) * df[df[d_col] == 0][y_col]
    x = df[df[d_col] == 0][x_cols]

    # OR model, only for S.E. estimation
    l_model = LassoCV(cv=5, random_state=rand, n_jobs=4)
    l_model.fit(x, y)
    lhat = l_model.predict(df[x_cols])

    G = - (1 - 2 * lambda_) * att / (lambda_ * (1 - lambda_)) - \
        (df[y_col] * (df[d_col]-ghat) / (1-ghat) / (lambda_ * (1 - lambda_)) / phat).mean()
    var = ((((df[t_col] - lambda_) * df[y_col] - lhat) * (df[d_col] - ghat) / (1 - ghat) / phat /
            (lambda_ * (1 - lambda_)) - df[d_col] * att / phat + G * (df[t_col] - lambda_)) ** 2).mean()
    se = np.sqrt(var / len(df))
    t = att / se
    p = scipy.stats.t.sf(abs(t), len(df) - 2) * 2
    t_crit = np.abs(scipy.stats.t.ppf((1 - 0.95) / 2, len(df) - 2))
    ci = (att - t_crit * se, att + t_crit * se)
    result = {'ATT': att, 'S.E.': se, 't-stat': t, 'p-value': p, 'C.I.': ci}
    return result


def semi_did(df: pd.DataFrame, d_col: str, x_cols: list, y_cols: list, t_col=None, eps=1e-5, rand=None, panel=True):
    """
    [input]
    - df : pd.DataFrame input data table
    - d_col : str column name of treatment variable
    - x_cols : list column names of control variables
    - y_cols : list column names of outcome(s), single-element for rcs
    - t_col : str column name of time variable, None for panel
    - eps : threshold for propensity score trimming
    - rand : seed for random number generation for replicability
    - panel : boolean indicating if it is panel
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """

    # wrapper function of panel and rcs models
    if panel and t_col is not None:
        raise Warning('T column should be None for panel model, ignored.')
    if panel and len(y_cols) != 2:
        raise ValueError('Y1 and Y0 should be specified in Y column list for panel model.')
    if not panel and t_col is None:
        raise ValueError('T column should not be None for repeated cross-section model.')
    if not panel and len(y_cols) != 1:
        raise ValueError('Y column list only needs one column name.')
    if panel:
        result = semi_did_panel(df, y_cols[0], y_cols[1], d_col, x_cols, eps, rand)
    else:
        result = semi_did_rcs(df, y_cols[0], t_col, d_col, x_cols, eps, rand)
    return result
