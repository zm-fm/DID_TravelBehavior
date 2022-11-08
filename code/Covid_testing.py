# to test DiD models with Covid data
# reference & data source
# Chauhan, R.S., Bhagat-Conway, M.W., Capasso da Silva, D. et al.
# A database of travel-related behaviors and attitudes before, during, and after COVID-19 in the United States.
# Sci Data 8, 245 (2021). https://doi.org/10.1038/s41597-021-01020-8

import pandas as pd
import numpy as np
import sys
import itertools
import time
from DMLDiD import dmldid
from DRDID import drdid
from DML_DRDID import dml_drdid
from baseline import twfe, semi_did


def did(df: pd.DataFrame, model: str, method, d_col: str, x_cols: list, y_cols: list, t_col=None, panel=True):
    """
    [input]
    - df : pd.DataFrame input data table
    - model : machine learning model used
    - method: estimator used
    - d_col : str column name of treatment variable
    - x_cols : list column names of control variables
    - y_col : list column names of outcome(s), single-element for rcs
    - t_col : str column name of time variable, None for panel
    - panel : boolean indicating if it is panel
    [return]
    - result : dictionary of estimated ATT, standard error, significance testing, and 95% confidence interval
    """
    if model == 'N.A.':
        result = method(df, d_col, x_cols, y_cols, t_col, panel=panel)
    else:
        result = method(df, d_col, x_cols, y_cols, t_col, model=model, panel=panel)
    return result


if __name__ == "__main__":
    pd.set_option('max_column', 500)
    pd.set_option('display.width', 500)
    np.set_printoptions(threshold=sys.maxsize)

    path = 'D:/中大工作/研究/GRF 2021/data/empirical data/dataverse_files/'
    # read processed data
    df = pd.read_csv(path + 'covid_pooled_processed.csv')
    # define X, Y0, Y1, D
    d_col = 'D_wfh'
    y_cols = [['pre_work_com_days', 'now_work_com_days'], ['pre_work_pri_time', 'now_work_pri_time']]
    y_cols.extend([['pre_mode_' + str(i + 1), 'now_mode_' + str(i + 1)] for i in range(5)])
    x_cols = ['hhveh_harm', 'hhsize', 'nchildren', 'tenure_harm', 'home_move', 'age', 'gender', 'studentjs',
              'driver', 'bike', 'att_covid_selfsevere', 'att_covid_stayhome', 'att_covid_commdisasters',
              'att_covid_overreact', 'att_wfh_likewfh', 'pre_work_com_dist']
    x_cols.extend(['race_' + str(i + 1) for i in range(5)])
    x_cols.extend(['inc_' + str(i + 1) for i in range(10)])
    x_cols.extend(['job_' + str(i + 1) for i in range(5)])
    x_cols.extend(['edu_' + str(i + 1) for i in range(4)])

    # list of methods and models used
    dct_method = {'DMLDiD': dmldid, 'DRDiD': drdid, 'DML_DRDiD': dml_drdid, 'Abadie': semi_did, 'TWFE': twfe}
    settings = list(itertools.product(['DMLDiD', 'DRDiD', 'DML_DRDiD'], ['Linear', 'Lasso', 'GB', 'RF', 'MLP']))
    settings.extend([('TWFE', 'N.A.'), ('Abadie', 'N.A.')])

    print('All workers')
    for y_col in y_cols:
        for setting in settings:
            t = time.time()
            method_name, model = setting
            print('Outcome: {}, Method: {}, ML model: {}'.format(y_col, method_name, model))
            method = dct_method.get(method_name)
            # TWFE changes data table, changing outcome0, outcome1 to Y1, Y0
            if method_name == 'TWFE':
                df_ = df.copy()
                result = did(df_, model, method, d_col, x_cols, y_col, panel=True)
            else:
                result = did(df, model, method, d_col, x_cols, y_col, panel=True)
            print(result)
            print('Time spent: {}'.format(time.time() - t))

    print('Only workers who went to work at least on one day')
    df = df[(df['pre_work_com_days'] > 0) & (df['now_work_com_days'] > 0)]
    for y_col in y_cols:
        for setting in settings:
            t = time.time()
            method_name, model = setting
            print('Outcome: {}, Method: {}, ML model: {}'.format(y_col, method_name, model))
            method = dct_method.get(method_name)
            # TWFE changes data table, changing outcome0, outcome1 to Y1, Y0
            if method_name == 'TWFE':
                df_ = df.copy()
                result = did(df_, model, method, d_col, x_cols, y_col, panel=True)
            else:
                result = did(df, model, method, d_col, x_cols, y_col, panel=True)
            print(result)
            print('Time spent: {}'.format(time.time() - t))
