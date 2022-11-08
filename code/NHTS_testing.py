# to test DiD models with Taxes NHTS data
# reference & data source
# U.S. Department of Transportation; Federal Highway Administration. (2009).
# 2009 National Household Travel Survey. http://nhts.ornl.gov
# U.S. Department of Transportation; Federal Highway Administration. (2017).
# 2017 National Household Travel Survey. http://nhts.ornl.gov

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

    path = '' # ignored for privacy
    # read data
    df = pd.read_csv(path + 'texas_combined.csv')
    # prepare variables
    for c in ['HHFAMINC', 'EDUC', 'OCCAT', 'LIF_CYC', 'PRMACT', 'FUELTYPE', 'VEHTYPE']:
        df[c] = df[c].apply(lambda x: 97 if x == -1 else x)
        df[c] = df[c].astype(str)
    for c in ['GCDWORK', 'CNTTDTR', 'MCUSED', 'NBIKETRP', 'NWALKTRP', 'PTUSED', 'TIMETOWK', 'WRKTRANS', 'YEARMILE', 'FUELECO']:
        df[c] = df[c].apply(lambda x: max(x, 0))
    for c in ['HOMEOWN', 'BORNINUS', 'WKFTPT', 'WORKER', 'DRIVER', 'FLEXTIME', 'URBRUR', 'R_SEX']:
        df[c] = df[c].apply(lambda x: 0 if x != 1 else 1)
    # one-hot coding
    df_dummy = pd.get_dummies(df[['HHFAMINC', 'EDUC', 'OCCAT', 'LIF_CYC', 'PRMACT', 'FUELTYPE', 'VEHTYPE']], drop_first=True)
    df = pd.merge(df, df_dummy, left_index=True, right_index=True)
    df.drop(columns=['HHFAMINC', 'EDUC', 'OCCAT', 'LIF_CYC', 'PRMACT', 'FUELTYPE', 'VEHTYPE'], inplace=True)
    # define X, Y, D, T
    x_cols = ['HHSIZE', 'HHVEHCNT', 'HOMEOWN', 'WRKCOUNT', 'HTPPOPDN', 'HTRESDN', 'URBRUR', 'GCDWORK',
              'BORNINUS', 'DRIVER', 'R_AGE', 'R_SEX', 'WORKER', 'SCHTYP']
    x_cols.extend([c for c in df.columns if 'HHFAMINC' in c or 'EDUC' in c or 'OCCAT' in c or 'LIF_CYC' in c
                   or 'FUELTYPE' in c or 'VEHTYPE' in c])
    y_cols = ['CNTTDTR', 'MCUSED', 'NBIKETRP', 'NWALKTRP', 'PTUSED', 'TIMETOWK', 'YEARMILE']
    d_col = 'D'
    t_col = 'T'

    # list of methods and models used
    dct_method = {'DMLDiD': dmldid, 'DRDiD': drdid, 'DML_DRDiD': dml_drdid, 'Abadie': semi_did, 'TWFE': twfe}
    settings = list(itertools.product(['DMLDiD', 'DRDiD', 'DML_DRDiD'], ['Linear', 'Lasso', 'GB', 'RF', 'MLP']))
    settings.extend([('TWFE', 'N.A.'), ('Abadie', 'N.A.')])

    for y_col in y_cols:
        for setting in settings:
            t = time.time()
            method_name, model = setting
            print('Outcome: {}, Method: {}, ML model: {}'.format(y_col, method_name, model))
            method = dct_method.get(method_name)
            result = did(df, model, method, d_col, x_cols, [y_col], t_col, panel=False)
            print(result)
            print('Time spent: {}'.format(time.time() - t))
