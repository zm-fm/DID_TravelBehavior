# to loosely tune hyperparameters of GB, RF, and MLP

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import time
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from DMLDiD import dmldid
from DRDID import drdid
from DML_DRDID import dml_drdid
from utils import model_selection
from DGP import simulate_data, simulate_linear_hetero


def did_simulation(df: pd.DataFrame, ps_model, or_model, method, true_att=0, panel=True):
    """
    [input]
    - df : pd.DataFrame input data table
    - model : machine learning model used
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

    result = method(df, ps_model, or_model, d_col, x_cols, y_cols, t_col, panel=panel)
    return result.get('ATT') - true_att, \
           int(result.get('C.I.')[0] < true_att < result.get('C.I.')[1]), \
           np.abs(result.get('C.I.')[1] - result.get('C.I.')[0]), result.get('PS'), result.get('OR')


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    plt.style.use('ggplot')

    # output of detailed results
    true_att = 0
    n = 100

    # list of methods and models used
    dct_method = {'DMLDiD': dmldid, 'DRDiD': drdid, 'DML_DRDiD': dml_drdid}
    settings = list(itertools.product(['DMLDiD', 'DRDiD', 'DML_DRDiD'], ['GBM', 'RF', 'MLP']))
    # default first
    dct_params = {'GBM': list(itertools.product([100, 300, 500], [-1, 5, 10], [0.1, 0.05, 0.01])),
                  'RF': list(itertools.product([100, 300, 500], [None, 5, 10], [1, 2, 4])),
                  'MLP': list(itertools.product([(100,), (2, 1, 1), (10, 10, 5, 5, 2, 1)], [0.0001, 0.001, 0.01]))}
    dct_names = {'GBM': ['n_estimators', 'max_depth', 'learning_rate'],
                 'RF': ['n_estimators', 'max_depth', 'min_samples_leaf'],
                 'MLP': ['hidden_layer_sizes', 'alpha']}
    for panel in [False, True]:
        for code in range(4):
            t = time.time()
            print('panel? {} scenario {}'.format(panel, code))
            results = {'Bias': [], 'Cover': [], 'Length': [], 'PS': [], 'OR': [], 'Method': [], 'Model': [], 'Param': []}
            for setting in settings:
                method_name, model = setting
                print('method {} model {}'.format(method_name, model))
                time.sleep(1)

                method = dct_method.get(method_name)
                params = dct_params.get(model)
                names = dct_names.get(model)
                for param in params:
                    dct_param = dict(zip(names, param))
                    ps_model, or_model = model_selection(model, None, **dct_param)
                    # run n rounds to account for randomness
                    for _ in tqdm(range(n)):
                        df = simulate_data(code=code, theta=true_att, panel=panel)
                        bias, cover, length, ps_score, or_score = did_simulation(df, ps_model, or_model, method, panel=panel)
                        results.get('Bias').append(bias)
                        results.get('Cover').append(cover)
                        results.get('Length').append(length)
                        results.get('PS').append(ps_score)
                        results.get('OR').append(or_score)
                        results.get('Method').append(method_name)
                        results.get('Model').append(model)
                        results.get('Param').append('-'.join(list(map(str, param))))
            df_result = pd.DataFrame(results)
            df_result.to_csv('tuning_' + str(panel) + str(code) + '.csv', index=False)
            # get means
            df_summary = df_result.groupby(['Method', 'Model', 'Param'])[['Bias', 'Cover', 'Length', 'PS', 'OR']].mean()
            # RMSE
            df_rmse = df_result.groupby(['Method', 'Model', 'Param'])['Bias'].apply(list).\
                apply(lambda x: mean_squared_error([true_att] * len(x), x, squared=False)).to_frame('RMSE')
            df_summary = pd.merge(df_summary, df_rmse, left_index=True, right_index=True)
            print(df_summary)
            print('time spent: {}'.format(time.time() - t))

    settings = list(itertools.product(['DMLDiD', 'DRDiD', 'DML_DRDiD'], ['GBM', 'RF', 'KNN', 'MLP'],
                                      [(True, False), (False, True)]))
    for panel in [False, True]:
        t = time.time()
        print('panel? {}'.format(panel))
        results = {'Bias': [], 'Cover': [], 'Length': [], 'PS': [], 'OR': [],
                   'Specs': [], 'Method': [], 'Model': [], 'Param': []}
        for setting in settings:
            method_name, model, (mis_ps, mis_or) = setting
            specs = (1 - int(mis_ps)) * 10 + (1 - int(mis_or))
            print('method {} model {} specs {}'.format(method_name, model, specs))
            time.sleep(1)

            params = dct_params.get(model)
            names = dct_names.get(model)
            method = dct_method.get(method_name)
            for param in params:
                dct_param = dict(zip(names, param))
                ps_model, or_model = model_selection(model, None, **dct_param)
                for _ in tqdm(range(n)):
                    df = simulate_linear_hetero(mis_ps=mis_ps, mis_or=mis_or, panel=panel)
                    bias, cover, length, ps_score, or_score = did_simulation(df, ps_model, or_model, method, panel=panel)
                    results.get('Bias').append(bias)
                    results.get('Cover').append(cover)
                    results.get('Length').append(length)
                    results.get('PS').append(ps_score)
                    results.get('OR').append(or_score)

                    results.get('Specs').append(specs)
                    results.get('Method').append(method_name)
                    results.get('Model').append(model)
                    results.get('Param').append('-'.join(list(map(str, param))))
        df_result = pd.DataFrame(results)
        df_result.to_csv('tuning_misspecs_' + str(panel), index=False)
        # get means
        df_summary = df_result.groupby(['Specs', 'Method', 'Model', 'Param'])[['Bias', 'Cover', 'Length', 'PS', 'OR']].mean()
        # RMSE
        df_rmse = df_result.groupby(['Specs', 'Method', 'Model', 'Param'])['Bias'].apply(list).\
            apply(lambda x: mean_squared_error([true_att] * len(x), x, squared=False)).to_frame('RMSE')
        df_summary = pd.merge(df_summary, df_rmse, left_index=True, right_index=True)
        print(df_summary)
        print('time spent: {}'.format(time.time() - t))
