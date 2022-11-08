# simulation analysis

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
from baseline import twfe, semi_did
from DGP import simulate_data
from utils import model_selection


def did_simulation(df: pd.DataFrame, ps_model, or_model, model: str, method, true_att=0, panel=True):
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
    if model == 'N.A.':
        result = method(df, ps_model, or_model, d_col, x_cols, y_cols, t_col, panel=panel)
    else:
        result = method(df, ps_model, or_model, d_col, x_cols, y_cols, t_col, panel=panel)
    return result.get('ATT') - true_att, \
           int(result.get('C.I.')[0] < true_att < result.get('C.I.')[1]), \
           np.abs(result.get('C.I.')[1] - result.get('C.I.')[0])


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    plt.style.use('ggplot')

    # output of detailed results
    output_file = 'full_result_scenes_1027.csv'
    true_att = 0
    n = 1000

    # list of methods and models used
    dct_best_param_rcs = {(0, 'DML_DRDiD', 'GBM'): [100, 5, 0.01], (0, 'DML_DRDiD', 'KNN'): [1, 10],
                          (0, 'DML_DRDiD', 'MLP'): [(2, 1, 1), 0.01], (0, 'DML_DRDiD', 'RF'): [500, None, 1],
                          (0, 'DMLDiD', 'GBM'): [500, 10, 0.01], (0, 'DMLDiD', 'KNN'): [10, 10],
                          (0, 'DMLDiD', 'MLP'): [(2, 1, 1), 0.01], (0, 'DMLDiD', 'RF'): [100, 5, 4],
                          (0, 'DRDiD', 'GBM'): [500, -1, 0.1], (0, 'DRDiD', 'KNN'): [10, 10],
                          (0, 'DRDiD', 'MLP'): [(2, 1, 1), 0.0001], (0, 'DRDiD', 'RF'): [500, None, 2],
                          (1, 'DML_DRDiD', 'GBM'): [500, 5, 0.05], (1, 'DML_DRDiD', 'KNN'): [1, 50],
                          (1, 'DML_DRDiD', 'MLP'): [(100,), 0.0001], (1, 'DML_DRDiD', 'RF'): [500, None, 2],
                          (1, 'DMLDiD', 'GBM'): [100, -1, 0.01], (1, 'DMLDiD', 'KNN'): [10, 50],
                          (1, 'DMLDiD', 'MLP'): [(100,), 0.001], (1, 'DMLDiD', 'RF'): [300, 5, 4],
                          (1, 'DRDiD', 'GBM'): [300, 5, 0.1], (1, 'DRDiD', 'KNN'): [10, 10],
                          (1, 'DRDiD', 'MLP'): [(100,), 0.001], (1, 'DRDiD', 'RF'): [500, None, 1],
                          (2, 'DML_DRDiD', 'GBM'): [100, -1, 0.01], (2, 'DML_DRDiD', 'KNN'): [1, 30],
                          (2, 'DML_DRDiD', 'MLP'): [(100,), 0.001], (2, 'DML_DRDiD', 'RF'): [100, 5, 4],
                          (2, 'DMLDiD', 'GBM'): [100, 5, 0.01], (2, 'DMLDiD', 'KNN'): [10, 50],
                          (2, 'DMLDiD', 'MLP'): [(100,), 0.001], (2, 'DMLDiD', 'RF'): [500, 5, 1],
                          (2, 'DRDiD', 'GBM'): [500, 5, 0.1], (2, 'DRDiD', 'KNN'): [1, 30],
                          (2, 'DRDiD', 'MLP'): [(100,), 0.0001], (2, 'DRDiD', 'RF'): [100, 10, 4],
                          (3, 'DML_DRDiD', 'GBM'): [500, 10, 0.01], (3, 'DML_DRDiD', 'KNN'): [10, 10],
                          (3, 'DML_DRDiD', 'MLP'): [(2, 1, 1), 0.01], (3, 'DML_DRDiD', 'RF'): [100, 10, 1],
                          (3, 'DMLDiD', 'GBM'): [100, 5, 0.01], (3, 'DMLDiD', 'KNN'): [5, 30],
                          (3, 'DMLDiD', 'MLP'): [(100,), 0.0001], (3, 'DMLDiD', 'RF'): [100, 5, 1],
                          (3, 'DRDiD', 'GBM'): [500, -1, 0.1], (3, 'DRDiD', 'KNN'): [1, 10],
                          (3, 'DRDiD', 'MLP'): [(2, 1, 1), 0.0001], (3, 'DRDiD', 'RF'): [300, 10, 2]}
    dct_best_param_panel = {(0, 'DML_DRDiD', 'GBM'): [100, -1, 0.05], (0, 'DML_DRDiD', 'KNN'): [10, 50],
                            (0, 'DML_DRDiD', 'MLP'): [(2, 1, 1), 0.0001], (0, 'DML_DRDiD', 'RF'): [100, 5, 4],
                            (0, 'DMLDiD', 'GBM'): [100, -1, 0.01], (0, 'DMLDiD', 'KNN'): [10, 50],
                            (0, 'DMLDiD', 'MLP'): [(2, 1, 1), 0.01], (0, 'DMLDiD', 'RF'): [100, 5, 1],
                            (0, 'DRDiD', 'GBM'): [300, -1, 0.05], (0, 'DRDiD', 'KNN'): [10, 50],
                            (0, 'DRDiD', 'MLP'): [(2, 1, 1), 0.0001], (0, 'DRDiD', 'RF'): [100, 10, 4],
                            (1, 'DML_DRDiD', 'GBM'): [100, -1, 0.01], (1, 'DML_DRDiD', 'KNN'): [10, 50],
                            (1, 'DML_DRDiD', 'MLP'): [(100,), 0.001], (1, 'DML_DRDiD', 'RF'): [500, 10, 4],
                            (1, 'DMLDiD', 'GBM'): [300, 5, 0.01], (1, 'DMLDiD', 'KNN'): [10, 30],
                            (1, 'DMLDiD', 'MLP'): [(100,), 0.001], (1, 'DMLDiD', 'RF'): [100, 10, 2],
                            (1, 'DRDiD', 'GBM'): [500, 5, 0.1], (1, 'DRDiD', 'KNN'): [5, 10],
                            (1, 'DRDiD', 'MLP'): [(2, 1, 1), 0.01], (1, 'DRDiD', 'RF'): [500, 5, 1],
                            (2, 'DML_DRDiD', 'GBM'): [500, 5, 0.01], (2, 'DML_DRDiD', 'KNN'): [10, 50],
                            (2, 'DML_DRDiD', 'MLP'): [(100,), 0.0001], (2, 'DML_DRDiD', 'RF'): [300, 5, 2],
                            (2, 'DMLDiD', 'GBM'): [500, -1, 0.01], (2, 'DMLDiD', 'KNN'): [5, 10],
                            (2, 'DMLDiD', 'MLP'): [(100,), 0.0001], (2, 'DMLDiD', 'RF'): [300, 5, 1],
                            (2, 'DRDiD', 'GBM'): [300, 5, 0.01], (2, 'DRDiD', 'KNN'): [1, 50],
                            (2, 'DRDiD', 'MLP'): [(100,), 0.01], (2, 'DRDiD', 'RF'): [100, 10, 2],
                            (3, 'DML_DRDiD', 'GBM'): [300, 10, 0.01], (3, 'DML_DRDiD', 'KNN'): [10, 50],
                            (3, 'DML_DRDiD', 'MLP'): [(100,), 0.001], (3, 'DML_DRDiD', 'RF'): [100, 5, 4],
                            (3, 'DMLDiD', 'GBM'): [300, 5, 0.01], (3, 'DMLDiD', 'KNN'): [10, 50],
                            (3, 'DMLDiD', 'MLP'): [(100,), 0.0001], (3, 'DMLDiD', 'RF'): [100, 5, 1],
                            (3, 'DRDiD', 'GBM'): [100, 5, 0.01], (3, 'DRDiD', 'KNN'): [5, 10],
                            (3, 'DRDiD', 'MLP'): [(100,), 0.01], (3, 'DRDiD', 'RF'): [100, 5, 4]}

    dct_best_param = {True: dct_best_param_panel, False: dct_best_param_rcs}

    dct_names = {'GBM': ['n_estimators', 'max_depth', 'learning_rate'],
                 'RF': ['n_estimators', 'max_depth', 'min_samples_leaf'],
                 'KNN': ['n_neighbors', 'leaf_size'],
                 'MLP': ['hidden_layer_sizes', 'alpha']}
    dct_method = {'DMLDiD': dmldid, 'DRDiD': drdid, 'DML_DRDiD': dml_drdid, 'Abadie': semi_did, 'TWFE': twfe}
    settings = list(itertools.product(['DMLDiD', 'DRDiD', 'DML_DRDiD'], ['GBM', 'RF', 'KNN', 'MLP']))
    # settings.extend([('TWFE', 'N.A.'), ('Abadie', 'N.A.')])
    results = {'Bias': [], 'Cover': [], 'Length': [], 'DGP': [], 'Panel': [], 'Method': [], 'Model': []}
    # first loop through all 4 DGPs
    for code in range(4):
        for panel in [True, False]:
            # # todo bypass rcs scenarios for now
            # if not panel:
            #     continue
            data_type = 'Panel' if panel else 'RCS'
            print('DGP {} {}'.format(code, data_type))
            time.sleep(1)
            # run n rounds to account for randomness
            for _ in tqdm(range(n)):
                df = simulate_data(code=code, theta=true_att, panel=panel)
                for setting in settings:
                    method_name, model = setting
                    method = dct_method.get(method_name)
                    param_names = dct_names.get(model)
                    params = dct_best_param.get(panel).get((code, method_name, model))
                    dct_param = dict(zip(param_names, params))
                    ps_model, or_model = model_selection(model, None, **dct_param)
                    bias, cover, length = did_simulation(df, ps_model, or_model, model, method, panel=panel)
                    results.get('Bias').append(bias)
                    results.get('Cover').append(cover)
                    results.get('Length').append(length)
                    results.get('DGP').append(code)
                    results.get('Panel').append(data_type)
                    results.get('Method').append(method_name)
                    results.get('Model').append(model)
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_file, index=False)
    # get means
    df_summary = df_result.groupby(['DGP', 'Panel', 'Method', 'Model'])[['Bias', 'Cover', 'Length']].mean()
    # RMSE
    df_rmse = df_result.groupby(['DGP', 'Panel', 'Method', 'Model'])['Bias'].apply(list).\
        apply(lambda x: mean_squared_error([true_att] * len(x), x, squared=False)).to_frame('RMSE')
    df_summary = pd.merge(df_summary, df_rmse, left_index=True, right_index=True)
    print(df_summary)
