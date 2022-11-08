# to test for misspecified PS or OR models by different approaches

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
from DGP import simulate_linear_hetero
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
    output_file = 'full_result_misspecs_1027.csv'
    true_theta = 0
    n = 1000

    dct_best_param = {(True, 1, 'DML_DRDiD', 'GBM'): [500, 5, 0.01], (True, 1, 'DML_DRDiD', 'KNN'): [10, 50],
                      (True, 1, 'DML_DRDiD', 'MLP'): [(100,), 0.0001], (True, 1, 'DML_DRDiD', 'RF'): [300, None, 1],
                      (True, 1, 'DMLDiD', 'GBM'): [300, 5, 0.01], (True, 1, 'DMLDiD', 'KNN'): [10, 50],
                      (True, 1, 'DMLDiD', 'MLP'): [(10, 10, 5, 5, 2, 1), 0.01], (True, 1, 'DMLDiD', 'RF'): [100, 5, 4],
                      (True, 1, 'DRDiD', 'GBM'): [100, -1, 0.1], (True, 1, 'DRDiD', 'KNN'): [5, 30],
                      (True, 1, 'DRDiD', 'MLP'): [(10, 10, 5, 5, 2, 1), 0.001], (True, 1, 'DRDiD', 'RF'): [500, None, 1],
                      (True, 10, 'DML_DRDiD', 'GBM'): [100, 5, 0.05], (True, 10, 'DML_DRDiD', 'KNN'): [10, 30],
                      (True, 10, 'DML_DRDiD', 'MLP'): [(100,), 0.001], (True, 10, 'DML_DRDiD', 'RF'): [500, None, 4],
                      (True, 10, 'DMLDiD', 'GBM'): [300, 10, 0.01], (True, 10, 'DMLDiD', 'KNN'): [10, 30],
                      (True, 10, 'DMLDiD', 'MLP'): [(100,), 0.0001], (True, 10, 'DMLDiD', 'RF'): [500, 10, 1],
                      (True, 10, 'DRDiD', 'GBM'): [100, 10, 0.05], (True, 10, 'DRDiD', 'KNN'): [10, 10],
                      (True, 10, 'DRDiD', 'MLP'): [(100,), 0.0001], (True, 10, 'DRDiD', 'RF'): [300, None, 4],
                      (False, 1, 'DML_DRDiD', 'GBM'): [100, 5, 0.05], (False, 1, 'DML_DRDiD', 'KNN'): [10, 10],
                      (False, 1, 'DML_DRDiD', 'MLP'): [(100,), 0.001], (False, 1, 'DML_DRDiD', 'RF'): [100, 10, 4],
                      (False, 1, 'DMLDiD', 'GBM'): [100, 10, 0.01], (False, 1, 'DMLDiD', 'KNN'): [10, 50],
                      (False, 1, 'DMLDiD', 'MLP'): [(100,), 0.01], (False, 1, 'DMLDiD', 'RF'): [300, 5, 2],
                      (False, 1, 'DRDiD', 'GBM'): [300, 10, 0.05], (False, 1, 'DRDiD', 'KNN'): [1, 30],
                      (False, 1, 'DRDiD', 'MLP'): [(100,), 0.01], (False, 1, 'DRDiD', 'RF'): [100, 5, 2],
                      (False, 10, 'DML_DRDiD', 'GBM'): [100, -1, 0.05], (False, 10, 'DML_DRDiD', 'KNN'): [10, 30],
                      (False, 10, 'DML_DRDiD', 'MLP'): [(100,), 0.01], (False, 10, 'DML_DRDiD', 'RF'): [100, 5, 2],
                      (False, 10, 'DMLDiD', 'GBM'): [100, 10, 0.01], (False, 10, 'DMLDiD', 'KNN'): [10, 10],
                      (False, 10, 'DMLDiD', 'MLP'): [(10, 10, 5, 5, 2, 1), 0.001], (False, 10, 'DMLDiD', 'RF'): [300, 10, 1],
                      (False, 10, 'DRDiD', 'GBM'): [300, 10, 0.01], (False, 10, 'DRDiD', 'KNN'): [10, 50],
                      (False, 10, 'DRDiD', 'MLP'): [(100,), 0.01], (False, 10, 'DRDiD', 'RF'): [500, 5, 4]}
    dct_names = {'GBM': ['n_estimators', 'max_depth', 'learning_rate'],
                 'RF': ['n_estimators', 'max_depth', 'min_samples_leaf'],
                 'KNN': ['n_neighbors', 'leaf_size'],
                 'MLP': ['hidden_layer_sizes', 'alpha']}

    # list of methods and models used
    dct_method = {'DMLDiD': dmldid, 'DRDiD': drdid, 'DML_DRDiD': dml_drdid, 'Abadie': semi_did, 'TWFE': twfe}
    settings = list(itertools.product(['DMLDiD', 'DRDiD', 'DML_DRDiD'], ['GBM', 'RF', 'KNN', 'MLP']))
    # settings.extend([('TWFE', 'N.A.'), ('Abadie', 'N.A.')])

    results = {'Bias': [], 'Cover': [], 'Length': [], 'Panel': [], 'Specs': [], 'Method': [], 'Model': []}
    for panel in [True, False]:
        data_type = 'Panel' if panel else 'RCS'
        for (mis_ps, mis_or) in [(True, False), (False, True)]:
            specs = (1 - int(mis_ps)) * 10 + (1 - int(mis_or))
            print('{} specification: {}'.format(data_type, specs))
            time.sleep(1)
            # run n rounds to account for randomness
            for _ in tqdm(range(n)):
                df = simulate_linear_hetero(mis_ps=mis_ps, mis_or=mis_or, panel=panel)
                for setting in settings:
                    method_name, model = setting
                    method = dct_method.get(method_name)
                    param_names = dct_names.get(model)
                    params = dct_best_param.get((panel, specs, method_name, model))
                    dct_param = dict(zip(param_names, params))
                    ps_model, or_model = model_selection(model, None, **dct_param)
                    bias, cover, length = did_simulation(df, ps_model, or_model, model, method, panel=panel)
                    results.get('Bias').append(bias)
                    results.get('Cover').append(cover)
                    results.get('Length').append(length)
                    results.get('Panel').append(data_type)
                    results.get('Specs').append(specs)
                    results.get('Method').append(method_name)
                    results.get('Model').append(model)
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_file, index=False)
    # get means
    df_summary = df_result.groupby(['Panel', 'Specs', 'Method', 'Model'])[['Bias', 'Cover', 'Length']].mean()
    # RMSE
    df_rmse = df_result.groupby(['Panel', 'Specs', 'Method', 'Model'])['Bias'].apply(list).\
        apply(lambda x: mean_squared_error([true_theta] * len(x), x, squared=False)).to_frame('RMSE')
    df_summary = pd.merge(df_summary, df_rmse, left_index=True, right_index=True)
    print(df_summary)
