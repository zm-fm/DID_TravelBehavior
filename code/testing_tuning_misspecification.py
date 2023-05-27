# to tune hyperparameters of GB, RF, and MLP for paper, with model misspecification

import pandas as pd
from tqdm import tqdm
import itertools
import time
from sklearn.metrics import mean_squared_error
from DMLDiD import dmldid
from DRDID import drdid
from DML_DRDID import dml_drdid
from utils import simulate_data, model_selection, did_simulation


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)

    # output of detailed results
    true_att = 0
    n = 100

    # list of methods and models used
    dct_method = {'DMLDiD': dmldid, 'DRDID': drdid, 'DML_DRDID': dml_drdid}
    # default first
    dct_params = {'GBM': list(itertools.product([100, 300, 500], [-1, 5, 10], [0.1, 0.05, 0.01])),
                  'RF': list(itertools.product([100, 300, 500], [None, 5, 10], [1, 2, 4])),
                  'MLP': list(itertools.product([(100,), (2, 1, 1), (10, 10, 5, 5, 2, 1)], [0.0001, 0.001, 0.01]))}
    dct_names = {'GBM': ['n_estimators', 'max_depth', 'learning_rate'],
                 'RF': ['n_estimators', 'max_depth', 'min_samples_leaf'],
                 'MLP': ['hidden_layer_sizes', 'alpha']}
    data_settings = list(itertools.product([True, False], [(True, False), (False, True)]))
    model_settings = list(itertools.product(['DMLDiD', 'DRDID', 'DML_DRDID'], ['GBM', 'RF', 'MLP']))
    for data_setting in data_settings:
        results = {'Bias': [], 'Cover': [], 'Length': [], 'PS': [], 'OR': [], 'Method': [], 'Model': [], 'Param': []}
        panel, (mis_ps, mis_or) = data_setting
        t = time.time()
        print('panel? {} PS mis? {} OR mis? {}'.format(panel, mis_ps, mis_or))

        for model_setting in model_settings:
            method_name, model = model_setting
            print('method {} model {}'.format(method_name, model))
            time.sleep(1)

            params = dct_params.get(model)
            names = dct_names.get(model)
            method = dct_method.get(method_name)
            for param in params:
                dct_param = dict(zip(names, param))
                ps_model, or_model = model_selection(model, None)
                ps_model.set_params(**dct_param)
                or_model.set_params(**dct_param)
                for _ in tqdm(range(n)):
                    df = simulate_data(theta=true_att, panel=panel, hetero=False, unobs=False, mis_ps=mis_ps, mis_or=mis_or)
                    try:
                        bias, cover, length, ps_score, or_score = did_simulation(df, ps_model, or_model, method, panel=panel)
                        results.get('Bias').append(bias)
                        results.get('Cover').append(cover)
                        results.get('Length').append(length)
                        results.get('PS').append(ps_score)
                        results.get('OR').append(or_score)
                        results.get('Method').append(method_name)
                        results.get('Model').append(model)
                        results.get('Param').append('-'.join(list(map(str, param))))
                    except ValueError as e:
                        print('method: {}, model: {}, message: {}'.format(method_name, model, e))
                        continue
        print('time spent: {}'.format(time.time() - t))
        df_result = pd.DataFrame(results)
        df_result.to_csv('tuning_misspecs_detailed_' + str(int(panel)) + str(1 - int(mis_ps)) + str(1 - int(mis_or)) + '.csv', index=False)
        # get means
        df_summary = df_result.groupby(['Method', 'Model', 'Param'])[['Bias', 'Cover', 'Length', 'PS', 'OR']].mean()
        df_median = df_result.groupby(['Method', 'Model', 'Param'])['Bias'].median().to_frame('Median')
        # RMSE
        df_rmse = df_result.groupby(['Method', 'Model', 'Param'])['Bias'].apply(list).\
            apply(lambda x: mean_squared_error([true_att] * len(x), x, squared=False)).to_frame('RMSE')
        df_summary = pd.merge(df_summary, df_median, left_index=True, right_index=True)
        df_summary = pd.merge(df_summary, df_rmse, left_index=True, right_index=True)
        df_summary.to_csv('tuning_misspecs_summary_' + str(int(panel)) + str(1 - int(mis_ps)) + str(1 - int(mis_or)) + '.csv')
