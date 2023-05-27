# to test for misspecified PS and/or OR models by different approaches

import pandas as pd
from tqdm import tqdm
import itertools
import time
from sklearn.metrics import mean_squared_error
from DMLDiD import dmldid
from DRDID import drdid
from DML_DRDID import dml_drdid
from baseline import twfe, semi_did
from utils import simulate_data, model_selection, did_simulation, parse_parameter


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)

    # output of detailed results
    output_full = 'full_result_misspecs_0520.csv'
    output_summary = 'summary_result_misspecs_0520.csv'
    param_file_header = 'tuning summary/tuning_misspecs_summary_'
    true_att = 0
    n = 1000

    dct_names = {'GBM': ['n_estimators', 'max_depth', 'learning_rate'],
                 'RF': ['n_estimators', 'max_depth', 'min_samples_leaf'],
                 'MLP': ['hidden_layer_sizes', 'alpha']}

    # list of methods and models used
    dct_method = {'DMLDiD': dmldid, 'DRDID': drdid, 'DML_DRDID': dml_drdid, 'Abadie': semi_did, 'TWFE': twfe}

    data_settings = list(itertools.product([True, False], [(True, False), (False, True)]))
    model_settings = list(itertools.product(['DMLDiD', 'DRDID', 'DML_DRDID'], ['Linear', 'Lasso', 'GBM', 'RF', 'MLP']))
    model_settings.extend([('TWFE', 'N.A.'), ('Abadie', 'N.A.')])

    results = {'Bias': [], 'Cover': [], 'Length': [], 'Panel': [], 'mis_PS': [], 'mis_OR': [], 'Method': [], 'Model': []}
    for data_setting in data_settings:
        panel, (mis_ps, mis_or) = data_setting
        dct_best_param = parse_parameter(param_file_header + str(int(panel)) + str(1 - int(mis_ps)) + str(1 - int(mis_or)) + '.csv')
        print('panel? {} PS mis? {} OR mis? {}'.format(panel, mis_ps, mis_or))
        time.sleep(1)
        # run n rounds to account for randomness
        for _ in tqdm(range(n)):
            df = simulate_data(theta=true_att, panel=panel, hetero=False, unobs=False, mis_ps=mis_ps, mis_or=mis_or)
            for model_setting in model_settings:
                method_name, model = model_setting
                method = dct_method.get(method_name)
                ps_model, or_model = model_selection(model, None)
                if model in ['GBM', 'RF', 'MLP']:
                    param_names = dct_names.get(model)
                    params = dct_best_param.get((method_name, model))
                    dct_param = dict(zip(param_names, params))
                    ps_model.set_params(**dct_param)
                    or_model.set_params(**dct_param)
                try:
                    bias, cover, length, ps_score, or_score = did_simulation(df, ps_model, or_model, method, panel=panel)
                    results.get('Bias').append(bias)
                    results.get('Cover').append(cover)
                    results.get('Length').append(length)
                    results.get('Panel').append(panel)
                    results.get('mis_PS').append(mis_ps)
                    results.get('mis_OR').append(mis_or)
                    results.get('Method').append(method_name)
                    results.get('Model').append(model)
                except ValueError as e:
                    print('method: {}, model: {}, message: {}'.format(method_name, model, e))
                    continue
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_full, index=False)
    # get means
    df_summary = df_result.groupby(['Panel', 'mis_PS', 'mis_OR', 'Method', 'Model'])[['Bias', 'Cover', 'Length']].mean()
    df_median = df_result.groupby(['Panel', 'mis_PS', 'mis_OR', 'Method', 'Model'])['Bias'].median().to_frame('Median')
    # RMSE
    df_rmse = df_result.groupby(['Panel', 'mis_PS', 'mis_OR', 'Method', 'Model'])['Bias'].apply(list).\
        apply(lambda x: mean_squared_error([true_att] * len(x), x, squared=False)).to_frame('RMSE')
    df_summary = pd.merge(df_summary, df_median, left_index=True, right_index=True)
    df_summary = pd.merge(df_summary, df_rmse, left_index=True, right_index=True)
    df_summary.to_csv(output_summary)
    print(df_summary)
