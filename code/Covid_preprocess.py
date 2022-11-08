# to process Covid panal data for case study
# reference & data source
# Chauhan, R.S., Bhagat-Conway, M.W., Capasso da Silva, D. et al.
# A database of travel-related behaviors and attitudes before, during, and after COVID-19 in the United States.
# Sci Data 8, 245 (2021). https://doi.org/10.1038/s41597-021-01020-8

import pandas as pd
import numpy as np
import sys

pd.set_option('max_column', 500)
pd.set_option('display.width', 500)
np.set_printoptions(threshold=sys.maxsize)

path = '' # ignored for privacy

df = pd.read_csv(path + 'wave1full/covid_pooled_public_1.0.0.csv', low_memory=False)

# data recoding dictionaries
dct_inc = {'Less than $10,000': 0, '$10,000 to $14,999': 1, '$15,000 to $24,999': 2, '$25,000 to $34,999': 3,
           '$35,000 to $49,999': 4, '$50,000 to $74,999': 5, '$75,000 to $99,999': 6, '$100,000 to $124,999': 7,
           '$125,000 to $149,999': 8, '$150,000 to $199,999': 9, '$200,000 or more': 10}
dct_job = {'I prefer not to answer': 0, 'Question not displayed to respondent': 0,
           'Manufacturing, construction, maintenance, or farming (does not includes maintenance and farming for those who answered jobcat_pre_w2)': 1,
           'Professional, managerial, or technical': 2, 'Clerical or administrative support': 3, 'Sales or service': 4,
           'Something else': 5}
dct_edu = {'Some grade/high school': 0, 'Completed high school or GED': 1, 'Some college or technical school': 2,
           'Bachelor\'s degree(s) or some graduate school': 3, 'Completed graduate degree(s)': 4}
dct_mode = {'Question not displayed to respondent': 0, 'Private vehicle': 1, 'Transit': 2,
            'Bicycle (or scooter in Wave 1B)': 3, 'Walk': 4, 'Other mode': 5}
# worker_now    no   yes
# worker_pre
# no          3175    89
# yes          726  4733

# data processing steps
df['now_sch_com_days'] = df[['now_sch_com_days_w1a', 'apr_sch_com_days_w1b']].max(axis=1)
df['now_sch_pri_time'] = df[['now_sch_pri_time_w1a', 'apr_sch_pri_time_w1b']].max(axis=1)
df.loc[(df['pre_work_com_days'].isna()) & (df['worker_pre'] == 'yes'), 'pre_work_com_days'] = df['pre_sch_com_days']
df.loc[(df['now_work_com_days'].isna()) & (df['worker_now'] == 'yes'), 'now_work_com_days'] = df['now_sch_com_days']

df.loc[(df['pre_work_pri_time'].isna()) & (df['worker_pre'] == 'yes'), 'pre_work_pri_time'] = df['pre_sch_pri_time']
df.loc[(df['now_work_pri_time'].isna()) & (df['worker_now'] == 'yes'), 'now_work_pri_time'] = df['now_sch_pri_time']

df.loc[(df['pre_work_pri_mode_harm'] == 'Question not displayed to respondent') & (df['worker_pre'] == 'yes'),
       'pre_work_pri_mode_harm'] = df['pre_sch_pri_mode_harm']
df.loc[(df['now_work_pri_mode_harm'] == 'Question not displayed to respondent') & (df['worker_now'] == 'yes'),
       'now_work_pri_mode_harm'] = df['now_sch_pri_mode_harm']
df.loc[(df['pre_work_com_dist'].isna()) & (df['worker_now'] == 'yes'), 'pre_work_com_dist'] = df['pre_sch_com_dist']

df = df[['resp_id', 'worker_pre', 'worker_now', 'wfh_pre', 'wfh_now', 'restric_2_w1b',
         'hhveh_harm', 'hhsize', 'hhincome', 'nchildren', 'tenure_harm', 'home_move',
         'age', 'gender', 'studentjs', 'jobcat_pre_harm', 'educ', 'driver', 'bike',
         'race_1', 'race_2', 'race_3', 'race_4', 'race_5',
         'att_covid_selfsevere', 'att_covid_stayhome', 'att_covid_commdisasters', 'att_covid_overreact',
         'att_wfh_likewfh', 'risk_percp_1_w1b',
         'pre_work_com_days', 'now_work_com_days', 'pre_work_pri_time', 'now_work_pri_time',
         'pre_work_pri_mode_harm', 'now_work_pri_mode_harm', 'pre_work_com_dist', 'wave']]
df.fillna(0, inplace=True)

# outcomes: travel behavior
# 1. go to work? (at least once):
# pre_work_com_days (Y0), now_work_com_days (Y1)
# 2. commuting time
# (if go to work at least once: pre_work_com_days>0 and now_work_com_days>0):
# pre_work_pri_time (Y0), now_work_pri_time (Y1)
# 3. mode change
# (if go to work at least once: pre_work_com_days>0 and now_work_com_days>0):
# pre_work_pri_mode_harm (Y0), now_work_pri_mode_harm (Y1)
# 4. commuting distance (?)  for now: use T0 distance as **control variable**
# pre_work_com_dist, now_com_dist (only for those who changed jobs, to exclude?)
# df = df[(df['pre_work_com_days'] > 0) & (df['now_work_com_days'] > 0)]
# print(df['pre_work_pri_mode_harm'].value_counts())
# print(df['now_work_pri_mode_harm'].value_counts())
# control variables:
# SES: jobcat_pre_harm, studentjs, hhveh_harm, age, gender, tenure_harm, home_move, educ, hhsize
# race_n (n=1-7, 6 is other, 7 is not answer), nchildren, hhincome, driver, bike
# attitudes: att_covid_selfsevere, att_covid_stayhome, att_covid_commdisasters, att_covid_overreact,
# att_wfh_likewfh, risk_percp_1_w1b (only for wave=1B)

# further recoding and processing
df['worker_pre'] = (df['worker_pre'] == 'yes').astype(int)
df['worker_now'] = (df['worker_now'] == 'yes').astype(int)
df['wfh_pre'] = (df['wfh_pre'] == 'Yes').astype(int)
df['wfh_now'] = (df['wfh_now'] == 'Yes').astype(int)
df['restric_2_w1b'] = (df['restric_2_w1b'] == 'Currently in place').astype(int)
df['hhveh_harm'] = (df['hhveh_harm'] != '0').astype(int)
df['hhincome'] = df['hhincome'].apply(lambda x: dct_inc.get(x))
df = pd.merge(df, pd.get_dummies(df['hhincome'], drop_first=True, prefix='inc'), left_index=True, right_index=True)
df['nchildren'] = (df['nchildren'] > 0).astype(int)
df['tenure_harm'] = df['tenure_harm'].isin(['Own with a mortgage', 'Own without a mortgage']).astype(int)
df['home_move'] = (df['home_move'] != 'No').astype(int)
df['gender'] = (df['gender'] == 'Female').astype(int)
df['studentjs'] = (df['studentjs'] == 'yes').astype(int)
df['jobcat_pre_harm'] = df['jobcat_pre_harm'].apply(lambda x: dct_job.get(x))
df = pd.merge(df, pd.get_dummies(df['jobcat_pre_harm'], drop_first=True, prefix='job'), left_index=True, right_index=True)
df['educ'] = df['educ'].apply(lambda x: dct_edu.get(x))
df = pd.merge(df, pd.get_dummies(df['educ'], drop_first=True, prefix='edu'), left_index=True, right_index=True)
df['driver'] = (df['driver'] == 'Yes').astype(int)
df['bike'] = (df['bike'] == 'Yes').astype(int)
df['race_1'] = (df['race_1'] == 'White/Caucasian').astype(int)
df['race_2'] = (df['race_2'] == 'Black/African American').astype(int)
df['race_3'] = (df['race_3'] == 'American Indian and Alaska Native').astype(int)
df['race_4'] = (df['race_4'] == 'Asian').astype(int)
df['race_5'] = (df['race_5'] == 'Native Hawaiian or Other Pacific Islander').astype(int)
for c in ['att_covid_selfsevere', 'att_covid_stayhome', 'att_covid_commdisasters', 'att_covid_overreact',
          'att_wfh_likewfh', 'risk_percp_1_w1b']:
    df[c] = df[c].isin(['Somewhat agree', 'Somewhat agree', 'Extremely high risk', 'High risk']).astype(int)

df['pre_work_pri_mode_harm'] = df['pre_work_pri_mode_harm'].apply(lambda x: dct_mode.get(x))
df = pd.merge(df, pd.get_dummies(df['pre_work_pri_mode_harm'], drop_first=True, prefix='pre_mode'),
              left_index=True, right_index=True)
df['now_work_pri_mode_harm'] = df['now_work_pri_mode_harm'].apply(lambda x: dct_mode.get(x))
df = pd.merge(df, pd.get_dummies(df['now_work_pri_mode_harm'], drop_first=True, prefix='now_mode'),
              left_index=True, right_index=True)
df['wave'] = (df['wave'] == '1B').astype(int)

# treatment: newly added work-from-home workers during pandemic
df['D_wfh'] = (1 - df['wfh_pre']) * (df['wfh_now'])
# alternative treatment: stay-at-home order restric_2_w1b (only for wave=1B)
df = df[(df['worker_pre'] == 1) & (df['worker_now'] == 1)]
df.drop(columns=['worker_pre', 'worker_now', 'hhincome', 'jobcat_pre_harm', 'educ',
                 'pre_work_pri_mode_harm', 'now_work_pri_mode_harm'], inplace=True)
print(df.head())
df.to_csv(path + 'covid_pooled_processed.csv', index=False)
