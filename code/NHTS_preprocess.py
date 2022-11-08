# to process extracted and formatted NHTS data for Texas case analysis
# reference & data source
# U.S. Department of Transportation; Federal Highway Administration. (2009).
# 2009 National Household Travel Survey. http://nhts.ornl.gov
# U.S. Department of Transportation; Federal Highway Administration. (2017).
# 2017 National Household Travel Survey. http://nhts.ornl.gov

import sys

import pandas as pd
import numpy as np

pd.set_option('max_column', 500)
pd.set_option('display.width', 500)
np.set_printoptions(threshold=sys.maxsize)

path = '' # ignored for privacy
# recoding dictionaries
dct_inc = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5, 10: 5, 11: 6, 12: 6, 13: 6, 14: 6, 15: 6,
           16: 7, 17: 7, 18: 8}
dct_mode_09 = {1: 2, 2: 3, 3: 2, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 5, 12: 5, 13: 6, 14: 5, 15: 6,
               16: 6, 17: 7, 18: 7, 19: 8, 20: 8, 21: 8, 22: 1, 23: 1, 24: 5, 97: 8}
dct_mode_17 = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 3, 10: 5, 11: 5, 12: 5, 13: 5, 14: 6, 15: 6,
               16: 7, 17: 8, 18: 2, 19: 8, 20: 8, 97: 8}
# IDs: 'HOUSEID', 'PERSONID', 'HH_CBSA'
# HH attributes: 'DRVRCNT', 'HHFAMINC', 'HHSIZE', 'HHVEHCNT', 'HH_RACE', 'HOMEOWN', 'LIF_CYC', 'WRKCOUNT'
# BE attributes: 'HTPPOPDN', 'HTRESDN', 'RAIL', 'URBRUR', 'GCDWORK'
# SES: 'BORNINUS', 'DRIVER', 'EDUC', 'FLEXTIME', 'OCCAT', 'PRMACT', 'R_AGE', 'R_SEX', 'WKFMHMXX', 'WKFTPT',
#      'WORKER', 'SCHTYP'
# travel attributes: 'CNTTDTR', 'MCUSED', 'NBIKETRP', 'NWALKTRP', 'PTUSED', 'TRAVDAY', 'USEPUBTR',
#                    'TIMETOWK', 'WRKTRANS', 'YEARMILE'
# $100 in 2009 is equivalent in purchasing power to about $114.26 in 2017 (to ignore)
# source: Bureau of Labor Statistics' Consumer Price Index (CPI), http://www.bls.gov/cpi/

# read extracted data for Texas
p_09 = pd.read_csv(path + 'texas_person_2009.csv')
p_17 = pd.read_csv(path + 'texas_person_2017.csv')

# processing and recoding
p_09['PID'] = p_09['HOUSEID'] * 100 + p_09['PERSONID']
p_17['PID'] = p_17['HOUSEID'] * 100 + p_17['PERSONID']

# HHFAMINC: mostly same as 2017, only merging 8-11 in 2017 to 8 (>=100000)
p_09['HHFAMINC'] = p_09['HHFAMINC'].apply(lambda x: dct_inc.get(x) if x in dct_inc.keys() else -1)
p_17['HHFAMINC'] = p_17['HHFAMINC'].apply(lambda x: max(min(x, 8), -1))

p_09['WRKTRANS'] = p_09['WRKTRANS'].apply(lambda x: dct_mode_09.get(x) if x in dct_mode_09.keys() else -1)
p_17['WRKTRANS'] = p_17['WRKTRANS'].apply(lambda x: dct_mode_17.get(x) if x in dct_mode_17.keys() else -1)

p_09['PRMACT'] = p_09['PRMACT'].apply(lambda x: max(min(x, 7), -1))
p_17['PRMACT'] = p_17['PRMACT'].apply(lambda x: max(min(x, 7), -1))

p_09['WKFTPT'] = p_09['WKFTPT'].apply(lambda x: max(min(x, 2), -1))
p_17['WKFTPT'] = p_17['WKFTPT'].apply(lambda x: max(min(x, 2), -1))

p_09['SCHTYP'] = p_09['SCHTYP'].isin([1, 2, 3]).astype(int)
p_17['SCHTYP'] = p_17['SCHTYP'].isin([1, 2]).astype(int)

for c in p_09.columns:
    p_09[c] = p_09[c].apply(lambda x: max(x, -1))
    p_17[c] = p_17[c].apply(lambda x: max(x, -1))

# 'HOUSEID', 'PERSONID', 'VEHID', 'HH_CBSA',
# 'ANNMILES', 'BESTMILE', 'FUELTYPE', 'GSYRGAL', 'HYBRID', 'VEHAGE', 'VEHTYPE', 'FUELECO'
# 'FUELECO' named by self, originally 'EPATMPG' and 'FEGEMPG'
v_09 = pd.read_csv(path + 'texas_veh_2009.csv')
v_17 = pd.read_csv(path + 'texas_veh_2017.csv')
v_09.loc[(v_09['PERSONID'] < 0) | (v_09['PERSONID'] == 99), 'PERSONID'] = 1
v_17.loc[(v_17['PERSONID'] < 0) | (v_17['PERSONID'] == 97), 'PERSONID'] = 1
v_09.loc[v_09['VEHAGE'] < 0, 'VEHAGE'] = -1
v_17.loc[v_17['VEHAGE'] < 0, 'VEHAGE'] = -1
v_17.loc[v_17['VEHAGE'] > 24, 'VEHAGE'] = 35
v_09.loc[v_09['VEHTYPE'] < 0, 'VEHTYPE'] = -1
v_09.loc[v_09['VEHTYPE'] == 8, 'VEHTYPE'] = 97
v_17.loc[v_17['VEHTYPE'] < 0, 'VEHTYPE'] = -1
v_09['FUELTYPE'] = v_09['FUELTYPE'].apply(lambda x: {1: 0, 2: 0, 3: 0, 4: 1}.get(x))
v_17['FUELTYPE'] = v_17['FUELTYPE'].apply(lambda x: {1: 1, 2: 0, 3: 0, 97: 0, -7: 0, -8: 0}.get(x))
v_09['PID'] = v_09['HOUSEID'] * 100 + v_09['PERSONID']
v_17['PID'] = v_17['HOUSEID'] * 100 + v_17['PERSONID']
# some person ID error
v_09 = v_09[v_09['PID'].isin(set(v_09['PID']).intersection(set(p_09['PID'])))]

v_sum_09 = v_09[['PID', 'BESTMILE', 'GSYRGAL']].copy()
v_sum_17 = v_17[['PID', 'BESTMILE', 'GSYRGAL']].copy()
v_sum_09.loc[v_sum_09['BESTMILE'] < 0, 'BESTMILE'] = 0
v_sum_09.loc[v_sum_09['GSYRGAL'] < 0, 'GSYRGAL'] = 0
v_sum_17.loc[v_sum_17['BESTMILE'] < 0, 'BESTMILE'] = 0
v_sum_17.loc[v_sum_17['GSYRGAL'] < 0, 'GSYRGAL'] = 0
v_sum_09 = v_sum_09.groupby('PID').sum()
v_sum_17 = v_sum_17.groupby('PID').sum()

v_09.sort_values(by='BESTMILE', ascending=False, inplace=True)
v_09.drop_duplicates(subset='PID', keep='first', inplace=True)
v_09 = v_09[['PID', 'FUELTYPE', 'VEHAGE', 'VEHTYPE', 'FUELECO']]
v_09 = pd.merge(v_09, v_sum_09, left_on='PID', right_index=True)
v_17.sort_values(by='BESTMILE', ascending=False, inplace=True)
v_17.drop_duplicates(subset='PID', keep='first', inplace=True)
v_17 = v_17[['PID', 'FUELTYPE', 'VEHAGE', 'VEHTYPE', 'FUELECO']]
v_17 = pd.merge(v_17, v_sum_17, left_on='PID', right_index=True)

# 'HOUSEID', 'PERSONID'
# no use: 'VEHID', 'HH_CBSA', 'STRTTIME', 'WHYTRP1S', 'NUMONTRP', 'TRPHHVEH', 'TDWKND', 'TRIPPURP', 'TRPTRANS'
# cap at 0 and sum: row count, 'TRVLCMIN', 'TRPMILES', 'VMT_MILE'
# classify and count: 'PUBTRANS', 'DRVR_FLG', 'PSGR_FLG'
# get mean: 'GASPRICE'
t_09 = pd.read_csv(path + 'texas_trip_2009.csv')
t_17 = pd.read_csv(path + 'texas_trip_2017.csv')

t_09.loc[t_09['WHYTRP1S'] == 60, 'WHYTRP1S'] = 97
t_09['TRPTRANS'] = t_09['TRPTRANS'].apply(lambda x: dct_mode_09.get(x) if x in dct_mode_09.keys() else -1)
t_17['TRPTRANS'] = t_17['TRPTRANS'].apply(lambda x: dct_mode_17.get(x) if x in dct_mode_17.keys() else -1)
t_09['PID'] = t_09['HOUSEID'] * 100 + t_09['PERSONID']
t_17['PID'] = t_17['HOUSEID'] * 100 + t_17['PERSONID']

t_09['TRVLCMIN'] = t_09['TRVLCMIN'].apply(lambda x: max(x, 0))
t_09['TRPMILES'] = t_09['TRPMILES'].apply(lambda x: max(x, 0))
t_09['VMT_MILE'] = t_09['VMT_MILE'].apply(lambda x: max(x, 0))
t_17['TRVLCMIN'] = t_17['TRVLCMIN'].apply(lambda x: max(x, 0))
t_17['TRPMILES'] = t_17['TRPMILES'].apply(lambda x: max(x, 0))
t_17['VMT_MILE'] = t_17['VMT_MILE'].apply(lambda x: max(x, 0))
t_09['PUBTRANS'] = (t_09['PUBTRANS'] == 1).astype(int)
t_09['DRVR_FLG'] = (t_09['DRVR_FLG'] == 1).astype(int)
t_09['PSGR_FLG'] = (t_09['PSGR_FLG'] == 1).astype(int)
t_17['PUBTRANS'] = (t_17['PUBTRANS'] == 1).astype(int)
t_17['DRVR_FLG'] = (t_17['DRVR_FLG'] == 1).astype(int)
t_17['PSGR_FLG'] = (t_17['PSGR_FLG'] == 1).astype(int)

t_09['COUNT'] = 1
t_09['PUBTIME'] = t_09['TRVLCMIN'] * t_09['PUBTRANS']
t_09['DRVRTIME'] = t_09['TRVLCMIN'] * t_09['DRVR_FLG']
t_09['PSGRTIME'] = t_09['TRVLCMIN'] * t_09['PSGR_FLG']
t_09['PUBMILE'] = t_09['TRPMILES'] * t_09['PUBTRANS']
t_09['PSGRMILE'] = t_09['TRPMILES'] * t_09['PSGR_FLG']

t_17['COUNT'] = 1
t_17['PUBTIME'] = t_17['TRVLCMIN'] * t_17['PUBTRANS']
t_17['DRVRTIME'] = t_17['TRVLCMIN'] * t_17['DRVR_FLG']
t_17['PSGRTIME'] = t_17['TRVLCMIN'] * t_17['PSGR_FLG']
t_17['PUBMILE'] = t_17['TRPMILES'] * t_17['PUBTRANS']
t_17['PSGRMILE'] = t_17['TRPMILES'] * t_17['PSGR_FLG']

t_09 = t_09[['PID', 'COUNT', 'TRVLCMIN', 'TRPMILES', 'PUBTRANS', 'PUBTIME', 'PUBMILE',
             'DRVR_FLG', 'DRVRTIME', 'VMT_MILE', 'PSGR_FLG', 'PSGRTIME', 'PSGRMILE']].groupby('PID').sum()
t_17 = t_17[['PID', 'COUNT', 'TRVLCMIN', 'TRPMILES', 'PUBTRANS', 'PUBTIME', 'PUBMILE',
             'DRVR_FLG', 'DRVRTIME', 'VMT_MILE', 'PSGR_FLG', 'PSGRTIME', 'PSGRMILE']].groupby('PID').sum()

p_09 = pd.merge(p_09, v_09, how='left', on='PID')
p_17 = pd.merge(p_17, v_17, how='left', on='PID')
p_09 = pd.merge(p_09, t_09, how='left', left_on='PID', right_index=True)
p_17 = pd.merge(p_17, t_17, how='left', left_on='PID', right_index=True)

# 2009 as t0, 2017 as t1
p_09['T'] = 0
p_17['T'] = 1

# Dallas area as treatment, others as control
p_09['D'] = (p_09['HH_CBSA'] == 19100).astype(int)
p_17['D'] = (p_17['HH_CBSA'] == 19100).astype(int)

# merge data
df = pd.concat([p_09, p_17], ignore_index=True)

df.drop(columns=['HOUSEID', 'PERSONID', 'HH_CBSA', 'RAIL', 'DRVRCNT', 'HH_RACE', 'WKFMHMXX',
                 'USEPUBTR', 'VEHAGE'], inplace=True)
df['TRAVDAY'] = (df['TRAVDAY'].isin([2, 3, 4, 5, 6])).astype(int)
df[['FUELTYPE', 'VEHTYPE', 'FUELECO']] = df[['FUELTYPE', 'VEHTYPE', 'FUELECO']].fillna(value=-1)
df[['BESTMILE', 'GSYRGAL', 'COUNT', 'TRVLCMIN', 'TRPMILES', 'PUBTRANS', 'PUBTIME', 'PUBMILE',
    'DRVR_FLG', 'DRVRTIME', 'VMT_MILE', 'PSGR_FLG', 'PSGRTIME', 'PSGRMILE']] = \
    df[['BESTMILE', 'GSYRGAL', 'COUNT', 'TRVLCMIN', 'TRPMILES', 'PUBTRANS', 'PUBTIME', 'PUBMILE',
    'DRVR_FLG', 'DRVRTIME', 'VMT_MILE', 'PSGR_FLG', 'PSGRTIME', 'PSGRMILE']].fillna(value=0)
lst = list(df.columns)
lst.remove('PID')
lst.insert(0, 'PID')
df = df[lst]
df.to_csv(path + 'texas_combined.csv', index=False)
print(df.head())
