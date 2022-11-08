# to read and pre-preprocess the NHTS data

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

# note to self
# HHFAMINC not consistent
# WRKTRANS not consistent
# PRMACT slightly inconsistent: 7 in 2009 equals 97 in 2017
# WKFTPT slightly inconsistent: 3 in 2009 (multiple jobs) more like 2 in 2009&2017 (part-time)
# SCHTYP only cares 'not in school': 4 in 2009, 3 in 2017
lst_person_col = ['HOUSEID', 'PERSONID', 'HH_CBSA',
                  'DRVRCNT', 'HHFAMINC', 'HHSIZE', 'HHVEHCNT', 'HH_RACE', 'HOMEOWN', 'LIF_CYC', 'WRKCOUNT', # HH attributes
                  'HTPPOPDN', 'HTRESDN', 'RAIL', 'URBRUR', 'GCDWORK', # BE attributes
                  'BORNINUS', 'DRIVER', 'EDUC', 'FLEXTIME', 'OCCAT', 'PRMACT', 'R_AGE', 'R_SEX', 'WKFMHMXX', 'WKFTPT', 'WORKER', 'SCHTYP', # SES
                  'CNTTDTR', 'MCUSED', 'NBIKETRP', 'NWALKTRP', 'PTUSED', 'TRAVDAY', 'USEPUBTR', 'TIMETOWK', 'WRKTRANS', 'YEARMILE'] # travel attributes
# note to self
# FUELTYPE not consistent
# VEHAGE slightly inconsistent: 35 as 25+ in 2009
# VEHTYPE slighltly inconsistent: 8 golf cart in 2009, merge to 97 as in 2017 as others
lst_veh_col = ['HOUSEID', 'PERSONID', 'VEHID', 'HH_CBSA',
               'ANNMILES', 'BESTMILE', 'FUELTYPE', 'GSYRGAL', 'HYBRID', 'VEHAGE', 'VEHTYPE', 'FUELECO'] # last one renamed by self
# note to self
# WHYTRP1S slightly inconsistent: 60 in 2009 not in 2017, change to 97
# TRPTRANS not consistent
lst_trip_col = ['HOUSEID', 'PERSONID', 'VEHID', 'HH_CBSA',
                'STRTTIME', 'TRVLCMIN', 'WHYTRP1S', 'TRIPPURP', 'TRPTRANS', 'TRPMILES', 'VMT_MILE', 'NUMONTRP',
                'TRPHHVEH', 'PUBTRANS', 'TDWKND', 'DRVR_FLG', 'PSGR_FLG', 'GASPRICE']
# area codes for metropolitan areas in Texas
lst_texas = ['12420', '19100', '26420', '41700']

# read data
person_2009 = pd.read_csv(path + 'NHTS 2009/PERV2PUB.csv')
person_2009 = person_2009[lst_person_col]
veh_2009 = pd.read_csv(path + 'NHTS 2009/VEHV2PUB.csv')
veh_2009.rename(columns={'EPATMPG': 'FUELECO'}, inplace=True)
veh_2009 = veh_2009[lst_veh_col]
trip_2009 = pd.read_csv(path + 'NHTS 2009/DAYV2PUB.csv')
trip_2009 = trip_2009[lst_trip_col]

person_2017 = pd.read_csv(path + 'NHTS 2017/perpub.csv')
# to update age/gender of 'missing' ones with imputed values
person_2017.loc[person_2017['R_AGE'] < 0, 'R_AGE'] = person_2017['R_AGE_IMP']
person_2017.loc[person_2017['R_SEX'] < 0, 'R_SEX'] = person_2017['R_SEX_IMP']
person_2017 = person_2017[lst_person_col]
veh_2017 = pd.read_csv(path + 'NHTS 2017/vehpub.csv')
veh_2017.rename(columns={'FEGEMPG': 'FUELECO'}, inplace=True)
veh_2017 = veh_2017[lst_veh_col]
trip_2017 = pd.read_csv(path + 'NHTS 2017/trippub.csv')
trip_2017 = trip_2017[lst_trip_col]

person_2009 = person_2009[person_2009['HH_CBSA'].isin(lst_texas)]
person_2017 = person_2017[person_2017['HH_CBSA'].isin(lst_texas)]
veh_2009 = veh_2009[veh_2009['HH_CBSA'].isin(lst_texas)]
veh_2017 = veh_2017[veh_2017['HH_CBSA'].isin(lst_texas)]
trip_2009 = trip_2009[trip_2009['HH_CBSA'].isin(lst_texas)]
trip_2017 = trip_2017[trip_2017['HH_CBSA'].isin(lst_texas)]

# export extracted data
person_2009.to_csv(path + 'NHTS processed/texas_person_2009.csv', index=False)
person_2017.to_csv(path + 'NHTS processed/texas_person_2017.csv', index=False)
veh_2009.to_csv(path + 'NHTS processed/texas_veh_2009.csv', index=False)
veh_2017.to_csv(path + 'NHTS processed/texas_veh_2017.csv', index=False)
trip_2009.to_csv(path + 'NHTS processed/texas_trip_2009.csv', index=False)
trip_2017.to_csv(path + 'NHTS processed/texas_trip_2017.csv', index=False)
