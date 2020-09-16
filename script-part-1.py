#%% PACKAGES

import pandas as pd
import numpy as np
import reverse_geocoder as rg
from time import time
import requests
import datetime
import math


#################
# PREPROCESSING #
#################

# rg_cities1000.csv doesn't need preprocessing.

#%% DATA READING

dtype = {"geonameid":np.int32,"name":str,"asciiname":str,"alternatenames":str,
          "latitude":np.float32,"longitude":np.float32,"featureclass":str,
          "featurecode":str,"countrycode":str,"cc2":str,"admin1code":str,
          "admin2code":str,"admin3code":str,"admin4code":str,
          "population":np.int32,"elevation":np.float32,"dem":np.int32,
          "timezone":str,"modificationdate":str}
usecols=[2,4,5,6,7,8,9]
cities500 = pd.read_table('cities500.txt', names=list(dtype.keys()),
                      encoding='utf-8', dtype=dtype, usecols=usecols)
allCountries = pd.read_table('allCountries.txt', names=list(dtype.keys()),
                              encoding='utf-8', dtype=dtype, usecols=usecols)

#%% DATA CHECK

a = cities500.describe(include='O')
b = cities500.describe(percentiles=[])

c = allCountries.describe(include='O')
d = allCountries.describe(percentiles=[])
# Exploring all the variables with the Spyder variable explorer everything
# seems reasonable.

#%% DATA CLEANING

### cities500 ###
# Remove duplicates: same (lat,lon)
cities500.drop_duplicates(subset=['latitude','longitude'], inplace=True)

# Remove rows with nas in lat, lon or cc
mask = cities500[['latitude','longitude','countrycode']].notna().apply(lambda x:
                                                          x.all(), axis=1)
cities500 = cities500.loc[mask,:].copy()
# 184 removed rows


### allCountries ###
allCountries.drop_duplicates(subset=['latitude','longitude'], inplace=True)
mask = allCountries[['latitude','longitude','countrycode']].notna().apply(
    lambda x: x.all(), axis=1)
allCountries = allCountries.loc[mask,:].copy()
# 300k removed rows


#%% RENAME VARIABLES AND SAVE CLEANED DATA TO PKL


is_populated = allCountries.featureclass=='P'
allCountries = allCountries[['latitude','longitude','asciiname',
                             'countrycode']].copy()
allCountries.rename(columns={'latitude':'lat','longitude':'lon',
                             'asciiname':'name','countrycode':'cc'},
                    inplace=True)
allCountries.insert(3,'admin1', 'x')
allCountries.insert(4,'admin2', 'x')
allCountries.to_pickle("rg_allCountries_clean.pkl")
allCountries.to_csv("rg_allCountries_clean.csv", index=False)
# 11.7M

allCountries.loc[is_populated,:].to_pickle("rg_allCountries_clean_onlypop.pkl")
allCountries.loc[is_populated,:].to_csv("rg_allCountries_clean_onlypop.csv",
                                        index=False)

# 4.7M


cities500 = cities500[['latitude','longitude','asciiname',
                       'countrycode']].copy()
cities500.rename(columns={'latitude':'lat','longitude':'lon',
                          'asciiname':'name','countrycode':'cc'},
                 inplace=True)
cities500.insert(3,'admin1', 'x')
cities500.insert(4,'admin2', 'x')
cities500.to_pickle("rg_cities500_clean.pkl")
cities500.to_csv("rg_cities500_clean.csv", index=False)
# 196k

dtype = {"lat":np.float32,"lon":np.float32,"name":str,"admin1":str,
         "admin2":str,"cc":str}
rg_cities1000 = pd.read_csv('rg_cities1000.csv', header=0, encoding='utf-8',
                            dtype=dtype)
rg_cities1000.to_pickle('rg_cities1000.pkl') # To save also the datatypes


dtype = {"lat":np.float32,"lon":np.float32,"name":str,"admin1":str,
         "admin2":str,"cc":str}
rg_cities1000 = pd.read_csv('rg_cities1000.csv', header=0, encoding='utf-8',
                            dtype=dtype)
rg_cities1000.to_pickle('rg_cities1000.pkl') # To save also the datatypes
# 145k

dtype = {'latitud_ga':np.float32,'longitud_ga':np.float32}
sample = pd.read_csv('sample.csv', index_col=0, sep=';', usecols=[0,1,2],
                     encoding='utf-8', dtype=dtype)
sample.rename(columns={'latitud_ga':'lat', 'longitud_ga':'lon'}, inplace=True)
sample.to_pickle('sample.pkl') # To save also the datatypes

#%% READ PKL DATA

rg_cities1000 = pd.read_pickle('rg_cities1000.pkl')
cities500 = pd.read_pickle('rg_cities500_clean.pkl')
allCountries = pd.read_pickle('rg_allCountries_clean_onlypop.pkl')



##############################################
# ACCURACY AND PERFORMANCE TEST WITH GN DATA #
##############################################

# The rg library using the default table rg_cities1000

#%% TEST WITH cities500 SET-DIFFERENCE cities1000

# I filter cities500 removing the shared rows with rg_cities1000
def all_conditions_ok(row):
    result = False
    if row['population'] <= 1000:
        if row['featurecode'] not in ['PPL','PPLA','PPLA2','PPLA3']:
            result = True
    return result
mask = cities500.apply(all_conditions_ok, axis=1)
filtered = cities500.loc[mask, :].copy()


# To make sure to remove every row shared with rg_cities1000, I also delete
# the ones sharing the same pair municipality-country
diff = pd.merge(filtered, rg_cities1000[['name','cc']], how='left',
                indicator=True,
                left_on = ['asciiname','countrycode'],
                right_on = ['name','cc'],
                suffixes=('','_'))
diff = diff[diff['_merge']=='left_only'].copy()

# I calculate right and wrong results with respect to country
coords = diff[['latitude','longitude']].to_numpy(copy=True).tolist()
coords = [tuple(i) for i in coords]

start_time = time()
results = rg.search(coords)
time_elapsed = time() - start_time
print('Elapsed time: {} secs'.format(time_elapsed)) # 0.3 s

results = pd.DataFrame(results)
have_different_cc = diff['countrycode'].to_numpy() != \
                    results['cc'].to_numpy()
print(np.mean(have_different_cc)) # 0.64 %
print(np.sum(have_different_cc)) # 7/1094 mistakes

#%% TEST WITH allCountries SET-DIFFERENCE rg_cities1000

# I filter allCountries so the rows that still remain are not in rg_cities1000
# and all of them must represent municipalities --no mountains, lakes, etc.
def all_conditions_ok(row):
    result = False
    if row['population'] <= 1000:
        if row['featurecode'] not in ['PPL','PPLA','PPLA2','PPLA3']:
            result = True
    return result

mask = allCountries.apply(all_conditions_ok, axis=1)
filtered = allCountries.loc[mask, :].copy()

# To make sure to remove every row shared with rg_cities1000, I also delete
# the ones sharing the same pair municipality-country
diff = pd.merge(filtered, rg_cities1000[['name','cc']], how='left',
                indicator=True,
                left_on = ['asciiname','countrycode'],
                right_on = ['name','cc'],
                suffixes=('','_'))
diff = diff[diff['_merge']=='left_only'].copy()

# I calculate right and wrong results with respect to country
coords = diff[['latitude','longitude']].to_numpy(copy=True).tolist()
coords = [tuple(i) for i in coords]

start_time = time()
results = rg.search(coords)
time_elapsed = time() - start_time
print('Elapsed time: {} secs'.format(time_elapsed)) # 0.7 s

results = pd.DataFrame(results)
have_different_cc = diff['countrycode'].to_numpy() != \
                    results['cc'].to_numpy()
print(np.mean(have_different_cc)) # 3.07 %
print(np.sum(have_different_cc)) # 10k/320k mistakes



###########################################
# REQUESTING COUNTRYCODES TO GEONAMES API #
###########################################

#%%
def get_cc(row):
    """Obtain 2-letter contrycode requesting Geonames API"""
    params = {'lat':row['latitud_ga'],'lng':row['longitud_ga'],
              'username':'marcoscastillo'}
    req = requests.get("http://api.geonames.org/countryCode", params=params)
    return req.text.strip()

sample = pd.read_csv('sample.csv', index_col=0, sep=';', usecols=[0,1,2])

# Obtain the countrycode of the 48,475 observations of the BS sample and save
# it to pkl
for i in np.arange(0,48000,1000):
    time_start = time.time()
    getcc = sample.iloc[i:i+1000,:].apply(get_cc, axis='columns')
    getcc.to_pickle('getcc{}_{}.pkl'.format(i,i+1000))
    print('Computed untill {} row'.format(i+1000))

    # Time to sleep (in secs) to make sure we don't exceed the usage limits
    time_sleep = 3600 - (time.time() - time_start)/3
    frac, whole = math.modf(time_sleep/60)
    print(datetime.datetime.now())
    print('Sleeping {} mins and {} secs'.format(int(whole), int(frac*60)))
    time.sleep(time_sleep)

getcc2 = sample.iloc[48000:48475,:].apply(get_cc, axis='columns')
getcc = pd.read_pickle('getcc0_48000.pkl')
getcc = pd.concat([getcc,getcc2])
getcc.to_pickle('sample_cc.pkl')



##############################################
# ACCURACY AND PERFORMANCE TEST WITH BS DATA #
##############################################

#%%
sample = pd.read_pickle('sample.pkl')
results_gn = pd.read_pickle('sample_cc.pkl')

# Exploring country distribution with Spyder variable explorer
cc_distrib = results_gn.value_counts()

# Sometimes the requests gets an error. When that happen, the resulting string
# is larger than 2, so like this we count how many errors are there.
n_errors = np.sum(results_gn.apply(lambda cc: len(cc)!=2))
print(n_errors) # 64

results_gn = results_gn.to_numpy(copy=True)

# Transforming the coordinates to the suitable format for being the input of
# the function rg.search
coords = sample[['lat','lon']].to_numpy(copy=True).tolist()
coords = [tuple(i) for i in coords]

time_start = time()
results_rg = rg.search(coords)
print(time()-time_start) # 0.4 s

results_rg = pd.DataFrame(results_rg).cc.to_numpy(copy=True)
n_rg_wrong = np.sum(results_rg != results_gn) - n_errors
print(n_rg_wrong) # 35

rg_accuracy = n_rg_wrong / (results_rg.size - n_errors)
print(rg_accuracy) # 0.0007

# Where do the wrong results happen?
wrong_gn = results_gn[results_rg != results_gn].reshape(99,1)
wrong_rg = results_rg[results_rg != results_gn].reshape(99,1)
wrong_concat = np.concatenate((wrong_gn, wrong_rg), axis=1)

wrong_concat = wrong_concat.tolist()
wrong_concat = [mylist[0] + "-" +mylist[1] for mylist in wrong_concat]
wrong_concat = pd.Series(wrong_concat)

wrong_types = wrong_concat.value_counts()
# Checking wrong_types with the Spyder variable explorer, I see that most of
# the wrong results happen near the borders of Spain.
