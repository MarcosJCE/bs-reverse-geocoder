#%% PACKAGES

import pandas as pd
import numpy as np
import reverse_geocoder as rg
import time
import io
import requests
import datetime
import math
import sklearn

# Packages for trying other methods differents to reverse_geocoder, like KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, preprocessing, tree, metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

#################
# PREPROCESSING #
#################

# rg_cities1000.csv doesn't need preprocessing.

#%% DATA READING (THE ORIGINAL)

dtype = {"geonameid":np.int64,"name":str,"asciiname":str,"alternatenames":str,
          "latitude":np.float64,"longitude":np.float64,"featureclass":str,
          "featurecode":str,"countrycode":str,"cc2":str,"admin1code":str,
          "admin2code":str,"admin3code":str,"admin4code":str,
          "population":np.int64,"elevation":np.float64,"dem":np.int64,
          "timezone":str,"modificationdate":str}
         
cities500 = pd.read_table('cities500.txt', names=dtype.keys(), 
                      encoding='utf-8', dtype=dtype)
allCountries = pd.read_table('allCountries.txt', names=dtype.keys(), 
                              encoding='utf-8', dtype=dtype)

#%% DATA CHECKING 
# To see if all the values are reasonable or there are inconsistencies.

a = cities500.describe(include='O')
b = cities500.describe(percentiles=[])

c = allCountries.describe(include='O')
d = allCountries.describe(percentiles=[])
# Exploring all the variables with the Spyder variable explorer everything 
# seems reasonable.

#%% DATA CLEANING

### cities500 ###
b = cities500.shape[0]

# Remove duplicates: same (lat,lon)
cities500.drop_duplicates(subset=['latitude','longitude'], inplace=True)

# Remove rows with nas in lat, lon or cc
mask = cities500[['latitude','longitude','countrycode']].notna().apply(lambda x: 
                                                          x.all(), axis=1)
cities500 = cities500.loc[mask,:]

a = cities500.shape[0]
print('{} rows removed'.format(b-a)) # 184


### allCountries ###
b = allCountries.shape[0]

# Remove duplicates: same coordinates (latitude, longitude)
allCountries.drop_duplicates(subset=['latitude','longitude'], inplace=True)

# Remove rows with not-a-numbers in latitude, longitude or countrycode
mask = allCountries[['latitude','longitude','countrycode']].notna().apply(
    lambda x: x.all(), axis=1)
allCountries = allCountries.loc[mask,:]

a = allCountries.shape[0]
print('{} rows removed'.format(b-a)) # ca 300,000


#%% SAVE CLEANED DATA

# To save time in future usage of that data, we save them cleaned and with just
# the variables we'll use. In .pkl and .csv

# Boolean Series for populated places
is_populated = allCountries.featureclass=='P'

allCountries = allCountries[['latitude','longitude','asciiname','admin1code',
                             'admin2code','countrycode']].copy()
allCountries.rename(columns={'latitude':'lat','longitude':'lon',
                             'asciiname':'name','admin1code':'admin1',
                             'admin2code':'admin2','countrycode':'cc'}, 
                    inplace=True)

allCountries.to_pickle("rg_allCountries_clean.pkl")
allCountries.to_csv("rg_allCountries_clean.csv", index=False)

allCountries.loc[is_populated,:].to_pickle("rg_allCountries_clean_onlypop.pkl")
allCountries.loc[is_populated,:].to_csv("rg_allCountries_clean_onlypop.csv", 
                                        index=False)


cities500 = cities500[['latitude','longitude','asciiname','admin1code',
                       'admin2code','countrycode']].copy()
cities500.rename(columns={'latitude':'lat','longitude':'lon',
                          'asciiname':'name','admin1code':'admin1',
                          'admin2code':'admin2','countrycode':'cc'}, 
                 inplace=True)
cities500.to_pickle("rg_cities500_clean.pkl")
cities500.to_csv("rg_cities500_clean.csv", index=False)


dtype = {"lat":np.float64,"lon":np.float64,"name":str,"admin1":str,
         "admin2":str,"cc":str}
rg_cities1000 = pd.read_csv('rg_cities1000.csv', header=0, encoding='utf-8',
                            dtype=dtype)
rg_cities1000.to_pickle('rg_cities1000.pkl') # To save also the datatypes

#%% READ DATA

rg_cities1000 = pd.read_pickle('rg_cities1000.pkl')
cities500 = pd.read_pickle('rg_cities500_clean.pkl')
allCountries = pd.read_pickle('rg_allCountries_clean_onlypop.pkl')

############
# ACCURACY #
############

#%% TESTING cities500 SET-DIFFERENCE cities1000

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

start_time = time.perf_counter()
rg.search(coords)
time_elapsed = time.perf_counter() - start_time
print('Elapsed time: {} secs'.format(time_elapsed)) # 0.3 s

results = pd.DataFrame(rg.search(coords))
have_different_cc = diff['countrycode'].to_numpy() != \
                    results['cc'].to_numpy()
print(np.mean(have_different_cc)) # 0.64 %
print(np.sum(have_different_cc)) # 7/1094 mistakes

#%% TESTING WITH allCountries SET-DIFFERENCE rg_cities1000

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

start_time = time.perf_counter()
rg.search(coords)
time_elapsed = time.perf_counter() - start_time
print('Elapsed time: {} secs'.format(time_elapsed)) # 0.7 s

results = pd.DataFrame(rg.search(coords))
have_different_cc = diff['countrycode'].to_numpy() != \
                    results['cc'].to_numpy()
print(np.mean(have_different_cc)) # 3.07 %
print(np.sum(have_different_cc)) # 10k/320k mistakes



##########################################################
# COMPUTACIONAL TIME DEPENDING ON THE NO. OF COORDINATES #
##########################################################

#%% More coordinates than in a real scenario 

#!!! WARNING: PC freezes with too many coordinates. 

ncoords = [int(10**i) for i in range(8)]
many_coords = coords*40 # size ~12M

for n in ncoords:
    choosen_coords = many_coords[0:n]
    start_time = time.perf_counter()
    results = rg.search(choosen_coords)
    time_elapsed = time.perf_counter() - start_time
    print('{} & {:.2f}'.format(n, time_elapsed))



##################
# WITH BS SAMPLE #
##################

#%% REQUESTING COUNTRYCODES TO GEONAMES API

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
    

    
#%% CHECKING ACCURACY OR REVERSE_GEOCODER WITH SAMPLE OF BS

sample = pd.read_csv('sample.csv', index_col=0, sep=';', usecols=[0,1,2])
results_gn = pd.read_pickle('sample_cc.pkl')

# Exploring country distribution with Spyder variable explorer
cc_distrib = results_gn.value_counts()
# Approx. 47,000 out of the 48,475 are 'ES'

# Sometimes the requests gets an error. When that happen, the resulting string
# is larger than 2, so like this we count how many errors are there.
n_errors = np.sum(results_gn.apply(lambda cc: len(cc)!=2))
print(n_errors) # 64

results_gn = results_gn.to_numpy()

# Transforming the coordinates to the si¡uitable format for being the input of
# the function rg.search
coords = sample[['latitud_ga','longitud_ga']].to_numpy().tolist()
coords = [tuple(i) for i in coords]


### With the default datasource: rg_cities1000.csv ###
results_rg = rg.search(coords)
results_rg = pd.DataFrame(results_rg).cc.to_numpy()

n_rg_wrong = np.sum(results_rg != results_gn) - n_errors 
print(n_rg_wrong) # 35

rg_accuracy = n_rg_wrong / (results_rg.size - n_errors)
print(rg_accuracy) # 0.0007


### With allCountries_clean_onlypop.csv (~4.7M rows) as datasource ###
stream = io.StringIO(open('rg_allCountries_clean_onlypop.csv', 
                          encoding='utf-8').read())
geo = rg.RGeocoder(mode=2, verbose=True, stream=stream)
results_rg2 = geo.query(coords)
results_rg2 = pd.DataFrame(results_rg2).cc.to_numpy()

n_rg2_wrong = np.sum(results_rg2 != results_gn) - n_errors 
print(n_rg2_wrong) # 35

rg2_accuracy = n_rg2_wrong / (results_rg2.size - n_errors)
print(rg2_accuracy) # 0.0007

### With allCountries_clean.csv (~11.7M rows) as datasource ###
stream = io.StringIO(open('rg_allCountries_clean.csv', 
                          encoding='utf-8').read())
geo = rg.RGeocoder(mode=2, verbose=True, stream=stream)
results_rg2 = geo.query(coords)
results_rg2 = pd.DataFrame(results_rg2).cc.to_numpy()

n_rg2_wrong = np.sum(results_rg2 != results_gn) - n_errors 
print(n_rg2_wrong) # 35

rg2_accuracy = n_rg2_wrong / (results_rg2.size - n_errors)
print(rg2_accuracy) # 0.0007

# Do both methods give the same results? 
print(np.sum(results_rg != results_rg2))
# Yes, exactly the same ones

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



#####################
# DIFFERENT METHODS #
#####################

#%% KNN

dtype = {"geonameid":np.int64,"name":str,"asciiname":str,"alternatenames":str,
          "latitude":np.float64,"longitude":np.float64,"featureclass":str,
          "featurecode":str,"countrycode":str,"cc2":str,"admin1code":str,
          "admin2code":str,"admin3code":str,"admin4code":str,
          "population":np.int64,"elevation":np.float64,"dem":np.int64,
          "timezone":str,"modificationdate":str}
         
cities500 = pd.read_csv('rg_allCountries_clean_onlypop.csv', names=dtype.keys(), 
                      encoding='utf-8', dtype=dtype)


cities500 = pd.read_pickle('rg_allCountries_clean_onlypop.pkl')

cities500 = cities500[['latitude','longitude','countrycode']].copy()
cities500.rename(columns={'latitude':'lat','longitude':'lon',
                          'countrycode':'cc'}, inplace=True)

is_notna = pd.notna(cities500).all(axis=1)
cities500 = cities500.loc[is_notna,:].copy()

X = cities500[['lat','lon']].to_numpy().copy()
y = cities500['cc'].to_numpy().copy()

### 3-NN

results_gn = pd.read_pickle('getcc0_48475.pkl').to_numpy()
sample = pd.read_csv('sample.csv', index_col=0, sep=';', usecols=[0,1,2])
sample = sample.to_numpy(copy=True)

neigh = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
neigh.fit(X, y)
results_knn = neigh.predict(sample)

print(np.sum(results_knn != results_gn))




