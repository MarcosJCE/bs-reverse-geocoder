#%% PACKAGES

import pandas as pd
import numpy as np
import reverse_geocoder as rg
import time
import io
import requests
import time
import datetime
import math
import datetime


#################
# PREPROCESSING #
#################



#%% LEER DATOS ORIGINALES ###

dtype = {"lat":np.float64,"lon":np.float64,"name":str,"admin1":str,
          "admin2":str,"cc":str}
rg1000 = pd.read_csv('rg_cities1000.csv', header=0, encoding='utf-8',
                      dtype=dtype)

dtype = {"geonameid":np.int64,"name":str,"asciiname":str,"alternatenames":str,
          "latitude":np.float64,"longitude":np.float64,"featureclass":str,
          "featurecode":str,"countrycode":str,"cc2":str,"admin1code":str,
          "admin2code":str,"admin3code":str,"admin4code":str,
          "population":np.int64,"elevation":np.float64,"dem":np.int64,
          "timezone":str,"modificationdate":str}
         
c500 = pd.read_table('cities500.txt', names=dtype.keys(), encoding='utf-8',
                      dtype=dtype)
allCountries = pd.read_table('allCountries.txt', names=dtype.keys(), 
                              encoding='utf-8', dtype=dtype)

#%% COMPROBAR DATOS ###

a = c500.describe(include='O')
b = c500.describe(percentiles=[])

c = allCountries.describe(include='O')
d = allCountries.describe(percentiles=[])
# I explore all the variables with the Spyder 'Variable explorer'. 
# Everything seems reasonable, except a negative population in allCountries.

print(allCountries.loc[allCountries['population']<0,'featureclass'])
# The feature class of them is H, so the 'negative-populated' locations will
# be removed when just keeping the populated places (featureclass==P).

#%% LIMPIAR DATOS ###

# CON c500

b = c500.shape[0]

# Remove duplicates: samehttp://api.geonames.org/countryCode?lat=47.03&lng=10.2&username=marcoscastillo (lat,lon)
c500.drop_duplicates(subset=['latitude','longitude'], inplace=True)

# Remove rows with nas in lat, lon or cc
mask = c500[['latitude','longitude','countrycode']].notna().apply(lambda x: 
                                                          x.all(), axis=1)
c500 = c500.loc[mask,:]

a = c500.shape[0]
print('{} rows removed'.format(b-a)) # 184


# CON allCountries
b = allCountries.shape[0]

# Nos quedamos solo con las localizaciones pobladas (featureclass = P).
allCountries = allCountries.loc[allCountries['featureclass']=='P',:].copy()

# Remove duplicates: same (lat,lon)
allCountries.drop_duplicates(subset=['latitude','longitude'], inplace=True)

# Remove rows with nas in lat, lon or cc
mask = allCountries[['latitude','longitude','countrycode']].notna().apply(
    lambda x: x.all(), axis=1)
allCountries = allCountries.loc[mask,:]

a = allCountries.shape[0]
print('{} rows removed'.format(b-a)) # 7.3M



#%% GUARDAR PKLS

allCountries.to_pickle("./allCountries.pkl")
c500.to_pickle("./c500.pkl")
rg1000.to_pickle("./rg1000.pkl")

#%% LEER PKLS

rg1000 = pd.read_pickle('./rg1000.pkl')
c500 = pd.read_pickle('./c500.pkl')
allCountries = pd.read_pickle('./allCountries.pkl')



#############
# PRECISIÓN #
#############



#%% TESTEAR CON c500 - rg1000

# Filtro c500 para que las filas que queden, en principio, no estén en rg1000

def all_conditions_ok(row):
    result = False
    if row['population'] <= 1000:
        if row['featurecode'] not in ['PPL','PPLA','PPLA2','PPLA3']:
            result = True
    return result
mask = c500.apply(all_conditions_ok, axis=1)
filtered = c500.loc[mask, :].copy()

# Para asegurarme de que no queda ninguna fila que esté en rg1000, elimino
# también las que coindidan con ciudad y país
diff = pd.merge(filtered, rg1000[['name','cc']], how='left', indicator=True,
                left_on = ['asciiname','countrycode'],
                right_on = ['name','cc'],
                suffixes=('','_'))
diff = diff[diff['_merge']=='left_only'].copy()

# Calcular cuántas veces se equivoca y cuántas acierta de país.
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
print(np.sum(have_different_cc)) # 7/1094 errores

#%% TESTEAR CON allCountries - rg1000

# Filtro allCountries para que las filas que queden, en principio, no estén en 
# rg1000 y solo haya municipios (no montañas, ni ríos...)
# def all_conditions_ok(row):
#     result = False
#     if row['population'] <= 1000:
#         if row['featurecode'] not in ['PPL','PPLA','PPLA2','PPLA3']:
#             result = True
#     return result
    
# mask = allCountries.apply(all_conditions_ok, axis=1)
# filtered = allCountries.loc[mask, :].copy()
# filtered.to_pickle('allCountries_filtered_rg_cities1000.pkl')

filtered = pd.read_pickle('allCountries_filtered_rg_cities1000.pkl')

# Para asegurarme de que no queda ninguna fila que esté en rg1000, elimino
# también las que coindidan con ciudad y país
diff = pd.merge(filtered, rg1000[['name','cc']], how='left', indicator=True,
                left_on = ['asciiname','countrycode'],
                right_on = ['name','cc'],
                suffixes=('','_'))
diff = diff[diff['_merge']=='left_only'].copy()

# Calcular cuántas veces se equivoca y cuántas acierta de país.
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
print(np.sum(have_different_cc)) # 10k/320k errores



########################
# TIEMPO COMPUTACIONAL #
########################



#%% CON rg_cities1000 COMO BASE DE DATOS

ncoords = [int(10**i) for i in range(8)]
many_coords = coords*40 # size ~12M

for n in ncoords:
    choosen_coords = many_coords[0:n]
    start_time = time.perf_counter()
    results = rg.search(choosen_coords)
    time_elapsed = time.perf_counter() - start_time
    print('{} & {:.2f}'.format(n, time_elapsed))
    
# Podemos permitirnos sacrificar performance para ganar precisión: 10M en solo
# 12.8 segundos, en una máquina poco potente.

#%% Con c500 COMO BASE DE DATOS

#!!! La máquina peta si le metes muchas coords. 

stream = io.StringIO(open('c500_stream.csv', encoding='utf-8').read())
geo = rg.RGeocoder(mode=2, verbose=True, stream=stream)

ncoords = [int(10**i) for i in range(8)]
many_coords = coords*40 # size ~12M

for n in ncoords:
    choosen_coords = many_coords[0:int(n)]
    start_time = time.perf_counter()
    results = geo.query(choosen_coords)
    time_elapsed = time.perf_counter() - start_time
    print('{} & {:.2f}'.format(n, time_elapsed)) 

# IMPORTANTE
# Queda probado que, aun con los 4.7M de filas del csv personalizado que le 
# hemos pasado a reverse:geocoder, la performance sigue siendo buenísima:
# 12.8 segundos para 10M de coordenadas.

########################################
# LO ANTERIOR, TODO, ES UN POCO... MEH #
########################################

###################################
# TESTEANDO CON LA MUESTRA DEL BS #
###################################

# Lo suyo sería testearlo con la librería por defecto y luego meterle el 
# allCountries como base de datos y quizá cambiar la distancia euclídea.

#%% CON allCountries COMO BASE DE DATOS
### BORRAR TODA ESTA CELDA CUANDO YA NO ME HAGA FALTA PORQUE LO QUE APERECE 
### ESTÉ EN LA SIGUIENTE CELDA.

# Testear reverese_geocoder con allCountries_stream.csv como base de datos.
# Utilizo la muestra del BS, obtengo lo cc con reverse_geocoder y con REST 
# requests a la api de geonames. Luego compruebo cuántas coinciden.

# LEER DATOS Y PASAR A LISTA DE TUPLAS LAS COORDS
sample_bs = pd.read_csv('Sample_BS.csv', index_col=0, sep=';', usecols=[0,1,2])
sample_bs.rename({'latitud_ga':'latitude','longitud_ga':'longitude'}, 
                 axis='columns', inplace=True)

coords = sample_bs.to_numpy(copy=True).tolist()
coords = [tuple(i) for i in coords]

# REVERSE_GEOCODER
stream = io.StringIO(open('allCountries_stream.csv', encoding='utf-8').read())
geo = rg.RGeocoder(mode=2, verbose=True, stream=stream)
results = geo.query(coords)
results = pd.DataFrame(results).cc

results.value_counts() # Only 1283 not in Spain
outspain = results.loc[results!='ES',:]
inspain = results.loc[results=='ES',:].iloc[0:1283,:]
dftest = pd.concat([inspain, outspain])

# REQUESTS TO GEONAMES
def get_cc(row):
    params = {'lat':row['lat'],'lng':row['lon'],'username':'marcoscastillo'}
    req = requests.get("http://api.geonames.org/countryCode", params=params)
    return req.text.strip()

getcc0_999 = dftest.iloc[:1000,:].apply(get_cc, axis='columns')
getcc1000_1999 = dftest.iloc[1000:2000,:].apply(get_cc, axis='columns')
getcc2000_2566 = dftest.iloc[2000:,:].apply(get_cc, axis='columns')
getcc0_2566 = pd.concat([getcc0_999, getcc1000_1999, getcc2000_2566])
getcc0_2566.to_pickle('getcc0_2566.pkl')

# CHECK ACCURACY
getcc0_2566 = pd.read_pickle('getcc0_2566.pkl')

different = dftest.cc.to_numpy() != getcc0_2566.to_numpy()
np.mean(different) # .007
np.sum(different) # 17/2566
# From the 17 errors, 14 come from different kind of errors:  
# ERR:15:no country code found, ERROR: canceling statement due to statement 
# timeout. The other 3 are actual errors: in the border between PT and ES, and
# FR and BE. So the accuracy is even better. 

# ¿Dónde están los errores?
dftest['ccapi'] = getcc0_2566
dftest.loc[different, ['lat','lon']]
dftest.loc[different, ['cc','ccapi']]
# Busco las coordenadas en Google Maps para salir de dudas: ¿son errores?
# V 40.61667, 0.6  ES
# V 42.03333, -8.65 ES
# V 36.59389, -6.23298 ES
# V -5.7333300000000005, 39.28333 TZ
# V -33.86847, 151.20033999999998 AU
# F (OCEAN) 0.251, 0.7794 YE
# V 18.433329999999998, -69.68333 DO
# V 10.20117, -64.68108000000001 VE
# F (ES) 42.056090000000005, -8.56749 PT
# V -16.501479999999997, -151.72135 PF
# F (OCEAN) 0.251, 0.7794 YE
# V 11.99928, 102.29564 TH
# V -2.2602700000000002, -79.86536 EC
# V 11.07567, -63.827909999999996 VE
# F (OCEAN) 0.251, 0.7794 YE
# V 10.609960000000001, -67.01039 VE
# F (FR) 50.4441, 3.65938 BE
# NOTE: the OCEAN coords are repeated 3 times: only 1 counts.
# ACTUAL ERRORS: 3/2566, 0.00117 TER.

#%% REQUESTS API GEONAMES

# Función para sacar el conutry code 'cc' de dos letras.
# Usernames: marcoscastillo, marcoscastevez, mascosestevez
def get_cc(row):
    params = {'lat':row['latitude'],'lng':row['longitude'],
              'username':'marcoscastillo'}
    req = requests.get("http://api.geonames.org/countryCode", params=params)
    return req.text.strip()

def get_cc_2(row):
    params = {'lat':row['latitude'],'lng':row['longitude'],
              'username':'marcoscastevez'}
    req = requests.get("http://api.geonames.org/countryCode", params=params)
    return req.text.strip()

def get_cc_3(row):
    params = {'lat':row['latitude'],'lng':row['longitude'],
              'username':'mascosestevez'}
    req = requests.get("http://api.geonames.org/countryCode", params=params)
    return req.text.strip()

def get_cc_4(row):
    params = {'lat':row['latitude'],'lng':row['longitude'],
              'username':'mellamomarcos'}
    req = requests.get("http://api.geonames.org/countryCode", params=params)
    return req.text.strip()


# Cargamos la muestra de BS y la barajamos de forma aleatoria. Como quizá no
# hagamos requests con las 48k instancias, si lo hacemos a submuestras, estas 
# deben ser aleatorias.
sample = pd.read_csv('Sample_BS.csv', index_col=0, sep=';', usecols=[0,1,2])
sample.rename({'latitud_ga':'latitude','longitud_ga':'longitude'}, 
                 axis='columns', inplace=True)
np.random.seed(0)
permut = np.random.permutation(sample.shape[0])
sample_shuffled = sample.iloc[permut,:]


# USER marcoscastillo (CONSOLE 2)
for i in np.arange(28000,34000,1000):
    time_start = time.time()
    print(i)
    getcc = sample_shuffled.iloc[i:i+1000,:].apply(get_cc, axis='columns')    
    getcc.to_pickle('getcc{}_{}.pkl'.format(i,i+1000))
    print('Computed untill {} row'.format(i+1000))
    
    time_sleep = 3600 - (time.time() - time_start)/3
    frac, whole = math.modf(time_sleep/60)
    print(datetime.datetime.now())
    print('Sleeping {} mins and {} secs'.format(int(whole), int(frac*60)))
    time.sleep(time_sleep)

# USER marcoscastevez (CONSOLE 3)
for i in np.arange(23000,34000,1000):
    time_start = time.time()
    print(i)
    getcc = sample_shuffled.iloc[i:i+1000,:].apply(get_cc_2, axis='columns')    
    getcc.to_pickle('getcc{}_{}.pkl'.format(i,i+1000))
    print('Computed untill {} row'.format(i+1000))
    
    time_sleep = 3600 - (time.time() - time_start)/3
    frac, whole = math.modf(time_sleep/60)
    print(datetime.datetime.now())
    print('Sleeping {} mins and {} secs'.format(int(whole), int(frac*60)))
    time.sleep(time_sleep)
    
# USER mascosestevez (CONSOLE 4)
for i in np.arange(34000,48000,1000):
    time_start = time.time()
    print(i)
    getcc = sample_shuffled.iloc[i:i+1000,:].apply(get_cc_3, axis='columns',)    
    getcc.to_pickle('getcc{}_{}.pkl'.format(i,i+1000))
    print('Computed untill {} row'.format(i+1000))
    
    time_sleep = 3600 - (time.time() - time_start)/3
    frac, whole = math.modf(time_sleep/60)
    print(datetime.datetime.now())
    print('Sleeping {} mins and {} secs'.format(int(whole), int(frac*60)))
    time.sleep(time_sleep)
    
# USER mellamomarcos (CONSOLE 5)
for i in np.arange(42000,48000,1000):
    time_start = time.time()
    print(i)
    getcc = sample_shuffled.iloc[i:i+1000,:].apply(get_cc_4, axis='columns')    
    getcc.to_pickle('getcc{}_{}.pkl'.format(i,i+1000))
    print('Computed untill {} row'.format(i+1000))
    
    time_sleep = 3600 - (time.time() - time_start)/3
    frac, whole = math.modf(time_sleep/60)
    print(datetime.datetime.now())
    print('Sleeping {} mins and {} secs'.format(int(whole), int(frac*60)))
    time.sleep(time_sleep)

    
#%% TESTEAR CON CITIES1000

b = pd.Series(dtype=str)    
for i in np.arange(0,17000,1000):
    a = pd.read_pickle('getcc{}_{}.pkl'.format(i,i+1000))
    b = pd.concat([b,a])
    
coords = subsample_20k.to_numpy(copy=True).tolist()
coords = [tuple(i) for i in coords]

results_rg = rg.search(coords)
results_rg = pd.DataFrame(results_rg).cc
results_rg = results_rg[:17000]
