#%% PACKAGES

import pandas as pd
import numpy as np
import reverse_geocoder as rg
import time
import io
import requests
import datetime
import math



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

## CON c500
b = c500.shape[0]

# Remove duplicates: same (lat,lon)
c500.drop_duplicates(subset=['latitude','longitude'], inplace=True)

# Remove rows with nas in lat, lon or cc
mask = c500[['latitude','longitude','countrycode']].notna().apply(lambda x: 
                                                          x.all(), axis=1)
c500 = c500.loc[mask,:]

a = c500.shape[0]
print('{} rows removed'.format(b-a)) # 184


## CON allCountries
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

# Para ahorrarnos el tiempo y leerlos así directamente la próxima vez, como
# hacemos en la siguiente celda
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
def all_conditions_ok(row):
    result = False
    if row['population'] <= 1000:
        if row['featurecode'] not in ['PPL','PPLA','PPLA2','PPLA3']:
            result = True
    return result
    
mask = allCountries.apply(all_conditions_ok, axis=1)
filtered = allCountries.loc[mask, :].copy()

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

# Con muchísimas coordenadas, más de las que tendría tratar en una situación 
# real

#%% Con muchísimas coordenadas, más que en situación real 

#!!! La máquina peta si le metes muchas coords. 

ncoords = [int(10**i) for i in range(8)]
many_coords = coords*40 # size ~12M

for n in ncoords:
    choosen_coords = many_coords[0:n]
    start_time = time.perf_counter()
    results = rg.search(choosen_coords)
    time_elapsed = time.perf_counter() - start_time
    print('{} & {:.2f}'.format(n, time_elapsed))



###################################
# TESTEANDO CON LA MUESTRA DEL BS #
###################################

#%% REQUESTS API GEONAMES

# Función para sacar el country code 'cc' de dos letras.
def get_cc(row):
    params = {'lat':row['latitud_ga'],'lng':row['longitud_ga'],
              'username':'marcoscastillo'}
    req = requests.get("http://api.geonames.org/countryCode", params=params)
    return req.text.strip()

sample = pd.read_csv('Sample_BS.csv', index_col=0, sep=';', usecols=[0,1,2])

# Generate all the country codes
for i in np.arange(0,48000,1000):
    time_start = time.time()
    getcc = sample.iloc[i:i+1000,:].apply(get_cc, axis='columns')    
    getcc.to_pickle('getcc{}_{}.pkl'.format(i,i+1000))
    print('Computed untill {} row'.format(i+1000))
    
    time_sleep = 3600 - (time.time() - time_start)/3
    frac, whole = math.modf(time_sleep/60)
    print(datetime.datetime.now())
    print('Sleeping {} mins and {} secs'.format(int(whole), int(frac*60)))
    time.sleep(time_sleep)

getcc2 = sample.iloc[48000:48475,:].apply(get_cc, axis='columns')
getcc = pd.read_pickle('getcc0_48000.pkl')
getcc = pd.concat([getcc,getcc2])
getcc.to_pickle('getcc0_48000.pkl')
    

    
#%% TESTS
sample = pd.read_csv('Sample_BS.csv', index_col=0, sep=';', usecols=[0,1,2])
results_gn = pd.read_pickle('getcc0_48475.pkl')

# Para ver la distribución de países con variable explorer
cc_distrib = results_gn.value_counts()

# number of no result (ERROR) answer
n_errors = np.sum(results_gn.apply(lambda cc: len(cc)!=2))
results_gn = results_gn.to_numpy()

coords = sample[['latitud_ga','longitud_ga']].to_numpy().tolist()
coords = [tuple(i) for i in coords]

# CON DATOS POR DEFECTO
results_rg = rg.search(coords)
results_rg = pd.DataFrame(results_rg).cc.to_numpy()

n_rg_errors = np.sum(results_rg != results_gn) - n_errors 
print(n_rg_errors) # 35

rg_accuracy = n_rg_errors / (results_rg.size - n_errors)
print(rg_accuracy) # 0.0007

# CON allCountries 'adaptado'
stream = io.StringIO(open('allCountries_stream.csv', encoding='utf-8').read())
geo = rg.RGeocoder(mode=2, verbose=True, stream=stream)
results_rg2 = geo.query(coords)
results_rg2 = pd.DataFrame(results_rg2).cc.to_numpy()

n_rg2_errors = np.sum(results_rg2 != results_gn) - n_errors 
print(n_rg2_errors) # 35

rg2_accuracy = n_rg2_errors / (results_rg2.size - n_errors)
print(rg2_accuracy) # 0.0007

# CON allCountries crudo
stream = io.StringIO(open('allCountries_stream_bigger.csv', 
                          encoding='utf-8').read())
geo = rg.RGeocoder(mode=2, verbose=True, stream=stream)
results_rg3 = geo.query(coords)
results_rg3 = pd.DataFrame(results_rg3).cc.to_numpy()

n_rg2_errors = np.sum(results_rg3 != results_gn) - n_errors 
print(n_rg2_errors) # 35

rg2_accuracy = n_rg2_errors / (results_rg3.size - n_errors)
print(rg2_accuracy) # 0.0007

# ¿DAN AMBOS EXACTAMENTE LOS MISMOS RESULTADOS?
np.sum(results_rg!=results_rg2)
# Sí, los resultados son los mismos.

# POR QUÉ SUCEDEN LOS ERRORES
e_gn = results_gn[results_rg != results_gn].reshape(99,1)
e_rg = results_rg[results_rg != results_gn].reshape(99,1)
concat = np.concatenate((e_gn, e_rg), axis=1)

concat = concat.tolist()
concat = [mylist[0] + "-" +mylist[1] for mylist in concat]
concat = pd.Series(concat)
error_types = concat.value_counts()
# Con el explorador de variables de spyder veo los errores más frecuentes.
# Confundir ES con GB, FR, PT los que más.
