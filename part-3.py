############
# INTERSECTION OF THE ERRORS
############

# Between the reverse_geocoder case and the best kNN

#%% PACKAGES

import pandas as pd
import numpy as np
import reverse_geocoder as rg
from time import time
import math
from sklearn import neighbors, tree
from sklearn.model_selection import KFold, GridSearchCV
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#%% RG

sample = pd.read_pickle('sample.pkl')
sample['cc'] = pd.read_pickle('sample_cc.pkl')

# Removing the errors in the queries to the API of geonames
mask = sample['cc'].str.len() == 2
sample = sample.loc[mask,:].copy()

# Transforming the coordinates to the suitable format for being the input of
# the function rg.search
coords = sample[['lat','lon']].to_numpy(copy=True).tolist()
coords = [tuple(i) for i in coords]

time_start = time()
results_rg = rg.search(coords)
print(time()-time_start) # 0.4 s

results_rg = pd.DataFrame(results_rg).cc.to_numpy(copy=True)
n_rg_wrong = np.sum(results_rg != sample.cc.to_numpy())
print(n_rg_wrong) # 35

rg_accuracy = n_rg_wrong / results_rg.size
print(rg_accuracy) # 0.0007

#%% KNN BEST

# READ DATA

cities500 = pd.read_pickle('rg_cities500_clean.pkl')
sample_cc = pd.read_pickle('sample_cc.pkl')


sample = pd.read_pickle('sample.pkl')

sample['cc'] = sample_cc

# Removing the errors in the queries to the API of geonames
mask = sample['cc'].str.len() == 2
sample = sample.loc[mask,:].copy()

cities500 = cities500[['lat','lon','cc']].copy()

# We used the same split between train and test than the one made for the
# reverse_geocoder library method and function
X_train = cities500[['lat','lon']].to_numpy(copy=True)
y_train = cities500['cc'].to_numpy(copy=True)

X_test = sample[['lat','lon']].to_numpy(copy=True)
y_test = sample['cc'].to_numpy(copy=True)


# KNN PROCESS

np.random.seed(0)

param_grid = {'n_neighbors':[1, 3, 5, 7],
              'metric':["euclidean","manhattan","chebyshev"],
              'weights':['uniform', 'distance']}

knn = GridSearchCV(neighbors.KNeighborsClassifier(),
                   param_grid=param_grid,
                   scoring='accuracy',
                   cv=KFold(n_splits=5, shuffle=True, random_state=0),
                   n_jobs=-1)
knn.fit(X_train, y_train)

start = time()
y_test_pred = knn.predict(X_test)
print("Time {:.2f} secs".format(time()-start))

ter = np.mean(y_test != y_test_pred)
print("TER {:.4f}%".format(ter*100))

print(knn.best_params_)

mask = y_test != y_test_pred
diffs = pd.DataFrame({'actual':y_test[mask], 'pred':y_test_pred[mask]})
diffs = diffs.sort_values(by=['actual','pred'])

#%% COMPARISON

df = pd.DataFrame({'actual_cc': y_test, 'rg':results_rg, 'knn':y_test_pred, 'lat': sample.lat,
                   'lon': sample.lon})

mask2 = (y_test != y_test_pred) & (results_rg != sample.cc.to_numpy())

df2 = df.loc[mask2,:].copy()
