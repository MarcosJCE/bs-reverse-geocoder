#%% PACKAGES

import pandas as pd
import numpy as np
from time import time
from sklearn import neighbors, tree
from sklearn.model_selection import KFold, GridSearchCV
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#%% READ DATA

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

##################
# Decision Trees #
##################


#%% DT
# We are just going to do hyperparameter tunning with criterion, as the rest of
# the parameters are set by default to the values for achieving higher
# accuracy.

np.random.seed(0)
dt = GridSearchCV(tree.DecisionTreeClassifier(),
                  param_grid={'criterion': ['gini', 'entropy']},
                  scoring='accuracy',
                  cv=KFold(n_splits=5, shuffle=True, random_state=0),
                  n_jobs=-1)
dt.fit(X_train, y_train)

start = time()
y_test_pred = dt.predict(X_test)
print("Time {:.2f} secs".format(time()-start))

ter = np.mean(y_test != y_test_pred)
print("TER {:.2f}%".format(ter*100))

print(dt.best_params_)

"""
Time 0.02 secs
TER 0.42%
{'criterion': 'entropy'}
"""


############
### k-NN ###
############

#%% KNN1

np.random.seed(0)

param_grid = {'n_neighbors':[1,3,5,7],
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

"""
Time 0.24 secs
TER 0.0620%
{'n_neighbors': 3, 'weights': 'distance'}
"""

#%% KNN2

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

"""
Time 0.23 secs
TER 0.0620%
{'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}
"""

#%% KNN3
# We tried something much different trying to reduce much more the TER
# as we were still far from the 0.07% TER of the reverse_geocoder library.
# We used the Harvesine distance.

np.random.seed(0)

param_grid = [{'n_neighbors': [1],
               'metric': ['haversine'],
               'algorithm':['ball_tree']},
              {'n_neighbors': [3,5],
               'metric': ['haversine'],
               'algorithm':['ball_tree'],
               'weights':['uniform','distance']}
              ]

knn = GridSearchCV(neighbors.KNeighborsClassifier(),
                   param_grid=param_grid,
                   scoring='accuracy',
                   cv=KFold(n_splits=5, shuffle=True, random_state=0),
                   n_jobs=-1)
knn.fit(X_train*np.pi/180, y_train)
dump(knn, 'knn_haversine.joblib')

start = time()
y_test_pred = knn.predict(X_test*np.pi/180)
print("Time {:.2f} secs".format(time()-start))

ter = np.mean(y_test != y_test_pred)
print("TER {:.4f}%".format(ter*100))

print(knn.best_params_)

"""
Time 23.62 secs
TER 0.0640%
{'algorithm': 'ball_tree', 'metric': 'haversine', 'n_neighbors': 3,
 'weights': 'distance'}
"""


###########################
# Support Vector Machines #
###########################

#%% SVM1

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', LinearSVC(C=.0001, random_state=0, verbose=1))
    ])

pipe.fit(X_train, y_train)

time_start = time()
y_pred = pipe.predict(X_test)
print("Time {:.1f} secs".format(time()-time_start))

ter = np.mean(y_pred != y_test)
print("TER {:.4f}%".format(100*ter))

"""
Time 0.1 secs
TER 99.5869%
"""

#%% SVM2

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', LinearSVC(C=1.0, tol=0.01, random_state=0, verbose=1))
    ])

pipe.fit(X_train, y_train)

time_start = time()
y_pred = pipe.predict(X_test)
print("Time {:.1f} secs".format(time()-time_start))

ter = np.mean(y_pred != y_test)
print("TER {:.4f}%".format(100*ter))

"""
Time 23 min 17 secs
TER 0.86%
"""

#%% SAVE THE BEST MODEL

# The best model was the KNN with k=3, metric='euclidean', weights='distance'

X_final = np.vstack((X_train, X_test))
y_final = np.hstack((y_train,y_test))

np.random.seed(0)

param_grid = {'n_neighbors':[1, 3, 5, 7],
              'metric':["euclidean","manhattan","chebyshev"],
              'weights':['uniform', 'distance']}

clf = GridSearchCV(neighbors.KNeighborsClassifier(),
                   param_grid,
                   scoring='accuracy',
                   cv=KFold(n_splits=5, shuffle=True, random_state=0),
                   n_jobs=-1)

clf.fit(X_train, y_train)

print(clf.best_params_)
# {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}

dump(clf, 'knn_final.joblib')

print(1 - clf.score(X_final, y_final))
# 0.00012 train TER



# CHECKING IF THE JOBLIB FILE IS OK

clf = load('knn_final.joblib')

print(clf.best_params_)
# {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}
print(1 - clf.score(X_final, y_final))
# 0.00012 train TER
# It seems that the file is ok.
