#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2

from __future__ import division
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sys
from sklearn.base import BaseEstimator, TransformerMixin
#from nolearn.dbn import DBN
from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.externals import joblib
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn import metrics
np.random.seed(42)

USE_CALIBRATION = True
GENERATE_PLOTS = False

data = pd.read_csv('data/modified_listings.csv')
extra_data = pd.read_csv('data/post_extra_data.csv')
extra_data = extra_data[['MlsNumber','LivingArea']]
merged = pd.merge(data, extra_data, on='MlsNumber', suffixes=['_left', '_right'])
merged.to_csv('all_data.csv')
for col in ['AP']:#, 'COP', 'AP', 'LS', 'MA']:#, 'PPR', '2X', '3X', '4X', '5X', 'AU', 'UNI', 'MEM']:
  print "*" * 80
  print col

  #X = merged[['LivingArea','WalkScore', 'NbPieces', 'NbChambres', 'NbSallesEaux', 'NbSallesBains', 'NbFoyerPoele', 'NbEquipements', 'NbGarages', 'NbStationnements', 'NbPiscines', 'NbBordEaux']]
  X = merged.drop(['MlsNumber', 'Lat', 'Lng', 'BuyPrice'], axis=1, inplace=False)
  Y = merged[['BuyPrice']]

  X = X[merged[col]==1]
  Y = Y[merged[col]==1]
  print 'X.shape: ', X.shape
  print 'Y.shape: ', Y.shape

  # filter rows with NaN
  mask = ~np.isnan(X).any(axis=1)
  X = X[mask]
  Y = Y[mask]
  print 'After NaN filter: ', X.shape

  # remove high-end listings
  mask = Y['BuyPrice'] < 500000
  X = X[mask]
  Y = Y[mask]
  print 'After upper-bound filter: ', X.shape

  # remove low-end listings
  mask = Y['BuyPrice'] > 10**5
  X = X[mask]
  Y = Y[mask]
  print 'After lower-bound filter: ', X.shape

  print "mean: ", Y['BuyPrice'].mean()
  print "median: ", Y['BuyPrice'].median()
  print "std: ", Y['BuyPrice'].std()

  # remove outliers
  mask = np.abs(Y['BuyPrice']-Y['BuyPrice'].median()) <= (3*Y['BuyPrice'].std())
  X = X[mask]
  Y = Y[mask]

  X = np.array(X)
  Y = np.array(Y)
  Y = Y.reshape(Y.shape[0])
  skf = KFold(n=X.shape[0], n_folds=10, shuffle=True, random_state=42)
  L = []
  for train_indices, test_indices in skf:
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    if USE_CALIBRATION:
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)

      clf1 = RandomForestRegressor()
      clf1.fit(X_train, Y_train)
      clf1_train = clf1.predict(X_train)
      clf1_test = clf1.predict(X_test)

      clf1_train = clf1_train.reshape((clf1_train.shape[0],1))
      clf1_test = clf1_test.reshape((clf1_test.shape[0],1))

      clf1_train = np.concatenate([clf1_train, X_train], axis=1)
      clf1_test = np.concatenate([clf1_test, X_test], axis=1)

      clf = GradientBoostingRegressor()
      clf.fit(clf1_train, Y_train)
      preds = clf.predict(clf1_test).astype(float)
    else:
      clf = Pipeline([
         ('scaler', StandardScaler()),
         # ('clf', AdaBoostRegressor()),
         # ('clf', ARDRegression()),
         # ('clf', BaggingRegressor()),
         # ('clf', BayesianRidge()),
         # ('clf', ElasticNet()),
         # ('clf', ExtraTreesRegressor()),
          ('clf', GradientBoostingRegressor()),
         # ('clf', KNeighborsRegressor(n_neighbors=5)),
         # ('clf', Lasso()),
         # ('clf', LinearRegression()),
         # ('clf', PassiveAggressiveRegressor()),
         # ('clf', RandomForestRegressor()),
         # ('clf', Ridge(alpha=0.5, normalize=False)),
         # ('clf', SVR()),
      ])
      clf.fit(X_train, Y_train)
      preds = clf.predict(X_test).astype(float)

    rmse = math.sqrt(metrics.mean_squared_error(preds, Y_test))
    print rmse
    L.append(rmse)

    if GENERATE_PLOTS:
      plt.plot(Y_test, preds, 'ro')
      plt.show()
      break
  print "Mean RMSE: ", np.array(L).mean()

