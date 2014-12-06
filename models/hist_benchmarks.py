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
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from scipy.stats.stats import pearsonr
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

USE_LOG = False
USE_CALIBRATION = False
GENERATE_PLOTS = False

data = pd.read_csv('data/hist_DUPROPRIO_v2.csv')
merged = data
for category in ['Condominium']:
  print "*" * 80
  print category

  X = merged[['NumberBedrooms', 'NumberBathrooms', 'LivingArea', 'SaleYYYY', 'AskingPrice']]
  Y = merged[['PriceSold']]

  X = X[merged['Category']==category]
  Y = Y[merged['Category']==category]
  print 'X.shape: ', X.shape
  print 'Y.shape: ', Y.shape

  # filter rows with NaN
  mask = ~np.isnan(X).any(axis=1)
  X = X[mask]
  Y = Y[mask]
  mask = ~np.isnan(Y).any(axis=1)
  X = X[mask]
  Y = Y[mask]
  print 'After NaN filter: ', X.shape

  # remove high-end listings
  mask = Y['PriceSold'] < 500000
  X = X[mask]
  Y = Y[mask]
  print 'After upper-bound filter: ', X.shape

  # remove low-end listings
  mask = Y['PriceSold'] > 10**5
  X = X[mask]
  Y = Y[mask]
  print 'After lower-bound filter: ', X.shape

  columns = X.columns.values
  X = np.array(X)
  Y = np.array(Y)
  if USE_LOG:
    Y = np.log(Y)
  Y = Y.reshape(Y.shape[0])

  print "mean: ", np.mean(Y)
  print "median: ", np.median(Y)
  print "std: ", Y.std()

  # remove outliers
  mask = np.abs(Y-np.mean(Y)) <= (3*Y.std())
  X = X[mask]
  Y = Y[mask]

  skf = KFold(n=X.shape[0], n_folds=10, shuffle=True, random_state=42)
  L = { 'rmse': [], 'corr': [], 'r2': [], 'diff': [], 'mae': [], 'explained_var': [], 'var': []}
  for train_indices, test_indices in skf:
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

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

    if USE_LOG:
      Y_test_10 = np.exp(Y_test)
      preds_10 = np.exp(preds)
    else:
      Y_test_10 = Y_test
      preds_10 = preds
    rmse = math.sqrt(metrics.mean_squared_error(Y_test_10, preds_10))
    corr = pearsonr(preds_10, Y_test_10)
    diff = np.array([abs(p-a)/a for (p,a) in zip(Y_test_10, preds_10)])
    mae = metrics.mean_absolute_error(Y_test_10, preds_10)
    explained_var = metrics.explained_variance_score(Y_test_10, preds_10)
    r2 = metrics.r2_score(Y_test_10, preds_10)
    var = np.var(diff)

    L['rmse'].append(rmse)
    L['corr'].append(corr[0])
    L['diff'].append(diff.mean())
    L['mae'].append(mae)
    L['explained_var'].append(explained_var)
    L['r2'].append(r2)
    L['var'].append(var)

    if GENERATE_PLOTS:
      plt.plot(Y_test_10, preds_10, 'ro')
      plt.show()
      break
  for key in L.keys():
    print "Mean %s: %f" % (key, np.array(L[key]).mean())
