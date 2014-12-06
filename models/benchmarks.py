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

USE_LOG = True
USE_NEURALNET = False
USE_CALIBRATION = False
GENERATE_PLOTS = False

data = pd.read_csv('data/modified_listings.csv')
extra_data = pd.read_csv('data/post_extra_data.csv')
extra_data = extra_data[['MlsNumber','LivingArea']]
merged = pd.merge(data, extra_data, on='MlsNumber', suffixes=['_left', '_right'])
merged.to_csv('data/all_data.csv')
for col in ['AP']:#, 'COP', 'AP', 'LS', 'MA']:#, 'PPR', '2X', '3X', '4X', '5X', 'AU', 'UNI', 'MEM']:
  print "*" * 80
  print col

  #X = merged[['LivingArea','AvgIncome', 'WalkScore', 'NbPieces', 'NbChambres', 'NbSallesEaux', 'NbSallesBains', 'NbFoyerPoele', 'NbEquipements', 'NbGarages', 'NbStationnements', 'NbPiscines', 'NbBordEaux']]
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
  #mask = Y['BuyPrice'] < 500000
  #X = X[mask]
  #Y = Y[mask]
  #print 'After upper-bound filter: ', X.shape

  # remove low-end listings
  #mask = Y['BuyPrice'] > 10**5
  #X = X[mask]
  #Y = Y[mask]
  #print 'After lower-bound filter: ', X.shape

  print "mean: ", Y['BuyPrice'].mean()
  print "median: ", Y['BuyPrice'].median()
  print "std: ", Y['BuyPrice'].std()

  columns = X.columns.values
  X = np.array(X)
  Y = np.array(Y)
  if USE_LOG:
    Y = np.log(Y)
  Y = Y.reshape(Y.shape[0])

  # remove outliers
  mask = np.abs(Y-np.mean(Y)) <= (3*Y.std())
  X = X[mask]
  Y = Y[mask]

  skf = KFold(n=X.shape[0], n_folds=10, shuffle=True, random_state=42)
  L = { 'rmse': [], 'corr': [], 'r2': [], 'diff': [], 'mae': [], 'explained_var': [], 'var': []}
  for train_indices, test_indices in skf:
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    if USE_CALIBRATION:
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)

      clf1 = GradientBoostingRegressor()
      clf1.fit(X_train_scaled, Y_train)
      clf1_train = clf1.predict(X_train_scaled)
      clf1_test = clf1.predict(X_test_scaled)

      clf1_train = clf1_train.reshape((clf1_train.shape[0],1))
      clf1_test = clf1_test.reshape((clf1_test.shape[0],1))

      clf1_train = np.concatenate([clf1_train, X_train_scaled], axis=1)
      clf1_test = np.concatenate([clf1_test, X_test_scaled], axis=1)

      clf = GradientBoostingRegressor()
      clf.fit(clf1_train, Y_train)
      preds = clf.predict(clf1_test).astype(float)
    elif USE_NEURALNET:
      Y_train, Y_test = Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)
      hidden_size = 100

      train_ds = SupervisedDataSet(X_train.shape[1], Y_train.shape[1])
      train_ds.setField('input', X_train)
      train_ds.setField('target', Y_train)
      net = buildNetwork(X_train.shape[1], hidden_size, Y_train.shape[1], bias=True)
      trainer = BackpropTrainer(net, train_ds)

      epochs = 10
      for i in xrange(epochs):
        mse = trainer.train()
        rmse = math.sqrt(mse)
        print "epoch: %d, rmse: %f" % (i, rmse)

      test_ds = SupervisedDataSet(X_test.shape[1], Y_test.shape[1])
      test_ds.setField('input', X_test)
      test_ds.setField('target', Y_test)
      preds = net.activateOnDataset(test_ds)
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
