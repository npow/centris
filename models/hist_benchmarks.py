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

"""
keys = [
  'AskingPrice', 'PriceSold',
  'NumberBedrooms', 'NumberBathrooms', 'LivingArea', 'LotSize', 'DaysOnMarket',
  'Category', 'Borough', 'Address',
  'SaleYYYY', 'SaleMM', 'SaleDD', 'SaleYYYYMMDD',
  'Apartment_Benchmark', 'Apartment_HPI',
  'Composite_Benchmark', 'Composite_HPI',
  'One_Storey_Benchmark', 'One_Storey_HPI',
  'Single_Family_Benchmark', 'Single_Family_HPI',
  'Townhouse_Benchmark', 'Townhouse_HPI',
  'Two_Storey_Benchmark', 'Two_Storey_HPI'
]
"""
data = pd.read_csv('data/hist_DUPROPRIO_v2.csv')
merged = data
categorical_columns = ['Borough', 'SaleYYYY', 'SaleMM', 'SaleDD']
for category in ['Condominium']:
  print "*" * 80
  print category

  X = merged[['NumberBedrooms', 'NumberBathrooms', 'LivingArea', 'DaysOnMarket', 'Composite_HPI']]
  X_cat = merged[categorical_columns]
  Y = merged[['PriceSold']]

  mask = merged['Category']==category
  X, X_cat, Y = X[mask], X_cat[mask], Y[mask]
  print 'X.shape: ', X.shape
  print 'Y.shape: ', Y.shape

  # filter rows with NaN
  mask = ~np.isnan(X).any(axis=1)
  X, X_cat, Y = X[mask], X_cat[mask], Y[mask]

  mask = ~np.isnan(Y).any(axis=1)
  X, X_cat, Y = X[mask], X_cat[mask], Y[mask]
  print 'After NaN filter: ', X.shape

  X, X_cat, Y = np.array(X), np.array(X_cat), np.array(Y)
  if USE_LOG:
    Y = np.log(Y)
  Y = Y.reshape(Y.shape[0])

  print "mean: ", np.mean(Y)
  print "median: ", np.median(Y)
  print "std: ", Y.std()

  # remove outliers
  mask = np.abs(Y-np.mean(Y)) <= (3*Y.std())
  X, X_cat, Y = X[mask], X_cat[mask], Y[mask]

  # one-hot encode categorical features
  X_cat_enc = []
  for i, cat in enumerate(categorical_columns):
    col = X_cat[:,i]
    col = LabelEncoder().fit_transform(col).reshape((-1,1))
    col_enc = OneHotEncoder(sparse=False).fit_transform(col)
    X_cat_enc.append(col_enc)
  X_cat = np.concatenate(X_cat_enc, axis=1)
  print 'X_cat.shape: ', X_cat.shape

  skf = KFold(n=X.shape[0], n_folds=10, shuffle=True, random_state=42)
  L = { 'rmse': [], 'corr': [], 'r2': [], 'pct_diff': [], 'mae': [], 'explained_var': [], 'var': []}
  for train_indices, test_indices in skf:
    X_train, X_train_cat, Y_train = X[train_indices], X_cat[train_indices], Y[train_indices]
    X_test, X_test_cat, Y_test = X[test_indices], X_cat[test_indices], Y[test_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.concatenate([X_train, X_train_cat], axis=1)
    X_test = np.concatenate([X_test, X_test_cat], axis=1)

    clf = GradientBoostingRegressor()
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
    L['pct_diff'].append(diff.mean())
    L['mae'].append(mae)
    L['explained_var'].append(explained_var)
    L['r2'].append(r2)
    L['var'].append(var)

    if GENERATE_PLOTS:
      plt.plot(Y_test_10, preds_10, 'ro')
      plt.show()
      break
  for key in sorted(L.keys()):
    print "Mean %s: %f" % (key, np.array(L[key]).mean())
