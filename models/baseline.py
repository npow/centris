#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2

from __future__ import division
import math
import numpy as np
import pandas as pd
import scipy
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from nolearn.dbn import DBN
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

data = pd.read_csv('data/modified_listings.csv')

for col in ['COP', 'AP', 'LS', 'MA']:#, 'PPR', '2X', '3X', '4X', '5X', 'AU', 'UNI', 'MEM']:
  print "*" * 80
  print col
  X = data[['WalkScore', 'NbPieces', 'NbChambres', 'NbSallesEaux', 'NbSallesBains', 'NbFoyerPoele', 'NbEquipements', 'NbGarages', 'NbStationnements', 'NbPiscines', 'NbBordEaux']]
  Y = data[['BuyPrice']]

  # filter rows with NaN
  mask = ~np.isnan(X).any(axis=1)
  X = X[mask]
  Y = Y[mask]

  # remove high-end listings
  mask = Y['BuyPrice'] < 10**6
  X = X[mask]
  Y = Y[mask]

  # remove low-end listings
  mask = Y['BuyPrice'] > 10**5
  X = X[mask]
  Y = Y[mask]

  print "mean: ", Y['BuyPrice'].mean()
  print "median: ", Y['BuyPrice'].median()
  print "std: ", Y['BuyPrice'].std()

  # remove outliers
  mask = np.abs(Y['BuyPrice']-Y['BuyPrice'].median()) <= (3*Y['BuyPrice'].std())
  X = X[mask]
  Y = Y[mask]

  X = np.array(X[data[col]==1])
  Y = np.array(Y[data[col]==1])
  print 'X.shape: ', X.shape
  print 'Y.shape: ', Y.shape

  Y = Y.reshape(Y.shape[0])
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

  clf = Pipeline([
      ('scaler', StandardScaler()),
      ('svm', SVR())
  ])
  clf.fit(X_train, Y_train)
  preds = clf.predict(X_test).astype(float)

  print 'RMSE: ', math.sqrt(metrics.mean_squared_error(preds, Y_test))
