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
GENERATE_PLOTS = False

data = pd.read_csv('data/modified_listings.csv')
extra_data = pd.read_csv('data/post_extra_data.csv')
extra_data = extra_data[['MlsNumber','LivingArea']]
merged = pd.merge(data, extra_data, on='MlsNumber', suffixes=['_left', '_right'])
merged.to_csv('data/all_data.csv')

categorical_columns = [
  'Outremont',
  'LaSalle',
  'Mont-Royal',
  'Ville-Marie',
  'Le Plateau-Mont-Royal',	
  'Hampstead',
  'Le Sud-Ouest',
  'Riviere-des-Prairies-Pointe-aux-Trembles',
  'Lachine',
  'Dorval',
  'Montreal-Nord',
  'Lile-Bizard-Sainte-Genevieve',
  'Kirkland',
  'Dollard-des-Ormeaux',
  'Senneville',
  'Ahuntsic-Cartierville',
  'Cote-Saint-Luc',
  'Saint-Leonard',
  'Montreal-Ouest',
  'Pointe-Claire',
  'Lile-Dorval',
  'Mercier-Hochelaga-Maisonneuve',
  'Cote-des-Neiges-Notre-Dame-de-Grace',
  'Rosemont-La Petite-Patrie',
  'Saint-Laurent',
  'Beaconsfield',
  'Villeray-Saint-Michel-Parc-Extension',
  'Westmount',
  'Montreal-Est',
  'Anjou',
  'Pierrefonds-Roxboro',
  'Sainte-Anne-de-Bellevue',
  'Verdun',
  'Baie-dUrfe',
  'AP',
  'C',
  'ME',	
  'VE',
  'I',
  '4X',
  'MA',
  'PP',
  'AU',
  'MPM',
  'TE',
  '2X',
  'TR',
  '3X',
  'MEM',
  'LS',
  'MM',
  '5X',
  'FE',
  'COP',
  'PCI',
  'UNI',
  'PPR',
  'TER',
  'FER'
]

numerical_columns = [
  'AvgIncome',
  'NbBordEaux',
  'NbChambres',
  'NbEquipements',
  'NbGarages',
  'NbFoyerPoele',
  'NbPieces',
  'NbPiscines',
  'NbSallesEaux',
  'NbSallesBains',
  'NbStationnements',
  'WalkScore',
  'Population',
  'Variation',	
  'Density',
  'avgAge',
  'below15',
  'below24',
  'below44',
  'below64',
  'below65',
  'below50000',
  'below80000',
  'below100000',
  'below150000',
  'above150001',
  'avgSize',
  'totalHouse',
  'size1',
  'size2',
  'size3',
  'size4',
  'size5',
  'totalFam',
  'Children',
  'NoChildren',
  'Single',
  'Unemploy',
  'Owner',
  'Renter',
  'totaldwelling',
  'before1960',
  'before1980',
  'before1990',
  'before2000',
  'before2005',
  'before2011',
  'Single',
  'Semi',
  'Duplex',
  'Buildings',	
  'Mobile',	
  'belowBach',
  'Bach',
  'abvBach',
  'University',
  'College',
  'Secondary',
  'Apprentice',
  'No',
  'NonImmigrant',
  'Immigrant',	
  'french',
  'English',
  'Others',
  'PoliceDist',
  'FireDist'
  ]

for col in ['AP']:#, 'COP', 'AP', 'LS', 'MA']:#, 'PPR', '2X', '3X', '4X', '5X', 'AU', 'UNI', 'MEM']:
  print "*" * 80
  print col

  X = merged[numerical_columns]
  #X = merged.drop(['MlsNumber', 'Lat', 'Lng', 'BuyPrice'], axis=1, inplace=False)
  X_cat = merged[categorical_columns]
  Y = merged[['BuyPrice']]

  mask = merged[col]==1
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
  L = { 'rmse': [], 'corr': [], 'r2': [], 'diff': [], 'mae': [], 'explained_var': [], 'var': []}
  for train_indices, test_indices in skf:
    X_train, X_train_cat, Y_train = X[train_indices], X_cat[train_indices], Y[train_indices]
    X_test, X_test_cat, Y_test = X[test_indices], X_cat[test_indices], Y[test_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.concatenate([X_train, X_train_cat], axis=1)
    X_test = np.concatenate([X_test, X_test_cat], axis=1)

    if USE_NEURALNET:
      Y_train, Y_test = Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)
      hidden_size = 10

      train_ds = SupervisedDataSet(X_train.shape[1], Y_train.shape[1])
      train_ds.setField('input', X_train)
      train_ds.setField('target', Y_train)
      net = buildNetwork(X_train.shape[1], hidden_size, Y_train.shape[1], bias=True)
      trainer = BackpropTrainer(net, train_ds)

      epochs = 100
      for i in xrange(epochs):
        mse = trainer.train()
        rmse = math.sqrt(mse)
        print "epoch: %d, rmse: %f" % (i, rmse)

      test_ds = SupervisedDataSet(X_test.shape[1], Y_test.shape[1])
      test_ds.setField('input', X_test)
      test_ds.setField('target', Y_test)
      preds = net.activateOnDataset(test_ds)
    else:
      #clf = AdaBoostRegressor()
      #clf = ARDRegression()
      #clf = BaggingRegressor()
      #clf = BayesianRidge()
      #clf = ElasticNet()
      clf = GradientBoostingRegressor()
      #clf = KNeighborsRegressor(n_neighbors=5)
      #clf = RandomForestRegressor()
      #clf = Ridge(alpha=0.5, normalize=True)
      #clf = SVR()
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
