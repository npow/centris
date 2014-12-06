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
#from pybrain.datasets.supervised import SupervisedDataSet
#from pybrain.tools.shortcuts import buildNetwork
#from pybrain.supervised.trainers import BackpropTrainer
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
def distance_on_unit_sphere(lat1, long1, lat2, long2):
 
  #source code at http://www.johndcook.com/blog/python_longitude_latitude/
    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0
         
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
         
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
         
    # Compute spherical distance from spherical coordinates.
         
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
     
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
           math.cos(phi1)*math.cos(phi2))
    if (cos>1):
      cos=1
    arc = math.acos( cos )
 
    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return arc*6373


def weight_vector(lat1,long1,points,sigma):
  weights=[]
  for point in points:
    distance=distance_on_unit_sphere(lat1,long1,point[0],point[1])
    weight=gaussian_weight(distance,sigma)
    weights.append(weight)
    #print "House at distance ",str(distance)," has weight ",str(weight)
  return weights

def gaussian_weight(distance,sigma):
  return math.exp(-distance**2/(2*sigma**2))
np.random.seed(42)


data = pd.read_csv('data/modified_listings.csv')
extra_data = pd.read_csv('data/post_extra_data.csv')
extra_data = extra_data[['MlsNumber','LivingArea']]
merged = pd.merge(data, extra_data, on='MlsNumber', suffixes=['_left', '_right'])
merged.to_csv('data/all_data.csv')
for col in ['AP']:#, 'COP', 'AP', 'LS', 'MA']:#, 'PPR', '2X', '3X', '4X', '5X', 'AU', 'UNI', 'MEM']:
  print "*" * 80
  print col

  #X = merged[['LivingArea','WalkScore', 'NbPieces', 'NbChambres', 'NbSallesEaux', 'NbSallesBains', 'NbFoyerPoele', 'NbEquipements', 'NbGarages', 'NbStationnements', 'NbPiscines', 'NbBordEaux']]
  geo_points=merged[['Lat','Lng']]
  X = merged.drop(['MlsNumber', 'Lat', 'Lng', 'BuyPrice'], axis=1, inplace=False)
  Y = merged[['BuyPrice']]

  X = X[merged[col]==1]
  Y = Y[merged[col]==1]
  geo_points=geo_points[merged[col]==1]
  print 'X.shape: ', X.shape
  print 'Y.shape: ', Y.shape
  print 'geo_points.shape', geo_points.shape

  # filter rows with NaN
  mask = ~np.isnan(X).any(axis=1)
  X = X[mask]
  Y = Y[mask]
  geo_points=geo_points[mask]
  print 'After NaN filter: ', X.shape

  # remove high-end listings
  mask = Y['BuyPrice'] < 500000
  X = X[mask]
  Y = Y[mask]
  geo_points=geo_points[mask]
  print 'After upper-bound filter: ', X.shape

  # remove low-end listings
  mask = Y['BuyPrice'] > 10**5
  X = X[mask]
  Y = Y[mask]
  geo_points=geo_points[mask]
  print 'After lower-bound filter: ', X.shape

  print "mean: ", Y['BuyPrice'].mean()
  print "median: ", Y['BuyPrice'].median()
  print "std: ", Y['BuyPrice'].std()

  # remove outliers
  mask = np.abs(Y['BuyPrice']-Y['BuyPrice'].median()) <= (3*Y['BuyPrice'].std())
  X = X[mask]
  Y = Y[mask]
  geo_points=geo_points[mask]

  columns = X.columns.values
  X = np.array(X)
  Y = np.array(Y)
  Y = Y.reshape(Y.shape[0])
  geo_points=np.array(geo_points)
  skf = KFold(n=X.shape[0], n_folds=10, shuffle=True, random_state=42)

  var_num=2
  num_neigh=range(3,10)
  var_range=np.array(range(16,80))/20.0
  for neigh in num_neigh:
    #for var_num in np.nditer(var_range):
      print neigh
      print var_num
      L_rmse = []
      L_corr = []
      L_diff = []
      k_clf=KNeighborsRegressor(n_neighbors=neigh)
      vari=var_num
    #clf=KNeighborsRegressor(n_neighbors=8)
      scaler=StandardScaler()
      clf = Pipeline([
         ('scaler', StandardScaler()),
         # ('clf', AdaBoostRegressor()),
         # ('clf', ARDRegression()),
         # ('clf', BaggingRegressor()),
         # ('clf', BayesianRidge()),
         # ('clf', ElasticNet()),
         # ('clf', ExtraTreesRegressor()),
         #('clf', GradientBoostingRegressor()),
         ('clf', k_clf),
         # ('clf', Lasso()),
         # ('clf', LinearRegression()),
         # ('clf', PassiveAggressiveRegressor()),
         # ('clf', RandomForestRegressor()),
         # ('clf', Ridge(alpha=0.5, normalize=False)),
         # ('clf', SVR()),
      ])
      num=1229
      X=scaler.fit_transform(X,Y)
      #clf.fit(X,Y)
      k_clf.fit(X,Y)
      y_pred_all=k_clf.predict(X).astype(float)
      dist,indic= k_clf.kneighbors(X[num])
      #print type(indic)
      indic= indic.tolist()[0]
      #print Y[indic]
      #print Y[num]
      #print geo_points[indic]
      #print geo_points[indic].tolist()
      weights= weight_vector(geo_points[num][0],geo_points[num][1],geo_points[indic].tolist(),5)
      #print Y[indic].mean()
      weights_array=np.array(weights)
      #print np.dot(weights_array,Y[indic])/weights_array.sum()
      #print(X[num])
      #print(X[indic])
      
      y_pred=[]
      for i in range(0,len(Y)):
        #print 'I am in the loop'
        dist,indic=k_clf.kneighbors(X[i])

        indic= indic.tolist()[0]
        weights= weight_vector(geo_points[i][0],geo_points[i][1],geo_points[indic[1:]].tolist(),vari)
        weights_array=np.array(weights)
        #y_pred.append(np.dot(weights_array,Y[indic[1:]])/weights_array.sum())
        y_pred.append(Y[indic[1:]].mean())


        #print y_pred[i]
        #print y_pred_all[i]
        

        # rmse = math.sqrt(metrics.mean_squared_error(preds, Y_test))
        # corr = pearsonr(preds, Y_test)
        # diff = np.array([abs(p-a)/a for (p,a) in zip(preds,Y_test)]).mean()

        # print rmse, corr
        # L_rmse.append(rmse)
        # L_corr.append(corr[0])
        # L_diff.append(diff)

        # if GENERATE_PLOTS:
        #   plt.plot(Y_test, preds, 'ro')
        #   plt.show()
        #   break
      print 'got it all'
      y_pred=np.array(y_pred)
      #print type(y_pred),y_pred.shape
      #print type(Y),Y.shape

      rmse = math.sqrt(metrics.mean_squared_error(y_pred, Y))
      corr = pearsonr(y_pred, Y)
      diff = np.array([abs(p-a)/a for (p,a) in zip(y_pred,Y)]).mean()
      print "RMSE: ", rmse
      print "corr: ", corr
      print "%diff: ", diff

