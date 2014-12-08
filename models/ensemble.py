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
  dist=[]
  for point in points:
    distance=distance_on_unit_sphere(lat1,long1,point[0],point[1])
    dist.append(distance)
    weight=gaussian_weight(distance,sigma)
    weights.append(weight)
    #print "House at distance ",str(distance)," has weight ",str(weight)
  return weights,dist

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
  #X=X[['LivingArea','NbChambres','NbSallesBains']]
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
  mask = Y['BuyPrice'] < 1000000
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
  XX=X[['LivingArea','NbChambres','NbSallesBains']]
  X = np.array(X)
  XX = np.array(XX)
  Y = np.array(Y)
  Y = Y.reshape(Y.shape[0])
  geo_points=np.array(geo_points)
  skf = KFold(n=X.shape[0], n_folds=10, shuffle=True, random_state=42)

  var_num=1
  num_neigh=range(2,75)
  #var_range=np.array(range(16,80))/20.0
  var_range=np.array([0.4])
  for neigh in [100]:
    j=-1
    for var_num in np.nditer(var_range):
      j=j+1
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

         ('clf', k_clf),

      ])
      num=1229
      X=scaler.fit_transform(X,Y)
      XX=scaler.fit_transform(XX,Y)
      #X=pca.fit_transform(X)
      #clf.fit(X,Y)
      k_clf.fit(XX,Y)
      y_pred_all=k_clf.predict(XX).astype(float)
      dist,indic= k_clf.kneighbors(XX[num])
      #print type(indic)
      indic= indic.tolist()[0]

      weights= weight_vector(geo_points[num][0],geo_points[num][1],geo_points[indic].tolist(),5)
      #print Y[indic].mean()
      weights_array=np.array(weights)

      
      y_pred=[]
      norm_dist=[]
      nbad = 0
      for i in range(0,len(Y)):
        #print 'I am in the loop'
        dist,indic=k_clf.kneighbors(XX[i])

        indic= indic.tolist()[0]
        weights,distance= weight_vector(geo_points[i][0],geo_points[i][1],geo_points[indic[1:]].tolist(),vari)
        n_nearby = len(filter(lambda x: x < 0.1, distance))
        if n_nearby > 0:
          weights_array=np.array(weights)
          dist_array=np.array(distance)
          y_pred.append(np.dot(weights_array,Y[indic[1:]])/weights_array.sum())
          #y_pred.append(Y[indic[1:]].mean())
          norm_dist.append(np.dot(dist_array,weights_array)/weights_array.sum())
          #norm_dist.append(dist_array.mean())
        else:
          clf = GradientBoostingRegressor()
          mask = np.in1d(np.arange(X.shape[0]), [i])
          clf.fit(X[~mask], Y[~mask])
          p = clf.predict(X[mask])
          y_pred.append(p)
          nbad += 1

      print 'got it all'
      print '*' * 80
      print nbad
      y_pred=np.array(y_pred)
      rmse = math.sqrt(metrics.mean_squared_error(y_pred, Y))
      corr = pearsonr(y_pred, Y)
      diff = np.array([abs(p-a)/a for (p,a) in zip(y_pred,Y)]).mean()
      print "RMSE: ", rmse
      print "corr: ", corr
      print "%diff: ", diff

"""
AP
X.shape:  (4425, 133)
Y.shape:  (4425, 1)
geo_points.shape (4425, 2)
After NaN filter:  (3092, 133)
After upper-bound filter:  (3057, 133)
After lower-bound filter:  (3054, 133)
mean:  326815.843157
median:  289000.0
std:  146918.0813
100
0.4
got it all
********************************************************************************
1331
RMSE:  50190.4479516
corr:  (0.90608458893495725, 0.0)
%diff:  0.0998519388801
"""
