#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2

from __future__ import division
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

data = pd.read_csv('data/modified_listings.csv')
Y = data[['BuyPrice']].values

data.drop(['MlsNumber', 'Lat', 'Lng', 'BuyPrice'], axis=1, inplace=True)
X = np.array(data)
print X.shape
print Y.shape
print X

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

svr = SVR(C=10000.0)
svr.fit(X_train, Y_train)
preds = svr.predict(X_test).astype(float)

print mean_squared_error(preds, Y_test)
