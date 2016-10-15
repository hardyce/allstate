# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:02:14 2016

@author: hardy_000
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from basemodel import basemodel
from pandas.stats.api import ols
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model
from sklearn.preprocessing import MinMaxScaler

class allstatemodel(basemodel):
    
    def loadtrain(self):
        path="C:\\Users\hardy_000\\Documents\\datasci\\allstate\\train.csv"
        data=pd.read_csv(path)
        return data
        
    def evaluate():
        return
        
    def loadtest(self):
        path="C:\\Users\hardy_000\\Documents\\datasci\\allstate\\test.csv"
        data=pd.read_csv(path)
        return data
    def predict():
        return
        
    def createDummies(dat):
        st=dat.select_dtypes(exclude=['floating'])
        dums=pd.get_dummies(st)

        res=pd.concat([dat.select_dtypes(exclude=['object']), dums], axis=1)
        return res
model=allstatemodel()       
data=model.loadtrain()
test=model.loadtest()
encoded = pd.get_dummies(pd.concat([data.drop('loss',axis=1).select_dtypes(include=['object']),test.select_dtypes(include=['object'])], axis=0))
train_rows = data.shape[0]
train_encoded = encoded.iloc[:train_rows, :]
test_encoded = encoded.iloc[train_rows:, :] 
    



y=data['loss']
x=pd.concat([train_encoded,data.select_dtypes(exclude=['object']).astype('float32')],axis=1)

train_encoded.join(data.select_dtypes(exclude=['object']))
pca = PCA(n_components=200)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns) 
pca.fit(df_scaled)
xpc=pca.transform(df_scaled)
regr = linear_model.LinearRegression()
regr.fit(xpc,y )


print('Variance score: %.2f' % regr.score(xpc, y))

test=model.loadtest()
ts=createDummies(test)
test_scaled = pd.DataFrame(scaler.fit_transform(ts), columns=ts.columns)
pred=regr.predict(pca.transform(test_encoded))