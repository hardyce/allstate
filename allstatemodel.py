# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:02:14 2016

@author: hardy_000
"""
#to do: take log of loss as it is skewed.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from basemodel import basemodel
from pandas.stats.api import ols
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, hstack
import xgboost
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import cross_val_score

#import seaborn as sns

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
        
    def preprocess():
        return
        
    def createDummies(dat):
        st=dat.select_dtypes(exclude=['floating'])
        dums=pd.get_dummies(st)

        res=pd.concat([dat.select_dtypes(exclude=['object']), dums], axis=1)
        return res
        
        
model=allstatemodel()       
train=model.loadtrain()
test=model.loadtest()
train=model.subset(train,len(train)/10)
y=np.log(train['loss'].values)
id_train=train['id'].values
id_test=test['id'].values

ntrain=train.shape[0]
traintest=pd.concat((train,test),axis=0)

sparse_data=[]
f_cat=[f for f in traintest.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(traintest[f].astype('category'))
    tmp=csr_matrix(dummy)
    sparse_data.append(tmp)
    
f_num=[f for f in traintest.columns if 'cont' in f]

scaler = MinMaxScaler()
tmp=csr_matrix(scaler.fit_transform(traintest[f_num]))

sparse_data.append(tmp)

xtraintest=hstack(sparse_data,format='csr')
xtrain=xtraintest[:ntrain,:]
xtest=xtraintest[ntrain:,:]
#clf = linear_model.LinearRegression()
#clf.fit(xtrain,y )
clf=sm.GLM(y,xtrain.toarray(),family=sm.families.Gaussian(sm.families.links.log))
clf.fit()
#clf = RandomForestRegressor(n_estimators = 100)
#clf = clf.fit(xtrain.toarray(),y)
cv = ShuffleSplit(len(xtrain.toarray()), random_state=11)
sum(cross_val_score(clf, xtrain.toarray(), y, cv=cv,scoring="median_absolute_error"))
plt.plot(np.exp(clf.predict(xtrain.toarray()))-np.exp(y))
b=np.exp(clf.predict(xtrain.toarray()))-np.exp(y)
c=b>40000
plt.plot(y)
plt.show()
#pca = PCA(n_components=10)
#pca.fit(xtrain.toarray())
#xpc=pca.transform(xtrain.toarray())


#res=model.fit()
#pca = PCA(n_components=500)
#xgb = xgboost.XGBClassifier()
#pca.fit(xtrain.toarray())
#xpc=pca.transform(xtrain.toarray())


#xgb.fit(xtrain,y)
#regr = linear_model.LinearRegression()
#regr.fit(xpc,y )
#plt.figure(figsize=(13,9))
#sns.distplot(train["loss"])
#sns.boxplot(train["loss"])
#print('Variance explained ratio: %.2f' %sum(pca.explained_variance_ratio_))
#print('Variance score: %.2f' % regr.score(xpc, y))
#pred=np.exp(forest.predict(xtest.toarray()))
#pred=np.exp(regr.predict(pca.transform(xtest.toarray())))
#pred=xgb.predict(pca.transform(xtest.toarray()))


submission=pd.DataFrame(np.array([id_test,pred]).T)
submission.columns=['id','loss']
submission.id=submission.id.astype('int')
submission.to_csv('submis.csv',sep=',',header=True,index=False)