# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:02:14 2016

@author: hardy_000
"""
#1208.1533 avg for unskew,drop feat xgb 9 50 11
#1261.392 without dropiing
#1206.3274 with just cont removed
#to do: take log of loss as it is skewed.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from basemodel import basemodel
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
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
        
    def plotCat(cat):
        for v in np.unique(train[cat]):
            plt.hist(train[train[cat]==v]['loss'],bins=100)
            plt.show()

    def encode(self,dat):
    

        catFeatureslist = []
        for colName,x in dat.iloc[1,:].iteritems():
            if(str(x).isalpha()):
                catFeatureslist.append(colName)


        for cf1 in catFeatureslist:
            le = LabelEncoder()
            le.fit(dat[cf1].unique())
            dat[cf1] = le.transform(dat[cf1])        
        return dat, le
    def encodeDummy(self,train,test):
    

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
        
        return xtrain,xtest
def score(fitmod,data,y):
    xgtest = xgb.DMatrix(data)
    preds=fitmod.predict(xgtest)
    return evalerror(preds,xgb.DMatrix(data,label=y))[1]
    

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))
def invboxcox(y,ld):
   if ld == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(ld*y+1)/ld))
      
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    con =2
    x =preds-labels
    grad =con*x / (np.abs(x)+con)
    hess =con**2 / (np.abs(x)+con)**2
    return grad, hess 
def unskew(joined):
    numeric_feats = joined.dtypes[joined.dtypes != "object"].index

    skewed_feats = joined[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    # transform features with skew > 0.25 (this can be varied to find optimal value)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index
    for feats in skewed_feats:
        joined[feats] = joined[feats] + 1
        joined[feats], lam = boxcox(joined[feats])
    return joined
    
def transformFeatures(joined):
    cat_feature = [n for n in joined.columns if n.startswith('cat')]     
    ###!!!!!unskews id
    joined=unskew(joined)
    for column in cat_feature:
        joined[column] = pd.factorize(joined[column].values, sort=True)[0]

    
    train = joined[joined['loss'].notnull()]
    test = joined[joined['loss'].isnull()]
    train=train.drop('loss',axis=1)
        
    return train,test
    
def crossValidate(data,y):
    results=[]
    cv = cross_validation.KFold(len(data), n_folds=10)
    results = []
    
    # "Error_function" can be replaced by the error function of your analysis
    for traincv, testcv in cv:
        xgtrain = xgb.DMatrix(data.iloc[traincv], label=y[traincv])
        fitmod = xgb.train(params, xgtrain,int(2012 / 0.9), obj=logregobj, feval=evalerror)
        scor=score(fitmod,data.iloc[testcv],y[testcv])
        results.append(scor)
        
    return results
        #label encoder broken
    
    
testing=True
model=allstatemodel()       
train=model.loadtrain()
test=model.loadtest()
if testing==True:
    train=model.subset(train,len(train)/50)
    train=train.reset_index()
y=np.log(train['loss'].values+200)
id_train=train['id'].values
train=train.drop('id',axis=1)
id_test=test['id'].values
test=test.drop('id',axis=1)
#cat9,cat50 and cont 11 high correlation with other vals
train=train.drop(['cont11','cont9'],axis=1)
test=test.drop(['cont11','cont9'],axis=1)
test['loss'] = np.nan
joined = pd.concat([train, test])
train,test=transformFeatures(joined)

RANDOM_STATE = 2016
params = {
        'min_child_weight': 1,
        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 12,
        'subsample': 0.8,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': RANDOM_STATE
    }

res=[]

if testing==False:
    
    xgtrain = xgb.DMatrix(train, label=y)
    xgtest = xgb.DMatrix(test.drop('loss',axis=1))

    mod = xgb.train(params, xgtrain,int(2012 / 0.9), obj=logregobj, feval=evalerror)

    pred=mod.predict(xgtest)
    pred=np.exp(pred)-200
    
else:
    
    res=crossValidate(train,y)
#clf = linear_model.LinearRegression()
#clf.fit(xtrain,y )
#clf=sm.GLM(y,xtrain.toarray(),family=sm.families.Gaussian(sm.families.links.log))
#clf.fit()
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
