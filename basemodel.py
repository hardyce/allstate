# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:23:32 2016

@author: hardy_000
"""

import numpy as np
import abc
class basemodel:
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def loadtrain(self):
        return
    @abc.abstractmethod    
    def loadtest(self):
        """load training data"""
        return
    def subset(self,data,numberofrows):
        return data.loc[np.random.choice(data.index, numberofrows, replace=False)]
      
    def crossvalidate(data,split):
        msk = np.random.rand(len(data)) < split
        train = data[msk]
        test=data[~msk]
        return train,test
        
    @abc.abstractmethod
    def evaluate():
        return
    @abc.abstractmethod
    def predict():
        return
