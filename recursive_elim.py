# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:19:33 2022

@author: wullems
"""

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.feature_selection
import sklearn.preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn import linear_model

#%% Import the dataset of measurements

FeaturesTable = pd.read_csv('C:\\Users\\wullems\\waterdata\\Features.csv',index_col=0)
# Interpolate missing values
FeaturesTable = FeaturesTable.interpolate()

dates = pd.to_datetime(FeaturesTable['Time'])

Vars = list(FeaturesTable)[1:33]

#%% Extract training data

Train = np.array(FeaturesTable[Vars][0:2557].astype(float))
Test = np.array(FeaturesTable[Vars][2557:].astype(float))
scaler = sklearn.preprocessing.StandardScaler()
scaler = scaler.fit(Train)
TrainScaled = scaler.transform(Train)
TestScaled = scaler.transform(Test)

# TrainScaled = Train
# TestScaled = Test
#%% Create input data as matrices with shape timesteps * variables

trainIn =[]
trainOut = []

n_future = 1
n_past = 7

for i in range(n_past, len(Train)):
    SaltInEl = np.concatenate(TrainScaled[i-n_past:i, 0:15].tolist())
    QtyInEl = np.concatenate(TrainScaled[i-n_past:i+1, 15:32].tolist())
    InEl = np.concatenate([SaltInEl,QtyInEl])
    trainIn.append(InEl)
    trainOut.append(TrainScaled[i,0:2])

trainIn, trainOut = np.array(trainIn), np.array(trainOut) 
#trainIn = sm.add_constant(trainIn)

#%% Set up model
model = linear_model.LinearRegression()
# cv=train_test_split(trainIn,trainOut,test_size=0.2)
rfe = sklearn.feature_selection.RFE(model,verbose=1)
fit = rfe.fit(trainIn, trainOut)

#%% Evaluate selected features
AllVars =[]
Timesteps = []
AllVars = n_past*Vars[0:15]+(n_past+1)*Vars[15:32]
Timesteps = np.concatenate([np.repeat(-6,15),np.repeat(-5,15),np.repeat(-4,15),np.repeat(-3,15),np.repeat(-2,15),np.repeat(-1,15),np.repeat(0,15),np.repeat(-6,17),np.repeat(-5,17),np.repeat(-4,17),np.repeat(-3,17),np.repeat(-2,17),np.repeat(-1,17),np.repeat(0,17),np.repeat(1,17)])
FeatureImportance = pd.DataFrame(data={'Variable':AllVars, 'Timestep':Timesteps, 'Rank':fit.ranking_, 'Keep':fit.support_.astype(int)})