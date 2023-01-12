# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:28:27 2022

@author: wullems
"""
#%%################# User settings ##########################################
# Customize directory in line 19

#%% Import packages
import numpy as np
import pandas as pd
import sklearn.preprocessing 
from boruta import BorutaPy
import xgboost as xgb

#%% Import the dataset of measurements

FeaturesTable = pd.read_csv('C:\\Users\\wullems\\waterdata\\Data\\Features.csv',index_col=0)
# Interpolate missing values
FeaturesTable = FeaturesTable.interpolate()
# Convert dates to datetime format
dates = pd.to_datetime(FeaturesTable['Time'])

Vars = list(FeaturesTable)[1:33]

#%% Extract training data

Train = np.array(FeaturesTable[Vars][0:2557].astype(float))
Test = np.array(FeaturesTable[Vars][2557:].astype(float))
scaler = sklearn.preprocessing.StandardScaler()
scaler = scaler.fit(Train)
TrainScaled = scaler.transform(Train)
TestScaled = scaler.transform(Test)

#%% Create input data as matrices with shape timesteps * variables

trainIn =[]
trainOut = []

n_future = 1
n_past = 7

for i in range(n_past, len(Train)):
# Include observations of  chloride from 7 days ago up to and including today
    SaltInEl = np.concatenate(TrainScaled[i-n_past:i, 0:15].tolist())
# Include observations of water level, discharge and wind from 7 days ago up to and including tomorrow
    QtyInEl = np.concatenate(TrainScaled[i-n_past:i+1, 15:32].tolist())
    InEl = np.concatenate([SaltInEl,QtyInEl])
    trainIn.append(InEl)
    trainOut.append(TrainScaled[i,0:15])

trainIn, trainOut = np.array(trainIn), np.array(trainOut) 

### Optionally, uncomment the line below to add a constant term
#trainIn = sm.add_constant(trainIn)

#%% Asses feature relevance (WARNING: RUNTIME ~1HR)

#Run a decision tree-based regression model
model = xgb.XGBRegressor()

# Replace features with shadow features, which are copies of random values within the feature's range.
# If this makes the regression model perform much worse, the variable is regarded as more relevant.
feat_selector = BorutaPy(model,perc=70,alpha=0.05,verbose=2)
feat_selector.fit(trainIn, trainOut[:,2])

#%% Create an overview of coefficients and inspect their relative importance
AllVars =[]
Timesteps = []

AllVars = n_past*Vars[0:15]+(n_past+1)*Vars[15:32]
Timesteps = np.concatenate([np.repeat(-6,15),np.repeat(-5,15),np.repeat(-4,15),np.repeat(-3,15),np.repeat(-2,15),np.repeat(-1,15),np.repeat(0,15),np.repeat(-6,17),np.repeat(-5,17),np.repeat(-4,17),np.repeat(-3,17),np.repeat(-2,17),np.repeat(-1,17),np.repeat(0,17),np.repeat(1,17)])
      
FeatureImportance = pd.DataFrame(data={'Variable':AllVars, 'Timestep':Timesteps, 'Rank':feat_selector.ranking_, 'Keep':feat_selector.support_.astype(int)})

# Inspecting the table 'FeatureImportance' gives some information on feature relevance for predicting ClKr400Mean. 
# By doing the same analysis for ClKr400Min and ClKr400Max, we obtained an overview of the most relevant parametners.
