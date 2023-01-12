# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:01:51 2022

@author: wullems
"""

#%%
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import sklearn.feature_selection
from sklearn import linear_model
import sklearn.preprocessing 
import statsmodels.api as sm
from sklearn.decomposition import PCA
import seaborn

#%% Import the dataset of measurements

FeaturesTable = pd.read_csv('C:\\Users\\wullems\\waterdata\\Features.csv',index_col=0)
# Interpolate missing values
FeaturesTable = FeaturesTable.interpolate()

dates = pd.to_datetime(FeaturesTable['Time'])

Vars = list(FeaturesTable)[1:]

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
    trainOut.append(TrainScaled[i,0:15])

trainIn, trainOut = np.array(trainIn), np.array(trainOut) 
#trainIn = sm.add_constant(trainIn)
#%% Fit linear model
#model = linear_model.LinearRegression()

model = sm.OLS(trainOut[:,1],trainIn)
model = model.fit()

forecast = model.predict(trainIn)
plt.figure()
plt.plot(forecast, label = 'predicted', marker=',', linestyle='')
plt.plot(trainOut[:,1], label = 'observed', marker=',',linestyle='')
plt.legend()
#%% Create an overview of coefficients and inspect their relative importance
AllVars =[]
Timesteps = []
Ps = model.pvalues
#coefs = model.coef_[1,:]
#coefs_abs = abs(coefs)

AllVars = n_past*Vars[0:15]+(n_past+1)*Vars[15:32]
Timesteps = np.concatenate([np.repeat(-6,15),np.repeat(-5,15),np.repeat(-4,15),np.repeat(-3,15),np.repeat(-2,15),np.repeat(-1,15),np.repeat(0,15),np.repeat(-6,17),np.repeat(-5,17),np.repeat(-4,17),np.repeat(-3,17),np.repeat(-2,17),np.repeat(-1,17),np.repeat(0,17),np.repeat(1,17)])
# for j in range(1,n_past+1):
#     for i in range(0,30):    
#         AllVars.append(Vars[i])
#         Timesteps.append(j-n_past)
# for j in range(n_past+1,n_past+2):
#     for i in range(15,30):    
#         AllVars.append(Vars[i])
#         Timesteps.append(j-n_past)
keep = (Ps <= 0.05).astype(int)        
FeatureImportance = pd.DataFrame(data={'Variable':AllVars, 'Timestep':Timesteps, 'P_value':Ps, 'Keep':keep})
FeatureImportance = FeatureImportance.sort_values(by='P_value')  

VarImportance = []

for i in range(len(Vars)):
    val = []
    var = Vars[i]
    for j in range(len(FeatureImportance['Variable'])):
        if FeatureImportance['Variable'][j] == var:
            val.append(FeatureImportance['P_value'][j])
    val = np.array(val)
    valmean = val.mean()
    VarImportance.append(valmean)
    

BundledImportance = pd.DataFrame(data={'Variable':Vars, 'Mean_P_value':VarImportance})

BundledImportance = BundledImportance.sort_values(by='Mean_P_value',ascending=False)
plt.figure()
plt.bar(BundledImportance['Variable'],BundledImportance['Mean_P_value'])

#%% Calculate performance metrics
TSS = sum(np.subtract(trainOut[:,1], trainOut[:,1].mean())**2)
MSS = sum(np.subtract(forecast[:], trainOut[:,1].mean())**2)
RSS = sum(np.subtract(forecast[:], trainOut[:,1])**2)

RSScheck = TSS-MSS;
R2 = 1-RSS/TSS
R2adj = 1-(((1-R2)*(trainIn.shape[0]-1))/(trainIn.shape[0]-np.count_nonzero(trainIn,axis=1).mean()-1))

RMSE = math.sqrt(np.square(np.subtract(trainOut[:,1],forecast[:])).mean())
print('RMSE = ' + str(RMSE) + 'R2adj = ' + str(R2adj))
################ MANUAL STEPS #################
# inspect the plot and dataframes to select features to delete. Use the following code:
#%% Set variables to remove
RemoveIn = [1]
#RemoveOut = [9,6,12,8,14,7,0,2] # (may only contain indices in range(0,15)
#(replace numbers with indices to remove)

RemoveInAll=[]

for i in RemoveIn:
    RemoveInAll.append(np.where(np.array(AllVars)==Vars[i])[0]) 
RemoveInAll = np.concatenate(RemoveInAll)


trainIn[:,RemoveInAll] = 0
#trainOut[:,RemoveOut] = 0

######### LOG of removed vars##########
  