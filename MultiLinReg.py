# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:23:37 2022

@author: wullems
"""
test = False
#%%
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from sklearn import linear_model
import sklearn.preprocessing 

#%% Import the dataset of measurements

FeaturesTable = pd.read_csv('C:\\Users\\wullems\\waterdata\\Features.csv')
# Interpolate missing values
FeaturesTable = FeaturesTable.interpolate()

dates = pd.to_datetime(FeaturesTable['Time'])



#%% Select the features with relevance according to Boruta analysis
FeaturesTable = FeaturesTable[['Time','ClKr400Min','ClKr400Mean','ClKr400Max','ClLkh700Min','ClLkh700Mean','ClLkh700Max','HDrdMean','HHvhMin','HHvhMean','HHvhMax','HKrMean','HVlaMean','QHagMean','QLobMean','QTielMean','WindEW','WindNS']]
Vars = list(FeaturesTable)[1:18]
#%% Extract training data

Train = np.array(FeaturesTable[Vars][0:2557].astype(float))
Test = np.array(FeaturesTable[Vars][2557:].astype(float))
scaler = sklearn.preprocessing.StandardScaler()
scaler = scaler.fit(Train)
#TrainScaled = scaler.transform(Train)
#TestScaled = scaler.transform(Test)

TrainScaled = Train
TestScaled = Test
Scaled = np.vstack((TrainScaled,TestScaled))
#%% Create input data as matrices with shape timesteps * variables

trainIn =[]
trainOut = []

n_future = 1
n_past = 5

for i in range(n_past, len(Train)):
    SaltInEl = np.concatenate(TrainScaled[i-n_past:i, 0:6].tolist())
    QtyInEl = np.concatenate(TrainScaled[i-n_past:i+1, 6:17].tolist())
    InEl = np.concatenate([SaltInEl,QtyInEl])
    trainIn.append(InEl)
    trainOut.append(TrainScaled[i,0:6])

trainIn, trainOut = np.array(trainIn), np.array(trainOut) 

#%% Fit linear model
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(trainIn, trainOut)

#%% create a one day ahead forecast
forecast = model.predict(trainIn)
#forecast_copies = np.hstack([forecast,forecast,forecast[:,0:3]])

#forecastReal = scaler.inverse_transform(forecast_copies)[:,0:6]
forecastReal = forecast

trainOutReal = Train[:,0:6]
plt.figure()
plt.plot(dates[n_past:len(Train[:,0])],forecastReal[:,1], label ='predicted', marker = ",", linestyle="")
plt.plot(dates[n_past:len(Train[:,0])], trainOutReal[n_past:,1], label ='observed', marker = ",", linestyle="")
plt.ylim(0,1500)
plt.legend()
plt.title('Forecast at t=+1')
RMSE = np.zeros(7)
Accuracy = np.zeros(7)
Precision = np.zeros(7)
Recall = np.zeros(7)
Threshold = 300
RMSE[0] = math.sqrt(np.square(np.subtract(trainOutReal[n_past:,1],forecastReal[:,1])).mean())
event = trainOutReal[n_past:,1] > Threshold
warning = forecastReal[:,1] > Threshold;
Accuracy[0] = (sum(np.logical_and(event, warning)) + sum(np.logical_and(np.logical_not(event),np.logical_not(warning))))/len(forecast)
Precision[0] = sum(np.logical_and(event, warning))/sum(warning)
Recall[0] = sum(np.logical_and(event, warning))/sum(event)
plt.figure()
plt.plot(trainOutReal[n_past:,1], forecastReal[:,1], marker = ',', linestyle="")
plt.plot([0,2000],[0,2000], color='black')
plt.xlim(0,2000)
plt.ylim(0,2000)
plt.xlabel('observed')
plt.ylabel('predicted')
plt.title('Fit at t=+1')

#%% replace observed values with forecast values to simulate multiple-day forecasts.

for j in range(2,8):
    for i in range(n_past, len(Train)):
        SaltInEl = trainIn[i-n_past, 6:n_past*6].tolist()
        SaltInEl = np.concatenate([SaltInEl, forecast[i-n_past,:].tolist()])
        QtyInEl = np.concatenate(Scaled[i-n_past+j-1:i+j, 6:17].tolist())
        InEl = np.concatenate([SaltInEl,QtyInEl])        
        trainIn[i-n_past] = InEl
    
    trainIn = np.array(trainIn)
            
    forecast = model.predict(trainIn)
    #forecast_copies = np.hstack([forecast,forecast,forecast[:,0:3]])
    # forecastReal = scaler.inverse_transform(forecast_copies)[:,0:6]
    forecastReal = forecast
    trainOutReal = Train[n_past:,0:6]
    plt.figure()
    plt.plot(dates[n_past+j-1:len(Train)],forecastReal[:-(j-1),1], label ='predicted', marker = ",", linestyle="")
    plt.plot(dates[n_past+j-1:len(Train)],trainOutReal[j-1:,1], label ='observed', marker = ",", linestyle="")
    plt.ylim(0,1500)
    plt.legend() 
    plt.title('Forecast at t=+'+ str(j))
    RMSE[j-1] = math.sqrt(np.square(np.subtract(trainOutReal[j-1:,1],forecastReal[:-(j-1),1])).mean())
    event = trainOutReal[j-1:,1] > Threshold
    warning = forecastReal[:-(j-1),1] > Threshold;
    Accuracy[j-1] = (sum(np.logical_and(event, warning)) + sum(np.logical_and(np.logical_not(event),np.logical_not(warning))))/len(forecast)
    Precision[j-1] = sum(np.logical_and(event, warning))/sum(warning)
    Recall[j-1] = sum(np.logical_and(event, warning))/sum(event)
    plt.figure()
    plt.plot(trainOutReal[j-1:,1], forecastReal[:-(j-1),1], marker = ',', linestyle="")
    plt.plot([0,2000],[0,2000], color='black')
    plt.xlim(0,2000)
    plt.ylim(0,2000)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Fit at t=+'+ str(j))
    
#%% Create a test input dataset
if test == True:
    testIn = []
    testOut = []
    
    n_past = 5

    for i in range(n_past, len(Test)):
        SaltInEl = np.concatenate(Test[i-n_past:i, 0:6].tolist())
        QtyInEl = np.concatenate(Test[i-n_past:i+1, 6:17].tolist())
        InEl = np.concatenate([SaltInEl,QtyInEl])
        testIn.append(InEl)
        # testOut.append(Test[i,0:6])
    
    testIn, testOut = np.array(testIn), np.array(testOut)
        
#%% create a one day ahead forecast
if test == True:
    forecast_test = model.predict(testIn)
    testOut = Test[n_past:,0:6]
    plt.figure()
    plt.plot(dates[2557+n_past:],forecast_test[:,1], label ='predicted', marker = ",", linestyle="")
    plt.plot(dates[2557+n_past:], testOut[:,1], label ='observed', marker = ",", linestyle="")
    plt.ylim(0,1500)
    plt.legend()
    plt.title('Forecast at t=+1')
    RMSE_test = np.array(tf.zeros(7))
    RMSE_test[0] = math.sqrt(np.square(np.subtract(testOut[:,1],forecast_test[:,1])).mean())
    Accuracy_test = np.zeros(7)
    Precision_test = np.zeros(7)
    Recall_test = np.zeros(7)
    Threshold = 300
    event = testOut[:,1] > Threshold
    warning = forecast_test[:,1] > Threshold;
    Accuracy_test[0] = (sum(np.logical_and(event, warning)) + sum(np.logical_and(np.logical_not(event),np.logical_not(warning))))/len(forecast_test)
    Precision_test[0] = sum(np.logical_and(event, warning))/sum(warning)
    Recall_test[0] = sum(np.logical_and(event, warning))/sum(event)
    plt.figure()
    plt.plot(testOut[:,1], forecast_test[:,1], marker = ',', linestyle="")
    plt.plot([0,2000],[0,2000], color='black')
    plt.xlim(0,2000)
    plt.ylim(0,2000)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Fit at t=+1')

#%% replace observed values with forecast values to simulate multiple-day forecasts.
if test == True:
    for j in range(2,8):
        for i in range(n_past, len(Test)-j):
            SaltInEl = testIn[i-n_past, 6:n_past*6].tolist()
            SaltInEl = np.concatenate([SaltInEl, forecast_test[i-n_past,:].tolist()])
            QtyInEl = np.concatenate(Test[i-n_past+j-1:i+j, 6:17].tolist())
            InEl = np.concatenate([SaltInEl,QtyInEl])        
            testIn[i-n_past] = InEl
        
        testIn = np.array(testIn)
                
        forecast_test = model.predict(testIn)
        testOut = Test[n_past:,0:6]
        plt.figure()
        plt.plot(dates[2557+n_past+j-1:],forecast_test[:-(j-1),1], label ='predicted', marker = ",", linestyle="")
        plt.plot(dates[2557+n_past+j-1:],testOut[j-1:,1], label ='observed', marker = ",", linestyle="")
        plt.ylim(0,1500)
        plt.legend() 
        plt.title('Forecast at t=+'+ str(j))
        RMSE_test[j-1] = math.sqrt(np.square(np.subtract(testOut[j-1:,1],forecast_test[:-(j-1),1])).mean())
        event = testOut[j-1:,1] > Threshold
        warning = forecast_test[:-(j-1),1] > Threshold;
        Accuracy_test[j-1] = (sum(np.logical_and(event, warning)) + sum(np.logical_and(np.logical_not(event),np.logical_not(warning))))/len(forecast_test)
        Precision_test[j-1] = sum(np.logical_and(event, warning))/sum(warning)
        Recall_test[j-1] = sum(np.logical_and(event, warning))/sum(event)
        plt.figure()
        plt.plot(testOut[j-1:,1], forecast_test[:-(j-1),1], marker = ',', linestyle="")
        plt.plot([0,2000],[0,2000], color='black')
        plt.xlim(0,2000)
        plt.ylim(0,2000)
        plt.xlabel('observed')
        plt.ylabel('predicted')
        plt.title('Fit at t=+'+ str(j))
        
#%%
plt.figure()
plt.plot(range(1,8),RMSE)
plt.xlabel('days ahead')
plt.ylabel('Cl [mg/l]')
plt.title('Root Mean Squared Error')
plt.figure()
plt.plot(range(1,8),Precision)
plt.xlabel('days ahead')
plt.title('Forecast precision')
plt.figure()
plt.plot(range(1,8),Recall)
plt.xlabel('days ahead')
plt.title('Forecast recall')

#%%
plt.figure()
plt.plot(dates[n_past+2000:n_past+2200],forecastReal[2000:2200,1], label ='predicted', marker = ",", linestyle="-")
plt.plot(dates[n_past+2000:n_past+2200], trainOutReal[n_past+2000:n_past+2200,1], label ='observed', marker = ",", linestyle="-")
plt.ylim(0,1500)
plt.legend()
plt.title('Forecast at t=+7')