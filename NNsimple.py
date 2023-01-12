# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:02:42 2022

@author: wullems
"""

test = True
#%%
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import metrics
from matplotlib import pyplot as plt
import sklearn.preprocessing
import seaborn as sns
import shap


#%% Import the dataset of measurements

FeaturesTable = pd.read_csv('C:\\Users\\wullems\\waterdata\\Features.csv')
# Interpolate missing values
FeaturesTable = FeaturesTable.interpolate()

dates = pd.to_datetime(FeaturesTable['Time'])

#%% Select the features with relevance according to Boruta analysis
FeaturesTable = FeaturesTable[['Time','ClKr400Min','ClKr400Mean','ClKr400Max','ClLkh700Min','ClLkh700Mean','ClLkh700Max','HDrdMean','HHvhMin','HHvhMean','HHvhMax','HKrMean','HVlaMean','QHagMean','QLobMean','QTielMean','WindEW','WindNS']]
Vars = list(FeaturesTable)[1:18]
#%% Extract and scale training data

Train = np.array(FeaturesTable[Vars][0:2557].astype(float))
Test = np.array(FeaturesTable[Vars][2557:].astype(float))
scaler = sklearn.preprocessing.StandardScaler()
#scaler = sklearn.preprocessing.PowerTransformer()
scaler = scaler.fit(Train)
TrainScaled = scaler.transform(Train)
TestScaled = scaler.transform(Test)
#Threshold = scaler.transform([np.repeat(300,np.shape(Train)[1]),np.repeat(300,np.shape(Train)[1])])[0,0:15]
Scaled = np.vstack((TrainScaled,TestScaled))
#%% Create input data as tensors with shape batchs * (timesteps * variables)

trainIn =[]
trainOut = []

n_future = 1
n_past = 5

for i in range(n_past, len(TrainScaled)):
    SaltInEl = np.concatenate(TrainScaled[i-n_past:i, 0:6].tolist())
    QtyInEl = np.concatenate(TrainScaled[i-n_past:i+1, 6:17].tolist())
    InEl = np.concatenate([SaltInEl,QtyInEl])
    trainIn.append(InEl)
    trainOut.append(TrainScaled[i,0:6])

trainIn, trainOut = np.array(trainIn), np.array(trainOut) 


#%% Create the structure of the machine learning model
InputData = keras.Input(n_past*6+(n_past+1)*11, name = 'Input_data')
Dense1 = keras.layers.Dense(16, activation = 'relu', name  = 'Dense_layer1')(InputData)
Dense2 = keras.layers.Dense(16, activation = 'relu', name  = 'Dense_layer2')(Dense1)
Dropout = keras.layers.Dropout(0.2)(Dense2)
SaltPred = keras.layers.Dense(6, name = 'Salt_Pred')(Dropout)

model = keras.Model(inputs = InputData, outputs= SaltPred)
tf.keras.utils.plot_model(model, to_file = 'SimpleNN1.png', show_shapes=True)
model.compile(optimizer='adam', loss = 'mean_squared_error')
callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min', restore_best_weights=True)
history = model.fit(trainIn, trainOut, epochs=1000, batch_size=16, validation_split=0.3, verbose=2, callbacks=callback)

plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.ylim(0.0,1.0)
plt.title('Paramset 8')
plt.legend()

#%% create a one day ahead forecast
forecast = model.predict(trainIn)
forecast_copies = np.hstack([forecast,forecast,forecast[:,0:5]])
forecastReal = scaler.inverse_transform(forecast_copies)[:,0:6]

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
RMSE[0] = math.sqrt(np.square(np.subtract(trainOutReal[n_past:,1],forecastReal[:,1])).mean())
Threshold = 300
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
        SaltInEl = trainIn[i-n_past, 6:6*(n_past)].tolist()
        SaltInEl = np.concatenate([SaltInEl, forecast[i-n_past,:].tolist()])
        QtyInEl = np.concatenate(Scaled[i-n_past+j-1:i+j, 6:17].tolist())
        InEl = np.concatenate([SaltInEl,QtyInEl])        
        trainIn[i-n_past] = InEl
    
    trainIn = np.array(trainIn)
            
    forecast = model.predict([trainIn])
    forecast_copies = np.hstack([forecast,forecast,forecast[:,0:5]])
    forecastReal = scaler.inverse_transform(forecast_copies)[:,0:6]
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
    
    n_future = 1
    n_past = 5
    
    for i in range(n_past, len(TestScaled)):
        SaltInEl = np.concatenate(TestScaled[i-n_past:i, 0:6].tolist()) 
        QtyInEl = np.concatenate(TestScaled[i-n_past:i+1, 6:17].tolist())
        InEl = np.concatenate([SaltInEl,QtyInEl])
        testIn.append(InEl)
        
    
    testIn = np.array(testIn) 

#%% create a one day ahead forecast
if test == True:
    forecast_test = model.predict(testIn)
    forecast_copies_test = np.hstack([forecast_test,forecast_test,forecast_test[:,0:5]])
    forecastReal_test = scaler.inverse_transform(forecast_copies_test)[:,0:6]
    testOutReal = Test[n_past:,0:6]
    plt.figure()
    plt.plot(dates[2557+n_past:],forecastReal_test[:,1], label ='predicted', marker = ",", linestyle="")
    plt.plot(dates[2557+n_past:], testOutReal[:,1], label ='observed', marker = ",", linestyle="")
    plt.ylim(0,1500)
    plt.legend()
    plt.title('Forecast at t=+1 (test)')
    RMSE_test = np.zeros(7)
    RMSE_test = np.zeros(7)
    Accuracy_test = np.zeros(7)
    Precision_test = np.zeros(7)
    Recall_test = np.zeros(7)
    RMSE_test[0] = math.sqrt(np.square(np.subtract(testOutReal[:,1],forecastReal_test[:,1])).mean())
    Threshold = 300
    event = testOutReal[:,1] > Threshold
    warning = forecastReal_test[:,1] > Threshold;
    Accuracy_test[0] = (sum(np.logical_and(event, warning)) + sum(np.logical_and(np.logical_not(event),np.logical_not(warning))))/len(forecast)
    Precision_test[0] = sum(np.logical_and(event, warning))/sum(warning)
    Recall_test[0] = sum(np.logical_and(event, warning))/sum(event)
    plt.figure()
    plt.plot(testOutReal[:,1], forecastReal_test[:,1], marker = ',', linestyle="")
    plt.plot([0,2000],[0,2000], color='black')
    plt.xlim(0,2000)
    plt.ylim(0,2000)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Fit at t=+1 (test)')

#%% replace observed values with forecast values to simulate multiple-day forecasts.
if test == True:
    for j in range(2,8):
        for i in range(n_past, len(TestScaled)-j):
            SaltInEl = testIn[i-n_past, 6:n_past*6].tolist()
            SaltInEl = np.concatenate([SaltInEl, forecast_test[i-n_past,:].tolist()])
            QtyInEl = np.concatenate(TestScaled[i-n_past+j-1:i+j, 6:17].tolist())
            InEl = np.concatenate([SaltInEl,QtyInEl])        
            testIn[i-n_past] = InEl
        
        testIn = np.array(testIn)
                
        forecast_test = model.predict(testIn)
        forecast_copies_test = np.hstack([forecast_test,forecast_test,forecast_test[:,0:5]])
        forecastReal_test = scaler.inverse_transform(forecast_copies_test)[:,0:6]
        testOutReal = Test[n_past:,0:6]
        plt.figure()
        plt.plot(dates[2557+n_past+j-1:],forecastReal_test[:-(j-1),1], label ='predicted', marker = ",", linestyle="")
        plt.plot(dates[2557+n_past+j-1:],testOutReal[j-1:,1], label ='observed', marker = ",", linestyle="")
        plt.ylim(0,1500)
        plt.legend() 
        plt.title('Forecast at t=+'+ str(j) + '(test)')
        RMSE_test[j-1] = math.sqrt(np.square(np.subtract(testOutReal[j-1:,1],forecastReal_test[:-(j-1),1])).mean())
        event = testOutReal[j-1:,1] > Threshold
        warning = forecastReal_test[:-(j-1),1] > Threshold;
        Accuracy_test[j-1] = (sum(np.logical_and(event, warning)) + sum(np.logical_and(np.logical_not(event),np.logical_not(warning))))/len(forecast)
        Precision_test[j-1] = sum(np.logical_and(event, warning))/sum(warning)
        Recall_test[j-1] = sum(np.logical_and(event, warning))/sum(event)
        plt.figure()
        plt.plot(testOutReal[j-1:,1], forecastReal_test[:-(j-1):,1], marker = ',', linestyle="")
        plt.plot([0,2000],[0,2000], color='black')
        plt.xlim(0,2000)
        plt.ylim(0,2000)
        plt.xlabel('observed')
        plt.ylabel('predicted')
        plt.title('Fit at t=+'+ str(j) + '(test)')
        
#%%
plt.figure()
plt.plot(range(1,8),RMSE)
plt.xlabel('days ahead')
plt.ylabel('Cl [mg/l]')
plt.title('Root Mean Squared Error (test)')
plt.figure()
plt.plot(range(1,8),Precision)
plt.xlabel('days ahead')
plt.title('Forecast precision (test)')
plt.figure()
plt.plot(range(1,8),Recall)
plt.xlabel('days ahead')
plt.title('Forecast recall (test)')
plt.figure()
plt.plot(range(1,8),RMSE_test)
plt.xlabel('days ahead')
plt.ylabel('Cl [mg/l]')
plt.title('Root Mean Squared Error (test)')
plt.figure()
plt.plot(range(1,8),Precision_test)
plt.xlabel('days ahead')
plt.title('Forecast precision (test)')
plt.figure()
plt.plot(range(1,8),Recall_test)
plt.xlabel('days ahead')
plt.title('Forecast recall(test)')

#%%
plt.figure()
plt.plot(dates[n_past+2000:n_past+2200],forecastReal[2000:2200,1], label ='predicted', marker = ",", linestyle="-")
plt.plot(dates[n_past+2000:n_past+2200], trainOutReal[n_past+2000:n_past+2200,1], label ='observed', marker = ",", linestyle="-")
plt.ylim(0,1500)
plt.legend()
plt.title('Forecast at t=+7')

#%%
AllVars =[]
Timesteps = []
#Ps = model.pvalues
#coefs = model.coef_[1,:]
#coefs_abs = abs(coefs)
AllVars = n_past*Vars[0:6]+(n_past+1)*Vars[6:17]
Timesteps = np.concatenate([np.repeat(-4,6),np.repeat(-3,6),np.repeat(-2,6),np.repeat(-1,6),np.repeat(0,6),np.repeat(-4,11),np.repeat(-3,11),np.repeat(-2,11),np.repeat(-1,11),np.repeat(0,11),np.repeat(1,11)])
FeatureImportance = pd.DataFrame(data={'Variable':AllVars, 'Timestep':Timesteps})
