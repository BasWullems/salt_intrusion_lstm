# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:49:07 2022

@author: wullems
"""

#%% Import packages
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.arima.model as arima
import pandas as pd
import matplotlib.pyplot as plt

#%% Load features
Features = pd.read_csv(r'C:\Users\wullems\waterdata\FeaturesNorm.csv')

#%% Create an input and output datset
Exog = Features[['ClLkh250Min', 'ClLkh250Mean', 'ClLkh250Max', 'ClLkh500Min', 'ClLkh500Mean', 'ClLkh500Max', 'ClLkh700Min', 'ClLkh700Mean', 'ClLkh700Max', 'HDrdMin', 'HDrdMean', 'HDrdMax', 'HHvhMin', 'HHvhMean', 'HHvhMax', 'HKrMin', 'HKrMean', 'HKrMax', 'HVlaMin', 'HVlaMean', 'HVlaMax', 'QHagMean', 'QLobMean', 'QTielMean']]
Endog = Features['ClKr400Mean']
Time = Features['Time']

# Remove NaNs
rem, _ = np.where(Exog.isna())
rem = np.unique(rem)
Rem = np.where(Endog.isna())
Rem = np.unique(Rem)
Remove = np.concatenate((rem,Rem))
Exog = Exog.drop(Remove,axis=0)
Endog = Endog.drop(Remove,axis=0)
Time = Time.drop(Remove, axis = 0)
Exog, Endog, Time = Exog.reset_index(drop=True), Endog.reset_index(drop=True), Time.reset_index(drop=True)
TrainEx, CvEx, TestEx = Exog[0:2013], Exog[2013:2598], Exog[2598:3162]
TrainEn, CvEn, TestEn = Endog[0:2013], Endog[2013:2598], Endog[2598:3162]
TrainTi, CvTi, TestTi = Time[0:2013], Time[2013:2598], Time[2598:3162]

#%% Fit a time series analysis model 
armamod = arima.ARIMA(TrainEn,TrainEx)
armares = armamod.fit()
armapredtrain = armares.fittedvalues
plt.plot(TrainTi, TrainEx,'r.')
plt.plot(TrainTi, armapredtrain,'b.')
#armapredtest = armares.predict(TestData)