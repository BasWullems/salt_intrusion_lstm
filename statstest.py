# -*- coding: utf-8 -*-

#%% Import packages
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

#%% Load features
FeaturesShift = pd.read_csv(r'C:\Users\wullems\waterdata\FeaturesShift.csv')
Features = pd.read_csv(r'C:\Users\wullems\waterdata\Features.csv')


#%% Create an input and output datset
Inputvars = ['ClKr400Min-1', 'ClKr400Min-2', 'ClKr400Min-3', 'ClKr400Min-4' , 'ClKr400Min-5', 'ClKr400Min-6', 'ClKr400Min-7', 'ClKr400Mean-1', 'ClKr400Mean-2', 'ClKr400Mean-3', 'ClKr400Mean-4' , 'ClKr400Mean-5', 'ClKr400Mean-6', 'ClKr400Mean-7','ClKr400Max-1', 'ClKr400Max-2', 'ClKr400Max-3', 'ClKr400Max-4' , 'ClKr400Max-5', 'ClKr400Max-6', 'ClKr400Max-7', 'ClKr550Min-1', 'ClKr550Min-2', 'ClKr550Min-3', 'ClKr550Min-4' , 'ClKr550Min-5', 'ClKr550Min-6', 'ClKr550Min-7', 'ClKr550Mean-1', 'ClKr550Mean-2', 'ClKr550Mean-3', 'ClKr550Mean-4' , 'ClKr550Mean-5', 'ClKr550Mean-6', 'ClKr550Mean-7', 'ClKr550Max-1', 'ClKr550Max-2', 'ClKr550Max-3', 'ClKr550Max-4' , 'ClKr550Max-5', 'ClKr550Max-6', 'ClKr550Max-7', 'ClLkh250Min-1', 'ClLkh250Min-2', 'ClLkh250Min-3', 'ClLkh250Min-4', 'ClLkh250Min-5', 'ClLkh250Min-6', 'ClLkh250Min-7', 'ClLkh250Mean-1', 'ClLkh250Mean-2', 'ClLkh250Mean-3', 'ClLkh250Mean-4', 'ClLkh250Mean-5', 'ClLkh250Mean-6', 'ClLkh250Mean-7', 'ClLkh250Max-1', 'ClLkh250Max-2', 'ClLkh250Max-3', 'ClLkh250Max-4', 'ClLkh250Max-5', 'ClLkh250Max-6', 'ClLkh250Max-7', 'ClLkh500Min-1', 'ClLkh500Min-2', 'ClLkh500Min-3', 'ClLkh500Min-4', 'ClLkh500Min-5', 'ClLkh500Min-6', 'ClLkh500Min-7', 'ClLkh500Mean-1', 'ClLkh500Mean-2', 'ClLkh500Mean-3', 'ClLkh500Mean-4', 'ClLkh500Mean-5', 'ClLkh500Mean-6', 'ClLkh500Mean-7', 'ClLkh500Max-1', 'ClLkh500Max-2', 'ClLkh500Max-3', 'ClLkh500Max-4', 'ClLkh500Max-5', 'ClLkh500Max-6', 'ClLkh500Max-7', 'ClLkh700Min-1', 'ClLkh700Min-2', 'ClLkh700Min-3', 'ClLkh700Min-4', 'ClLkh700Min-5', 'ClLkh700Min-6', 'ClLkh700Min-7', 'ClLkh700Mean-1', 'ClLkh700Mean-2', 'ClLkh700Mean-3', 'ClLkh700Mean-4', 'ClLkh700Mean-5', 'ClLkh700Mean-6', 'ClLkh700Mean-7', 'ClLkh700Max-1', 'ClLkh700Max-2', 'ClLkh700Max-3', 'ClLkh700Max-4', 'ClLkh700Max-5', 'ClLkh700Max-6', 'ClLkh700Max-7', 'HDrdMin-1', 'HDrdMin-2', 'HDrdMin-3', 'HDrdMean-1', 'HDrdMean-2', 'HDrdMean-3', 'HDrdMax-1', 'HDrdMax-2', 'HDrdMax-3', 'HHvhMin-1', 'HHvhMin-2', 'HHvhMin-3', 'HHvhMean-1', 'HHvhMean-2', 'HHvhMean-3', 'HHvhMax-1', 'HHvhMax-2', 'HHvhMax-3', 'HKrMin-1', 'HKrMin-2', 'HKrMin-3', 'HKrMean-1', 'HKrMean-2', 'HKrMean-3', 'HKrMax-1', 'HKrMax-2', 'HKrMax-3', 'HVlaMin-1', 'HVlaMin-2', 'HVlaMin-3', 'HVlaMean-1', 'HVlaMean-2', 'HVlaMean-3', 'HVlaMax-1', 'HVlaMax-2', 'HVlaMax-3', 'QHagMean-1', 'QHagMean-2', 'QHagMean-3', 'QLobMean-1', 'QLobMean-2', 'QLobMean-3', 'QTielMean-1', 'QTielMean-2', 'QTielMean-3']
Input = FeaturesShift[Inputvars]
Input.insert(0, 'Intercept', 1)
Output = FeaturesShift['ClKr400Mean']
Time = FeaturesShift['Time']

# Remove NaNs
rem, _ = np.where(Input.isna())
rem = np.unique(rem)
Rem = np.where(Output.isna())
Rem = np.unique(Rem)
Remove = np.concatenate((rem,Rem))
Input = Input.drop(Remove,axis=0)
Output = Output.drop(Remove,axis=0)
Time = Time.drop(Remove, axis = 0)
Input, Output, Time = Input.reset_index(drop=True), Output.reset_index(drop=True), Time.reset_index(drop=True)

#%% Partition dataset in training, cross-validation and test datasets
TrainIn, CvIn, TestIn = Input[0:1897], Input[1897:2438], Input[2438:2917]
TrainOut, CvOut, TestOut = Output[0:1897], Output[1897:2438], Output[2438:2917]
TrainTime, CvTime, TestTime = Time[0:1897], Time[1897:2438], Time[2438:2917]
#%% Fit a linear model
linmod = sm.OLS(TrainOut, TrainIn)
result = linmod.fit()
print(result.summary())
pred = result.fittedvalues
plt.figure(0)
plt.plot(TrainTime,TrainOut,'r.')
plt.plot(TrainTime,pred,'b.')
plt.legend('observed','predicted')
plt.title('Linear regression for mean chloride at Krimpen a/d IJssel: training')
plt.xlabel('Year')
plt.ylabel('Cl [mg/l]')
newpred = result.predict(TestIn)
plt.figure(1)
plt.plot(TestTime, TestOut, 'r.')
plt.plot(TestTime, newpred, 'b.')
plt.legend('observed', 'predicted')
plt.title('Linear regression for mean chloride at Krimpen a/d IJssel: test')
plt.xlabel('Year')
plt.ylabel('Cl [mg/l]')

#%%
corrmat = np.corrcoef(TrainIn.T.iloc[1:,:])
demo = TrainIn.values
#%% Try a stepwise regression
#sklearn.feature_selection.SequentialFeatureSelector(