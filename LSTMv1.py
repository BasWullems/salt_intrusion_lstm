# -*- coding: utf-8 -*-
"""
Build and/or evaluate an LSTM model to predict chloride concentrations.

Created on Thu Jun 30 14:46:31 2022.
@author: wullems
"""

# %% Import packages

import math
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import sklearn.preprocessing
from datetime import datetime
import random

# Fix the random seed to ensure reproducible results.
np.random.seed(1)
tf.random.set_seed(2)
random.seed(3)


# %% USER SETTINGS

# set TEST to True if the model must be tested on the seperate test dataset.
# Set to False for training only.
TEST = True

# Enter 'File' to reload a saved model suite or
# 'New' to recalculate and save models.
SOURCE = 'File'

# Set the location to save the models to. Must contain a format placeholder.
# Default = '.\\Models2\\LSTM_{}'
MODELPATH = '..\\Models\\LSTM_{}'

# TODO: fix inconsistent behaviour of file location ##########################

# Set the location to save figures to.
FIGPATH = ('C:\\Users\\wullems\\OneDrive - Stichting Deltares\\Pictures\\'
           'Analyse\\')

SAVEFIGS = False  # True if figures should be saved
NUMMODELS = 15  # number of models in the ensemble
N_PAST = 5  # number of days in the past used to make a prediction
N_FUTURE = 7  # number of days in the future for which to make a prediction

# Set the date at which the train-test split occurs
# (default is datetime(2018, 1, 1, 0, 0, 0)
SPLIT = datetime(2018, 1, 1, 0, 0, 0)

# Set the threshold value for chloride concentrations above which a salt
# intrusion event is defined
THRESHOLD = 300


# %% Import the dataset of measurements

features_table = pd.read_csv('..\\Data\\Features.csv', index_col=0)
# Interpolate missing values
features_table = features_table.interpolate()
# Convert dates to datetime format
dates = pd.to_datetime(features_table.Time)
features_table.Time = pd.to_datetime(features_table['Time'])
# Index observations by their date
features_table = features_table.set_index('Time')


# %% Select the features with relevance according to Boruta analysis
# (see script boruta_procedure.py)

features_table = features_table[['ClKr400Min', 'ClKr400Mean', 'ClKr400Max',
                                 'ClKr550Min', 'ClKr550Mean', 'ClKr550Max',
                                 'ClLkh250Min', 'ClLkh250Mean', 'ClLkh250Max',
                                 'ClLkh700Min', 'ClLkh700Mean', 'ClLkh700Max',
                                 'HDrdMean', 'HHvhMean',
                                 'HKrMin', 'HKrMean', 'HKrMax',
                                 'HVlaMean',
                                 'QHagMean', 'QLobMean', 'QTielMean',
                                 'WindEW', 'WindNS']]
vars_ = features_table.columns  # Extract variable names


# %% Extract, split and scale training data

split = features_table.index.get_loc(SPLIT)
train = features_table.iloc[:split, :]
test = features_table.iloc[split:, :]

scaler = sklearn.preprocessing.StandardScaler()
scaler = scaler.fit(train)

train_scaled = scaler.transform(train)
train_scaled = pd.DataFrame(train_scaled, index=dates[:split], columns=vars_)
test_scaled = scaler.transform(test)
test_scaled = pd.DataFrame(test_scaled, index=dates[split:], columns=vars_)


# %% Create input data as tensors with shape:
# issue date * preceding timesteps * variables

# We have 12 features with chloride concentrations and 11 features with other
# variables (here called quantity or qty variables). These must be treated
# differently in the model and are therefore split into two separate tensors.
# For the salt data, the input consists of measurements of the N_PAST preceding
# days. For the quantity data, we also include the measurement of the next day,
# as a proxy for a forecast.
# Salt data of the next day form the output dataset on which we train the
# model.

salt_in = np.empty((train_scaled.shape[0]-N_PAST, N_PAST, 12))
salt_in.fill(np.nan)
qty_in = np.empty((train_scaled.shape[0]-N_PAST, N_PAST+1, 11))
qty_in.fill(np.nan)
train_out = np.empty((train_scaled.shape[0]-N_PAST, 12))
train_out.fill(np.nan)

for i in range(N_PAST, train_scaled.shape[0]):
    salt_in[i-N_PAST, :, :] = np.array(train_scaled[vars_[0:12]][i-N_PAST:i])
    qty_in[i-N_PAST, :, :] = np.array(train_scaled[vars_[12:23]][i-N_PAST:i+1])
    train_out[i-N_PAST, :] = np.array(train_scaled.iloc[i, 0:12])


# %% Create and train the machine learning model, or load from file


def createmodel(path, nummodels):
    """
    Create and save an ensemble of Long Short Term Memory models.

    Parameters
    ----------
    path : str
        Path to which the model should be saved.
    nummodels : int
        Number of models in the ensemble.

    Returns
    -------
    models : list of Functional
        ensemble of models to be used directly in the Python envorionment
    or reloaded from the location specified by 'path'.

    """
    models = []

    # A number of models are trained, with different initial conditions
    # randomly set by the algorithm.
    # This creates an ensemble and accounts for some of the spread in
    # model solutions.
    # Randomness is restricted by a fixed seed to  ensure reproducibility.

    for m in range(nummodels):
        print('Fitting model ' + str(m))
        SaltData = keras.Input(shape=(N_PAST, 12), name='salt_data')
        QtyData = keras.Input(shape=(N_PAST+1, 11), name='quantity_data')
        SaltSeq = keras.layers.LSTM(
            32, activation='relu', name='salt_seq',
            return_sequences=False, return_state=False)(SaltData)

        # Input data are fed into an LSTM layer with relu activation. Relu
        # activation is most suitable for regression problems.
        # Only cell outputs (no internal states or sequences) should be
        # returned, to make concatenation possible.

        # TODO: add source ###################################################
        # TODO: consider using the built-in dropout of the LSTM layer. #######

        SaltSeq2 = keras.layers.Dropout(0.3, name='salt_seq_2')(SaltSeq)
        QtySeq = keras.layers.LSTM(32, activation='relu', name='qty_seq',
                                   return_sequences=False, return_state=False
                                   )(QtyData)
        QtySeq2 = keras.layers.Dropout(0.3, name='qty_seq_2')(QtySeq)
        pattern = keras.layers.concatenate([SaltSeq2, QtySeq2], name='pattern')

        #######################################################################
        # Uncomment these lines for experimentation with the model structure.
        # If Connector and Connector2 are used, SaltPred must be connected to
        # (Connector2) instead of (pattern)

        # Connector = keras.layers.Dense(64, name='Connector')(pattern)
        # Connector2 = keras.layers.Dropout(0.5)(Connector)

        # HAVE YOU CHANGED THE LAYER CONNECTIONS? ############################
        ######################################################################

        # The outputs are finally fed into an output layer that relates LSTM
        # outputs to predicted salt concentrations.
        SaltPred = keras.layers.Dense(12, name='salt_pred')(pattern)
        model = keras.Model(inputs=(SaltData, QtyData), outputs=SaltPred)

        # Set the criteria for model optimization: algorithm, metric and
        # weights.
        # This model is fit to predict the twelve salt variables (%line 74-77).
        # Using loss_weights, the fit to one variable can be made more
        # important in the fitting process than the others.
        # The adam optimizer adjusts model weights by a variable amount, trying
        # to avoid the risks of getting stuck in local minima or skipping over
        # the global minimum of the loss function.
        # See: Kingma & Ba, 2017. Adam: a mehthod for stochastic optimization.
        # https://arxiv.org/pdf/1412.6980.pdf

        model.compile(optimizer='adam', loss='mse',
                      loss_weights=[2, 3, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1])
        tf.keras.utils.plot_model(model, show_shapes=True)

        # Set the number of iterations without improvement to the validation
        # dataset (generated by algorithm, see below), before concluding
        # maximum accuracy has been reached and restoring model weights to the
        # epoch with the best results.
        callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=100, verbose=1,
            mode='min', restore_best_weights=True)

        # Fit model to the training dataset.
        # Fitting is done in iterations called epochs.
        # In each epoch, all instances of SaltData and QtyData are used to
        # predict the values in train_out, in batches of size 'batch_size'.
        # The last 30% of the training dataset is kept separate for validation.
        history = model.fit((salt_in, qty_in), train_out,
                            epochs=500, batch_size=64, validation_split=0.3,
                            verbose=2, callbacks=callback, shuffle=False)

        plt.figure()
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.ylim(0.0, 1.0)
        plt.xlabel('iterations')
        plt.ylabel('loss (mean squared error)')
        plt.legend()
        plt.show()
        models.append(model)
        model.save(path.format(m))
    return models

# TODO: from here, start a new script ########################################


def loadmodel(path, nummodels):
    """
    Load LSTM salt intrusion model from specified location.

    Parameters
    ----------
    path : str
        Path from which the model should be loaded.
    nummodels : int
        Number of models in the ensemble.
    (Cannot exceed the number of models stored in path)

    # TODO: assert condition #################################################

    Returns
    -------
    models : list of Functional
        ensemble of models to be used directly in the Python envorionment.

    """
    models = []
    for m in range(nummodels):
        model = keras.models.load_model(path.format(m))
        models.append(model)
    return models


if SOURCE == 'New':
    models = createmodel(MODELPATH, NUMMODELS)
elif SOURCE == 'File':
    models = loadmodel(MODELPATH, NUMMODELS)

# %% Use each of the models to create a forecast


def ensemble_forecast(models, data, nfuture, saltvars, qtyvars):
    """
    Create a number of forecasts using an ensemble of models.

    Parameters
    ----------
    models : list of Functional
        Ensemble of LSTM models.
    data : DataFrame
        Input data with dates as indices and variables as columns.
    nfuture : int
        Maximum number of days to predict ahead.
    saltvars : Index
        Columns in 'data' to be used as salt input variables. Must match the
        number of salt variables used in the training of 'models'.

    # TODO: assert condition ##################################################

    qtyvars : Index
        Columns in 'data' to be used as quantity input variables. Must match
        the number of quantity variables used in the training of 'models'.

    # TODO: assert condition ##################################################

    Returns
    -------
    forecast : Array of float
        Array with shape models * lead time * issue times * salt variables.
        Contains predicted values of chloride concentrations, ginven in
        normalized values with respect to the statistics of the training
        dataset.
    forecast_real : Array of float
        Array with shape models * lead time * issue times * salt variables.
        Transformed back to real chloride concentrations, using statistics of
        the training dataset .

    # TODO: try to make the scaler an integral part of the model, so it doesn't
    have to be supplied by the user.'##########################################

    """
    # set up tensor structure as model*lead time*issue time*variables
    forecast = np.empty(
        (len(models), nfuture, data.shape[0]-nfuture-N_PAST, len(saltvars)))
    forecast.fill(np.nan)
    forecast_real = np.empty(
        (len(models), nfuture, data.shape[0]-nfuture-N_PAST, len(saltvars)))
    forecast_real.fill(np.nan)

    # We have 12 features with chloride concentrations and 11 features with
    # other variables (here called quantity or qty variables). These must be
    # treated differently in the model and are therefore split into two
    # separate tensors.
    # For the salt data, the input consists of measurements of the N_PAST
    # preceding days. For the quantity data, we also include the measurement of
    # the next day, as a proxy for a forecast.

    for m in range(NUMMODELS):
        model = models[m]
        salt_in = np.empty(
            (data.shape[0]-N_PAST-nfuture, N_PAST, len(saltvars)))
        salt_in.fill(np.nan)
        qty_in = np.empty(
            (data.shape[0]-N_PAST-nfuture, N_PAST+1, len(qtyvars)))
        qty_in.fill(np.nan)

        for i in range(N_PAST, data.shape[0]-nfuture):
            salt_in[i-N_PAST, :, :] = np.array(data[saltvars][i-N_PAST:i])
            qty_in[i-N_PAST, :, :] = np.array(data[qtyvars][i-N_PAST:i+1])

        # Create a one day ahead forecast
        forecast[m, 0, :, :] = model.predict([salt_in, qty_in])
        # Backtransform the forecast from normalized scores to real
        # concentrations.
        forecast_copies = np.hstack(
            [forecast[m, 0, :, :], forecast[m, 0, :, 1:]])
        forecast_real[m, 0, :, :] = scaler.inverse_transform(
            forecast_copies)[:, 0:12]

        # We create a new salt_in dataset by taking all but the first
        # observation used for that issue time. Then append the forecast we
        # just made to this observation sequence. We repeat this procedure
        # until we have reached the desired forecasting horizon given by
        # nfuture. We hereby shift the window of observations one day at the
        # time, with the last value being the result of the previous
        # forecasting step.

        for j in range(1, nfuture):
            for i in range(N_PAST, data.shape[0]-nfuture):
                salt_in[i-N_PAST, :, :] = np.vstack(
                    (salt_in[i-N_PAST, 1:, :], forecast[m, j-1, i-N_PAST, :]))
                qty_in[i-N_PAST, :, :] = np.array(
                    data[qtyvars][i-N_PAST+j:i+j+1])
            forecast[m, j, :, :] = model.predict([salt_in, qty_in])
            forecast_copies = np.hstack(
                [forecast[m, j, :, :], forecast[m, j, :, 1:]])
            forecast_real[m, j, :, :] = scaler.inverse_transform(
                forecast_copies)[:, 0:12]

    return forecast, forecast_real


# make an ensemble forecast for the training dataset
forecast, forecast_real = ensemble_forecast(
    models, train_scaled, N_FUTURE, vars_[0:12], vars_[12:23])

# %% Set up an alternative model to compare skill

# The persistence forecast takes the most recent observation at issue time and
# uses that for all the predicted values until the forecasting horizon.


def build_persistence_model(npast, nfuture, data, col=1):
    """
    Copy the most recent observation at issue time into the future.

    Parameters
    ----------
    npast : int
        Number of past observations used by the LSTM model (Needed to ensure
        consistent shape).
    # TODO: check if this is really necessary.#################################
    nfuture : int
        Maximum number of days to predict ahead.
    data : DataFrame
        Input data with dates as indices and variables as columns.
    col : int
        Index of variable for which to create a persistence forecast.

    Returns
    -------
    persistence_model : array of float
        array of shape lead time * issue time.
        Contains predicted salt concentrations.
    """
    persistence_model = np.empty((nfuture, data.shape[0]-nfuture-npast))
    persistence_model.fill(np.nan)
    for j in range(nfuture):
        for i in range(npast, data.shape[0]-nfuture):
            persistence_model[j, i-npast] = data.iloc[i-1, col]
    return persistence_model


persistence_model = build_persistence_model(N_PAST, N_FUTURE, train, 1)


# %% Plot forecasts and fits

forecast_fig, forecast_axs = plt.subplots(3, 2, figsize=(10, 10),
                                          sharex=False, sharey=False)
forecast_1 = forecast_axs[0, 0]
forecast_4 = forecast_axs[1, 0]
forecast_7 = forecast_axs[2, 0]

# Calculate the upper and lower bound and median of predicted values
lowbound1 = np.amin(forecast_real[:, 0, :365, 1], axis=0)
highbound1 = np.amax(forecast_real[:, 0, :365, 1], axis=0)
medianval1 = np.median(forecast_real[:, 0, :365, 1], axis=0)

# Plot predicted and observed values in the first year of the training dataset
# for lead time 1.
forecast_1.fill_between(train.index[N_PAST:N_PAST+365], lowbound1, highbound1,
                        label='predicted', color='pink')
forecast_1.plot(train.index[N_PAST:N_PAST+365], medianval1,
                label='predicted', linestyle="-", lw=0.5, color='red')
forecast_1.plot(train.index, train.ClKr400Mean,
                label='observed', linestyle="-", lw=0.5, color='blue')
forecast_1.set_xlim(datetime(2011, 1, 1, 0, 0, 0),
                    datetime(2012, 1, 1, 0, 0, 0))
forecast_1.set_xticks(ticks=[datetime(2011, 1, 1, 0, 0, 0),
                             datetime(2011, 4, 1, 0, 0, 0),
                             datetime(2011, 7, 1, 0, 0, 0),
                             datetime(2011, 10, 1, 0, 0, 0)], labels=[])
forecast_1.set_ylim(0, 1500)
forecast_1.set_ylabel('[Cl] (mg/L)')
forecast_1.set(title='Forecast (2011)')
forecast_1.annotate('(a) t+1', (datetime(2011, 1, 10, 0, 0, 0), 1400))
forecast_1.legend(['predicted range', 'predicted median', 'observed'],
                  loc=(0.05, 0.65), frameon=False)

# Repeat procedure for lead times 4 and 7.
lowbound4 = np.amin(forecast_real[:, 3, :365, 1], axis=0)
highbound4 = np.amax(forecast_real[:, 3, :365, 1], axis=0)
medianval4 = np.median(forecast_real[:, 3, :365, 1], axis=0)

forecast_4.fill_between(train.index[N_PAST+3:N_PAST+3+365],
                        lowbound4, highbound4,
                        label='predicted', color='pink')
forecast_4.plot(train.index[N_PAST+3:N_PAST+3+365], medianval4,
                label='predicted', linestyle="-", lw=0.5, color='red')
forecast_4.plot(train.index, train.ClKr400Mean,
                label='observed', linestyle="-", lw=0.5, color='blue')
forecast_4.set_xlim(datetime(2011, 1, 1, 0, 0, 0),
                    datetime(2012, 1, 1, 0, 0, 0))
forecast_4.set_xticks(ticks=[datetime(2011, 1, 1, 0, 0, 0),
                             datetime(2011, 4, 1, 0, 0, 0),
                             datetime(2011, 7, 1, 0, 0, 0),
                             datetime(2011, 10, 1, 0, 0, 0)], labels=[])
forecast_4.set_ylim(0, 1500)
forecast_4.set_ylabel('[Cl] (mg/L)')
forecast_4.annotate('(c) t+4', (datetime(2011, 1, 10, 0, 0, 0), 1400))

lowbound7 = np.amin(forecast_real[:, 6, :365, 1], axis=0)
highbound7 = np.amax(forecast_real[:, 6, :365, 1], axis=0)
medianval7 = np.median(forecast_real[:, 6, :365, 1], axis=0)

forecast_7.fill_between(train.index[N_PAST+6:N_PAST+6+365],
                        lowbound7, highbound7, label='predicted', color='pink')
forecast_7.plot(train.index[N_PAST+6:N_PAST+6+365], medianval7,
                label='predicted', linestyle="-", lw=0.5, color='red')
forecast_7.plot(train.index, train.ClKr400Mean,
                label='observed', linestyle="-", lw=0.5, color='blue')
forecast_7.set_xlim(datetime(2011, 1, 1, 0, 0, 0),
                    datetime(2012, 1, 1, 0, 0, 0))
forecast_7.set_xticks(ticks=[datetime(2011, 1, 3, 0, 0, 0),
                             datetime(2011, 4, 1, 0, 0, 0),
                             datetime(2011, 7, 1, 0, 0, 0),
                             datetime(2011, 10, 1, 0, 0, 0)],
                      labels=['Jan', 'Apr', 'Jul', 'Oct'])
forecast_7.set_ylim(0, 1500)
forecast_7.set_xlabel(2011)
forecast_7.set_ylabel('[Cl] (mg/L)')
forecast_7.annotate('(e) t+7', (datetime(2011, 1, 10, 0, 0, 0), 1400))

# Plot predicted values against observed values.
fit_1 = forecast_axs[0, 1]
fit_4 = forecast_axs[1, 1]
fit_7 = forecast_axs[2, 1]

for m in range(NUMMODELS):
    fit_1.plot(train.ClKr400Mean[N_PAST:-N_FUTURE], forecast_real[m, 0, :, 1],
               marker=',', color='red', linestyle="")
fit_1.plot([0, 2000], [0, 2000], color='black',)
fit_1.set_aspect('equal', 'box')
fit_1.set_xlim(0, 1500)
fit_1.set_ylim(0, 1500)
fit_1.set_ylabel('[Cl] (mg/L) predicted')
fit_1.set_title('Fit (2011-2017)')
fit_1.annotate('(b) t+1', (100, 1400))
fit_1.set_xticks(ticks=[0, 500, 1000, 1500], labels=[])
fit_1.set_yticks(ticks=[0, 500, 1000, 1500])

for m in range(NUMMODELS):
    fit_4.plot(train.ClKr400Mean[N_PAST+3:-N_FUTURE+3],
               forecast_real[m, 3, :, 1],
               marker=',', color='red', linestyle="")
fit_4.plot([0, 2000], [0, 2000], color='black')
fit_4.set_aspect('equal', 'box')
fit_4.set_xlim(0, 1500)
fit_4.set_ylim(0, 1500)
fit_4.set_ylabel('[Cl] (mg/L) predicted')
fit_4.annotate('(d) t+4', (100, 1400))
fit_4.set_xticks(ticks=[0, 500, 1000, 1500], labels=[])
fit_4.set_yticks(ticks=[0, 500, 1000, 1500])

for m in range(NUMMODELS):
    fit_7.plot(train.ClKr400Mean[N_PAST+6:-N_FUTURE+6],
               forecast_real[m, 6, :, 1],
               marker=',', color='red', linestyle="")
fit_7.plot([0, 2000], [0, 2000], color='black')
fit_7.set_aspect('equal', 'box')
fit_7.set_xlim(0, 1500)
fit_7.set_ylim(0, 1500)
fit_7.set_xlabel('[Cl] (mg/l) observed')
fit_7.set_ylabel('[Cl] (mg/L) predicted')
fit_7.annotate('(f) t+7', (100, 1400))
fit_7.set_xticks(ticks=[0, 500, 1000, 1500])
fit_7.set_yticks(ticks=[0, 500, 1000, 1500])

forecast_fig.suptitle('Forecast of mean [Cl] at Krimpen aan de IJssel, '
                      'depth=-4.0 m a.m.s.l.\n (training)')

if SAVEFIGS is True:
    forecast_fig.savefig(FIGPATH+'Forecast_2011_train.png')
    forecast_fig.savefig(FIGPATH+'Forecast_2011_train.pdf')


# %% Compute metrics

RMSE = np.empty((N_FUTURE, NUMMODELS))
RMSE.fill(np.nan)
Accuracy = np.empty((N_FUTURE, NUMMODELS))
Accuracy.fill(np.nan)
Precision = np.empty((N_FUTURE, NUMMODELS))
Precision.fill(np.nan)
Recall = np.empty((N_FUTURE, NUMMODELS))
Recall.fill(np.nan)

RMSE_persistence = np.empty(N_FUTURE)
RMSE_persistence.fill(np.nan)
Accuracy_persistence = np.empty(N_FUTURE)
Accuracy_persistence.fill(np.nan)
Precision_persistence = np.empty(N_FUTURE)
Precision_persistence.fill(np.nan)
Recall_persistence = np.empty(N_FUTURE)
Recall_persistence.fill(np.nan)

# Calculate Root mean squared error for persistence model vs. observations
RMSE_persistence[0] = math.sqrt(np.square(np.subtract(
    train.ClKr400Mean[N_PAST:-N_FUTURE], persistence_model[0, :])).mean())

# Define an event as a day where mean chloride concentrations exceeded a
# threshold level.
# Define a warning as a day where the model predicts exceedence of the
# threshold level.

event = train.ClKr400Mean[N_PAST:-N_FUTURE] > THRESHOLD
warning_persistence = persistence_model[0, :] > THRESHOLD

# Calculate accuracy, precision and recall for the persistence model.
Accuracy_persistence[0] = (
    sum(np.logical_and(event, warning_persistence))
    + sum(np.logical_and(np.logical_not(event),
                         np.logical_not(warning_persistence)))
    )/(forecast.shape[2])
Precision_persistence[0] = sum(np.logical_and(event, warning_persistence)
                               )/sum(warning_persistence)
Recall_persistence[0] = sum(np.logical_and(event, warning_persistence)
                            )/sum(event)

# Calculate RMSE, accuracy, precision and recall for the persistence model.
for m in range(NUMMODELS):
    RMSE[0, m] = math.sqrt(np.square(np.subtract(
            train.ClKr400Mean[N_PAST:-N_FUTURE], forecast_real[m, 0, :, 1])
        ).mean())
    warning_LSTM = forecast_real[m, 0, :, 1] > THRESHOLD
    Accuracy[0, m] = (
        sum(np.logical_and(event, warning_LSTM))
        + sum(np.logical_and(np.logical_not(event),
                             np.logical_not(warning_LSTM)))
        )/(forecast.shape[2])
    Precision[0, m] = sum(np.logical_and(event, warning_LSTM)
                          )/sum(warning_LSTM)
    Recall[0, m] = sum(np.logical_and(event, warning_LSTM))/sum(event)

# Repeat procedure for each of the lead times.
for j in range(1, N_FUTURE):
    event = train.ClKr400Mean[N_PAST+j:-N_FUTURE+j] > THRESHOLD
    warning_persistence = persistence_model[j, :] > THRESHOLD
    RMSE_persistence[j] = math.sqrt(np.square(np.subtract(
        train.ClKr400Mean[N_PAST+j:-N_FUTURE+j], persistence_model[j, :])
        ).mean())
    Accuracy_persistence[j] = (
        sum(np.logical_and(event, warning_persistence))
        + sum(np.logical_and(np.logical_not(event),
                             np.logical_not(warning_persistence)))
        )/(forecast.shape[2])
    Precision_persistence[j] = sum(np.logical_and(
        event, warning_persistence))/sum(warning_persistence)
    Recall_persistence[j] = sum(np.logical_and(
        event, warning_persistence))/sum(event)

    for m in range(NUMMODELS):
        RMSE[j, m] = math.sqrt(np.square(np.subtract(
            train.ClKr400Mean[N_PAST+j:-N_FUTURE+j],
            forecast_real[m, j, :, 1])).mean())
        warning_LSTM = forecast_real[m, j, :, 1] > THRESHOLD
        Accuracy[j, m] = (
            sum(np.logical_and(event, warning_LSTM))
            + sum(np.logical_and(np.logical_not(event),
                                 np.logical_not(warning_LSTM)))
            )/(forecast.shape[2])
        Precision[j, m] = sum(np.logical_and(
            event, warning_LSTM))/sum(warning_LSTM)
        Recall[j, m] = sum(np.logical_and(event, warning_LSTM))/sum(event)


# %% Plot metrics
metrics_fig, metrics_axs = plt.subplots(
    1, 3, figsize=(10, 3), sharex=False, sharey=False)
metrics_fig.tight_layout(pad=2)
RMSE_plot = metrics_axs[0]
Precision_plot = metrics_axs[1]
Recall_plot = metrics_axs[2]

for m in range(NUMMODELS):
    RMSE_plot.plot(range(1, 8), RMSE[:, m], color='red', lw=0.5, label='LSTM')
RMSE_plot.plot(range(1, 8), RMSE_persistence,
               color='cyan', lw=1.5, label='Persistence')
RMSE_plot.set_ylabel('RMSE [mg/l]')
RMSE_plot.set_xticks(ticks=range(1, 8))
RMSE_plot.annotate('(a)', (1, 70))

for m in range(NUMMODELS):
    Precision_plot.plot(
        range(1, 8), Precision[:, m], color='red', lw=0.5, label='LSTM')
Precision_plot.plot(range(1, 8), Precision_persistence,
                    color='cyan', lw=1.5, label='Persistence')
Precision_plot.set_xlabel('lead time (days)')
Precision_plot.set_ylabel('Precision')
Precision_plot.set_xticks(ticks=range(1, 8))
Precision_plot.set_ylim(0, 1)
Precision_plot.annotate('(b)', (1, 0.9))

for m in range(NUMMODELS):
    Recall_plot.plot(
        range(1, 8), Recall[:, m], color='red', lw=0.5, label='LSTM')
Recall_plot.plot(range(1, 8), Recall_persistence,
                 color='cyan', lw=1.5, label='reference (persistence)')
Recall_plot.set_ylabel('Recall')
Recall_plot.set_xticks(ticks=range(1, 8))
Recall_plot.set_ylim(0, 1)
Recall_plot.annotate('(c)', (1, 0.9))
plt.legend(Recall_plot.get_legend_handles_labels()[0][-2:],
           Recall_plot.get_legend_handles_labels()[1][-2:], frameon=False)

if SAVEFIGS is True:
    metrics_fig.savefig(FIGPATH+'metrics_train.png')
    metrics_fig.savefig(FIGPATH+'metrics_train.pdf')


# %% Run the analysis on a separate test dataset

if TEST is True:
    forecast_test, forecast_real_test = ensemble_forecast(
        models, test_scaled, N_FUTURE, vars_[0:12], vars_[12:23])


# %% Set up an alternative model to compare skill (TEST)

if TEST is True:
    persistence_model_test = build_persistence_model(N_PAST, N_FUTURE, test, 1)


# %% Plot forecasts and fits (test)

if TEST is True:
    forecast_fig_test, forecast_axs_test = plt.subplots(
        3, 2, figsize=(10, 10), sharex=False, sharey=False)
    forecast_1_test = forecast_axs_test[0, 0]
    forecast_4_test = forecast_axs_test[1, 0]
    forecast_7_test = forecast_axs_test[2, 0]

    lowbound1_test = np.amin(forecast_real_test[:, 0, :365, 1], axis=0)
    highbound1_test = np.amax(forecast_real_test[:, 0, :365, 1], axis=0)
    medianval1_test = np.median(forecast_real_test[:, 0, :365, 1], axis=0)

    forecast_1_test.fill_between(test.index[N_PAST:N_PAST+365],
                                 lowbound1_test, highbound1_test,
                                 label='predicted', color='pink')
    forecast_1_test.plot(test.index[N_PAST:N_PAST+365], medianval1_test,
                         label='predicted', linestyle="-", lw=0.5, color='red')
    forecast_1_test.plot(test.index, test.ClKr400Mean,
                         label='observed', linestyle="-", lw=0.5, color='blue')
    forecast_1_test.set_xlim(
        datetime(2018, 1, 1, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0))
    forecast_1_test.set_xticks(ticks=[datetime(2018, 1, 1, 0, 0, 0),
                                      datetime(2018, 4, 1, 0, 0, 0),
                                      datetime(2018, 7, 1, 0, 0, 0),
                                      datetime(2018, 10, 1, 0, 0, 0)],
                               labels=[])
    forecast_1_test.set_ylim(0, 2000)
    forecast_1_test.set_ylabel('[Cl] (mg/L)')
    forecast_1_test.set(title='Forecast (2018)')
    forecast_1_test.annotate('(a) t+1', (datetime(2018, 1, 10, 0, 0, 0), 1800))
    forecast_1_test.legend(['predicted range', 'predicted median', 'observed'],
                           loc=(0.05, 0.6), frameon=False)

    lowbound4_test = np.amin(forecast_real_test[:, 3, :365, 1], axis=0)
    highbound4_test = np.amax(forecast_real_test[:, 3, :365, 1], axis=0)
    medianval4_test = np.median(forecast_real_test[:, 3, :365, 1], axis=0)

    forecast_4_test.fill_between(test.index[N_PAST+3:N_PAST+3+365],
                                 lowbound4_test, highbound4_test,
                                 label='predicted', color='pink')
    forecast_4_test.plot(test.index[N_PAST+3:N_PAST+3+365],
                         medianval4_test,
                         label='predicted', linestyle="-", lw=0.5, color='red')
    forecast_4_test.plot(test.index, test.ClKr400Mean,
                         label='observed', linestyle="-", lw=0.5, color='blue')
    forecast_4_test.set_xlim(
        datetime(2018, 1, 1, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0))
    forecast_4_test.set_xticks(ticks=[datetime(2018, 1, 1, 0, 0, 0),
                                      datetime(2018, 4, 1, 0, 0, 0),
                                      datetime(2018, 7, 1, 0, 0, 0),
                                      datetime(2018, 10, 1, 0, 0, 0)],
                               labels=[])
    forecast_4_test.set_ylim(0, 2000)
    forecast_4_test.set_ylabel('[Cl] (mg/L)')
    forecast_4_test.annotate('(c) t+4', (datetime(2018, 1, 10, 0, 0, 0), 1800))

    lowbound7_test = np.amin(forecast_real_test[:, 6, :365, 1], axis=0)
    highbound7_test = np.amax(forecast_real_test[:, 6, :365, 1], axis=0)
    medianval7_test = np.median(forecast_real_test[:, 6, :365, 1], axis=0)

    forecast_7_test.fill_between(test.index[N_PAST+6:N_PAST+6+365],
                                 lowbound7_test, highbound7_test,
                                 label='predicted', color='pink')
    forecast_7_test.plot(test.index[N_PAST+6:N_PAST+6+365],
                         medianval7_test,
                         label='predicted', linestyle="-", lw=0.5, color='red')
    forecast_7_test.plot(test.index, test.ClKr400Mean,
                         label='observed', linestyle="-", lw=0.5, color='blue')
    forecast_7_test.set_xlim(
        datetime(2018, 1, 1, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0))
    forecast_7_test.set_xticks(ticks=[datetime(2018, 1, 1, 0, 0, 0),
                                      datetime(2018, 4, 1, 0, 0, 0),
                                      datetime(2018, 7, 1, 0, 0, 0),
                                      datetime(2018, 10, 1, 0, 0, 0)],
                               labels=['Jan', 'Apr', 'Jul', 'Oct'])
    forecast_7_test.set_ylim(0, 2000)
    forecast_7_test.set_xlabel(2018)
    forecast_7_test.set_ylabel('[Cl] (mg/L)')
    forecast_7_test.annotate('(e) t+7', (datetime(2018, 1, 10, 0, 0, 0), 1800))

    fit_1_test = forecast_axs_test[0, 1]
    fit_4_test = forecast_axs_test[1, 1]
    fit_7_test = forecast_axs_test[2, 1]

    for m in range(NUMMODELS):
        fit_1_test.plot(test.ClKr400Mean[N_PAST:-N_FUTURE],
                        forecast_real_test[m, 0, :, 1],
                        marker=',', color='red', linestyle="")
    fit_1_test.plot([0, 2000], [0, 2000], color='black')
    fit_1_test.set_aspect('equal', 'box')
    fit_1_test.set_xlim(0, 2000)
    fit_1_test.set_ylim(0, 2000)
    fit_1_test.set_ylabel('[Cl] (mg/L) predicted')
    fit_1_test.set_title('Fit (2018-2020)')
    fit_1_test.annotate('(b) t+1', (100, 1800))
    fit_1_test.set_xticks(ticks=[0, 500, 1000, 1500, 2000], labels=[])
    fit_1_test.set_yticks(ticks=[0, 500, 1000, 1500, 2000])

    for m in range(NUMMODELS):
        fit_4_test.plot(test.ClKr400Mean[N_PAST+3:-N_FUTURE+3],
                        forecast_real_test[m, 3, :, 1],
                        marker=',', color='red', linestyle="")
    fit_4_test.plot([0, 2000], [0, 2000], color='black')
    fit_4_test.set_aspect('equal', 'box')
    fit_4_test.set_xlim(0, 2000)
    fit_4_test.set_ylim(0, 2000)
    fit_4_test.set_ylabel('[Cl] (mg/L) predicted')
    fit_4_test.annotate('(d) t+4', (100, 1800))
    fit_4_test.set_xticks(ticks=[0, 500, 1000, 1500, 2000], labels=[])
    fit_4_test.set_yticks(ticks=[0, 500, 1000, 1500, 2000])

    for m in range(NUMMODELS):
        fit_7_test.plot(test.ClKr400Mean[N_PAST+6:-N_FUTURE+6],
                        forecast_real_test[m, 6, :, 1],
                        marker=',', color='red', linestyle="")
    fit_7_test.plot([0, 2000], [0, 2000], color='black')
    fit_7_test.set_aspect('equal', 'box')
    fit_7_test.set_xlim(0, 2000)
    fit_7_test.set_ylim(0, 2000)
    fit_7_test.set_xlabel('[Cl] (mg/l) observed')
    fit_7_test.set_ylabel('[Cl] (mg/L) predicted')
    fit_7_test.annotate('(f) t+7', (100, 1800))
    fit_7_test.set_xticks(ticks=[0, 500, 1000, 1500, 2000])
    fit_7_test.set_yticks(ticks=[0, 500, 1000, 1500, 2000])

    forecast_fig_test.suptitle(
        '''Forecast of mean [Cl] at Krimpen aan de IJssel,
        depth=-4.00m a.m.s.l. \n (test)''')
    forecast_fig_test.tight_layout()

if SAVEFIGS is True:
    forecast_fig_test.savefig(FIGPATH+'Forecast_2018_test.png')
    forecast_fig_test.savefig(FIGPATH+'Forecast_2018_test.pdf')


# %% Compute metrics (test)
if TEST is True:
    RMSE_test = np.empty((N_FUTURE, NUMMODELS))
    RMSE_test.fill(np.nan)
    Accuracy_test = np.empty((N_FUTURE, NUMMODELS))
    Accuracy_test.fill(np.nan)
    Precision_test = np.empty((N_FUTURE, NUMMODELS))
    Precision_test.fill(np.nan)
    Recall_test = np.empty((N_FUTURE, NUMMODELS))
    Recall_test.fill(np.nan)

    RMSE_persistence_test = np.empty(N_FUTURE)
    RMSE_persistence_test.fill(np.nan)
    Accuracy_persistence_test = np.empty(N_FUTURE)
    Accuracy_persistence_test.fill(np.nan)
    Precision_persistence_test = np.empty(N_FUTURE)
    Precision_persistence_test.fill(np.nan)
    Recall_persistence_test = np.empty(N_FUTURE)
    Recall_persistence_test.fill(np.nan)

    RMSE_persistence_test[0] = math.sqrt(np.square(np.subtract(
        test.ClKr400Mean[N_PAST:-N_FUTURE],
        persistence_model_test[0, :])).mean())

    event = test.ClKr400Mean[N_PAST:-N_FUTURE] > THRESHOLD
    warning_persistence = persistence_model_test[0, :] > THRESHOLD
    Accuracy_persistence_test[0] = (
        sum(np.logical_and(event, warning_persistence))
        + sum(np.logical_and(np.logical_not(event),
                             np.logical_not(warning_persistence)))
        )/(forecast_test.shape[2])
    Precision_persistence_test[0] = sum(np.logical_and(
        event, warning_persistence))/sum(warning_persistence)
    Recall_persistence_test[0] = sum(np.logical_and(
        event, warning_persistence))/sum(event)

    for m in range(NUMMODELS):
        RMSE_test[0, m] = math.sqrt(np.square(np.subtract(
                test.ClKr400Mean[N_PAST:-N_FUTURE],
                forecast_real_test[m, 0, :, 1])).mean())
        warning_LSTM = forecast_real_test[m, 0, :, 1] > THRESHOLD
        Accuracy_test[0, m] = (
            sum(np.logical_and(event, warning_LSTM))
            + sum(np.logical_and(np.logical_not(event),
                                 np.logical_not(warning_LSTM)))
            )/(forecast.shape[2])
        Precision_test[0, m] = sum(np.logical_and(
            event, warning_LSTM))/sum(warning_LSTM)
        Recall_test[0, m] = sum(np.logical_and(event, warning_LSTM))/sum(event)

    for j in range(1, N_FUTURE):
        event = test.ClKr400Mean[N_PAST+j:-N_FUTURE+j] > THRESHOLD
        warning_persistence = persistence_model_test[j, :] > THRESHOLD
        RMSE_persistence_test[j] = math.sqrt(np.square(np.subtract(
            test.ClKr400Mean[N_PAST+j:-N_FUTURE+j],
            persistence_model_test[j, :])).mean())
        Accuracy_persistence_test[j] = (
            sum(np.logical_and(event, warning_persistence))
            + sum(np.logical_and(np.logical_not(event),
                                 np.logical_not(warning_persistence)))
            )/(forecast.shape[2])
        Precision_persistence_test[j] = sum(np.logical_and(
            event, warning_persistence))/sum(warning_persistence)
        Recall_persistence_test[j] = sum(np.logical_and(
            event, warning_persistence))/sum(event)

        for m in range(NUMMODELS):
            RMSE_test[j, m] = math.sqrt(np.square(np.subtract(
                test.ClKr400Mean[N_PAST+j:-N_FUTURE+j],
                forecast_real_test[m, j, :, 1])).mean())
            warning_LSTM = forecast_real_test[m, j, :, 1] > THRESHOLD
            Accuracy_test[j, m] = (
                sum(np.logical_and(event, warning_LSTM))
                + sum(np.logical_and(np.logical_not(event),
                                     np.logical_not(warning_LSTM)))
                )/(forecast.shape[2])
            Precision_test[j, m] = sum(np.logical_and(
                event, warning_LSTM))/sum(warning_LSTM)
            Recall_test[j, m] = sum(np.logical_and(
                event, warning_LSTM))/sum(event)

# %% Plot metrics (test)

if TEST is True:
    metrics_fig_test, metrics_axs_test = plt.subplots(
        1, 3, figsize=(10, 3), sharex=False, sharey=False)
    metrics_fig_test.tight_layout(pad=2)
    RMSE_plot_test = metrics_axs_test[0]
    Precision_plot_test = metrics_axs_test[1]
    Recall_plot_test = metrics_axs_test[2]

    for m in range(NUMMODELS):
        RMSE_plot_test.plot(
            range(1, 8), RMSE_test[:, m], color='red', lw=0.5, label='LSTM')
    RMSE_plot_test.plot(range(1, 8), RMSE_persistence_test,
                        color='cyan', lw=1.5, label='Persistence')
    RMSE_plot_test.set_ylabel('RMSE [mg/l]')
    RMSE_plot_test.set_xticks(ticks=range(1, 8))
    RMSE_plot_test.annotate('(a)', (1, 170))

    for m in range(NUMMODELS):
        Precision_plot_test.plot(range(1, 8), Precision_test[:, m],
                                 color='red', lw=0.5, label='LSTM')
    Precision_plot_test.plot(range(1, 8), Precision_persistence_test,
                             color='cyan', lw=1.5, label='Persistence')
    Precision_plot_test.set_xlabel('lead time')
    Precision_plot_test.set_ylabel('Precision')
    Precision_plot_test.set_xticks(ticks=range(1, 8))
    Precision_plot_test.set_ylim(0, 1)
    Precision_plot_test.annotate('(b)', (1, 0.9))

    for m in range(NUMMODELS):
        Recall_plot_test.plot(range(1, 8), Recall_test[:, m],
                              color='red', lw=0.5, label='LSTM')
    Recall_plot_test.plot(range(1, 8), Recall_persistence_test, color='cyan',
                          lw=1.5, label='Reference (persistence)')
    Recall_plot_test.set_ylabel('Recall')
    Recall_plot_test.set_xticks(ticks=range(1, 8))
    Recall_plot_test.set_ylim(0, 1)
    Recall_plot_test.annotate('(c)', (1, 0.9))
    plt.legend(Recall_plot_test.get_legend_handles_labels()[0][-2:],
               Recall_plot_test.get_legend_handles_labels()[1][-2:],
               loc='lower left', frameon=False)

if SAVEFIGS is True:
    metrics_fig_test.savefig(FIGPATH+'metrics_test.png')
    metrics_fig_test.savefig(FIGPATH+'metrics_test.pdf')

# %% Perform sensitivity analysis

# TODO: build function ########################################################

if TEST is True:
    meandiff = np.empty((NUMMODELS, len(vars_)))
    meandiff.fill(np.nan)

    # To test the models' sensitivity to changes in each of the input
    # variables, increase this variable by 0.2. Increasing one variable
    # independently of the others is not physically realistic, but it does show
    # how a model uses its inputs to make predictions. Then rerun the model and
    # compare how much the variable of interest changes with respect to the
    # original forecast. Here the variable of interest is Clkr400Mean, or
    # the variable with column no. 1.

    for v in range(len(vars_)):
        Perturbation = test_scaled.copy()
        Perturbation[vars_[v]] += 0.2
        forecast_new, forecast_new_real = ensemble_forecast(
            models, Perturbation, N_FUTURE, vars_[0:12], vars_[12:23])
        Perturbed = forecast_new[:, 0, :, 1]
        for n in range(NUMMODELS):
            meandiff[n, v] = np.mean(np.subtract(
                Perturbed[n, :], forecast_test[n, 0, :, 1]))

    meandiff2 = pd.DataFrame(meandiff,
                             columns=[r'$Cl_{Krimpen.a.d.IJ,min,-4.00m}$',
                                      r'$Cl_{Krimpen.a.d.IJ,mean,-4.00m}$',
                                      r'$Cl_{Krimpen.a.d.IJ,max,-4.00m}$',
                                      r'$Cl_{Krimpen.a.d.IJ,min,-5.50m}$',
                                      r'$Cl_{Krimpen.a.d.IJ,mean,-5.50m}$',
                                      r'$Cl_{Krimpen.a.d.IJ,max,-5.50m}$',
                                      r'$Cl_{Lekhaven,min,-2.50m}$',
                                      r'$Cl_{Lekhaven,mean,-2.50m}$',
                                      r'$Cl_{Lekhaven,max,-2.50m}$',
                                      r'$Cl_{lekhaven,min,-7.00m}$',
                                      r'$Cl_{Lekhaven,mean,-7.00m}$',
                                      r'$Cl_{Lekhaven,max,-7.00m}$',
                                      r'$H_{Dordrecht,mean}$',
                                      r'$H_{Hoek.v.H.,mean}$',
                                      r'$H_{Krimpen.a.d.IJ,min}$',
                                      r'$H_{Krimpen.a.d.IJ,mean}$',
                                      r'$H_{Krimpen.a.d.IJ,max}$',
                                      r'$H_{Vlaardingen,mean}$',
                                      r'$Q_{Hagestein,mean}$',
                                      r'$Q_{Lobith,mean}$',
                                      r'$Q_{Tiel,mean}$',
                                      r'$Wind_{Rotterdam,EW,mean}$',
                                      r'$Wind_{Rotterdam,NS,mean}$'])

    sens_fig = plt.figure()
    sensitivity = meandiff2[meandiff2.columns[::-1]].boxplot(vert=False)

    sensitivity.set_xlim(-0.08, 0.08)
    sensitivity.set_xlabel(
        r"""Mean normalized change in $Cl_{Krimpen.a.d.IJ,min,4.00}$
        when input variable is changed by 0.2 (normalized)""")
    sensitivity.axvline(0)

    if SAVEFIGS is True:
        sens_fig.savefig(FIGPATH+'sensitivity.png')
        sens_fig.savefig(FIGPATH+'sensitivity.pdf', bbox_inches='tight')
