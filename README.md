# salt_intrusion_lstm
Machine learning model for predicting salt concentrations in the Rhine-Meuse delta.

LSTMv1.py creates an ensemble of LSTM models from Rijkswaterstaat and KNMI measurements.
The folder 'Data'  contains processed data, identical to 'Features.csv'  in the raw dataset.
The folder 'Models' contains an ensemble of LSTM models created with the script 'LSTMv1.py'.
The script 'preprocessing.py' was used to convert the raw data to the daily data in 'Features.csv'. (LONG runtime)
