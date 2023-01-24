# -*- coding: utf-8 -*-
"""
Preprocess Rijkswaterstaat data for use in a machine learning model.

Created on Mon Jan 23 10:23:06 2023

@author: wullems
"""

# %% Import packages

import pandas as pd
import numpy as np
import datetime

# %% Import raw data and remove metadata


def import_raw_data(filename, var, depth):
    """
    Import raw data and remove metadata.

    Parameters
    ----------
    filename : str
        Path of the raw data file.
    var : str
        Variable to be processed: 'Cl' for chloride, 'Q' for discharge,
        'H' for water level.
    depth : int
        Measuremnt depth in cm relative to NAP if var == 'Cl'.
        enter 0 if var == 'Q' or 'H'.

    Returns
    -------
    newtable : DataFrame
        table of cleaned data, metadata removed.

    """
    table = pd.read_csv(filename)
    table.columns = table.columns.str.replace(".", "_")

    if var == 'Cl':
        newtable = pd.DataFrame({
            'Time': pd.to_datetime(table.Tijdstip[
                (table.WaarnemingMetadata_BemonsteringshoogteLijst == depth) &
                (table.Parameter_Omschrijving == "chloride")]),
            'Value': table.Meetwaarde_Waarde_Numeriek[
                (table.WaarnemingMetadata_BemonsteringshoogteLijst == depth) &
                (table.Parameter_Omschrijving == "chloride")]})
        newtable = newtable.drop(newtable[newtable.Value > 10000].index)

    if var == 'Q':
        newtable = pd.DataFrame({
            'Time': pd.to_datetime(table.Tijdstip[:]),
            'Value': table.Meetwaarde_Waarde_Numeriek[:]})
        newtable = newtable.drop(newtable[newtable.Value > 100000].index)

    if var == 'H':
        newtable = pd.DataFrame({
            'Time': pd.to_datetime(table.Tijdstip[:]),
            'Value': table.Meetwaarde_Waarde_Numeriek[:]})
        newtable = newtable.drop(newtable[newtable.Value > 10000].index)
        newtable = newtable[0:-1]

    newtable = newtable.drop_duplicates().sort_values(by=['Time'])
    newtable = newtable.reset_index(drop=True)
    return newtable


ClKr400 = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\CONCTTE_KRIMPADIJSLK2011_2020_8.csv',
    'Cl', -400)

ClKr550 = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\CONCTTE_KRIMPADIJSLK2011_2020_8.csv',
    'Cl', -550)

ClLkh250 = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\CONCTTE_LEKHVRTOVR2011_2020_9.csv',
    'Cl', -250)

ClLkh500 = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\CONCTTE_LEKHVRTOVR2011_2020_9.csv',
    'Cl', -500)

ClLkh700 = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\CONCTTE_LEKHVRTOVR2011_2020_9.csv',
    'Cl', -700)

HKrimpen = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\WATHTE_KRIMPADIJSL2011_2020_42.csv',
    'H', 0)

HHvh = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\WATHTE_HOEKVHLD2011_2020_34.csv',
    'H', 0)

HDordrecht = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\WATHTE_DORDT2011_2020_43.csv',
    'H', 0)

HVlaardingen = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\WATHTE_VLAARDGN2011_2020_38.csv',
    'H', 0)

QHagestein = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\Q_HAGSBVN2011_2020_18.csv',
    'Q', 0)

QTiel = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\Q_TIELWL2011_2020_16.csv',
    'Q', 0)

QLobith = import_raw_data(
    'C:\\Users\\wullems\\waterdata\\Data\\Q_LOBH2011_2020_14.csv',
    'Q', 0)

# %% Calculate mean daily statistics

timesteps = []
starttime = datetime.date(2011, 1, 1)
endtime = datetime.date(2020, 12, 31)

for n in range((endtime-starttime).days+1):
    timesteps.append(starttime + datetime.timedelta(n))


def dailystats(table, timesteps):
    """
    Calculate daily minimum, mean and maximum of a variable.

    Parameters
    ----------
    table : DataFrame
        10-minute data of a single variable at a single location.
    timesteps : list of datetime
        days for wich daily statistics are calculated.

    Returns
    -------
    dailytable: Dataframe
        daily min, mean and max of the input variable in 'table'.

    """
    mins = np.empty(len(timesteps))
    means = np.empty(len(timesteps))
    maxes = np.empty(len(timesteps))
    mins.fill(np.nan)
    means.fill(np.nan)
    maxes.fill(np.nan)

    for n in range(len(timesteps)):
        today = table.Value[table.Time.dt.date == timesteps[n]]
        if len(today) != 0:
            mins[n] = min(today)
            means[n] = np.mean(today)
            maxes[n] = max(today)

    dailytable = pd.DataFrame({
        'Time': timesteps, 'Min': mins, 'Mean': means, 'Max': maxes})
    return dailytable


ClKr400Daily = dailystats(ClKr400, timesteps)
ClKr550Daily = dailystats(ClKr550, timesteps)
ClLkh250Daily = dailystats(ClLkh250, timesteps)
ClLkh500Daily = dailystats(ClLkh500, timesteps)
ClLkh700Daily = dailystats(ClLkh700, timesteps)
HDordrechtDaily = dailystats(HDordrecht, timesteps)
HHvhDaily = dailystats(HHvh, timesteps)
HKrimpenDaily = dailystats(HKrimpen, timesteps)
HVlaardingenDaily = dailystats(HVlaardingen, timesteps)
QHagesteinDaily = dailystats(QHagestein, timesteps)
QLobithDaily = dailystats(QLobith, timesteps)
QTielDaily = dailystats(QTiel, timesteps)

# %% Add wind data
Weatherdata = pd.read_csv(
    'C:\\Users\\wullems\\waterdata\\Data\\etmgeg_344.txt', sep=',', header=50)
Winddir = np.array(Weatherdata.DDVEC[
    (Weatherdata.YYYYMMDD >= 20110101) & (Weatherdata.YYYYMMDD < 20210101)
    ].astype(float))
Winddir_deg = 450-Winddir
Winddir_rad = np.radians(Winddir_deg)
Windspeed = np.array(Weatherdata.FHVEC[
    (Weatherdata.YYYYMMDD >= 20110101) & (Weatherdata.YYYYMMDD < 20210101)
    ].astype(float))
WindEW = -np.cos(Winddir_rad)*Windspeed
WindNS = -np.sin(Winddir_rad)*Windspeed
Date = Weatherdata.YYYYMMDD[
    (Weatherdata.YYYYMMDD >= 20110101) & (Weatherdata.YYYYMMDD < 20210101)]
Wind = pd.DataFrame(data={'Date': Date, 'WindEW': WindEW, 'WindNS': WindNS},
                    dtype='float').reset_index()

# %% Merge data into a single table
Features = pd.DataFrame({'Time': timesteps,
                         'ClKr400Min': ClKr400Daily.Min,
                         'ClKr400Mean': ClKr400Daily.Mean,
                         'ClKr400Max': ClKr400Daily.Max,
                         'ClKr550Min': ClKr400Daily.Min,
                         'ClKr550Mean': ClKr400Daily.Mean,
                         'ClKr550Max': ClKr400Daily.Max,
                         'ClLkh250Min': ClLkh250Daily.Min,
                         'ClLkh250Mean': ClLkh250Daily.Mean,
                         'ClLkh250Max': ClLkh250Daily.Max,
                         'ClLkh500Min': ClLkh500Daily.Min,
                         'ClLkh500Mean': ClLkh500Daily.Mean,
                         'ClLkh500Max': ClLkh500Daily.Max,
                         'ClLkh700Min': ClLkh700Daily.Min,
                         'ClLkh700Mean': ClLkh700Daily.Mean,
                         'ClLkh700Max': ClLkh700Daily.Max,
                         'HDrdMin': HDordrechtDaily.Min,
                         'HDrdMean': HDordrechtDaily.Mean,
                         'HDrdMax': HDordrechtDaily.Max,
                         'HHvhMin': HHvhDaily.Min,
                         'HHvhMean': HHvhDaily.Mean,
                         'HHvhMax': HHvhDaily.Max,
                         'HKrMin': HKrimpenDaily.Min,
                         'HKrMean': HKrimpenDaily.Mean,
                         'HKrMax': HKrimpenDaily.Max,
                         'HVlaMin': HVlaardingenDaily.Min,
                         'HVlaMean': HVlaardingenDaily.Mean,
                         'HVlaMax': HVlaardingenDaily.Max,
                         'QHagMin': QHagesteinDaily.Min,
                         'QHagMean': QHagesteinDaily.Mean,
                         'QHagMax': QHagesteinDaily.Max,
                         'QLobMin': QLobithDaily.Min,
                         'QLobMean': QLobithDaily.Mean,
                         'QLobMax': QLobithDaily.Max,
                         'QTielMin': QTielDaily.Min,
                         'QTielMean': QTielDaily.Mean,
                         'QTielMax': QTielDaily.Max,
                         'WindEW': Wind.WindEW,
                         'WindNS': Wind.WindNS})

# %% Save file
Features.to_csv('C:\\Users\\wullems\\waterdata\\Data\\Features2.csv')
