# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:05:05 2022

@author: wullems
"""

import pandas as pd
import numpy as np

FeaturesTable = pd.read_csv('C:\\Users\\wullems\\waterdata\\Features.csv',index_col=0)
Weatherdata = pd.read_csv('C:\\Users\\wullems\\waterdata\\etmgeg_344.txt', sep = ',', header=50)
Winddir = np.array(Weatherdata.DDVEC[(Weatherdata.YYYYMMDD >= 20110101) & (Weatherdata.YYYYMMDD < 20210101)].astype(float))
Winddir_deg = 450-Winddir
Winddir_rad = np.radians(Winddir_deg)
Windspeed = np.array(Weatherdata.FHVEC[(Weatherdata.YYYYMMDD >= 20110101) & (Weatherdata.YYYYMMDD < 20210101)].astype(float))
WindEW = -np.cos(Winddir_rad)*Windspeed
WindNS = -np.sin(Winddir_rad)*Windspeed
Date = Weatherdata.YYYYMMDD[(Weatherdata.YYYYMMDD >= 20110101) & (Weatherdata.YYYYMMDD < 20210101)]
Wind = pd.DataFrame(data={'Date':Date, 'WindEW':WindEW, 'WindNS':WindNS}, dtype='float').reset_index()
FeaturesTable = pd.concat([FeaturesTable,Wind.WindEW,Wind.WindNS],axis=1)
FeaturesTable.to_csv('C:\\Users\\wullems\\waterdata\\Features.csv')
