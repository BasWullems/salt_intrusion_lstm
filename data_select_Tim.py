"""
This is a minimal example on how to retrieve data from water info.
Make sure to set up a path to store the resulting dataframe. Also, decomment
the line of code
"""

from ddlpy import ddlpy
import datetime
import matplotlib
import pandas
import os
import numpy

OutputPath = "C:\\Users\\wullems\\waterdata\\maasmond2021\\"

# get all locations
locations = ddlpy.locations()

#select a set of parameters 
# and a set of stations
codes= ['Q']#, 'WATHTE', 'SALNTT', 'CONCTTE']
units = ['m3/s']#, 'cm', 'DIMSLS', 'mg/l']
parameters = ['NVT']#, 'NVT', 'NVT', 'Cl'] 

# code = Q
station1 = ['VG03', 'VG02', 'MAASSS', 'KEIZVR', 'BREINOD', 'PUTTHK',
            'HARVSZBNN', 'MAASMD', 'HARVSS']


stations = [station1]

for i in range(len(codes)):
    code = codes[i]
    parameter = parameters[i]
    unit = units[i]
    station = stations[i]

# Filter the locations dataframe with the desired parameters and stations.
    selected= locations[locations.index.isin(station)]

    selected = selected[(selected['Grootheid.Code'] == code) 
                         & (selected['Parameter.Code'] == parameter)
                         & (selected['Eenheid.Code'] == unit)].reset_index()
    
    # Obtain measurements per parameter row
    for j in range(len(selected)):
        location= selected.loc[j]
        STATION = location['Code']
        startYear = 2020
        endYear = 2020
        
        start_date = datetime.datetime(startYear, 1, 1) # also inputs for the code
        end_date = datetime.datetime(endYear, 12, 31)
        
        print('Searching %s at %s from %s to %s'%(code, STATION, start_date, end_date))
        
        measurements = ddlpy.measurements(location, start_date=start_date, end_date=end_date)
        if (len(measurements) > 0):
            print('Data was found in Waterbase')
            measurements.to_csv("%s%s_%s%s_%s.csv"%(OutputPath, location.Code, code, startYear, endYear), index= False)
        else:
            print('No Data!')

    
    
    
    
    
