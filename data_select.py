"""
This is a minimal example on how to retrieve data from water info.
Make sure to set up a path to store the resulting dataframe. Also, decomment
the line of code
"""

#%% Import packages and set ouput path

from ddlpy import ddlpy
import datetime
import matplotlib
import pandas
import os
import numpy

OutputPath = "C:\\Users\\wullems\\waterdata\\"


#%% get all locations
locations = ddlpy.locations()

#%%
#select a set of parameters 
# and a set of stations
codes= ['Q', 'WATHTE', 'SALNTT', 'CONCTTE']
units = ['m3/s', 'cm', 'DIMSLS', 'mg/l']
parameters = ['NVT', 'NVT', 'NVT', 'Cl'] 

# code = Q
station1 = ['ARNH', 'BORD', 'DRIB', 'HAGB', 'LOBI', 'MEGE', 'TIEW', 'VG03', 
          'VG02', 'MAASSS', 'KEISVR', 'BRIENOD', 'PUTTHK', 'LOBH', 'PANNDSKP',
          'TIELWL', 'DRIELBVN', 'HAGSBVN', 'PANNDN', 'HARVSZBNN', 'MAASMD',
          'LITH', 'KEIZVR', 'HARVSS', 'HAGSN', 'GOUDVHVN']


# code = WATHTE
  
           
station2 = ['NWGN', 'MLK1', 'MLK2', 'GOUD', 'AMRO', 'AMRB', 'ARNH', 'HAGO',
            'BERN', 'CULB', 'DODE', 'DRIB', 'DRIO', 'HEES', 'HELL', 'HAGB',
            'HOEK', 'KEIZ', 'KRIL', 'KRIY', 'LOBI', 'MAAS', 'MOER', 'ROTT',
            'SCHH', 'SPIJ', 'STEL', 'TENN', 'TIEW', 'VLAA', 'WERK', 'WIJK',
            'ZALT', 'CREV', 'HOEKVHLD', 'STELLDBTN', 'HARVT10', 'ROTTDM',
            'VLAARDGN', 'MAASSS', 'KRIMPADLK', 'SCHOONHVN', 'KRIMPADIJSL',
            'DORDT', 'GOUDBG', 'EEMHVN', 'EURPHAVN', 'GEULHVN', 'HARMSBG',
            'HARTBG', 'HARTHVN', 'PARKSS', 'GOIDSOD', 'HAGSBNDN', 'VURN',
            'WERKDBTN', 'MOERDK', 'RAKND', 'HELLVSS', 'HEESBN', 'KEIZVR',
            'MAASMSMPL', 'AMLAHVN', 'ROZBSSZZDE', 'TENNSHVN', 'MAESLKRZZDE',
            'DODWD', 'ZALTBML', 'TIELWL', 'DRIELBVN', 'DRIELBNDN', 'AMRGBVN',
            'AMRGBNDN', 'CULBBG', 'HAGSBVN', 'ALBSDRTOVR', 'HARVSZBNN',
            'MIDDHNS', 'GEERTDBG', 'HOEKVHLSSDM', 'HARVBG01', 'WAAL8712R',
            'NEDRN9042R', 'WAAL8758L', 'NEDRN8961L', 'NEDRN9003L', 'WAAL8801R',
            'WAAL8819R', 'WAAL8830R', 'WAAL8900R', 'WAAL9051R', 'WAAL9092R',
            'WAAL9170R', 'WAAL9220R', 'WAAL9298L', 'WAAL9399R', 'WAAL9460R',
            'NEDRN8801R', 'NEDRN8870L', 'NEDRN9134R', 'NEDRN9182R', 'LEK9281R',
            'LEK9312R', 'LEK9434R', 'LEK9526R', 'LEK9566R', 'LEK9628R',
            'LEK9669R', 'NEDRN8827L', 'LEK9349R', 'WAAL8894R', 'WAAL8896R',
            'WAAL8894L', 'WAAL8897L', 'WAAL8932L', 'WAAL8932R', 'WAAL8934R',
            'WAAL8934L', 'WAAL8978R', 'WAAL8980R', 'WAAL8978L', 'WAAL8980L',
            'WAAL9018R', 'WAAL9018L', 'WAAL9020L', 'WAAL9045R', 'WAAL9056R',
            'WAAL9090R', 'WAAL9375R', 'WAAL9045L', 'WAAL9090L', 'WAAL9091L',
            'WAAL9375L', 'WAAL9378R', 'WAAL9376L', 'WAAL9257L', 'NEDRN8790R',
            'NEDRN8790L', 'NEDRN8795R', 'NEDRN8795L', 'NEDRN8800R', 'NEDRN8800L',
            'HAASTT', 'HARTKWT', 'SCHEURHVN', 'DINTHVN', 'WIJKBDSDE', 'GORCM',
            'HARVMD', 'WATWGMD02', 'HARVSZBNNKT', 'HARVSZBNZKT', 'SINTADWBVN',
            'SINTADWBNDN', 'WILLDP', 'HEUSDN', 'CAPSVR', 'HARVSZBTN', 'GOUDHNPSS',
            'GOUDA', 'SLIEDT', 'ZUIDOD', 'MISSSPHVN', 'BENLHVN', 'BRIELLE',
            'WILLSD', 'NOORDWD', 'AMHV']

#code = 'SALNTT'

station3 = ['HVH25', 'HVH45', 'HVH90', 'STBUb', 'STBUo', 'HOEKVHLD', 'MAASSS',
            'BRIENOD','HARVSS', 'PUTTHK', 'BOVSS', 'GOUDVHVN', 'BEERKNMDN',
            'SCHEELHK', 'HOEKVHLBSD', 'ROCKJBSD', 'VOLKRK02',
            'PETLUHVN7', 'NOORDDM', 'SCHENKDK', 'HARDVD', 'HEINOD', 'HOOGVT',
            'ZUIDHVN', 'BRIELSGT', 'POORTGL', 'OUDBELD', 'SUURHBG', 'HELLGT',
            'MOERDBGN', 'GEERTDBG', 'HOEKVHLDMDN', 'HOEKVHLDNOVR', 'HOEKVHLZOVR',
            'tHOEK']

#code = 'CONCTTE'

station4 = ['ZUIDLD', 'SPIJKNSBWTLK', 'KIER1', 'KIER3', 'BEERPLKOVR', 'BRIENOBRTOVR'
            'HARVWT', 'KINDDLKOVR', 'KRIMPADIJSLK', 'LEKHVRTOVR', 'MIDDHNSMB',
            'HOEKVHLRTOVR', 'INLSU', 'STELLDBNN', 'KIER4', 'VOLKRSZSSHLD',
            'HARVDPAGMH', 'HARVDPADZD', 'HARVDPABZD', 'HARVDPDE', 'HARVDPADND',
            'HARVDPDF', 'HARVDPABND', 'HOEKVHLD', 'VLAARDGN', 'MAASSS',
            'KRIMPADIJSL', 'HARMSBG', 'HELLVSS', 'KEIZVR', 'STADAHHRVTDP',
            'BRIENOD', 'HARVSS', 'BOVSS', 'GOUDVHVN', 'BEERKNMDN', 'SCHEELHK',
            'GOUDRK', 'HARVBG', 'VOLKRK02', 'PETLUHVN7', 'AMRKHVZD', 'SCHENKDK',
            'HARDVD', 'HEINOD', 'HOOGVT', 'KINDDADLK', 'KINDDK', 'ZUIDHVN',
            'BRIELSGT', 'PAPDT', 'PAULZD', 'PERNS', 'POORTGL', 'RIETPT',
            'SLIKKVR', 'OUDBELD', 'SUURHBG', 'HELLGT', 'STELLDM', 'MIDDHNS',
            'MOERDBGN', 'DORDDWTILT', 'KLUNDT', 'PETLUHVN4', 'GEERTDBG',
            'HOEKVHLDMDN', 'HOEKVHLDNOVR', 'HOEKVHLZOVR', 'BER30', 'BRB25', 'BRB65',
            'HVH25', 'HVH45', 'HVH90', 'KDD50', 'KR1b', 'KR1m', 'KR1o', 'KR3m',
            'KR2m', 'KR2b', 'KR3b', 'KR3o', 'KR2o', 'KRY40', 'LEK25', 'LEK50',
            'MH020', 'LEK70', 'MH080', 'MH150', 'SPI90', 'SPI45', 'SPI25',
            'SPU10', 'SPU50', 'STB02', 'STB11', 'STB06', 'STBUb', 'STBUo',
            'KRY55', 'SUUHBNZDE', 'KEIZVR', 'ZUIDLD', 'NOORDVKBINV3', 'ALBSDRTOVR',
            'SPIJKNSBWTLK', 'WIELDRTOVR', 'BEERPLKOVR', 'BRIENOBRTOVR',
            'KINDDLKOVR', 'KRIMPADIJSLK', 'LEKHVRTOVR', 'MIDDHNSMB', 'HOEKVHLRTOVR',
            'INLSU', 'VOLKRSZSSHLD', 'HOEKVHLBSD', 'ROCKJBSD', 'POORTHVRTOVR',
            'BEERPLWL', 'HARTKRG', 'OUDMMBOM1', 'NIEUWWTWNW2', 'KR4b', 'KR4m',
            'KR4o', 'KR4b', 'HVL08', 'HVL13', 'HVL02', 'tHOEK', 'VOLKRK02', 'HELLGT']

stations = [station1, station2, station3, station4]

#%% set up points overview
X = []
Y = []
system = []
name = []
var = []


#%% Download parameters
for i in range(1,len(codes)):
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
        startYear = 2011
        endYear = 2020
        
        start_date = datetime.datetime(startYear, 1, 1) # also inputs for the code
        end_date = datetime.datetime(endYear, 12, 31)
        
        print('Searching %s at %s from %s to %s'%(code, STATION, start_date, end_date))
        
        measurements = ddlpy.measurements(location, start_date=start_date, end_date=end_date)
        if (len(measurements) > 0):
            print('Data was found in Waterbase')
            # Save file
            measurements.to_csv("%s%s_%s%s_%s_%s.csv"%(OutputPath, code, location.Code,  startYear, endYear, j), index= False)
            # Save coordinates
            X.append(location.X)
            Y.append(location.Y)
            system.append(location.Coordinatenstelsel)
            name.append(location.Naam)
            var.append(location['Grootheid.Code'])
        else:
            print('No Data!')

    
#%% Save coordinates

coordinates = {'X':X,'Y':Y,'system':system,'name':name,'var':var}
Coordinates = pandas.DataFrame(coordinates)
Coordinates.to_csv("%scoordinates.csv"%(OutputPath))
    
    
    
    
