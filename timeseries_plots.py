# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:19:37 2022

@author: wullems
"""
#%% Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import pytz
import statistics

#%% Load and clean data
ClKr = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\CONCTTE_KRIMPADIJSLK2011_2020_8.csv',sep=',')
ClKr.columns = ClKr.columns.str.replace(".", "_")
ClKr.columns = ClKr.columns.str.replace('WaarnemingMetadata_BemonsteringshoogteLijst','Depth')
ClKr = ClKr.drop(ClKr[ClKr.Meetwaarde_Waarde_Numeriek>1000*statistics.median(ClKr.Meetwaarde_Waarde_Numeriek)].index)
ClLkh = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\CONCTTE_LEKHVRTOVR2011_2020_9.csv',sep=',')
ClLkh.columns = ClLkh.columns.str.replace(".", "_")
ClLkh.columns = ClLkh.columns.str.replace('WaarnemingMetadata_BemonsteringshoogteLijst','Depth')
ClLkh = ClLkh.drop(ClLkh[ClLkh.Meetwaarde_Waarde_Numeriek>1000*statistics.median(ClLkh.Meetwaarde_Waarde_Numeriek)].index)
HKr = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\WATHTE_KRIMPADIJSL2011_2020_42.csv',sep=',')
HKr.columns = HKr.columns.str.replace(".", "_")
HKr = HKr.drop(HKr[HKr.Meetwaarde_Waarde_Numeriek>10000].index)
HHvh = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\WATHTE_HOEKVHLD2011_2020_34.csv',sep=',')
HHvh.columns = HHvh.columns.str.replace(".", "_")
HHvh = HHvh.drop(HHvh[HHvh.Meetwaarde_Waarde_Numeriek>10000].index)
HVla = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\WATHTE_VLAARDGN2011_2020_38.csv',sep=',')
HVla.columns = HVla.columns.str.replace(".", "_")
HVla = HVla.drop(HVla[HVla.Meetwaarde_Waarde_Numeriek>10000].index)
HDrd = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\WATHTE_DORDT2011_2020_43.csv',sep=',')
HDrd.columns = HDrd.columns.str.replace(".", "_")
HDrd = HDrd.drop(HDrd[HDrd.Meetwaarde_Waarde_Numeriek>10000].index)
HMaes = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\WATHTE_MAESLKRZZDE2011_2020_64.csv',sep=',')
HMaes.columns = HMaes.columns.str.replace(".", "_")
HMaes = HMaes.drop(HMaes[HMaes.Meetwaarde_Waarde_Numeriek>10000].index)
HMaas = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\WATHTE_MAASSS2011_2020_39.csv',sep=',')
HMaas.columns = HMaas.columns.str.replace(".", "_")
HMaas = HMaas.drop(HMaas[HMaas.Meetwaarde_Waarde_Numeriek>10000].index)
HRdam = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\WATHTE_ROTTDM2011_2020_37.csv',sep=',')
HRdam.columns = HRdam.columns.str.replace(".", "_")
HRdam = HRdam.drop(HRdam[HRdam.Meetwaarde_Waarde_Numeriek>10000].index)
QLob = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\Q_LOBH2011_2020_14.csv',sep=',')
QLob.columns = QLob.columns.str.replace(".", "_")
QLob = QLob.drop(QLob[QLob.Meetwaarde_Waarde_Numeriek>1000*statistics.median(QLob.Meetwaarde_Waarde_Numeriek)].index)
QHag = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\Q_HAGSBVN2011_2020_18.csv',sep=',')
QHag.columns = QHag.columns.str.replace(".", "_")
QHag = QHag.drop(QHag[QHag.Meetwaarde_Waarde_Numeriek>1000*statistics.median(QHag.Meetwaarde_Waarde_Numeriek)].index)
QTiel = pd.read_table('C:\\Users\\wullems\\waterdata\\Data\\Q_TIELWL2011_2020_16.csv',sep=',')
QTiel.columns = QTiel.columns.str.replace(".", "_")
QTiel = QTiel.drop(QTiel[QTiel.Meetwaarde_Waarde_Numeriek>1000*statistics.median(QTiel.Meetwaarde_Waarde_Numeriek)].index)
Weatherdata = pd.read_csv('C:\\Users\\wullems\\waterdata\\Data\\etmgeg_344.txt', sep = ',', header=50)
Weatherdata.columns = Weatherdata.columns.str.strip()

ClKr.insert(0,'Time',pd.to_datetime(ClKr.t,format='%Y-%m-%d %H:%M:%S%z'))
ClLkh.insert(0,'Time',pd.to_datetime(ClLkh.t,format='%Y-%m-%d %H:%M:%S%z'))
HKr.insert(0,'Time',pd.to_datetime(HKr.t,format='%Y-%m-%d %H:%M:%S%z'))
HHvh.insert(0,'Time',pd.to_datetime(HHvh.t,format='%Y-%m-%d %H:%M:%S%z'))
HVla.insert(0,'Time',pd.to_datetime(HVla.t,format='%Y-%m-%d %H:%M:%S%z'))
HDrd.insert(0,'Time',pd.to_datetime(HDrd.t,format='%Y-%m-%d %H:%M:%S%z'))
HMaas.insert(0,'Time',pd.to_datetime(HMaas.t,format='%Y-%m-%d %H:%M:%S%z'))
HMaes.insert(0,'Time',pd.to_datetime(HMaes.t,format='%Y-%m-%d %H:%M:%S%z'))
HRdam.insert(0,'Time',pd.to_datetime(HRdam.t,format='%Y-%m-%d %H:%M:%S%z'))

QLob.insert(0,'Time',pd.to_datetime(QLob.t,format='%Y-%m-%d %H:%M:%S%z'))
QHag.insert(0,'Time',pd.to_datetime(QHag.t,format='%Y-%m-%d %H:%M:%S%z'))
QTiel.insert(0,'Time',pd.to_datetime(QTiel.t,format='%Y-%m-%d %H:%M:%S%z'))
Weatherdata.insert(2,'Time',pd.to_datetime(Weatherdata.YYYYMMDD,format='%Y%m%d'))
Weatherdata = Weatherdata.replace(r'\s*$',np.nan,regex=True)
Weatherdata.FHX=Weatherdata.FHX.astype(float)

FeaturesTable = pd.read_csv('C:\\Users\\wullems\\waterdata\\Data\\Features.csv',index_col=0)
# Interpolate missing values
FeaturesTable = FeaturesTable.interpolate()
# Convert dates to datetime format
dates = pd.to_datetime(FeaturesTable['Time'])
FeaturesTable['Time'] =dates
#%% Plot timeseries
fig,axs=plt.subplots(5,1,figsize=(10,15),sharex=True)
starttime=datetime(2017,1,1,0,0,0,0,pytz.timezone('Europe/Amsterdam'))
endtime=datetime(2018,1,1,0,0,0,0,pytz.timezone('Europe/Amsterdam'))
starttime2=datetime(2017,1,1,0,0,0,0)
endtime2=datetime(2018,1,1,0,0,0,0)
ClKr2017 = ClKr[(ClKr.Time>=starttime) & (ClKr.Time<endtime)]
ClLkh2017 = ClLkh[(ClLkh.Time>=starttime) & (ClLkh.Time<endtime)]
HHvh2017 = HHvh[(HHvh.Time>=starttime) & (HHvh.Time<endtime)]
QLob2017 = QLob[(QLob.Time>=starttime) & (QLob.Time<endtime)].sort_values('Time')
Weather2017 = Weatherdata[(Weatherdata.Time>=starttime2) & (Weatherdata.Time<endtime2)]
Features2017 = FeaturesTable[(FeaturesTable.Time>=starttime2) & (FeaturesTable.Time<endtime2)]
axs[0].plot(Features2017.Time, Features2017.ClKr400Mean, lw=0.7, label='-4.0')
axs[0].plot(Features2017.Time, Features2017.ClKr550Mean, lw=0.7, label='-5.5')
axs[0].set_ylabel('[Cl](mg/l)')
axs[0].set_ylim(0,1500)
axs[0].set_xlim(starttime,endtime)
axs[0].legend(title='Depth (m a.m.s.l.)', frameon=False)
axs[0].set_title('(a) Daily mean chloride concentration at Krimpen aan den IJssel')

axs[1].plot(Features2017.Time, Features2017.ClLkh250Mean,lw=0.7, label='-2.5')
axs[1].plot(Features2017.Time, Features2017.ClLkh500Mean,lw=0.7, label='-5.0')
axs[1].plot(Features2017.Time, Features2017.ClLkh700Mean,lw=0.7, label='-7.0')
axs[1].set_ylabel('[Cl](mg/l)')
axs[1].set_ylim(0,6000)
axs[1].set_xlim(starttime,endtime)
axs[1].set_title('(b) Daily mean chloride concentration at Lekhaven')
axs[1].legend(title='Depth (m a.m.s.l.)',frameon=False)

axs[2].plot(Features2017.Time, Features2017.HHvhMean)
axs[2].set_ylabel('Water level (cm a.m.s.l.)')
axs[2].set_title('(c) Daily mean water level at Hoek van Holland')
axs[2].set_ylim(-50,150)
axs[2].set_xlim(starttime,endtime)

axs[3].plot(QLob2017.Time, QLob2017.Meetwaarde_Waarde_Numeriek)
axs[3].set_title('(d) Discharge at Lobith')
axs[3].set_ylabel('Discharge (m\u00b3/s)')
axs[3].set_ylim(0,6000)
axs[3].set_xlim(starttime,endtime)

axs[4].plot(Weather2017.Time,Weather2017.FHVEC)
axs[4].set_title('(e) Daily mean wind speed at Rotterdam')
axs[4].set_ylabel('Wind speed (m/s)')
axs[4].set_ylim(0,150)
axs[4].set_xlim(starttime,endtime)
axs[4].set_xticks(ticks=[datetime(2017,1,1,0,0,0),datetime(2017,2,1,0,0,0),datetime(2017,3,1,0,0,0),datetime(2017,4,1,0,0,0),datetime(2017,5,1,0,0,0),datetime(2017,6,1,0,0,0),datetime(2017,7,1,0,0,0),datetime(2017,8,1,0,0,0),datetime(2017,9,1,0,0,0),datetime(2017,10,1,0,0,0),datetime(2017,11,1,0,0,0),datetime(2017,12,1,0,0,0)], labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
axs[4].set_xlabel('2017')

#%% Save
fig.savefig('C:\\Users\\wullems\\OneDrive - Stichting Deltares\\Pictures\\Tijdseries\\Timeseries_2017.pdf',bbox_inches='tight')

#%% Calculate correlation between water levels
HDrd=pd.DataFrame({'Time':HDrd.Time,'Value':HDrd.Meetwaarde_Waarde_Numeriek})
HHvh=pd.DataFrame({'Time':HHvh.Time,'Value':HHvh.Meetwaarde_Waarde_Numeriek})
HKr=pd.DataFrame({'Time':HKr.Time,'Value':HKr.Meetwaarde_Waarde_Numeriek})
HMaas=pd.DataFrame({'Time':HMaas.Time,'Value':HMaas.Meetwaarde_Waarde_Numeriek})
HMaes=pd.DataFrame({'Time':HMaes.Time,'Value':HMaes.Meetwaarde_Waarde_Numeriek})
HRdam=pd.DataFrame({'Time':HRdam.Time,'Value':HRdam.Meetwaarde_Waarde_Numeriek})
HVla=pd.DataFrame({'Time':HVla.Time,'Value':HVla.Meetwaarde_Waarde_Numeriek})

MainTimeseries=pd.DataFrame({'Time':pd.date_range(start=datetime(2011,1,1,0,0,0),end=datetime(2021,1,1,0,0,0),freq=timedelta(minutes=10), tz=pytz.FixedOffset(60))})
Waterlevels=pd.merge_asof(MainTimeseries,HHvh.dropna().sort_values(by='Time').drop_duplicates(),on='Time',direction='nearest')
Waterlevels=pd.merge_asof(Waterlevels,HMaes.dropna().sort_values(by='Time').drop_duplicates(),on='Time',direction='nearest')
Waterlevels=pd.merge_asof(Waterlevels,HMaas.dropna().sort_values(by='Time').drop_duplicates(),on='Time',direction='nearest')
Waterlevels=pd.merge_asof(Waterlevels,HVla.dropna().sort_values(by='Time').drop_duplicates(),on='Time',direction='nearest')
Waterlevels=pd.merge_asof(Waterlevels,HRdam.dropna().sort_values(by='Time').drop_duplicates(),on='Time',direction='nearest')
Waterlevels=pd.merge_asof(Waterlevels,HKr.dropna().sort_values(by='Time').drop_duplicates(),on='Time',direction='nearest')
Waterlevels=pd.merge_asof(Waterlevels,HDrd.dropna().sort_values(by='Time').drop_duplicates(),on='Time',direction='nearest')

Waterlevels.columns=['Time','Hoek_van_Holland','Maeslantkering','Maassluis','Vlaardingen','Rotterdam','Krimpen','Dordrecht']
#%%
Corr = Waterlevels.corr()
mask = np.triu(np.ones_like(Corr, dtype=bool))
fig=plt.figure(figsize=(5,5))
sns.heatmap(Corr, cmap='cool_r', annot=True, vmin=0.5,vmax=1,mask=mask)
fig.savefig('C:\\Users\\wullems\\OneDrive - Stichting Deltares\\Pictures\\Analyse\\correlation_plot_waterlevels.pdf', bbox_inches='tight')