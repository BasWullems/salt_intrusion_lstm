# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 17:12:55 2022

@author: wullems
"""

from ddlpy import ddlpy
import datetime
import matplotlib
import pandas
import os
import numpy

# get all locations
locations = ddlpy.locations()

# create lists of variables
Codes = locations['Grootheid.Code'].unique()

Omschrijvingen = locations['Grootheid.Omschrijving'].unique()
variables=numpy.column_stack((Codes,Omschrijvingen))
parameters = locations['Parameter_Wat_Omschrijving'].unique()

#select locations with waterlevel measurements
waterlevels = locations[(locations['Grootheid.Code']=='WATHTE')]

#select locations with discharge measurements
discharges = locations[(locations['Grootheid.Code']=='Q')]

#select locations with salinity measurements
salinities = locations[(locations['Grootheid.Code']=='SALNTT')]
chloride = locations[(locations['Grootheid.Code']=='CONCTTE') & 
                     (locations['Parameter.Code']=='Cl')]
