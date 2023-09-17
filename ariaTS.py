#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:03:45 2022

@author: km
"""
import numpy as np
import glob
import os
from datetime import date
import isce.components.isceobj as isceobj
import matplotlib.pyplot as plt
import makeMap
import cartopy.crs as ccrs
from mroipac.looks.Looks import Looks
from scipy.interpolate import griddata 
import cv2
from scipy import signal
import localParams
import util
import netCDF4 as nc
import PyPS2.util
from PyPS2.util import show

fn = '/d/S1-GUNW-A-R-064-tops-20220606_20220513-015117-37215N_35340N-PP-8858-v2_0_4.nc'
ds = nc.Dataset(fn)
print(ds.__dict__)
for var in ds.variables.values():
    print(var)
    
unw = ds['science']['grids']['data']['unwrappedPhase']
show(unw)
