#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:20:01 2018
  Make dates and pairs dictionaries

@author: kdm95
"""
import numpy as np
import glob
import pickle
from osgeo import gdal

#<><><><><><><><><><><><>Set these variables><><><><><><><><><><><><><><><
# Define area of interest
#bbox = list([35.8, 36.9, -120.3, -118.7]) #minlat,maxlat,minlon,maxlon
#maxlat = bbox[0]; minlat = bbox[1]; minlon = bbox[2]; maxlon = bbox[3]
workdir = '/data/kdm95/AK/' # working directory (should be where merged is)
alks = int(3) # number of looks in azimuth
rlks = int(7) # number of looks in range
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

lam = 0.056 # 0.056 for c-band
mergeddir=workdir + 'merged/'
intdir = mergeddir + 'interferograms/'
tsdir = workdir + 'TS/'

pairs1=list()
pairs2=list()
pairs = list()
flist = glob.glob(intdir + '2*_2*')
[pairs.append(f[-17:]) for f in flist]
[pairs1.append(f[-17:-9]) for f in flist]
[pairs2.append(f[-8:]) for f in flist]
pairs.sort();pairs1.sort();pairs2.sort()


f_hgt = mergeddir + 'geom_master/hgt.rdr.vrt'
f_lat = mergeddir + 'geom_master/lat.rdr.vrt'
f_lon = mergeddir + 'geom_master/lon.rdr.vrt'
#hgt_ifg = gdal.Open(f_hgt).ReadAsArray()
lon_ifg = gdal.Open(f_lon).ReadAsArray()
#lat_ifg = gdal.Open(f_lat).ReadAsArray()

# Image dimensions before and after downlooking
nd = len(pairs) # number of pairs (number of dates minus one?)
ny,nx = lon_ifg.shape
nxl = int(np.floor(nx/rlks))
nyl = int(np.floor(ny/alks))

# Saving the objects:
with open(tsdir + 'params.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks], f)

del(lon_ifg)