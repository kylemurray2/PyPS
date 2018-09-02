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
import os
import isceobj
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#<><><><><><><><><><><><>Set these variables><><><><><><><><><><><><><><><
# Define area of interest
#bbox = list([35.8, 36.9, -120.3, -118.7]) #minlat,maxlat,minlon,maxlon
#maxlat = bbox[0]; minlat = bbox[1]; minlon = bbox[2]; maxlon = bbox[3]
workdir = '/data/kdm95/Kern/SENT/P144_stack/' # working directory (should be where merged is)
alks = int(4) # number of looks in azimuth
rlks = int(10) # number of looks in range
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

lam = 0.056 # 0.056 for c-band
mergeddir=workdir + 'merged/'
intdir = mergeddir + 'interferograms/'
tsdir = workdir + 'TS/'

if not os.path.isdir(tsdir):
    os.mkdir(tsdir)

pairs1=list()
pairs2=list()
pairs = list()
flist = glob.glob(intdir + '2*_2*')
[pairs.append(f[-17:]) for f in flist]
[pairs1.append(f[-17:-9]) for f in flist]
[pairs2.append(f[-8:]) for f in flist]
pairs.sort();pairs1.sort();pairs2.sort()

# Get width and length
f_lon = mergeddir + 'geom_master/lon.rdr.full.vrt'
lon_info = gdal.Info(f_lon)
src = gdal.Open(f_lon)
ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
nx = int(ulx + (src.RasterXSize * xres))
ny = int(uly + (src.RasterYSize * yres))

nd = len(pairs) # number of pairs (number of dates minus one?)
nxl = int(np.floor(nx/rlks))
nyl = int(np.floor(ny/alks))

# Saving the objects:
with open(tsdir + 'params.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks], f)

def downLook(infile, outfile,alks,rlks):
    inImage = isceobj.createImage()
    inImage.load(infile + '.xml')
    inImage.filename = infile

    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inImage)
    lkObj.setOutputFilename(outfile)
    lkObj.looks()

file_list = list(['lat','lon','hgt','los','incLocal'])
for f in file_list:
    infile = mergeddir + 'geom_master/' + f + '.rdr.full'
    outfile = mergeddir + 'geom_master/' + f + '_lk.rdr'
    downLook(infile, outfile,alks,rlks)
    
    
# Get bounding coordinates (Frame)
f_lon_lk = mergeddir + 'geom_master/lon_lk.rdr'
f_lat_lk = mergeddir + 'geom_master/lat_lk.rdr'

Image = isceobj.createImage()
Image.load(f_lon_lk + '.xml')
lon_ifg = Image.memMap()[:,:,0]
lon_ifg = lon_ifg.copy().astype(np.float32)
lon_ifg[lon_ifg==0]=np.nan
Image.finalizeImage()

Image = isceobj.createImage()
Image.load(f_lat_lk + '.xml')
lat_ifg = Image.memMap()[:,:,0]
lat_ifg = lat_ifg.copy().astype(np.float32)
lat_ifg[lat_ifg==0]=np.nan
Image.finalizeImage()

for l in np.arange(0,nyl):
    ll = lon_ifg[l,:]
    if not np.isnan(ll.max()):
        break

for p in np.arange(l+1,nyl):
    ll = lon_ifg[p,:]
    if np.isnan(ll.max()):
        break
l+=1

ul = (lon_ifg[l,0],lat_ifg[l,0])
ur = (lon_ifg[l,-1],lat_ifg[l,-1])
ll = (lon_ifg[p-1,0],lat_ifg[p-1,0])
lr = (lon_ifg[p-1,-1],lat_ifg[p-1,-1])

(32.6,32.85,32.85,38.7,38.7,38.45,38.45,32.6)

lons = np.array([ul[0],ur[0],ur[0],lr[0],lr[0],ll[0],ll[0],ul[0]])
lats = np.array([ul[1],ur[1],ur[1],lr[1],lr[1],ll[1],ll[1],ul[1]])


pad=2
plt.close()
plt.rc('font',size=14)
fig = plt.figure(figsize=(6,6))
m = Basemap(llcrnrlat=lats.min()-pad,urcrnrlat=lats.max()+pad,\
        llcrnrlon=lons.min()-pad,urcrnrlon=lons.max()+pad,resolution='i',epsg=3395)
m.arcgisimage(service='World_Shaded_Relief',xpixels=1000)
m.drawstates(linewidth=1.5,zorder=1,color='white')
m.drawcountries(linewidth=1.5,zorder=1,color='white')
m.drawparallels(np.arange(np.floor(lats.min()-pad), np.ceil(lats.max()+pad), 2), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
m.drawmeridians(np.arange(np.floor(lons.min()-pad), np.ceil(lons.max()+pad),2), linewidth=0,labels=[1,0,0,1])
m.plot(lons,lats,linewidth=2,latlon=True,color='red',zorder=10)
plt.title('Extent of stack')
plt.show()
#plt.savefig('areamap.png',transparent=True,dpi=300 )







