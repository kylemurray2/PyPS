#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 12:37:39 2018

@author: kdm95
"""

import numpy as np
import isceobj
import pickle
import gdal
import os
from mpl_toolkits.basemap import Basemap
from datetime import date
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from scipy import signal
import glob


with open(tsdir + '/params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks = pickle.load(f)   


lon_gac_min=-118
lon_gac_max=-116
lat_gac_min=33
lat_gac_max=35

xdim_gac = 2401
ydim_gac = 2401
gac_lon_vec = np.linspace(lon_gac_min, lon_gac_max, xdim_gac)
gac_lat_vec = np.linspace(lat_gac_min, lat_gac_max, ydim_gac)

gac_lat,gac_lon = np.meshgrid(gac_lat_vec, gac_lon_vec, sparse=False, indexing='ij')


def inpaint_nans(im):
    """
     Function for filling nan values
    """
    ipn_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]) # kernel for inpaint_nans
    nans = np.isnan(im)
    while np.sum(nans)>0:
        im[nans] = 0
        vNeighbors = signal.convolve2d((nans==False),ipn_kernel,mode='same',boundary='symm')
        im2 = signal.convolve2d(im,ipn_kernel,mode='same',boundary='symm')
        im2[vNeighbors>0] = im2[vNeighbors>0]/vNeighbors[vNeighbors>0]
        im2[vNeighbors==0] = np.nan
        im2[(nans==False)] = im[(nans==False)]
        im = im2
        nans = np.isnan(im)
    return im

mergeddir=workdir + 'merged/'
f_lat = mergeddir + 'geom_master/lat_lk.rdr'
f_lon = mergeddir + 'geom_master/lon_lk.rdr'
f_los = mergeddir + 'geom_master/los_lk.rdr'

#CROP limits for the geom files
ymin=146
ymax=2780

Image = isceobj.createImage()
Image.load(f_lon + '.xml')
lon_ifg = Image.memMap()[ymin:ymax,:,0]
lon_ifg = lon_ifg.copy().astype(np.float32)
lon_ifg[lon_ifg==0]=np.nan
Image.finalizeImage()

Image = isceobj.createImage()
Image.load(f_lat + '.xml')
lat_ifg = Image.memMap()[ymin:ymax,:,0]
lat_ifg = lat_ifg.copy().astype(np.float32)
lat_ifg[lat_ifg==0]=np.nan
Image.finalizeImage()

Image = isceobj.createImage()
Image.load(f_los + '.xml')
los_ifg = Image.memMap()[ymin:ymax,:,0]
los_ifg = lon_ifg.copy().astype(np.float32)
los_ifg[lon_ifg==0]=np.nan
Image.finalizeImage()


# Load IFG
pair = pairs[33]
f = intdir + pair + '/fine_lk.unw'
Image = isceobj.createImage()
Image.load(f + '.xml')
phs_ifg = Image.memMap()[ymin:ymax,:,0]
phs_ifg = phs_ifg.copy().astype(np.float32)*lam/(4*np.pi)*100
phs_ifg[phs_ifg==0]=np.nan
Image.finalizeImage()

phs_ifg-=np.nanmean(phs_ifg)

#idxrow,idxcol = np.where((lat_ifg>lat_gac_max) | (lat_ifg <lat_gac_min) | (lon_ifg<lon_gac_min) | (lon_ifg>lon_gac_max-.5))
#lat_ifg[idxrow,idxcol] = np.nan
#lon_ifg[idxrow,idxcol] = np.nan
#lat_ifg = inpaint_nans(lat_ifg)
#lon_ifg = inpaint_nans(lon_ifg)
dates=list()
flist = glob.glob(intdir + '2*_2*')
[dates.append(f[-17:-9]) for f in flist]
dates.append(flist[-1][-8:])

gac_stack = list()

for ii in np.arange(0,nd):
    date1 = dates[ii]
    date2 =dates[ii+1]
    gf1 = workdir + 'GACOS/' + date1 + '.ztd'
    gf2 = workdir + 'GACOS/' + date2 + '.ztd' 
    gac1 = np.fromfile(gf1,dtype=np.float32)
    gac2 = np.fromfile(gf2,dtype=np.float32)
    gac = gac2-gac1
    gac_stack.append(np.reshape(gac,(ydim_gac,xdim_gac)).astype(np.float32))
    
    
plt.imshow(gac)

gac_grid =griddata((gac_lon.flatten(),gac_lat.flatten()),gac.flatten(), (lon_ifg,lat_ifg), method='nearest')
gac_grid = np.flipud(gac_grid)*100
gac_grid[gac_grid==0]=np.nan

gac_grid-=np.nanmean(gac_grid)

phs_c = phs_ifg-gac_grid
phs_c-=np.nanmean(phs_c)

pad=1
plt.figure()
m = Basemap(epsg=3395, llcrnrlat=(lat_ifg.min()-pad), urcrnrlat=(lat_ifg.max()+pad),\
            llcrnrlon=(lon_ifg.min()-pad), urcrnrlon=(lon_ifg.max()+pad), resolution='i')
m.drawstates(linewidth=0.5,zorder=6,color='white')
m.arcgisimage(service='World_Shaded_Relief',xpixels=2000)
cf = m.pcolormesh(lon_ifg,lat_ifg,gac_grid,shading='flat',cmap=plt.cm.Spectral.reversed(),latlon=True, zorder=8)
#cbar = m.colorbar(cf,location='bottom',pad="10%")
#cbar.set_label('cm')
plt.show()

plt.figure()
m = Basemap(epsg=3395, llcrnrlat=(lat_ifg.min()-pad), urcrnrlat=(lat_ifg.max()+pad),\
            llcrnrlon=(lon_ifg.min()-pad), urcrnrlon=(lon_ifg.max()+pad), resolution='i')
m.drawstates(linewidth=0.5,zorder=6,color='white')
m.arcgisimage(service='World_Shaded_Relief',xpixels=2000)
cf = m.pcolormesh(lon_ifg,lat_ifg,phs_ifg,shading='flat',cmap=plt.cm.Spectral.reversed(),latlon=True, zorder=8)
#cbar = m.colorbar(cf,location='bottom',pad="10%")
#cbar.set_label('cm')
plt.show()

plt.figure()
m = Basemap(epsg=3395, llcrnrlat=(lat_ifg.min()-pad), urcrnrlat=(lat_ifg.max()+pad),\
            llcrnrlon=(lon_ifg.min()-pad), urcrnrlon=(lon_ifg.max()+pad), resolution='i')
m.drawstates(linewidth=0.5,zorder=6,color='white')
m.arcgisimage(service='World_Shaded_Relief',xpixels=2000)
cf = m.pcolormesh(lon_ifg,lat_ifg,phs_c,shading='flat',cmap=plt.cm.Spectral.reversed(),latlon=True, zorder=8)
#cbar = m.colorbar(cf,location='bottom',pad="10%")
#cbar.set_label('cm')
plt.show()