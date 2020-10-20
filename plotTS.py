#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:29:24 2019

This is a custom script that plots profiles and time series from the rate map 
and deformation stack created with refDef.py and invertRates.py. This should be
copied and modified for your specific area. 

@author: kdm95
"""
import numpy as np
import isceobj
import os
import makeMap
from matplotlib import pyplot as plt
from skimage.measure import profile_line as pl
#import fitSine

params = np.load('params.npy').item()
locals().update(params)
geom = np.load('geom.npy').item()

seasonal = False

# Load the deformation stack
alld_flat_topo=np.load('alld_flat_topo.npy')

# Load the rate map
rates = np.load('')

## Example pixel for check
example_ts = list([[113,279],[111,208]]) # these are all the locations you want to plot a time series

dn0 = dn - dn[0]
d1=0
period = 365.25
#c,r = 3707,808
for ii,point in enumerate(example_ts):
    c=point[0];r=point[1]
    idx = ((r)*nxl)+c #finds the index of flattened array based on row col in image
    if seasonal:
        y=-alld_flat_topo[:,idx]*lam/(4*np.pi)*100
        phase,amplitude,bias,slope = fitSine.fitSine1d(dn0,y,period)
        yEst = amplitude*np.sin(dn0*(1/period)*2*np.pi + phase * (np.pi/180.0)) + slope*dn0 + bias
        plt.figure()
        plt.plot(dec_year,y[d1:],'.')
        plt.plot(dec_year[d1:],yEst[d1:])
        plt.xlabel('Year'); plt.ylabel('Displacement (cm)')
        plt.title(str(np.round((slope*365), decimals=2)) + ' cm/yr in LOS')
        plt.show()
    else:
        
#    plt.savefig(workdir + 'Figs/timeseries' + str(ii) + '.svg',transparent=True,dpi=200)

# Plot rate map
makeMap.makeImg(rates,lon_ifg,lat_ifg,vmin,vmax,pad,'rates (cm)')
plt.rc('font',size=12)
plt.figure(figsize=(12,12))
m = Basemap(epsg=3395, llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
        llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='l')
m.drawparallels(np.arange(np.floor(minlat-pad), np.ceil(maxlat+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
m.drawmeridians(np.arange(np.floor(minlon-pad), np.ceil(maxlon+pad),1), linewidth=0,labels=[1,0,0,1])
m.arcgisimage(service='World_Shaded_Relief',xpixels=1500)
cf = m.pcolormesh(lo,la,rates,shading='flat',latlon=True, cmap=plt.cm.Spectral_r,zorder=8,vmin=-10,vmax=10)
m.readshapefile('/data/kdm95/qfaults/qfaults_la', 'qfaults_la',zorder=30)
#m.plot(lo_p1,la_p1,color='red',zorder=40,latlon=True)
cbar = m.colorbar(cf,location='bottom',pad="10%")
cbar.set_label('cm')
plt.show()
#plt.savefig('Figs/rate_map.png', transparent=True, dpi=500)     
        
# Profiles
# Example profile: pl(image, src, dst, linewidth=1, order=1, mode='constant', cval=0.0)

# PROFILE 1 
x0,y0 = 164,2412
x1,y1 = 364,2604
# Convert to lat lon
lo_p4=(lo[y0,x0],lo[y1,x1])
la_p4=(la[y0,x0],la[y1,x1])
#find total distance
prof_dist = np.sqrt(np.square(lo_p4[0]-lo_p4[1]) + np.square(la_p4[0]-la_p4[1])) *111
fig, ax1 = plt.subplots(figsize=(6,4))
zii = pl(rates,(y0,x0),(y1,x1),linewidth=2)
prof_vec = np.linspace(0,prof_dist,len(zii))
ax1.plot(prof_vec,zii*10,'k.')
#ax1.vlines(8.4,ymin=10*zii.min(),ymax=10*zii.max(),color='red')
#plt.ylim([-1,.5])
ax1.set_xlabel('Profile Distance (km)')
ax1.tick_params(axis='y')
ax1.set_ylabel('LOS Rate (mm/yr)')
# topo profile
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
zii_topo = pl(hgt,(y0,x0),(y1,x1),linewidth=2)
ax2.plot(prof_vec,zii_topo)
#ax2.vlines(8.4,ymin=zii_topo.min(),ymax=zii_topo.max(),color='red')
#plt.ylim([-1,.5])
ax2.set_xlabel('Profile Distance (km)')
ax2.set_ylabel('Elevation (m)',color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
#plt.savefig(workdir + 'Figs/prof4_topo.svg',transparent=True,dpi=100)
#lonpt, latpt = m(16558,67529,inverse=True)
#
#fault_listx = list([161895,129528,122949,115581,113213,51636,40058])
#fault_listy = list([135211,120474,117580,115475,113633,86528,81528])
#dst=list()
#lonpt, latpt = m(fault_listx[0],fault_listy[0],inverse=True)
#
#for ii in np.arange(1,len(fault_listx)):
#    lonpt2, latpt2 = m(fault_listx[ii],fault_listy[ii],inverse=True)
#    dst.append(np.sqrt(np.square(lonpt-lonpt2)+np.square(latpt-latpt2)))