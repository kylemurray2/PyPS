#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:45:20 2018

@author: kdm95
"""

import numpy as np
import isceobj
#import os
#from datetime import date
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
from skimage.measure import profile_line as pl
import fitSine

#********************************
# Set these paramaters
#********************************
seasonal = False
mcov_flag = True
water_elevation = -103
#********************************

params = np.load('params.npy').item()
locals().update(params)
geom = np.load('geom.npy').item()

skip=3
# Run refDef.py before this to make the alld_flat_topo.npy file
alld_flat_topo=np.load('alld_flat_topo.npy')
pairs = np.load('pairs_cut.npy')

# Need to make the crop bounds stored in params.npy or geom.npy**************
ymin=406
ymax=3415
xmin=2
xmax=3966

nxl = xmax-xmin
nyl = ymax-ymin

# Get the geom files
hgt = geom['hgt_ifg'][ymin:ymax,xmin:xmax]
la = geom['lat_ifg'][ymin:ymax,xmin:xmax]
lo = geom['lon_ifg'][ymin:ymax,xmin:xmax]
gam = np.load('gam.npy')[ymin:ymax,xmin:xmax]

dn0 = dn - dn[0]
d1=0
period = 365.25

if seasonals:
    # Invert for seasonal plus long term rates
    phases,amplitudes,biases,slopes = fitSine.fitSine(dn0,alld_flat_topo,period)
    rates = np.reshape(slopes,(nyl,nxl)).astype(np.float32)*lam/(4*np.pi)*100*365
    r_nomsk = rates
    amps = np.reshape(amplitudes,(nyl,nxl)).astype(np.float32)*lam/(4*np.pi)*100
    a_nomsk = amps
    plt.figure();plt.imshow(np.flipud(r_nomsk),vmin=-2,vmax=2)
    plt.figure();plt.imshow(np.flipud(a_nomsk),vmin=0,vmax=2)
else:
    G = np.vstack([dn0, np.ones((len(dn0),1)).flatten()]).T
    Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
    mod   = np.dot(Gg, alld_flat_topo)
    rates = np.reshape(mod[0,:],(nyl,nxl))*lam/(4*np.pi)*100*365 # cm/yr
    #offs  = np.reshape(mod[1,:],(nyl, nxl))
    synth  = np.dot(G,mod);
    res    = (alld_flat_topo-synth)*lam/(4*np.pi)*100 # cm
    resstd = np.std(res,axis=0)
    resstd = np.reshape(resstd,(nyl, nxl))
    
    if mcov_flag:
        rate_uncertainty = []
        for d in alld:
            co=cov(d);
            mcov=np.diag(np.dot(Gg,np.dot(co,Gg.T)));
            rate_uncertainty.append(1.96*mcov[1]**.5)
        rate_uncertainty=np.reshape(rate_uncertainty,(nyl,nxl)) *lam/(4*pi)*100*365 #cm/yr
        plt.figure()
        ax,fig = plt.subplots(1,2,figsize=(4,5))
        ax[0].imshow(rates)
        ax[1].imshow(rate_uncertainty)




# Mask the rates matrix
gam[np.isnan(gam)]=0
rates[np.where(gam<.2)]=np.nan #masks the low coherence areas
rates[np.where( (hgt<water_elevation) ) ]=np.nan # masks the water
#rates=np.fliplr(rates)
#rates[np.isnan(rates)]=0

# Save rates
fname = tsdir + '/rates_flat.unw'
out = isceobj.createIntImage() # Copy the interferogram image from before
out.dataType = 'FLOAT'
out.filename = fname
out.width = nxl
out.length = nyl
out.dump(fname + '.xml') # Write out xml
rates.tofile(out.filename) # Write file out
out.renderHdr()
out.renderVRT()

# GEOCODE
#cmd = 'geocodeIsce.py -f ' + tsdir + 'rates_flat.unw -d ' + workdir + 'DEM/demLat_N33_N35_Lon_W119_W116.dem -m ' + workdir + 'master/ -s ' + workdir + 'pawns/20150514 -r ' + str(rlks) + ' -a ' + str(alks) + ' -b "'" 33 35 -118 -116"'" '
#os.system(cmd)

for l in np.arange(0,nyl):
    ll = lo[l,:]
    if not np.isnan(ll.max()):
        break

for p in np.arange(l+1,nyl):
    ll = lo[p,:]
    if np.isnan(ll.max()):
        break
l+=1
ul = (lo[l,0],la[l,0])
ur = (lo[l,-1],la[l,-1])
ll = (lo[p-1,0],la[p-1,0])
lr = (lo[p-1,-1],la[p-1,-1])

lons = np.array([ul[0],ur[0],ur[0],lr[0],lr[0],ll[0],ll[0],ul[0]])
lats = np.array([ul[1],ur[1],ur[1],lr[1],lr[1],ll[1],ll[1],ul[1]])
pad=0
minlat=lats.min()
maxlat=lats.max()
minlon=lons.min()
maxlon=lons.max()


# Plot rate map
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
#
# Plot rate std
plt.rc('font',size=12)
plt.figure(figsize=(12,12))
m = Basemap(epsg=3395, llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
        llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='l')
m.drawparallels(np.arange(np.floor(minlat-pad), np.ceil(maxlat+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
m.drawmeridians(np.arange(np.floor(minlon-pad), np.ceil(maxlon+pad),1), linewidth=0,labels=[1,0,0,1])
m.arcgisimage(service='World_Shaded_Relief',xpixels=1500)
cf = m.pcolormesh(lo,la,resstd,shading='flat',latlon=True, cmap=plt.cm.Spectral_r,zorder=8,vmin=0,vmax=8)
m.readshapefile('/data/kdm95/qfaults/qfaults_la', 'qfaults_la',zorder=30)
#m.plot(lo_p1,la_p1,color='red',zorder=40,latlon=True)
cbar = m.colorbar(cf,location='bottom',pad="10%")
cbar.set_label('cm')
plt.show()
