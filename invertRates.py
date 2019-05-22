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
#seasonal = False
#mcov_flag = True
#water_elevation = -103
#********************************

def invertRates(data,params, seasonals=False,mcov_flag=False,water_elevation=-103):
    ''' data is a flattened stack of inverted displacements shape=[nd,(nx*ny)]
        you can invert for a sinusoidal fit at each pixel with seasonals = True
        mcov_flag is the model covariance and works only with seasonals=False for now.
        Water elevation is usually not zero (relative to wgs84 ellipsoid.'''

    nyl = params['ymax'] - params['ymin']
    nxl = params['xmax'] - params['xmin']

    lam = params['lam']
    
    dn0 = params['dn'] - params['dn'][0]
    d1=0
    period = 365.25
    rate_uncertainty = []

    if seasonals:
        if mcov_flag:
            mcov_flag=False
            print('model covariance only works with seasonals=False for now')
            
        # Invert for seasonal plus long term rates
        phases,amplitudes,biases,slopes = fitSine.fitSine(dn0,data,period)
        rates = np.reshape(slopes,(nyl,nxl)).astype(np.float32)*lam/(4*np.pi)*100*365
        r_nomsk = rates
        amps = np.reshape(amplitudes,(nyl,nxl)).astype(np.float32)*lam/(4*np.pi)*100
        a_nomsk = amps
#        plt.figure();plt.imshow(np.flipud(r_nomsk),vmin=-2,vmax=2)
#        plt.figure();plt.imshow(np.flipud(a_nomsk),vmin=0,vmax=2)
        return rates, amps
    else:
        G = np.vstack([dn0, np.ones((len(dn0),1)).flatten()]).T
        Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
        mod   = np.dot(Gg, data)
        rates = np.reshape(mod[0,:],(nyl,nxl))*lam/(4*np.pi)*100*365 # cm/yr
        #offs  = np.reshape(mod[1,:],(nyl, nxl))
        synth  = np.dot(G,mod);
        res    = (data-synth)*lam/(4*np.pi)*100 # cm
        resstd = np.std(res,axis=0)
        resstd = np.reshape(resstd,(nyl, nxl))
        
        if mcov_flag:
            for ii in np.arange(0,len(data[0,:])):
                co=np.cov(data[:,ii]);
                mcov=np.diag(np.dot(Gg,np.dot(co,Gg.T)));
                rate_uncertainty.append(1.96*mcov[0]**.5)
            rate_uncertainty = np.asarray(rate_uncertainty,dtype=np.float32)
            rate_uncertainty = np.reshape(rate_uncertainty,(nyl,nxl))
            rate_uncertainty= rate_uncertainty*lam/(4*np.pi)*100*365 #cm/yr

        return rates,resstd

#
#
## Mask the rates matrix
#gam[np.isnan(gam)]=0
#rates[np.where(gam<.2)]=np.nan #masks the low coherence areas
#rates[np.where( (hgt<water_elevation) ) ]=np.nan # masks the water
##rates=np.fliplr(rates)
##rates[np.isnan(rates)]=0
#
## Save rates
#fname = tsdir + '/rates_flat.unw'
#out = isceobj.createIntImage() # Copy the interferogram image from before
#out.dataType = 'FLOAT'
#out.filename = fname
#out.width = nxl
#out.length = nyl
#out.dump(fname + '.xml') # Write out xml
#rates.tofile(out.filename) # Write file out
#out.renderHdr()
#out.renderVRT()
#
## GEOCODE
##cmd = 'geocodeIsce.py -f ' + tsdir + 'rates_flat.unw -d ' + workdir + 'DEM/demLat_N33_N35_Lon_W119_W116.dem -m ' + workdir + 'master/ -s ' + workdir + 'pawns/20150514 -r ' + str(rlks) + ' -a ' + str(alks) + ' -b "'" 33 35 -118 -116"'" '
##os.system(cmd)
#
#for l in np.arange(0,nyl):
#    ll = lo[l,:]
#    if not np.isnan(ll.max()):
#        break
#
#for p in np.arange(l+1,nyl):
#    ll = lo[p,:]
#    if np.isnan(ll.max()):
#        break
#l+=1
#ul = (lo[l,0],la[l,0])
#ur = (lo[l,-1],la[l,-1])
#ll = (lo[p-1,0],la[p-1,0])
#lr = (lo[p-1,-1],la[p-1,-1])
#
#lons = np.array([ul[0],ur[0],ur[0],lr[0],lr[0],ll[0],ll[0],ul[0]])
#lats = np.array([ul[1],ur[1],ur[1],lr[1],lr[1],ll[1],ll[1],ul[1]])
#pad=0
#minlat=lats.min()
#maxlat=lats.max()
#minlon=lons.min()
#maxlon=lons.max()
#
#
## Plot rate map
#plt.rc('font',size=12)
#plt.figure(figsize=(12,12))
#m = Basemap(epsg=3395, llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
#        llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='l')
#m.drawparallels(np.arange(np.floor(minlat-pad), np.ceil(maxlat+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
#m.drawmeridians(np.arange(np.floor(minlon-pad), np.ceil(maxlon+pad),1), linewidth=0,labels=[1,0,0,1])
#m.arcgisimage(service='World_Shaded_Relief',xpixels=1500)
#cf = m.pcolormesh(lo,la,rates,shading='flat',latlon=True, cmap=plt.cm.Spectral_r,zorder=8,vmin=-10,vmax=10)
#m.readshapefile('/data/kdm95/qfaults/qfaults_la', 'qfaults_la',zorder=30)
##m.plot(lo_p1,la_p1,color='red',zorder=40,latlon=True)
#cbar = m.colorbar(cf,location='bottom',pad="10%")
#cbar.set_label('cm')
#plt.show()
##plt.savefig('Figs/rate_map.png', transparent=True, dpi=500)
##
## Plot rate std
#plt.rc('font',size=12)
#plt.figure(figsize=(12,12))
#m = Basemap(epsg=3395, llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
#        llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='l')
#m.drawparallels(np.arange(np.floor(minlat-pad), np.ceil(maxlat+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
#m.drawmeridians(np.arange(np.floor(minlon-pad), np.ceil(maxlon+pad),1), linewidth=0,labels=[1,0,0,1])
#m.arcgisimage(service='World_Shaded_Relief',xpixels=1500)
#cf = m.pcolormesh(lo,la,resstd,shading='flat',latlon=True, cmap=plt.cm.Spectral_r,zorder=8,vmin=0,vmax=8)
#m.readshapefile('/data/kdm95/qfaults/qfaults_la', 'qfaults_la',zorder=30)
##m.plot(lo_p1,la_p1,color='red',zorder=40,latlon=True)
#cbar = m.colorbar(cf,location='bottom',pad="10%")
#cbar.set_label('cm')
#plt.show()
