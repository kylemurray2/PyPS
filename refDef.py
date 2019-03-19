#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:35:54 2018

@author: kdm95
"""

import numpy as np
import isceobj
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from mpl_toolkits.basemap import Basemap
import invertRates 


params = np.load('params.npy').item()
locals().update(params)
geom = np.load('geom.npy').item()
locals().update(geom)

atm_flag = 0 # set to 1 if you did a gacos correction


nxl = params['nxl']
nyl  = ymax-ymin  

#initial reference point
r,c = 200,200

# Get the ifgs
stack = []
for p in pairs:
    unw_file = intdir + '/' + p + '/fine_lk.unw'
    unwImage = isceobj.createIntImage()
    unwImage.load(unw_file + '.xml')
    stack.append(unwImage.memMap()[:,:,0])
stack = np.asarray(stack,dtype=np.float32)

# SBAS Inversion to get displacement at each date
## Make G matrix for dates inversion
G = np.zeros((nd+1,len(dn)))
for ii,pair in enumerate(pairs):
    a = dates.index(pair[0:8])
    b = dates.index(pair[9:17])
    G[ii,a] = -1
    G[ii,b] = 1
G[-1,0]=1

Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
N = np.dot(G,Gg)
R = np.dot(Gg,G)

# Do dates inversion
alld=np.zeros((len(dn),nxl*nyl))
for ii in np.arange(0,nyl-1): #iterate through rows
    tmp = np.zeros((nd+1,nxl))
    for jj,pair in enumerate(pairs): #loop through each ifg and append to alld 
        tmp[jj,:] = stack[jj,ii,:]
    alld[:,ii*nxl:nxl*ii+nxl] = np.dot(Gg, tmp)
del(tmp)  


plt.figure()
for ii in np.arange(0,len(alld[:,0]),4):
    plt.plot(np.reshape(alld[ii,:],(nyl,nxl))[:,560]) 

# MASKING______________________________
gam = np.load('gam.npy')[ymin:ymax,:]
gamflat = gam.flatten()


#
## Filter each date before calculating the std for referencing.
#def butter_lowpass(cutoff, fs, order=5):
#    nyq = 0.5 * fs
#    normal_cutoff = cutoff / nyq
#    b, a = butter(order, normal_cutoff, btype='low', analog=False)
#    return b, a
#
#def butter_lowpass_filter(data, cutoff, fs, order=5):
#    b, a = butter_lowpass(cutoff, fs, order=order)
#    y = lfilter(b, a, data, axis=1) #Check this axis!
#    return y
#
#order = 8
#fs = 0.066      # sample rate, Hz
#cutoff = 1/180  # desired cutoff frequency of the filter, Hz
#alld_filt = butter_lowpass_filter(alld, cutoff, fs, order) # check the function for the axis in lfilter
#alld_filt[alld_filt==0]==np.nan
#alld[alld==0]==np.nan
#std_img = np.nanstd(alld_filt,axis=0) # Temporal std
#std_img = np.reshape(std_img,(nyl,nxl))
#std_img[std_img==0]=np.nan
#
#plt.figure()
#plt.hist(std_img[~np.isnan(std_img)], 40, edgecolor='black', linewidth=.2)
    

#plt.figure()
#plt.hist( gamma0_lk.flatten()[~np.isnan(gamma0_lk.flatten())], 40, edgecolor='black', linewidth=.2)
#plt.title('Phase stability histogram')
#plt.xlabel('Phase stability (1 is good, 0 is bad)')
#msk = np.where( (std_img.flatten() < std_thresh) & (gamma0_lk.flatten() > gamma0_thresh) & (hgt.flatten() > -100) )
#alld[:,msk] = np.nan


# Remove mean from each image using just the nondeforming pixels
alld_flat=np.empty(alld.shape)
for ii in np.arange(0,len(alld[:,0])):
    a = alld[ii,:]
    alld_flat[ii,:] = alld[ii,:] - np.nanmedian(a[gamflat>.3])
#    alld_flat[ii,np.where(hgt.flatten()==-1)]=np.nan
plt.figure()
for ii in np.arange(0,len(alld[:,0]),5):
    plt.plot(np.reshape(alld_flat[ii,:],(nyl,nxl))[:,560])


# Plot phase-elevation 
#plt.figure()
#plt.plot(hgt.flatten()[msk], alld_flat[30,msk].flatten(),'.',markersize=1)

## Do phase-elevation correction
#x=hgt.flatten()[msk]
##x*=crop_mask.flatten()[msk]
#G = np.vstack([x, np.ones((len(x),1)).flatten()]).T
#Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
#alld_flat_topo = np.empty(alld_flat.shape)
#
#for ii in np.arange(0,len(alld[:,0])):
#    y=alld_flat[ii,msk].flatten()
#    mod    = np.dot(Gg, y)
#    y2 = mod[0]*hgt.flatten() + mod[1]
#    alld_flat_topo[ii,:] = alld_flat[ii,:].flatten()-y2
##    alld_flat[ii,np.where(hgt.flatten()==-1)]=np.nan
#    
## Plot example
#plt.figure()
#plt.plot(x,y,'.',markersize=1)
#plt.plot(hgt.flatten(),y2)
#plt.ylabel('phs')
#plt.xlabel('elevation (m)')
#plt.title('uncorrected phase and best fit')
#plt.savefig(workdir + 'Figs/phs_elev_uncorrected.png',transparent=True,dpi=200)
#
#plt.figure()
#plt.plot(hgt.flatten(),alld_flat_topo[ii,:],'.',markersize=1)
#plt.ylabel('phs')
#plt.xlabel('elevation (m)')
#plt.title('corrected phase')
#plt.savefig(workdir + 'Figs/phs_elev_corrected.png',transparent=True,dpi=200)
#
#
#s=std_img
#std_imgs[hgt<.1]=np.nan
#std_img[gamma0_lk<.2]=np.nan
#
#plt.rc('font',size=12)
#pad=1
#plt.figure(figsize=(12,12))
#m = Basemap(epsg=3395, llcrnrlat=lat_bounds.min()-pad,urcrnrlat=lat_bounds.max()+pad,\
#        llcrnrlon=lon_bounds.min()-pad,urcrnrlon=lon_bounds.max()+pad,resolution='l')
#m.drawparallels(np.arange(np.floor(lat_bounds.min()-pad), np.ceil(lat_bounds.max()+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
#m.drawmeridians(np.arange(np.floor(lon_bounds.min()-pad), np.ceil(lon_bounds.max()+pad),1), linewidth=0,labels=[1,0,0,1])
#m.arcgisimage(service='World_Shaded_Relief',xpixels=800)
#cf = m.pcolormesh(lo,la,std_imgs,vmax=80,shading='flat',latlon=True, zorder=8)
#cbar = m.colorbar(cf,location='bottom',pad="10%")
#cbar.set_label('cm')
#plt.show()
#plt.savefig(workdir + 'Figs/std_map.png',transparent=True,dpi=200)

#del(alld,alld_filt,Gg,G,std_img,x,y,y2)

data = -alld_flat.astype(np.float32) # Make subsidence negative
np.save('alld_flat.npy', data)

#data=np.load('alld_flat.npy')

water_elevation=-103
rates,rate_uncertainty = invertRates.invertRates(data,params, seasonals=False,mcov_flag=True,water_elevation=water_elevation)

rates = np.asarray(rates,dtype=np.float32)
np.save('rates.npy', rates)
np.save('rate_uncertainty.npy', rate_uncertainty)

rates[gam<.5]=np.nan
rate_uncertainty[gam<.3]=np.nan

#fig,ax = plt.subplots(2,2,figsize=(8,5))
#ax[0,0].imshow(-rates,vmin=-25,vmax=25)
#ax[0,1].imshow(rate_uncertainty,vmin=0,vmax=25)
#ax[1,0].imshow(-rates_c,vmin=-25,vmax=25)
#ax[1,1].imshow(rate_uncertainty_c,vmin=0,vmax=25)

lo = geom['lon_ifg'][ymin:ymax,:]
la = geom['lat_ifg'][ymin:ymax,:]
pad=0
minlat=la.min()
maxlat=la.max()
minlon=lo.min()
maxlon=lo.max()

# Plot rate map

plt.rc('font',size=12)
plt.figure(figsize=(12,12))
m = Basemap(epsg=3395, llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
        llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='l')
m.drawparallels(np.arange(np.floor(minlat-pad), np.ceil(maxlat+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
m.drawmeridians(np.arange(np.floor(minlon-pad), np.ceil(maxlon+pad),1), linewidth=0,labels=[1,0,0,1])
m.arcgisimage(service='World_Shaded_Relief',xpixels=500)
cf = m.pcolormesh(lo,la,rates,latlon=True, cmap=plt.cm.Spectral_r, zorder=8,vmin=0,vmax=2)
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
m.arcgisimage(service='World_Shaded_Relief',xpixels=500)
cf = m.pcolormesh(lo,la,rate_uncertainty,shading='flat',latlon=True, cmap=plt.cm.Spectral_r,zorder=8,vmin=0,vmax=1)
m.readshapefile('/data/kdm95/qfaults/qfaults_la', 'qfaults_la',zorder=30)
#m.plot(lo_p1,la_p1,color='red',zorder=40,latlon=True)
cbar = m.colorbar(cf,location='bottom',pad="10%")
cbar.set_label('cm')
plt.show()


## Mask the rates matrix
#gamma0_lk[np.isnan(gamma0_lk)]=0
#rates[np.where(gamma0_lk<.2)]=np.nan #masks the low coherence areas
#rates[np.where( (hgt_ifg<water_elevation) ) ]=np.nan # masks the water
#rate_uncertainty[np.where(gamma0_lk<.2)]=np.nan #masks the low coherence areas
#rate_uncertainty[np.where( (hgt_ifg<water_elevation) ) ]=np.nan # masks the water
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
