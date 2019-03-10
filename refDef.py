#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:35:54 2018

@author: kdm95
"""

import numpy as np
import isceobj
import pickle
from datetime import date
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from mpl_toolkits.basemap import Basemap

params = np.load('params.npy').item()
locals().update(params)
geom = np.load('geom.npy').item()
locals().update(geom)

atm_flag = 0 # set to 1 if you did a gacos correction

#badpairs = list(['20161130_20161212','','']) 
#pairs = np.setxor1d(badpairs,pairs).flatten()

nd=len(pairs)

##Crop images
#ymin=85
#ymax=3555
#xmin=45
#xmax=1720

ymin=0
ymax=nyl
xmin=0
xmax=nxl

nxl = xmax-xmin
nyl = ymax-ymin

#intitial_reference_point
c=3500
r=360
idx = ((r-1)*nxl)+c #finds the index of flattened array based on row col in image

gamma0_thresh = .3
std_thresh = 8

stds = []
for ii,pair in enumerate(pairs): #loop through each ifg and append to alld 
    unw_file = intdir + '/' + pair + '/fine_lk_int.unw'
    unwImage = isceobj.createIntImage()
    unwImage.load(unw_file + '.xml')
    unwifg = unwImage.memMap()[ymin:ymax,xmin:xmax,0]
    u = unwifg-np.nanmedian(unwifg)
    stds.append(1/np.nanstd(u.flatten()))
#    print(str(u.flatten().min()) + ' ' + str(u.flatten().max()) + ' ' + pair)
    print(str(np.nanstd(u.flatten())) + ' ' + pair)
stds.append(1)

#W = np.append(W,np.zeros((1,len(pairs))),axis=0)
#W[-1,0] = 1
# SBAS Inversion to get displacement at each date
## Make G matrix for dates inversion
G = np.zeros((nd+1,len(dn)))
for ii,pair in enumerate(pairs):
    a = np.asarray(np.where(dates==pair[0:8]),dtype=int)[0][0]
    b = np.asarray(np.where(dates==pair[9:17]),dtype=int)[0][0]
    G[ii,a] = -1
    G[ii,b] = 1

Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
N = np.dot(G,Gg)
R = np.dot(Gg,G)

# Do dates inversion
alld=np.zeros((len(dn),nxl*nyl))
for ii in np.arange(0,nyl-1): #iterate through rows
    tmp = np.zeros((nd+1,nxl))
    for jj,pair in enumerate(pairs): #loop through each ifg and append to alld 
#        jj+=1
        unw_file = intdir + '/' + pair + '/fine_lk_int.unw'
        unwImage = isceobj.createIntImage()
        unwImage.load(unw_file + '.xml')
        ref = unwImage.memMap()[r,c,0]
        tmp[jj,:] = unwImage.memMap()[ymin+ii,xmin:xmax,0] - ref
        dw = np.dot(W,tmp)
    alld[:,ii*nxl:nxl*ii+nxl] = np.dot(Gg, dw)
    
#    alld[:,ii*nyl:nyl*ii+nyl] = np.dot(Gg, tmp)

del(tmp)

    
# Filter each date before calculating the std for referencing.
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=1) #Check this axis!
    return y

order = 8
fs = 0.066      # sample rate, Hz
cutoff = 1/180  # desired cutoff frequency of the filter, Hz
alld_filt = butter_lowpass_filter(alld, cutoff, fs, order) # check the function for the axis in lfilter
alld_filt[alld_filt==0]==np.nan
alld[alld==0]==np.nan

std_img = np.nanstd(alld_filt,axis=0) # Temporal std
std_img = np.reshape(std_img,(nyl,nxl))
#std_img *= crop_mask
std_img[std_img==0]=np.nan
#plt.imshow(s,vmin=0,vmax=8)

plt.figure()
for ii in np.arange(0,len(alld[:,0]),4):
    plt.plot(np.reshape(alld[ii,:],(nyl,nxl))[200,:])
    
# MASKING______________________________
# Load gamma0_lk
f = tsdir  + '/gamma0_lk.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0_lk= intImage.memMap()[ymin:ymax,xmin:xmax,0] # * crop_mask

plt.figure()
plt.hist( gamma0_lk.flatten()[~np.isnan(gamma0_lk.flatten())], 40, edgecolor='black', linewidth=.2)
plt.title('Phase stability histogram')
plt.xlabel('Phase stability (1 is good, 0 is bad)')
plt.figure()
plt.hist(std_img[~np.isnan(std_img)], 40, edgecolor='black', linewidth=.2)


# Load height file
h = workdir + '/merged/geom_master/hgt_lk.rdr' #make this manually for now with looks.py and run fixImage.py 
hImg = isceobj.createImage()
hImg.load(h + '.xml')
hgt = hImg.memMap()[ymin:ymax,xmin:xmax,0].astype(np.float32) #*crop_mask
#hgt[hgt<0]=-1
## elevations at 4 of the main lakes that we'll mask out
#l1 = hgt[2012,133]

msk = np.where( (std_img.flatten() < std_thresh) & (gamma0_lk.flatten() > gamma0_thresh) & (hgt.flatten() > -100) )
#alld[:,msk] = np.nan

#a=np.reshape(alld,(len(pairs)+1,nyl,nxl))
#af=np.reshape(alld_filt,(len(pairs)+1,nyl,nxl))

# Remove mean from each image using just the nondeforming pixels
alld_flat=np.empty(alld.shape)
for ii in np.arange(0,len(alld[:,0])):
    alld_flat[ii,:] = alld[ii,:] - np.nanmedian(alld[ii,:])
#    alld_flat[ii,np.where(hgt.flatten()==-1)]=np.nan
plt.figure()
for ii in np.arange(0,len(alld[:,0]),5):
    plt.plot(np.reshape(alld_flat[ii,:],(nyl,nxl))[200,:])
    
    
a = np.reshape(alld_flat[44,:],(nyl,nxl))
plt.figure();plt.imshow(a)
    
#alld_flat=np.reshape(alld,(len(pairs)+1,nyl,nxl))
# Saving alld_flat :
#with open(tsdir + 'alld_flat.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump(alld_flat, f)

# Plot phase-elevation 
#plt.figure()
#plt.plot(hgt.flatten()[msk], alld_flat[30,msk].flatten(),'.',markersize=1)

# Do phase-elevation correction
x=hgt.flatten()[msk]
#x*=crop_mask.flatten()[msk]
G = np.vstack([x, np.ones((len(x),1)).flatten()]).T
Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
alld_flat_topo = np.empty(alld_flat.shape)

for ii in np.arange(0,len(alld[:,0])):
    y=alld_flat[ii,msk].flatten()
    mod    = np.dot(Gg, y)
    y2 = mod[0]*hgt.flatten() + mod[1]
    alld_flat_topo[ii,:] = alld_flat[ii,:].flatten()-y2
#    alld_flat[ii,np.where(hgt.flatten()==-1)]=np.nan
    
# Plot example
plt.figure()
plt.plot(x,y,'.',markersize=1)
plt.plot(hgt.flatten(),y2)
plt.ylabel('phs')
plt.xlabel('elevation (m)')
plt.title('uncorrected phase and best fit')
plt.savefig(workdir + 'Figs/phs_elev_uncorrected.png',transparent=True,dpi=200)

plt.figure()
plt.plot(hgt.flatten(),alld_flat_topo[ii,:],'.',markersize=1)
plt.ylabel('phs')
plt.xlabel('elevation (m)')
plt.title('corrected phase')
plt.savefig(workdir + 'Figs/phs_elev_corrected.png',transparent=True,dpi=200)


s=std_img
h = workdir + '/merged/geom_master/lon_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
lo = hImg.memMap()[ymin:ymax,xmin:xmax,0].astype(np.float32)
lo[lo==0]=np.nan
h = workdir + '/merged/geom_master/lat_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
la = hImg.memMap()[ymin:ymax,xmin:xmax,0].astype(np.float32)
la[la==0]=np.nan


#s*=crop_mask
#s=s[ymin:ymax,:]
s[hgt<.1]=np.nan
s[gamma0_lk<.2]=np.nan

#s[2195:,:4056]=np.nan
#s[2041:,:2810]=np.nan

plt.rc('font',size=12)
pad=1
plt.figure(figsize=(12,12))
m = Basemap(epsg=3395, llcrnrlat=lat_bounds.min()-pad,urcrnrlat=lat_bounds.max()+pad,\
        llcrnrlon=lon_bounds.min()-pad,urcrnrlon=lon_bounds.max()+pad,resolution='l')
m.drawparallels(np.arange(np.floor(lat_bounds.min()-pad), np.ceil(lat_bounds.max()+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
m.drawmeridians(np.arange(np.floor(lon_bounds.min()-pad), np.ceil(lon_bounds.max()+pad),1), linewidth=0,labels=[1,0,0,1])
m.arcgisimage(service='World_Shaded_Relief',xpixels=800)
cf = m.pcolormesh(lo,la,s,vmax=80,shading='flat',latlon=True, zorder=8)
cbar = m.colorbar(cf,location='bottom',pad="10%")
cbar.set_label('cm')
plt.show()
plt.savefig(workdir + 'Figs/std_map.png',transparent=True,dpi=200)


#del(alld,alld_filt,Gg,G,std_img,x,y,y2)

#alld_flat_topo = -alld_flat_topo.astype(np.float32) # Make subsidence negative
alld_flat_topo = -alld_flat.astype(np.float32) # Make subsidence negative
np.save('alld_flat_topo_W.npy', alld_flat_topo)
np.save('pairs_cut',pairs)


