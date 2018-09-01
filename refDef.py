#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:35:54 2018

@author: kdm95
"""

import numpy as np
import isceobj
import pickle
import os
from datetime import date
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, freqz

with open(tsdir + '/params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks = pickle.load(f)


#badpairs = list(['20161130_20161212','','']) 
#pairs = np.setxor1d(badpairs,pairs).flatten()
    
badpairs = list(['20151005_20151029',
'20151029_20151122',
'20151122_20151216',
'20161210_20170103',
'20170115_20170127',
'20170127_20170220',
'20170220_20170304',
'20170819_20170831',
'20170831_20170912',
'20171229_20180110',
'20180110_20180122',
'20180311_20180323',
'20180323_20180404',])
pairs = np.setxor1d(badpairs,pairs).flatten()


gamma0_thresh = .2
std_thresh = 10

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=1) #Check this axis!
    return y


# Filter requirements.
order = 8
fs = 0.066      # sample rate, Hz
cutoff = 1/180  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)
# Plot the frequency response.
#w, h = freqz(b, a, worN=8000)
#plt.subplot(2, 1, 1)
#plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
#plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
#plt.axvline(cutoff, color='k')
#plt.xlim(0, 0.5*fs)
#plt.title("Lowpass Filter Frequency Response")
#plt.xlabel('Frequency [Hz]')
#plt.grid()


# Convert pairs to dates
dn = list()
dn.append( date.toordinal(date(int(pairs[0][9:13]), int(pairs[0][13:15]), int(pairs[0][15:]))) )
for pair in pairs:
    yr = pair[9:13]
    mo = pair[13:15]
    day = pair[15:]
    dn.append(date.toordinal(date(int(yr), int(mo), int(day))))
dn = np.asarray(dn)
dn0 = dn-dn[0] # make relative to first date

# Make a stack of the unwrapped images (memory mapped)
alld = list()
alld.append(np.zeros((nxl*nyl,1)).flatten()) #the first image is all zeros
print('Min phase  Max phase  Pair')
for ii,pair in enumerate(pairs): #loop through each ifg and append to alld 
    unw_file = intdir + pair + '/fine_lk.unw'
    unwImage = isceobj.createIntImage()
    unwImage.load(unw_file + '.xml')
    unwifg = unwImage.memMap()[:,:,0]
    print(str(unwifg.min()) + ' ' + str(unwifg.max()) + ' ' + pair)
    alld.append(unwifg.flatten()+alld[ii]) # cumulatively add to the previous image
 
# Sum to make cumulative sum stack image
alld = np.asarray(alld) 
alld_filt = butter_lowpass_filter(alld, cutoff, fs, order) # check the function for the axis in lfilter

 # Filter each image
std_img = np.nanstd(alld_filt,axis=0) # Temporal std
plt.imshow(np.reshape(std_img,(nyl,nxl)))

# MASKING______________________________
# Load gamma0_lk
f = tsdir + 'gamma0_lk.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0_lk= intImage.memMap()[:,:,0] 

plt.figure()
plt.hist( gamma0_lk.flatten()[~np.isnan(gamma0_lk.flatten())], 40, edgecolor='black', linewidth=.2)
plt.title('Phase stability histogram')
plt.xlabel('Phase stability (1 is good, 0 is bad)')
plt.figure()
plt.hist(std_img, 40, edgecolor='black', linewidth=.2)


# Load height file
h = workdir + 'merged/geom_master/hgt_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
hgt = hImg.memMap()[:,:,0].astype(np.float32)
hgt[hgt<0]=-1
## elevations at 4 of the main lakes that we'll mask out
#l1 = hgt[2012,133]

msk = np.where( (std_img < std_thresh) & (gamma0_lk.flatten() > gamma0_thresh) & (hgt.flatten() > 0.1) )
#alld[:,msk] = np.nan

#a=np.reshape(alld,(len(pairs)+1,nyl,nxl))
#af=np.reshape(alld_filt,(len(pairs)+1,nyl,nxl))

# Remove mean from each image using just the nondeforming pixels
alld_flat=np.empty(alld.shape)
for ii in np.arange(0,len(pairs)+1):
    alld_flat[ii,:] = alld [ii,:] - alld[ii,msk].mean()
    alld_flat[ii,np.where(hgt.flatten()==-1)]=np.nan
    


#alld_flat=np.reshape(alld,(len(pairs)+1,nyl,nxl))
# Saving alld_flat :
#with open(tsdir + 'alld_flat.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump(alld_flat, f)

# Plot phase-elevation 
plt.figure()
plt.plot(alld_flat[30,msk].flatten(),hgt.flatten()[msk],'.',markersize=1)

# Do phase-elevation correction
x=hgt.flatten()[msk]
G = np.vstack([x, np.ones((len(x),1)).flatten()]).T
Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
alld_flat_topo = np.empty(alld_flat.shape)

for ii in np.arange(0,len(pairs)+1):
    y=alld_flat[ii,msk].flatten()
    mod    = np.dot(Gg, y)
    y2 = mod[0]*hgt.flatten() + mod[1]
    alld_flat_topo[ii,:] = alld_flat[ii,:].flatten()-y2
#    alld_flat[ii,np.where(hgt.flatten()==-1)]=np.nan

    
# Plot example
plt.plot(x,y,'.',markersize=1)
plt.plot(hgt.flatten(),y2)
plt.ylabel('phs')
plt.xlabel('elevation (m)')
plt.title('uncorrected phase and best fit')
plt.figure()
plt.plot(hgt.flatten(),alld_flat_topo[ii,:],'.',markersize=1)
plt.ylabel('phs')
plt.xlabel('elevation (m)')
plt.title('corrected phase')
del(alld,alld_filt,alld_flat,Gg,G,std_img,x,y,y2)
alld_flat_topo = -alld_flat_topo # Make subsidence negative


