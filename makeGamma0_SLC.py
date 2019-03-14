#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:39:22 2018
calcGamma and smart downlooking

Saves fine_diff.int files in respective ifg directories used to make the gamma0 
file (they can be deleted after it is made)

Saves a file called gamma0 in the tsdir (TS directory)
Saves downlooked ifgs in the respective ifg directories.

@author: kdm95
"""

import numpy as np
import isceobj
from matplotlib import pyplot as plt
import cv2
import os
#from mroipac.filter.Filter import Filter

overwrite = True

params = np.load('params.npy').item()
locals().update(params)
dates = params['dates']

# Make the gaussian filter we'll convolve with ifg
rx = 10
ry = 10
rx2 = np.floor(rx*3)
ry2 = np.floor(ry*3)
gausx = np.exp( np.divide( -np.square(np.arange(-rx2,rx2)), np.square(rx)));
gausy = np.exp( np.divide( -np.square(np.arange(-ry2,ry2)), np.square(ry)));
gaus = gausx[:, np.newaxis] * gausy[np.newaxis, :]
gaus -= gaus.min()
gaus  /= np.sum(gaus.flatten())

# get slc example image (extract from an xml file)
f = params['slcdir'] +'/'+ dates[0] + '/' + dates[0] + '.slc.full'
slcImage = isceobj.createSlcImage()
slcImage.load(f + '.xml')
intimg = isceobj.createIntImage()
intimg.width = slcImage.width
intimg.length = slcImage.length
for ii,d in enumerate(dates[:-1]): 
    if not os.path.isfile(params['slcdir'] + '/' + d + '/fine_diff.int') or overwrite:
        print('working on ' + d)
        d2 = dates[ii+1]
        #load ifg real and imaginary parts
        f = params['slcdir'] +'/'+ d + '/' + d + '.slc.full'
        slcImage = isceobj.createSlcImage()
        slcImage.load(f + '.xml')
        slc1 = slcImage.memMap()[:,:,0]
        f = params['slcdir'] +'/'+ d2 + '/' + d2 + '.slc.full'
        slcImage = isceobj.createSlcImage()
        slcImage.load(f + '.xml')
        slc2 = slcImage.memMap()[:,:,0]
        ifg = np.multiply(slc1,np.conj(slc2))
        ifg_real = np.real(ifg)
        ifg_imag = np.imag(ifg)
#        ifg_real[np.where(ifg_real==0)] = np.nan
#        ifg_imag[np.where(ifg_real==0)] = np.nan
        #filter real and imaginary parts    
        ifg_real_filt = cv2.filter2D(ifg_real,-1, gaus)
        ifg_imag_filt = cv2.filter2D(ifg_imag,-1, gaus)  
        phs_filt = np.arctan2(ifg_imag_filt, ifg_real_filt).astype(np.float32)

        # Difference them 
        cpx0    = ifg_real      + 1j * ifg_imag
        cpxf    = ifg_real_filt + 1j * ifg_imag_filt
        cpx0   /= abs(cpx0)
        cpxf   /= abs(cpxf)
        phsdiff = np.multiply(cpx0, np.conj(cpxf))
        
        #save diff ifg
        intImage = intimg.clone() # Copy the interferogram image from before
        intImage.filename = params['slcdir'] + '/' + d + '/fine_diff.int'
        intImage.dump(params['slcdir']  + '/' + d + '/fine_diff.int.xml') # Write out xml
        phsdiff.tofile(params['slcdir'] + '/' + d + '/fine_diff.int') # Write file out

    else:
        print(d + ' already exists.')

del(ifg_imag,ifg_imag_filt,ifg_real,ifg_real_filt,cpx0,cpxf,phsdiff)

#mad = lambda x: np.sqrt(np.nanmedian(abs(x - np.nanmedian(x,axis=0))**2),axis=0d)


gamma0 =list()
# Make a stack of the diff images (memory mapped )
# We have to do this in 20 subsections to save on memory
chunks = np.linspace(0,params['ny'],17,dtype=int)
for ii in np.arange(0,len(chunks)-1):
    diff_stack = list()
    for jj,d in enumerate(dates[:-1]): 
        diff_file = params['slcdir'] + '/' + d + '/fine_diff.int'
        diffImage = intimg.clone() 
        diffImage.load(diff_file + '.xml')
        img = diffImage.memMap()[chunks[ii]:chunks[ii+1],:,0]
        ph = abs(np.arctan2(np.imag(img), np.real(img)).astype(np.float32))
        ph[ph==0]=np.nan
        diff_stack.append(ph)
    # Find phase variance 
#    gamma0.append(np.abs( np.nansum( np.asarray(diff_stack), axis=0)/len(dates))) 
        gamma0.append(np.nanvar(np.asarray(diff_stack), axis=0))
b=np.empty((params['ny'],params['nx']))
for ii in np.arange(0,len(chunks)-1):
    b[chunks[ii]:chunks[ii+1]]=gamma0[ii]
gamma02 = a
gamma02 = np.asarray(gamma02)
gamma02=np.reshape(np.asarray(gamma02),(params['ny'],params['nx']))
gamma02 /= gamma02.max()
gamma02[np.where(gamma02==0)]=np.nan
gamma02=np.asarray(gamma02, dtype=np.float32)


# Save gamma0 file

out = intimg.clone() # Copy the interferogram image from before
out.filename = params['tsdir'] + '/gamma0.int'
out.dump(params['tsdir'] + '/gamma0.int.xml') # Write out xml
gamma02.tofile(out.filename) # Write file out

plt.imshow(gamma02,vmin=0.1,vmax=.4)
plt.figure()
plt.hist( gamma02.flatten()[~np.isnan(gamma02.flatten())], 40, edgecolor='black', linewidth=.2)
plt.title('Phase stability histogram')
plt.xlabel('Phase stability (1 is good, 0 is bad)')
plt.show()
