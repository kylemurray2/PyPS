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

params = np.load('params.npy').item()
locals().update(params)


# Make the gaussian filter we'll convolve with ifg
skip = 3 # how many redundant pairs did you make?
rx = 5
ry = 5
rx2 = np.floor(rx*3)
ry2 = np.floor(ry*3)
gausx = np.exp( np.divide( -np.square(np.arange(-rx2,rx2)), np.square(rx)));
gausy = np.exp( np.divide( -np.square(np.arange(-ry2,ry2)), np.square(ry)));
gaus = gausx[:, np.newaxis] * gausy[np.newaxis, :]
gaus -= gaus.min()
gaus  /= np.sum(gaus.flatten())


for pair in params['pairs'][0:len(params['pairs']):skip]: 
    if not os.path.isfile(params['intdir'] + '/' + pair + '/fine_diff.int'):
        print('working on ' + pair)
        #load ifg real and imaginary parts
        f = params['intdir'] +'/'+ pair + '/fine.int'
        intImage = isceobj.createIntImage()
        intImage.load(f + '.xml')
        ifg_real = np.real(intImage.memMap()[:,:,0] )
        ifg_imag = np.imag(intImage.memMap()[:,:,0] )
        ifg_real = ifg_real.copy() # mmap is readonly, so we need to copy it.
        ifg_imag = ifg_imag.copy()
        ifg_real[np.where(ifg_real==0)] = np.nan
        ifg_imag[np.where(ifg_real==0)] = np.nan
#        phs = np.angle(ifg_real+(1j*ifg_imag))
        phs = np.arctan2(ifg_imag, ifg_real).astype(np.float32)
        
        #filter real and imaginary parts    
        ifg_real_filt = cv2.filter2D(ifg_real,-1, gaus)
        ifg_imag_filt = cv2.filter2D(ifg_imag,-1, gaus)  
#        phs_filt = np.angle(ifg_real_filt+(1j*ifg_imag_filt))
        phs_filt = np.arctan2(ifg_imag_filt, ifg_real_filt).astype(np.float32)

        # Difference them 
        cpx0    = ifg_real + 1j*ifg_imag
        cpxf    = ifg_real_filt + 1j * ifg_imag_filt
        cpx0   /= abs(cpx0)
        cpxf   /= abs(cpxf)
        phsdiff = np.multiply(cpx0, np.conj(cpxf))
        
        #save diff ifg
        out = intImage.clone() # Copy the interferogram image from before
        out.scheme =  'BIP' #'BIP'/ 'BIL' / 'BSQ' 
        out.filename = params['intdir'] + '/' + pair + '/fine_diff.int'
        out.dump(params['intdir']  + '/' + pair + '/fine_diff.int.xml') # Write out xml
        phsdiff.tofile(params['intdir'] + '/' + pair+'/fine_diff.int') # Write file out


del(ifg_imag,ifg_imag_filt,ifg_real,ifg_real_filt,phs,cpx0,cpxf,phs_filt,phsdiff)

gamma0 =list()
# Make a stack of the diff images (memory mapped )
# We have to do this in 20 subsections to save on memory
chunks = np.linspace(0,params['ny'],20,dtype=int)
for ii in np.arange(0,len(chunks)-1):
    diff_stack = list()
    for pair in params['pairs'][0:len(params['pairs']):skip]: #loop through each ifg and save to 
        diff_file = params['intdir'] + '/' + pair + '/fine_diff.int'
        diffImage = isceobj.createIntImage()
        diffImage.load(diff_file + '.xml')
        diff_stack.append(diffImage.memMap()[chunks[ii]:chunks[ii+1],:,0])
    # Find phase variance 
    gamma0.append(np.abs( np.nansum( np.asarray(diff_stack), axis=0)/len(dates))) 

a=np.empty((params['ny'],params['nx']))
for ii in np.arange(0,len(chunks)-1):
    a[chunks[ii]:chunks[ii+1]]=gamma0[ii]
gamma0 = a
gamma0 = np.asarray(gamma0)
gamma0=np.reshape(np.asarray(gamma0),(params['ny'],params['nx']))
gamma0[np.where(gamma0==0)]=np.nan
gamma0=gamma0.astype(np.float32)


# Save gamma0 file
out = intImage.clone() # Copy the interferogram image from before
out.dataType = 'FLOAT'
out.filename = params['tsdir'] + '/gamma0.int'
out.dump(params['tsdir'] + '/gamma0.int.xml') # Write out xml
gamma0.tofile(out.filename) # Write file out

plt.imshow(gamma0,vmin=0.1,vmax=.8)
plt.figure()
plt.hist( gamma0.flatten()[~np.isnan(gamma0.flatten())], 40, edgecolor='black', linewidth=.2)
plt.title('Phase stability histogram')
plt.xlabel('Phase stability (1 is good, 0 is bad)')


