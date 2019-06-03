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

overwrite =False

params = np.load('params.npy',allow_pickle=True).item()
locals().update(params)
dates = params['dates']

# Make the gaussian filter we'll convolve with ifg
rx = 5
ry = 5
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
#        os.system('fixImageXml.py -i ' + f + ' -f')
        slcImage = isceobj.createSlcImage()
        slcImage.load(f + '.xml')
        slc1 = slcImage.memMap()[:,:,0]
        f = params['slcdir'] +'/'+ d2 + '/' + d2 + '.slc.full'
#        os.system('fixImageXml.py -i ' + f + ' -f')

        slcImage = isceobj.createSlcImage()
        slcImage.load(f + '.xml')
        slc2 = slcImage.memMap()[:,:,0]
        ifg = np.multiply(slc1,np.conj(slc2))
        del(slc1,slc2)
        ifg_real = np.real(ifg)
        ifg_imag = np.imag(ifg)
        del(ifg)
#        ifg_real[np.where(ifg_real==0)] = np.nan
#        ifg_imag[np.where(ifg_real==0)] = np.nan
        #filter real and imaginary parts    
        ifg_real_filt = cv2.filter2D(ifg_real,-1, gaus)
        ifg_imag_filt = cv2.filter2D(ifg_imag,-1, gaus)  
#        phs_filt = np.arctan2(ifg_imag_filt, ifg_real_filt).astype(np.float32)
        
        # Difference them 
        cpx0    = ifg_real      + 1j * ifg_imag
        del(ifg_real,ifg_imag)
        cpxf    = ifg_real_filt + 1j * ifg_imag_filt
        del(ifg_real_filt,ifg_imag_filt)
        cpx0   /= abs(cpx0)
        cpxf   /= abs(cpxf)
        phsdiff = np.multiply(cpx0, np.conj(cpxf))
        del(cpxf,cpx0)
        #save diff ifg
        intImage = intimg.clone() # Copy the interferogram image from before
        intImage.filename = params['slcdir'] + '/' + d + '/fine_diff.int'
        intImage.dump(params['slcdir']  + '/' + d + '/fine_diff.int.xml') # Write out xml
        phsdiff.tofile(params['slcdir'] + '/' + d + '/fine_diff.int') # Write file out
        del(phsdiff)
    else:
        print(d + ' already exists.')
#mad = lambda x: np.sqrt(np.nanmedian(abs(x - np.nanmedian(x,axis=0))**2),axis=0d)


gamma0 =np.zeros((ny,nx)).astype(np.float32)
# Make a stack of the diff images (memory mapped )
# We have to do this in 20 subsections to save on memory
nblocks = 60


diffImage = intimg.clone()   
blocks = np.linspace(0,params['ny'],nblocks).astype(int)
print('Processing gamma0 image in 20 separate blocks.')
for ii in np.arange(0,len(blocks)-1):
    print(str(ii) + '/' + str(len(blocks)) + ' blocks processed.')
    diff_stack = np.zeros((len(dates[:-1]),nx,(blocks[ii+1]-blocks[ii]))).astype(np.float32)
    for jj,d in enumerate(dates[:-1]): 
        diff_file = params['slcdir'] + '/' + d + '/fine_diff.int'
        diffImage.load(diff_file + '.xml')
        img = diffImage.memMap()[blocks[ii]:blocks[ii+1],:,0]
        ph = abs(np.arctan2(np.imag(img), np.real(img)).astype(np.float32))
        diff_stack[jj,:,:] = ph.T
    diff_stack[diff_stack==0] = np.nan
    gamma0[blocks[ii]:blocks[ii+1],0:nx] =np.nanvar(np.asarray(diff_stack), axis=0).T

plt.imshow(gamma0)



gamma0 = 1-gamma0
gamma0 += abs( gamma0[~np.isnan(gamma0)].min())
gamma0 /= gamma0[~np.isnan(gamma0)].max()

out = isceobj.createImage()
out.filename = params['tsdir'] + '/gamma0.int'
out.bands = 1
out.length = ny
out.width = nx
out.dataType = 'Float'
out.dump(out.filename + '.xml') # Write out xml
gamma0.tofile(out.filename) # Write file out

#gamma0 *= np.sqrt(gamma0)

plt.imshow(gamma0)
plt.figure()
plt.hist( gamma0.flatten()[~np.isnan(gamma0.flatten())], 40, edgecolor='black', linewidth=.2)
plt.title('Phase stability histogram')
plt.xlabel('Phase stability (1 is good, 0 is bad)')
plt.show()
