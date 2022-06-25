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
import time
import util
from util import show
#from mroipac.filter.Filter import Filter

doPS = True # Setting this to False will make gamma0 all ones.  

if doPS:
    
    overwrite =False
    plot = False
    params = np.load('params.npy',allow_pickle=True).item()
    locals().update(params)
    dates = params['dates']
    
    # Make the gaussian filter we'll convolve with ifg
    gaus = util.gaussian_kernel(8,8,4,4)
    #show(gaus)
        
    # get slc example image (extract from an xml file)
    f = params['slcdir'] +'/'+ dates[0] + '/' + dates[0] + '.slc.full'
    slcImage = isceobj.createSlcImage()
    slcImage.load(f + '.xml')
    intimg = isceobj.createIntImage()
    intimg.width = params['nx'] # if cropping, this will be the crop width
    intimg.length = params['ny']
    
    
    # Find SLCs with zero size or that don't exist.  And find what size they should be. 
    fSizes = []
    for ii,d in enumerate(dates[:-1]): 
        if os.path.isfile(params['slcdir'] + '/' + d + '/' + d + '.slc.full'):       
            if os.path.getsize(params['slcdir'] + '/' + d + '/' + d + '.slc.full')==0:
                print('WARNING: ' + params['slcdir'] + '/' + d + '/.slc.full. File size too small. May be corrupt.' )
            else:
                fSizes.append(os.path.getsize(params['slcdir'] + '/' + d + '/' + d + '.slc.full'))
        else:
            print(d + '/.slc.full does not exist')
    medSize = np.nanmedian(fSizes)
    
    # Find SLCs with zero size or that don't exist.  And find what size they should be. 
    for ii,d in enumerate(dates[:-1]): 
        if os.path.isfile(params['slcdir'] + '/' + d + '/' + d + '.slc.full'):       
            if os.path.getsize(params['slcdir'] + '/' + d + '/' + d + '.slc.full')!=medSize:
                # os.system('rm -r ' + params['slcdir'] + '/' + d )
                print('WARNING: ' + params['slcdir'] + '/' + d + '/.slc.full. File size not right. May be corrupt.' )
        else:
            print(d + '/.slc.full does not exist')
    
    # Fine fine_diff files that are too small and delete them. 
    for ii,d in enumerate(dates[:-1]): 
        if os.path.isfile(params['slcdir'] + '/' + d + '/fine_diff.int'):   
            if os.path.getsize(params['slcdir'] + '/' + d + '/fine_diff.int')<medSize:
                # os.system('rm ' + params['slcdir'] + '/' + d + '/fine*')
                print('removed ' + params['slcdir'] + '/' + d + '/fine_diff.int. File size too small. May be corrupt.' )
    
    print('\nFile sizes should be ' + str(medSize))
    
    #ii=0
    #d = dates[ii]
    
    for ii,d in enumerate(dates[:-1]): 
        if not os.path.isfile(params['slcdir'] + '/' + d + '/fine_diff.int') or overwrite:
            start_time=time.time()
            d2 = dates[ii+1]
    
            #save diff ifg
            intImage = intimg.clone() # Copy the interferogram image from before
            intImage.filename = params['slcdir'] + '/' + d + '/fine_diff.int'
            
            intImage.dump(intImage.filename + '.xml') # Write out xml
            fid=open(intImage.filename,"wb+")
            print('working on ' + d)
            
            #load ifg real and imaginary parts
            f = params['slcdir'] +'/'+ d + '/' + d + '.slc.full'
    #        os.system('fixImageXml.py -i ' + f + ' -f')
            slcImage = isceobj.createSlcImage()
            slcImage.load(f + '.xml')
            slc1 = slcImage.memMap()[cropymin:cropymax,cropxmin:cropxmax,0]
            f = params['slcdir'] +'/'+ d2 + '/' + d2 + '.slc.full'
    #        os.system('fixImageXml.py -i ' + f + ' -f')
    
            slcImage = isceobj.createSlcImage()
            slcImage.load(f + '.xml')
            slc2 = slcImage.memMap()[cropymin:cropymax,cropxmin:cropxmax,0]
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
            
            # Difference them and append it to the file
            fid.write(np.multiply(cpx0, np.conj(cpxf)))
            fid.close()
        else:
            print(d + ' already exists.')
        #mad = lambda x: np.sqrt(np.nanmedian(abs(x - np.nanmedian(x,axis=0))**2),axis=0d)
    
    nblocks = 6 # increase the nblocks because this part is more memory intensive
    
    gamma0 =np.zeros((ny,nx)).astype(np.float32)
    # Make a stack of the diff images (memory mapped )
    # We have to do this in 20 subsections to save on memory
    
    
    diffImage = intimg.clone()   
    blocks = np.linspace(0,params['ny'],nblocks).astype(int)
    print('Processing gamma0 image in separate blocks.')
    for ii in np.arange(0,len(blocks)-1):
        diff_stack = np.zeros((len(dates[:-1]),nx,(blocks[ii+1]-blocks[ii]))).astype(np.float32)
        for jj,d in enumerate(dates[:-1]): 
            diff_file = params['slcdir'] + '/' + d + '/fine_diff.int'
            diffImage.load(diff_file + '.xml')
            img = diffImage.memMap()[blocks[ii]:blocks[ii+1],:,0]
            ph = abs(np.arctan2(np.imag(img), np.real(img)).astype(np.float32))
            diff_stack[jj,:,:] = ph.T
        diff_stack[diff_stack==0] = np.nan
    #    if np.isnan(diff_stack).any():
    #        gamma0[blocks[ii]:blocks[ii+1],0:nx] = np.zeros(gamma0[blocks[ii]:blocks[ii+1],0:nx].shape)
    #    else:
        gamma0[blocks[ii]:blocks[ii+1],0:nx] =np.nanvar(np.asarray(np.exp(diff_stack)), axis=0).T
        print(str(ii+1) + '/' + str(len(blocks)) + ' blocks processed.')

    
    # plt.imshow( -(gamma0/np.nanmax(gamma0)) +1 )
    
    
    # gamma0 = gamma0[~np.isnan(gamma0)].max()-gamma0
    gamma0 = -(gamma0/np.nanmax(gamma0)) +1
    # gamma0 += abs( gamma0[~np.isnan(gamma0)].min())
    # gamma0 /= gamma0[~np.isnan(gamma0)].max()
else:
    gamma0=np.ones((ny,nx))
    
out = isceobj.createImage()
out.filename = params['tsdir'] + '/gamma0.int'
out.bands = 1
out.length = ny
out.width = nx
out.dataType = 'Float'
out.dump(out.filename + '.xml') # Write out xml
gamma0.tofile(out.filename) # Write file out

#gamma0 *= np.sqrt(gamma0)
if plot:
    plt.imshow(gamma0)
    plt.figure()
    plt.hist( gamma0.flatten()[~np.isnan(gamma0.flatten())], 40, edgecolor='black', linewidth=.2)
    plt.title('Phase stability histogram')
    plt.xlabel('Phase stability (1 is good, 0 is bad)')
    plt.show()

# Remove the phase stability images
os.system('rm ' + mergeddir + '/SLC/*/fine*')
