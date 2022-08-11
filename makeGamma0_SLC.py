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

#from mroipac.filter.Filter import Filter

doPS = True # Setting this to False will make gamma0 all ones.  

if doPS:
    
    overwrite =False
    plot = False
    
    ps = np.load('./ps.npy',allow_pickle=True).all()
    
    # Make the gaussian filter we'll convolve with ifg
    rx = 8
    rx2 = np.floor(rx*3)
    gausx = np.exp( np.divide( -np.square(np.arange(-rx2,rx2)), np.square(rx)));
    gausy = np.exp( np.divide( -np.square(np.arange(-rx2,rx2)), np.square(rx)));
    gaus = gausx[:, np.newaxis] * gausy[np.newaxis, :]
    gaus -= gaus.min()
    gaus  /= np.sum(gaus.flatten())
    del(rx,rx2,gausx,gausy)
    # get slc example image (extract from an xml file)
    f = ps.slcdir +'/'+ ps.dates[0] + '/' + ps.dates[0] + '.slc.full'
    slcImage = isceobj.createSlcImage()
    slcImage.load(f + '.xml')
    intimg = isceobj.createIntImage()
    intimg.width = slcImage.width
    intimg.length = slcImage.length
    
    
    # Find SLCs with zero size or that don't exist.  And find what size they should be. 
    fSizes = []
    for ii,d in enumerate(ps.dates[:-1]): 
        if os.path.isfile(ps.slcdir + '/' + d + '/' + d + '.slc.full'):       
            if os.path.getsize(ps.slcdir + '/' + d + '/' + d + '.slc.full')==0:
                print('WARNING: ' + ps.slcdir + '/' + d + '/.slc.full. File size too small. May be corrupt.' )
            else:
                fSizes.append(os.path.getsize(ps.slcdir + '/' + d + '/' + d + '.slc.full'))
        else:
            print(d + '/.slc.full does not exist')
    medSize = np.nanmedian(fSizes)
    
    # Find SLCs with zero size or that don't exist.  And find what size they should be. 
    for ii,d in enumerate(ps.dates[:-1]): 
        if os.path.isfile(ps.slcdir + '/' + d + '/' + d + '.slc.full'):       
            if os.path.getsize(ps.slcdir + '/' + d + '/' + d + '.slc.full')!=medSize:
                # os.system('rm -r ' + ps.slcdir + '/' + d )
                print('WARNING: ' + ps.slcdir + '/' + d + '/.slc.full. File size not right. May be corrupt.' )
        else:
            print(d + '/.slc.full does not exist')
    
    # Fine fine_diff files that are too small and delete them. 
    for ii,d in enumerate(ps.dates[:-1]): 
        if os.path.isfile(ps.slcdir + '/' + d + '/fine_diff.int'):   
            if os.path.getsize(ps.slcdir + '/' + d + '/fine_diff.int')<medSize:
                os.system('rm ' + ps.slcdir + '/' + d + '/fine*')
                print('removed ' + ps.slcdir + '/' + d + '/fine_diff.int. File size too small. May be corrupt.' )
    
    print('\nFile sizes should be ' + str(medSize))
    
    #ii=0
    #d = dates[ii]
    
    for ii,d in enumerate(ps.dates[:-1]): 
        if not os.path.isfile(ps.slcdir + '/' + d + '/fine_diff.int') or overwrite:
            start_time=time.time()
            d2 = ps.dates[ii+1]
    
            #save diff ifg
            intImage = isceobj.createIntImage()
            intImage.filename = ps.slcdir + '/' + d + '/fine_diff.int'
            intImage.width = ps.nx
            intImage.length = ps.ny
            intImage.dump(intImage.filename + '.xml') # Write out xml

            fid=open(intImage.filename,"wb+")
            print('working on ' + d)
            
            if ps.crop:
                f = ps.slcdir +'/'+ d + '/' + d + '.slc.full.crop'
            else:
                f = ps.slcdir +'/'+ d + '/' + d + '.slc.full'
            slcImage = isceobj.createSlcImage()
            slcImage.load(f + '.xml')
            slc1 = slcImage.memMap()[:,:,0]
            
            if ps.crop:
                f = ps.slcdir +'/'+ d2 + '/' + d2 + '.slc.full.crop'
            else:
                f = ps.slcdir +'/'+ d2 + '/' + d2 + '.slc.full'
            slcImage = isceobj.createSlcImage()
            slcImage.load(f + '.xml')
            slc2 = slcImage.memMap()[:,:,0]
            ifg = np.multiply(slc1,np.conj(slc2))
            ifg_real = np.real(ifg)
            ifg_imag = np.imag(ifg)
            
            ifg_real_filt = cv2.filter2D(ifg_real,-1, gaus)
            ifg_imag_filt = cv2.filter2D(ifg_imag,-1, gaus)  
            phs_filt = np.arctan2(ifg_imag_filt, ifg_real_filt).astype(np.float32)
            
            cpx0    = ifg_real      + 1j * ifg_imag
            cpxf    = ifg_real_filt + 1j * ifg_imag_filt
            cpx0   /= abs(cpx0)
            cpxf   /= abs(cpxf)
            
            fid.write(np.multiply(cpx0, np.conj(cpxf)))
            fid.close()
        else:
            print(d + ' already exists.')
        #mad = lambda x: np.sqrt(np.nanmedian(abs(x - np.nanmedian(x,axis=0))**2),axis=0d)
    
    nblocks = 40 # increase the nblocks because this part is more memory intensive
    
    gamma0 =np.zeros((ps.ny,ps.nx)).astype(np.float32)
    gamma1 =np.zeros((ps.ny,ps.nx)).astype(np.float32)

    # Make a stack of the diff images (memory mapped )
    # We have to do this in 20 subsections to save on memory
    
    
    diffImage = intImage.clone()   
    blocks = np.linspace(0,ps.ny,nblocks).astype(int)
    print('Processing gamma0 image in separate blocks.')
    for ii in np.arange(0,len(blocks)-1):
        print(str(ii) + '/' + str(len(blocks)) + ' blocks processed.')
        diff_stack = np.empty((len(ps.dates[:-1]),(blocks[ii+1]-blocks[ii]),ps.nx)).astype('complex64')
        for jj,d in enumerate(ps.dates[:-1]): 
            diff_file = ps.slcdir + '/' + d + '/fine_diff.int'
            diffImage.load(diff_file + '.xml')
            img = diffImage.memMap()[blocks[ii]:blocks[ii+1],:,0]
            ph = np.angle(img) #abs(np.arctan2(np.imag(img), np.real(img)).astype(np.float32))
            diff_stack[jj,:,:] = img
        diff_stack[diff_stack==0] = np.nan       
        gamma0[blocks[ii]:blocks[ii+1],0:ps.nx] = np.nanvar(np.asarray(np.angle(diff_stack)), axis=0)


# The maximum possible variance of the phase would be 2pi. We want to normalize between 0 and 1.
    gamma0/=-2*np.pi
    gamma0+=1

else:
    gamma0=np.ones((ps.ny,ps.nx))

util.writeISCEimg(gamma0,ps.tsdir + '/gamma0.int',1,ps.nx,ps.ny,'Float')

if plot:
    plt.figure()
    plt.imshow(gamma0,vmin=.5,vmax=.8)
    plt.figure()
    plt.hist( gamma0.flatten()[~np.isnan(gamma0.flatten())], 40, edgecolor='black', linewidth=.2)
    plt.title('Phase stability histogram')
    plt.xlabel('Phase stability (1 is good, 0 is bad)')
    plt.show()

# Remove the phase stability images
os.system('rm ' + ps.mergeddir + '/SLC/*/fine*')
