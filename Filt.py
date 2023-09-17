#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:08:09 2020
 Filter coherent areas into nan values for better unwrapping
@author: kdm95
"""

import numpy as np
import astropy
import isceobj
from matplotlib import pyplot as plt
import cv2
import os
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve


gamThresh = .45
connCompCompleteness = 0.9
mincor = .5


ps = np.load('./ps.npy',allow_pickle=True).all()

gam = np.load('Npy/gam.npy')
msk = np.load('Npy/msk.npy')
corAvgMap = np.load('Npy/cor.npy')
connSum = np.load('Npy/connSum.npy')


# Make masks based on 4 criteria
gamMsk = np.ones(gam.shape)
gamMsk[gam<gamThresh] = 0
connMsk = np.ones(gam.shape)
connMsk[connSum<round(.9*ps.nd)] = 0
corMsk = np.ones(gam.shape)
corMsk[corAvgMap<mincor] = 0


# # Make the final msk
msk = np.ones(gam.shape)
msk[gamMsk==0]  = 0
msk[connMsk==0] = 0
msk[corMsk==0]  = 0

plt.figure();plt.imshow(msk)

# Load ifg
pairID = 10
pair=ps.pairs[pairID]
kernel = Gaussian2DKernel(x_stddev=1)
    
for pair in ps.pairs:

    
    ifgimg = isceobj.createIntImage()
    ifgimg.load(ps.intdir + '/' + pair + '/fine_lk.int.xml')
    ifg_real = np.copy(np.real( ifgimg.memMap()[:,:,0]))
    ifg_imag = np.copy(np.imag( ifgimg.memMap()[:,:,0]))
    
    corimg = isceobj.createImage()
    corimg.load(ps.intdir + '/' + pair + '/cor_lk.r4.xml')
    cor = np.copy(corimg.memMap()[:,:,0])
    cor[cor==0]=np.nan
    msk2 = np.zeros(msk.shape)
    msk2[np.where((gam>gamthresh) & (cor>0.35))] = 1
    # convert low gamma areas to nans
    ifg_real[msk2==0] = np.nan
    ifg_real[msk2==0] = np.nan
    
    # Do the filtering

    
    # astropy's convolution replaces the NaN pixels with a kernel-weighted
    # interpolation from their neighbors
    astropy_conv_r = convolve(ifg_real, kernel)
    astropy_conv_i = convolve(ifg_imag, kernel)
    
    #Now add back in the good data
    astropy_conv_r[msk2==1] = ifg_real[msk2==1]
    astropy_conv_i[msk2==1] = ifg_imag[msk2==1]
    
    ifg_filt = astropy_conv_i*1j + astropy_conv_r
    ifg_filt[np.isnan(ifg_filt)] = 0

    out1 = ifgimg.copy('read')
    out1.filename = ps.intdir +'/' + pair + '/filt.int'
    out1.dump(out1.filename + '.xml') # Write out xml
    ifg_filt.tofile(out1.filename) # Write file out
    out1.renderHdr()
    out1.renderVRT()