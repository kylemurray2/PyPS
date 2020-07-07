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
from mpl_toolkits.basemap import Basemap
import cv2
import os
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve

#from mroipac.filter.Filter import Filter
params = np.load('params.npy',allow_pickle=True).item()
locals().update(params)
geom = np.load('geom.npy',allow_pickle=True).item()
gam = np.load('gam.npy')
gam[gam==0] = np.nan
gamthresh = np.nanmean(gam) - np.nanstd(gam)
msk = np.zeros(gam.shape)
msk[gam>gamthresh] = 1
plt.figure();plt.imshow(msk)
# Load ifg
pairID = 1
pair=pairs[pairID]

kernel = Gaussian2DKernel(x_stddev=2)
    
for pair in pairs:
    ifgimg = isceobj.createIntImage()
    ifgimg.load(params['intdir'] + '/' + pair + '/fine_lk.int.xml')
    ifg_real = np.copy(np.real( ifgimg.memMap()[:,:,0]))
    ifg_imag = np.copy(np.imag( ifgimg.memMap()[:,:,0]))
    
    # convert low gamma areas to nans
    ifg_real[gam<gamthresh] = np.nan
    ifg_real[gam<gamthresh] = np.nan
    
    # Do the filtering

    
    # astropy's convolution replaces the NaN pixels with a kernel-weighted
    # interpolation from their neighbors
    astropy_conv_r = convolve(ifg_real, kernel)
    astropy_conv_i = convolve(ifg_imag, kernel)
    
    ifg_filt = astropy_conv_i*1j + astropy_conv_r
    
    out1 = ifgimg.copy('read')
    out1.filename = params['intdir'] +'/' + pair + '/filt.int'
    out1.dump(out1.filename + '.xml') # Write out xml
    ifg_filt.tofile(out1.filename) # Write file out
    out1.renderHdr()
    out1.renderVRT()
