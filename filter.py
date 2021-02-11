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

#from mroipac.filter.Filter import Filter
params = np.load('params.npy',allow_pickle=True).item()
locals().update(params)
geom = np.load('geom.npy',allow_pickle=True).item()
locals().update(geom)

gam = np.load('gam.npy')
gam[gam==0] = np.nan
gamthresh =.48#np.nanmedian(gam) #- np.nanstd(gam)
msk = np.zeros(gam.shape)
msk[gam>gamthresh] = 1
msk[hgt_ifg<16] = 0
plt.figure();plt.imshow(msk);plt.show()
# Load ifg
pairID = 0
pair=pairs[pairID]
kernel = Gaussian2DKernel(x_stddev=1)
    
for pair in pairs:

    
    ifgimg = isceobj.createIntImage()
    ifgimg.load(params['intdir'] + '/' + pair + '/fine_lk.int.xml')
    ifg_real = np.copy(np.real( ifgimg.memMap()[:,:,0]))
    ifg_imag = np.copy(np.imag( ifgimg.memMap()[:,:,0]))
    
    corimg = isceobj.createImage()
    corimg.load(params['intdir'] + '/' + pair + '/cor_lk.r4.xml')
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
    out1.filename = params['intdir'] +'/' + pair + '/filt.int'
    out1.dump(out1.filename + '.xml') # Write out xml
    ifg_filt.tofile(out1.filename) # Write file out
    out1.renderHdr()
    out1.renderVRT()

# from scipy import fftpack
# im_fft = fftpack.fft2(unw)
# # In the lines following, we'll make a copy of the original spectrum and
# # truncate coefficients.
# # Define the fraction of coefficients (in each direction) we keep
# keep_fraction = 0.001
# # Call ff a copy of the original transform. Numpy arrays have a copy
# # method for this purpose.
# im_fft2 = im_fft.copy()
# # Set to zero all rows with indices between r*keep_fraction and
# # r*(1-keep_fraction):
# im_fft2[int(nyl*keep_fraction):int(nyl*(1-keep_fraction))] = 0
# # Similarly with the columns:
# im_fft2[:, int(nxl*keep_fraction):int(nxl*(1-keep_fraction))] = 0
# im_new = fftpack.ifft2(im_fft2).real
# fig,ax = plt.subplots(1,3)
# ax[0].imshow(unw)
# ax[1].imshow(im_new)
# ax[2].imshow(unw-im_new)
# keep_fraction = 0.005
# stack = []
# for p in params['pairs']:
#     unw_file = params['intdir'] + '/' + p + '/filt.unw'
#     unwImage = isceobj.createIntImage()
#     unwImage.load(unw_file + '.xml')
#     # unw = unwImage.memMap()[:,:,0] - np.nanmean(unwImage.memMap()[:,:,0][msk==1])
#     unw = unwImage.memMap()[:,:,0]
#     im_fft = fftpack.fft2(unw)
#     im_fft2 = im_fft.copy()
#     im_fft2[int(nyl*keep_fraction):int(nyl*(1-keep_fraction))] = 0
#     im_fft2[:, int(nxl*keep_fraction):int(nxl*(1-keep_fraction))] = 0
#     unwfilt = unw -fftpack.ifft2(im_fft2).real
#     unwfilt = unwfilt-unwfilt[r,c]
#     stack.append(unwfilt)
# # stack.append(np.zeros(unw.shape))
# stack = np.asarray(stack,dtype=np.float32)

# from astropy.convolution import convolve
# dim = 31
# x, y = np.meshgrid(np.linspace(-1,1,dim), np.linspace(-1,1,dim))
# d = np.sqrt(x*x+y*y)
# sigma, mu = 30, 0.0
# g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
# filt = convolve(unw,g)
# fig,ax = plt.subplots(1,3)
# ax[0].imshow(unw)
# ax[1].imshow(filt)
# ax[2].imshow(unw-filt)
