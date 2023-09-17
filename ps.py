#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:32:18 2022

Persistent Scatterer type approach

@author: km
"""

import numpy as np
import isce.components.isceobj as isceobj
from matplotlib import pyplot as plt
import cv2   
import os
import timeit
import time
import glob as glob
import util
from util import show
import skimage
minGam = .65

ps = np.load('./ps.npy',allow_pickle=True).all()

# Load the gamma0 file
f = ps.tsdir + '/gamma0.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0= intImage.memMap()[:,:,0]
gamma0=gamma0.copy() # mmap is readonly, so we need to copy it.
gamma0[np.isnan(gamma0)] = 0

msk = np.ones(gamma0.shape)
msk[gamma0>minGam] = 0
show(msk)

stack = []
ymin,ymax,xmin,xmax = 1200,2300,500,2000
mskCrop = msk[ymin:ymax,xmin:xmax]
for ii in range(len(ps.dates)-1):
    # load SLCS and make an ifg
    d=ps.dates[ii]
    d2 = ps.dates[ii+1]
    if ps.crop:
        
        f = ps.slcdir +'/'+ d + '/' + d + '.slc.full.crop'
    else:
        f = ps.slcdir +'/'+ d + '/' + d + '.slc.full'
    slcImage1 = isceobj.createSlcImage()
    slcImage1.load(f + '.xml')
    
    if ps.crop:
        f = ps.slcdir +'/'+ d2 + '/' + d2 + '.slc.full.crop'
    else:
        f = ps.slcdir +'/'+ d2 + '/' + d2 + '.slc.full'
    slcImage2 = isceobj.createSlcImage()
    slcImage2.load(f + '.xml')
    
    ifg = np.multiply(slcImage1.memMap()[ymin:ymax,xmin:xmax,0],np.conj(slcImage2.memMap()[ymin:ymax,xmin:xmax,0]))
    ifgMA = np.ma.masked_array(ifg,mask=mskCrop)
    stack.append(ifgMA)

stack = np.asarray(stack)
mskCropCube = np.expand_dims(mskCrop,2)

# Convert to masked array by extending the dimension of the mask and repeating
stackMA = np.ma.masked_array(stack,mask=np.tile(mskCropCube,(1,stack.shape[0])))


start_time=time.time()
samp_unw = skimage.restoration.unwrap_phase(np.angle(stack))
totalTime = time.time()-start_time
print('It took ' + str(np.round(totalTime,3)) + ' seconds.')
show(samp_unw[10,:,:])

from scipy.interpolate import griddata 




xx = np.arange(0,samp.shape[1])
yy = np.arange(0,samp.shape[0])
XX,YY = np.meshgrid(xx,yy)
ids = np.where(~np.isnan(samp))
reaGrid = griddata((XX[ids],YY[ids]),np.real(samp[ids]), (XX,YY), method='cubic')
imaGrid = griddata((XX[ids],YY[ids]),np.imag(samp[ids]), (XX,YY), method='cubic')
ifgGrid = cpx = imaGrid*1j + reaGrid

plt.figure();plt.imshow(np.angle(ifgGrid));plt.title('PS interpolated')

