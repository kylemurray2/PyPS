#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:54:40 2019
VV VH analysis for optimizing time series
@author: kdm95
"""

import numpy as np
import isceobj
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import glob
import cv2
import os
#from mroipac.filter.Filter import Filter
params = np.load('params.npy').item()
locals().update(params)
#params['slcdir'] = '/data/kdm95/Delta/p42/merged/SLC_VV'
#np.save('params.npy',params)
nxl= params['nxl']
nyl = params['nyl']
tsdir = params['tsdir']
dates = params['dates']

# Load the gamma0 file
f = params['tsdir'] + '/gamma0.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0= intImage.memMap()[:,:,0] 

vhdir = '/data/kdm95/Delta/p42/merged/SLC_VH'
vvdir = '/data/kdm95/Delta/p42/merged/SLC_VV'

# Run this to fix the file paths in the xml files if you changed the dir names (you probably did)
#for f in glob.glob(vhdir + '/*'):
#    os.system('fixImageXml.py -i ' + f + '/*.full -f' )
#for f in glob.glob(vvdir + '/*'):
#    os.system('fixImageXml.py -i ' + f + '/*.full -f' )
    
#ii=20 # start at date 20 just because
ymin=5000
ymax=7000
xmin=44000
xmax=46000


gamImage = isceobj.createSlcImage()
gamImage.load(tsdir + '/gamma0.int.xml')
gam = gamImage.memMap()[ymin:ymax,xmin:xmax,0]

vh_list = glob.glob(vhdir + '/*')
dates_vh = []
for d in vh_list:
    dates_vh.append(d[-8:])

vvvh_stack = np.empty((ymax-ymin,xmax-xmin,len(dates_vh)),dtype=np.complex64)
vvvh_stack_amp = np.empty((ymax-ymin,xmax-xmin,len(dates_vh)))

for ii in np.arange(0,len(dates_vh)-1):
    d1 = dates_vh[ii]
    d2 = dates_vh[ii+1]
    
    f1vv = vvdir +'/'+ d1 + '/' + d1 + '.slc.full'
    f2vv = vvdir +'/'+ d2 + '/' + d2 + '.slc.full'
    
    slcImage = isceobj.createSlcImage()
    slcImage.load(f1vv + '.xml')
    SLC1VV = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
    
    slcImage = isceobj.createSlcImage()
    slcImage.load(f2vv + '.xml')
    SLC2VV = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
    
    slcDiffVV = np.multiply(SLC1VV,np.conj(SLC2VV))
#    vvphs =np.arctan2(np.imag(slcDiffVV), np.real(slcDiffVV)).astype(np.float32)
#    phs1    = np.exp(1j*vvphs);
#    vvamp = abs(slcDiffVV)
#    vvamp/=vvamp.max()
    
    f1vh = vhdir +'/'+ d1 + '/' + d1 + '.slc.full'
    f2vh = vhdir +'/'+ d2 + '/' + d2 + '.slc.full'
    
    slcImage = isceobj.createSlcImage()
    slcImage.load(f1vh + '.xml')
    SLC1VH = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
    
    slcImage = isceobj.createSlcImage()
    slcImage.load(f2vh + '.xml')
    SLC2VH = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
    
    slcDiffVH = np.multiply(SLC1VH,np.conj(SLC2VH)) 
    
    vvvh = np.multiply(slcDiffVV,np.conj(slcDiffVH))
    
#    vvvh = np.multiply(SLC1VV, np.multiply(np.conj(SLC2VV), np.multiply(np.conj(SLC1VH),SLC2VH)))
    vvvh_stack[:,:,ii] = np.divide( vvvh, np.multiply(abs(vvvh), np.sqrt(np.sqrt(abs(vvvh)))))

#    vhphs =np.arctan2(np.imag(slcDiffVH), np.real(slcDiffVH)).astype(np.float32)
#    phs2    = np.exp(1j*vhphs);
#    vhamp = abs(slcDiffVH)
#    vhamp/=vhamp.max()
#    vvvh1_2 =np.multiply(slcDiffVV,np.conj(slcDiffVH))  
#    vvvh_stack[:,:,ii] = np.arctan2(np.imag(vvvh1_2), np.real(vvvh1_2)).astype(np.float32)
#    vvvh_stack[:,:,ii] = abs(np.angle(np.multiply(phs1,np.conj(phs2))))
#    vvvh_stack_amp[:,:,ii] = abs(vvamp-vhamp)

cpxm0=np.divide( np.nanmean(vvvh_stack,axis=2), np.nanmean(abs(vvvh_stack),axis=2) )
phsvvvh = np.arctan2( np.imag(cpxm0),  np.real(cpxm0)).astype(np.float32)

# Write to file
out = isceobj.createIntImage() # Copy the interferogram image from before
out.filename = params['tsdir'] + '/vvvh_dif'
out.width = xmax-xmin
out.length = ymax-ymin
out.dump(out.filename + '.xml') # Write out xml
cpxm0.tofile(out.filename) # Write file out

mask = np.zeros(cpxm0.shape)

ampStab = abs(cpxm0)
ampStab/=ampStab.max()

phsStab = 1-abs(phsvvvh)
phsStab+=abs(phsStab.min())
phsStab /= phsStab.max()

mask[np.where((phsStab>.5)&(ampStab>.5))] = 1

Stab = np.add(ampStab,phsStab)
Stab /= Stab.max()
plt.figure();plt.imshow(mask);plt.title('VVVH stability mask')
plt.figure();plt.imshow(phsStab);plt.title('VVVH Diff phase')
plt.figure();plt.imshow(ampStab);plt.title('VVVH Diff amp')
plt.figure();plt.imshow(Stab);plt.title('VVVH Stab')

plt.figure();plt.imshow(gam);plt.title('gamma0 phase stability')


phs_stability = np.nansum(vvvh_stack,axis = 2)
phs_stability /= phs_stability.max()
phs_med = np.nanmedian(phs_stability.flatten())
phs_std = np.nanstd(phs_stability.flatten())
vmin = phs_med-phs_std
vmax = phs_med+phs_std
plt.figure();plt.imshow(phs_stability,vmin=vmin,vmax=vmax)

amp_stability = np.nansum(vvvh_stack_amp,axis = 2)
amp_stability += abs(amp_stability.min())
amp_stability /= amp_stability.max()
amp_med = np.nanmedian(amp_stability.flatten())
amp_std = np.nanstd(amp_stability.flatten())
vmin = amp_med-amp_std
vmax = amp_med+amp_std
plt.figure();plt.imshow(amp_stability,vmin=vmin,vmax=vmax)
plt.figure();plt.imshow(gam)
#plt.figure();plt.imshow(vvvh_stack[:,:,6])


#vvphs =np.arctan2(np.imag(slcDiffVV), np.real(slcDiffVV)).astype(np.float32)
#vhphs =np.arctan2(np.imag(slcDiffVH), np.real(slcDiffVH)).astype(np.float32)
#plt.figure();plt.imshow(vvphs-vhphs)


#
#d1 =dates[ii]
#d2 =dates[ii+1]
#d3 =dates[ii+2]
#d4 =dates[ii+3]
# 
#f1 = vvdir +'/'+ d1 + '/' + d1 + '.slc.full'
#f2 = vvdir +'/'+ d2 + '/' + d2 + '.slc.full'
#f3 = vvdir +'/'+ d3 + '/' + d3 + '.slc.full'
#f4 = vvdir +'/'+ d4 + '/' + d4 + '.slc.full'
#f5 = vhdir +'/'+ d1 + '/' + d1 + '.slc.full'
#f6 = vhdir +'/'+ d2 + '/' + d2 + '.slc.full'
#f7 = vhdir +'/'+ d3 + '/' + d3 + '.slc.full'
#f8 = vhdir +'/'+ d4 + '/' + d4 + '.slc.full'
#
#
#slcImage = isceobj.createSlcImage()
#slcImage.load(f1 + '.xml')
#SLC1VV = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
#slcImage = isceobj.createSlcImage()
#slcImage.load(f2 + '.xml')
#SLC2VV = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
#slcImage = isceobj.createSlcImage()
#slcImage.load(f3 + '.xml')
#SLC3VV = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
#slcImage = isceobj.createSlcImage()
#slcImage.load(f4 + '.xml')
#SLC4VV = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
#slcImage = isceobj.createSlcImage()
#slcImage.load(f5 + '.xml')
#SLC1VH = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
#slcImage = isceobj.createSlcImage()
#slcImage.load(f6 + '.xml')
#SLC2VH = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
#slcImage = isceobj.createSlcImage()
#slcImage.load(f7 + '.xml')
#SLC3VH = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
#slcImage = isceobj.createSlcImage()
#slcImage.load(f8 + '.xml')
#SLC4VH = slcImage.memMap()[ymin:ymax,xmin:xmax,0]
#
#
#ifg1_2_VV_VV = np.multiply(SLC1VV,np.conj(SLC2VV))
#ifg1_2_VH_VH = np.multiply(SLC1VH,np.conj(SLC2VH))
#o= np.multiply(ifg1_2_VV_VV,np.conj(ifg1_2_VH_VH))
#ifg1_3_VV_VV = np.multiply(SLC1VV,np.conj(SLC3VV))
#ifg1_4_VV_VV = np.multiply(SLC1VV,np.conj(SLC4VV))
#del(SLC2VV,SLC3VV,SLC4VV)
#ifg1_1_VV_VH = np.multiply(SLC1VV,np.conj(SLC1VH))
#
#p = np.arctan2(np.imag(a), np.real(a)).astype(np.float32)
#plt.figure();plt.imshow(p)
##ifg1_2_VV_VH = np.multiply(SLC1VV,np.conj(SLC2VH))
##ifg1_3_VV_VH = np.multiply(SLC1VV,np.conj(SLC3VH))
##ifg1_4_VV_VH = np.multiply(SLC1VV,np.conj(SLC4VH))
#ifg1_2_VH_VH = np.multiply(SLC1VH,np.conj(SLC2VH))
#ifg1_3_VH_VH = np.multiply(SLC1VH,np.conj(SLC3VH))
#ifg1_4_VH_VH = np.multiply(SLC1VH,np.conj(SLC4VH))
#del(SLC1VH,SLC2VH,SLC3VH,SLC4VH)
#
#dif1 = np.multiply(ifg1_2_VV_VV,np.conj(ifg1_2_VH_VH))
#ph1 = np.arctan2(np.imag(dif1), np.real(dif1)).astype(np.float32)
#del(dif1,ifg1_2_VV_VV,ifg1_2_VH_VH)
#dif2 = np.multiply(ifg1_3_VV_VV,np.conj(ifg1_3_VH_VH))
#ph2 = np.arctan2(np.imag(dif2), np.real(dif2)).astype(np.float32)
#del(dif2,ifg1_3_VV_VV,ifg1_3_VH_VH)
#dif3 = np.multiply(ifg1_4_VV_VV,np.conj(ifg1_4_VH_VH))
#ph3 = np.arctan2(np.imag(dif3), np.real(dif3)).astype(np.float32)
#del(dif3,ifg1_4_VV_VV,ifg1_4_VH_VH)
#
#plt.figure();plt.imshow(ph1)
#plt.figure();plt.imshow(ph2)
#plt.figure();plt.imshow(ph3)
#
#
#
#
#
