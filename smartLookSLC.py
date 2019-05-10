#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:16:54 2018
DOWNLOOKING
Saves downlooked ifgs in the respective ifg directories.

FILTERING
work in progress

@author: kdm95
"""

import numpy as np
import isceobj
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import cv2
import os

sea = -1 # sealevel


#from mroipac.filter.Filter import Filter
params = np.load('params.npy',allow_pickle=True).item()
locals().update(params)
geom = np.load('geom.npy',allow_pickle=True).item()
locals().update(geom)
#params['slcdir'] = '/data/kdm95/Delta/p42/merged/SLC_VV'
#np.save('params.npy',params)
nxl= params['nxl']
nyl = params['nyl']
tsdir = params['tsdir']

# Creat window and downlooking vectors
win1 =np.ones((params['alks'],params['rlks']))
win=win1/sum(win1.flatten())
rangevec=np.arange(0,nxl) * params['rlks']
azvec=np.arange(0,params['nyl']) * params['alks']
yy,xx= np.meshgrid(azvec,rangevec,sparse=False, indexing='ij')
y=yy.flatten()
x=xx.flatten()

# Load the gamma0 file
f = params['tsdir'] + '/gamma0.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0= intImage.memMap()[:,:,0] 

if not os.path.isfile('gam.npy'):
    # Perform smart_looks first on gamma0
    gam=gamma0.copy() # mmap is readonly, so we need to copy it.
    #gam[np.where(gam==0)]=np.nan
    gam = cv2.filter2D(gam,-1, win)
    gam = np.reshape(gam[y,x],(nyl,nxl))
    gam[np.isnan(gam)] = 0
    # Save gamma0 file
    out = isceobj.createIntImage() # Copy the interferogram image from before
    out.dataType = 'FLOAT'
    out.filename = params['tsdir'] + '/gamma0_lk.int'
    out.width = nxl
    out.length = nyl
    out.dump(out.filename + '.xml') # Write out xml
    gam.tofile(out.filename) # Write file out
    out.renderHdr()
    out.renderVRT()
    gam[geom['hgt_ifg'] < sea] = 0
    np.save('gam.npy',gam)
    del(gam)
else: 
    print('gam.npy already exists')

if not os.path.isdir(params['intdir']):
    os.system('mkdir ' + params['intdir'])

for pair in params['pairs']: #loop through each ifg and save to 
    if not os.path.isdir(params['intdir'] + '/' + pair):
        os.system('mkdir ' + params['intdir']+ '/' + pair)
    if not os.path.isfile(params['intdir'] + '/' + pair + '/fine_lk.int'):
        print('working on ' + pair)
        d2 = pair[9:]
        d = pair[0:8]
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
        
        del(slc1,slc2)
        
        ifg_real = np.real(ifg)
        ifg_imag = np.imag(ifg)
        
        del(ifg)
    
        ifg_real_filt0 = cv2.filter2D(ifg_real,-1, win)
        ifg_imag_filt0 = cv2.filter2D(ifg_imag,-1, win)
        ifg_real = ifg_real * gamma0
        ifg_imag = ifg_imag * gamma0
        ifg_real_filt = cv2.filter2D(ifg_real,-1, win)
        ifg_imag_filt = cv2.filter2D(ifg_imag,-1, win)
        
        del(ifg_real,ifg_imag)
        
        rea_lk = np.reshape(ifg_real_filt[y,x],(params['nyl'],params['nxl']))
        ima_lk = np.reshape(ifg_imag_filt[y,x],(params['nyl'],params['nxl']))
        
        del(ifg_real_filt,ifg_imag_filt)
        
        phs_lk = np.arctan2(ima_lk, rea_lk)
        phs_lk[np.isnan(phs_lk)] = 0
        phs_lk[geom['hgt_ifg'] < sea] = 0
        
                # Save downlooked ifg
        out = isceobj.createImage() # Copy the interferogram image from before
        out.dataType = 'FLOAT'
        out.filename = params['intdir'] + '/' + pair + '/fine_lk.r4'
        out.width = params['nxl']
        out.length = params['nyl']
        out.dump(params['intdir'] + '/' + pair + '/fine_lk.r4.xml') # Write out xml
        phs_lk.tofile(params['intdir'] + '/' + pair + '/fine_lk.r4') # Write file out
        out.renderHdr()
        out.renderVRT() 
        
        
    if not os.path.isfile(params['intdir'] + '/' + pair + '/cor_lk.int'):
        cor_lk = np.log(  np.abs(  (rea_lk+(1j*ima_lk)).astype(np.complex64)) )
        cor_lk /= cor_lk[~np.isnan(cor_lk)].max()
        cor_lk[np.isinf(cor_lk)] = 0
        cor_lk[np.isnan(cor_lk)] = 0
        cor_lk[geom['hgt_ifg'] < sea] = 0
        
        out = isceobj.createImage() # Copy the interferogram image from before
        out.dataType = 'FLOAT'
        out.filename = params['intdir'] + '/' + pair + '/cor_lk.r4'
        out.width = params['nxl']
        out.length = params['nyl']
        out.dump(params['intdir'] + '/' + pair + '/cor_lk.r4.xml') # Write out xml
        cor_lk.tofile(params['intdir'] + '/' + pair + '/cor_lk.r4') # Write file out
        
        out.renderHdr()
        out.renderVRT()  