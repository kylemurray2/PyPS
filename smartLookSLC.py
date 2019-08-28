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

filterFlag = True
filterStrength = '0.6'

nblocks = 40

#from mroipac.filter.Filter import Filter
params = np.load('params.npy',allow_pickle=True).item()
locals().update(params)
geom = np.load('geom.npy',allow_pickle=True).item()
seaLevel=-10
#locals().update(geom)
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
del(xx,yy)

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
    gam[geom['hgt_ifg'] < seaLevel] = 0
    np.save('gam.npy',gam)
    del(gam)
else: 
    print('gam.npy already exists')

if not os.path.isdir(params['intdir']):
    os.system('mkdir ' + params['intdir'])

msk_filt = cv2.filter2D(gamma0,-1, win)

pair = params['pairs'][0]

for pair in params['pairs']: #loop through each ifg and save to 
    if not os.path.isdir(params['intdir'] + '/' + pair):
        os.system('mkdir ' + params['intdir']+ '/' + pair)
    if not os.path.isfile(params['intdir'] + '/' + pair + '/fine_lk_filt.int'):
        print('working on ' + pair)
        
        #Open a file to save stuff to
        out = isceobj.createImage() # Copy the interferogram image from before
        out.dataType = 'CFLOAT'
        of = params['intdir'] + '/' + pair + '/fine_lk.int'
        out.filename = of
        out.width = params['nxl']
        out.length = params['nyl']
        out.dump(of + '.xml') # Write out xml
        fid=open(of,"ab+")
        # break it into 20 blocks
        for kk in np.arange(0,nblocks):
            print(str(kk))
            idy = int(np.floor(ny/nblocks))
            start = int(kk*idy)
            stop = start+idy+1
            rangevec=np.arange(0,nxl) * params['rlks']
            idl = int(np.floor(params['nyl']/nblocks))
            azvec=np.arange(0,idl) * params['alks']
            
            yy,xx= np.meshgrid(azvec,rangevec,sparse=False, indexing='ij')
            y=yy.flatten()
            x=xx.flatten()            
            d2 = pair[9:]
            d = pair[0:8]
            #load ifg real and imaginary parts
            f = params['slcdir'] +'/'+ d + '/' + d + '.slc.full'
    #        os.system('fixImageXml.py -i ' + f + ' -f')
    
            slcImage = isceobj.createSlcImage()
            slcImage.load(f + '.xml')
            slc1 = slcImage.memMap()[start:stop,:,0]
            f = params['slcdir'] +'/'+ d2 + '/' + d2 + '.slc.full'
    #        os.system('fixImageXml.py -i ' + f + ' -f')
    
            slcImage = isceobj.createSlcImage()
            slcImage.load(f + '.xml')
            slc2 = slcImage.memMap()[start:stop,:,0]
            ifg = np.multiply(slc1,np.conj(slc2))
            
            del(slc1,slc2)
            
            ifg_real = np.real(ifg)
            ifg_imag = np.imag(ifg)
            
            del(ifg)
        
            ifg_real_filt0 = cv2.filter2D(ifg_real,-1, win)
            ifg_real = ifg_real * gamma0[start:stop,:]
            ifg_real_filt = cv2.filter2D(ifg_real,-1, win)
            del(ifg_real)
            rea_lk = np.reshape((ifg_real_filt/msk_filt[start:stop,:])[y,x],(idl,params['nxl']))
            del(ifg_real_filt)
    
            ifg_imag_filt0 = cv2.filter2D(ifg_imag,-1, win)
            ifg_imag = ifg_imag * gamma0[start:stop,:]
            ifg_imag_filt = cv2.filter2D(ifg_imag,-1, win)
            del(ifg_imag)
            ima_lk = np.reshape((ifg_imag_filt/msk_filt[start:stop,:])[y,x],(idl,params['nxl']))
            del(ifg_imag_filt)
            
    #        phs_lk = np.arctan2(ima_lk, rea_lk)
    #        phs_lk[np.isnan(phs_lk)] = 0
    #        phs_lk[geom['hgt_ifg'] < seaLevel] = 0
            cpx = ima_lk*1j + rea_lk
            cpx[np.isnan(cpx)] = 0
#            cpx[geom['hgt_ifg'] < seaLevel] = 0
                    # Save downlooked ifg
    
#            cpx.tofile(of) # Write file out
            fid.write(cpx)
        out.renderHdr()
        out.renderVRT() 
        fid.close()
        
        if filterFlag:
            offilt =  params['intdir'] + '/' + pair + '/fine_lk_filt.int'
            command = 'python /home/kdm95/Software/isce2/contrib/stack/topsStack/FilterAndCoherence.py -i ' + of + ' -f ' +  offilt + ' -s ' + filterStrength
            os.system(command)
            
        if not os.path.isfile(params['intdir'] + '/' + pair + '/cor_lk.r4'):
            cor_lk = np.log(  np.abs(  (rea_lk+(1j*ima_lk)).astype(np.complex64)) )
            cor_lk /= cor_lk[~np.isnan(cor_lk)].max()
            cor_lk[np.isinf(cor_lk)] = 0
            cor_lk[np.isnan(cor_lk)] = 0
            cor_lk[geom['hgt_ifg'] < seaLevel] = 0
            
            out = isceobj.createImage() # Copy the interferogram image from before
            out.dataType = 'FLOAT'
            out.filename = params['intdir'] + '/' + pair + '/cor_lk.r4'
            out.width = params['nxl']
            out.length = params['nyl']
            out.dump(params['intdir'] + '/' + pair + '/cor_lk.r4.xml') # Write out xml
            cor_lk.tofile(params['intdir'] + '/' + pair + '/cor_lk.r4') # Write file out
            
            out.renderHdr()
            out.renderVRT()  

# Downlook geom files this way too

# Get bounding coordinates (Frame)
#f_lon = mergeddir + '/geom_master/lon.rdr.full'
#f_lat = mergeddir + '/geom_master/lat.rdr.full'
#f_hgt = mergeddir + '/geom_master/hgt.rdr.full'
#
#Image = isceobj.createImage()
#Image.load(f_lon + '.xml')
#lon_ifg2 = Image.memMap()[:,:,0].copy().astype(np.float32)
#lon_lk2 = cv2.filter2D(lon_ifg2,-1, win)
#lonlk = np.reshape(lon_lk2[y,x],(params['nyl'],params['nxl']))
#
#
#Image = isceobj.createImage()
#Image.load(f_lat + '.xml')
#lat_ifg2 = Image.memMap()[:,:,0].copy().astype(np.float32)
#lat_lk2 = cv2.filter2D(lat_ifg2,-1, win)
#latlk = np.reshape(lat_lk2[y,x],(params['nyl'],params['nxl']))
#
#Image = isceobj.createImage()
#Image.load(f_hgt + '.xml')
#hgt_ifg2 = Image.memMap()[:,:,0].copy().astype(np.float32)
#hgt_lk2 = cv2.filter2D(hgt_ifg2,-1, win)
#hgtlk = np.reshape(hgt_lk2[y,x],(params['nyl'],params['nxl']))
#
#geom = {}
#geom['lon_ifg'] = lonlk
#geom['lat_ifg'] = latlk
#geom['hgt_ifg'] = hgtlk
#np.save('geom.npy',geom)
#        
