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
import cv2
import os
import timeit
import glob as glob
import util

filterFlag = True
unwrap = True
filterStrength = '.3'
fixImage = False

nblocks = 1

params = np.load('params.npy',allow_pickle=True).item()
locals().update(params)
geom = np.load('geom.npy',allow_pickle=True).item()
seaLevel=-10
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

# for p in pairs:
#     os.system('rm -r ' + intdir + '/' + p)

if fixImage:
    slcList = glob.glob(slcdir + '/*/*full')
    for fname in slcList:
        os.system('fixImageXml.py -i ' + fname + ' -f')

# Load the gamma0 file
f = params['tsdir'] + '/gamma0.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')

gamma0= intImage.memMap()[:,:,0] 
gamma0=gamma0.copy() # mmap is readonly, so we need to copy it.
gamma0[np.isnan(gamma0)] = 0


if not os.path.isfile('gam.npy'):
    # Perform smart_looks first on gamma0
    # gam=gamma0.copy() # mmap is readonly, so we need to copy it.
    #gam[np.where(gam==0)]=np.nan
    gam = cv2.filter2D(gamma0,-1, win)
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
    # gam[geom['hgt_ifg'] < seaLevel] = 0
    np.save('gam.npy',gam)
    # del(gam)
else: 
    print('gam.npy already exists')



if not os.path.isdir(params['intdir']):
    os.system('mkdir ' + params['intdir'])

gam_filt = cv2.filter2D(gamma0,-1, win)

pair = params['pairs2'][0]

for pair in params['pairs2']: #loop through each ifg and save to 
    if not os.path.isdir(params['intdir'] + '/' + pair):
        print(pair)
        os.system('mkdir ' + params['intdir']+ '/' + pair)
    if not os.path.isfile(params['intdir'] + '/' + pair + '/fine_lk.int'):
        print('working on ' + pair)
        
        starttime = timeit.default_timer()
        #Open a file to save stuff to
        out = isceobj.createImage() # Copy the interferogram image from before
        out.dataType = 'CFLOAT'
        out.filename = params['intdir'] + '/' + pair + '/fine_lk.int'
        out.width = params['nxl']
        out.length = params['nyl']
        out.dump(out.filename + '.xml') # Write out xml
        fid=open(out.filename,"ab+")
        
        # open a cor file too
        outc = isceobj.createImage() # Copy the interferogram image from before
        outc.dataType = 'FLOAT'
        outc.filename = params['intdir'] + '/' + pair + '/cor_lk.r4'
        outc.width = params['nxl']
        outc.length = params['nyl']
        outc.dump(outc.filename + '.xml') # Write out xml
        fidc=open(outc.filename,"ab+")
        
        # break it into blocks
        for kk in np.arange(0,nblocks):
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
            slc1 = slcImage.memMap()[cropymin:cropymax,cropxmin:cropxmax,0][start:stop,:]

            f = params['slcdir'] +'/'+ d2 + '/' + d2 + '.slc.full'
    #        os.system('fixImageXml.py -i ' + f + ' -f')
    
            slcImage = isceobj.createSlcImage()
            slcImage.load(f + '.xml')
            slc2 = slcImage.memMap()[cropymin:cropymax,cropxmin:cropxmax,0][start:stop,:]
            ifg = np.multiply(slc1,np.conj(slc2))
            
            # del(slc1,slc2)

            # multiply by gamma0 (weight the values)
            ifg_real = np.real(ifg) * gamma0[start:stop,:]
            ifg_imag = np.imag(ifg) * gamma0[start:stop,:]
            

            # del(ifg)
            
            # simple filter of the ifg just like we did with the gamma0 file (now msk_filt)
            # This is the mean phase of pixels in each box
            ifg_real_filt = cv2.filter2D(ifg_real,-1, win)
            # now reverse the multiplication we did by dividing by the msk_filt
            #   and take the center pixel from each box
            rea_lk = np.reshape((ifg_real_filt/gam_filt[start:stop,:])[y,x],(idl,params['nxl']))    
           
            ifg_imag_filt = cv2.filter2D(ifg_imag,-1, win)
            ima_lk = np.reshape((ifg_imag_filt/gam_filt[start:stop,:])[y,x],(idl,params['nxl']))
            
            # del(ifg_imag_filt,ifg_imag,ifg_real_filt,ifg_real)
    
            cpx = ima_lk*1j + rea_lk
            cpx[np.isnan(cpx)] = 0
#            cpx[geom['hgt_ifg'] < seaLevel] = 0
#            cpx.tofile(of) # Write file out
            fid.write(cpx)
            
            cor_lk = np.log(  np.abs(  (rea_lk+(1j*ima_lk)).astype(np.complex64)) )
            # cor_lk /= cor_lk[~np.isnan(cor_lk)].max()
            cor_lk[np.isinf(cor_lk)] = 0
            cor_lk[np.isnan(cor_lk)] = 0
#            cor_lk[geom['hgt_ifg'] < seaLevel] = 0
            fidc.write(cor_lk)
        
        out.renderHdr()
        out.renderVRT()  
        outc.renderHdr()
        outc.renderVRT() 
        fid.close()
        fidc.close()
    # for pair in params['pairs2']: #loop through each ifg and save to 
        if filterFlag:
            name = params['intdir'] + '/' + pair + '/fine_lk.int'
            corname = params['intdir'] + '/' + pair + '/cor.r4'
            offilt =  params['intdir'] + '/' + pair + '/fine_lk_filt.int'
            command = 'python /home/km/Software/test/isce2/contrib/stack/topsStack/FilterAndCoherence.py -c ' + corname + ' -i ' + name + ' -f ' +  offilt + ' -s ' + filterStrength + ' > log'
            os.system(command)
            # if unwrap:
            #     unwName = params['intdir']+ '/' + pair + '/filt.unw'
            #     util.unwrap_snaphu(name,corname,unwName,nyl,nxl)
        # print("The time difference is :", timeit.default_timer() - starttime)      

# plt.figure();plt.imshow(np.angle(cpx),cmap='hsv')        

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




