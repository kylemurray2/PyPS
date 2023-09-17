#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:16:54 2018

Distributed scatterer type approach

Downlooking, filtering, and coherence
Saves downlooked ifgs in the respective ifg directories.

Note:
    when it makes ifgs, it's subtracting the secondary from the reference.  So 
       negative values are going away from the satellite (e.g., subsidence) and
       positive values are going towards the satellite (e.g., uplift).  

@author: kdm95
"""

import isce.components.iscesys.Component.ProductManager as pm



import numpy as np
import isce.components.isceobj as isceobj
from matplotlib import pyplot as plt
import cv2   
import os
import timeit
import glob as glob
import util
from util import show
import FilterAndCoherence


filterFlag      = True
unwrap          = False # Usually better to leave False and use runSnaphu.py for more options and outputs
filterStrength  = '.3'
fixImage        = False  #Do this in case you renamed any of the directories or moved the SLCs since they were made
nblocks         = 1
seaLevel        = -10

ps = np.load('./ps.npy',allow_pickle=True).all()

# Creat window and downlooking vectors
win1 =np.ones((ps.alks,ps.rlks))
win=win1/sum(win1.flatten())
win = np.asarray(win,dtype=np.float32)
rangevec=np.arange(0,ps.nxl) * ps.rlks
azvec=np.arange(0,ps.nyl) * ps.alks
yy,xx= np.meshgrid(azvec,rangevec,sparse=False, indexing='ij')
y=yy.flatten()
x=xx.flatten()
del(xx,yy)

# for p in pairs:
#     os.system('rm -r ' + intdir + '/' + p)

if fixImage:
    slcList = glob.glob(ps.slcdir + '/*/*full')
    for fname in slcList:
        os.system('fixImageXml.py -i ' + fname + ' -f')

# Load the gamma0 file
# f = ps.tsdir + '/gamma0.int'
# intImage = isceobj.createIntImage()
# intImage.load(f + '.xml')
# gamma0= intImage.memMap()
gamma0= np.ones((ps.ny,ps.nx))

gamma0=gamma0.copy() # mmap is readonly, so we need to copy it.
gamma0[np.isnan(gamma0)] = 0



if not os.path.isfile('Npy/gam.npy'):
    # Perform smart_looks first on gamma0
    # gam=gamma0.copy() # mmap is readonly, so we need to copy it.
    #gam[np.where(gam==0)]=np.nan
    gam = cv2.filter2D(gamma0,-1, win)
    gam = np.reshape(gam[y,x],(ps.nyl,ps.nxl))
    gam[np.isnan(gam)] = 0
    # Save gamma0 file
    out = isceobj.createIntImage() # Copy the interferogram image from before
    out.dataType = 'FLOAT'
    out.filename = ps.tsdir + '/gamma0_lk.int'
    out.width = ps.nxl
    out.length = ps.nyl
    out.dump(out.filename + '.xml') # Write out xml
    gam.tofile(out.filename) # Write file out
    out.renderHdr()
    out.renderVRT()
    # gam[geom['hgt_ifg'] < seaLevel] = 0
    np.save('Npy/gam.npy',gam)
    # del(gam)
else: 
    print('gam.npy already exists')


if not os.path.isdir(ps.intdir):
    os.system('mkdir ' + ps.intdir)

# gam_filt = cv2.filter2D(gamma0,-1, win)

gamma0 = gamma0**2
rangevec=np.arange(0,ps.nxl) * ps.rlks
idl = int(np.floor(ps.nyl/nblocks))
idy = int(np.floor(ps.ny/nblocks))
azvec=np.arange(0,idl) * ps.alks
yy,xx= np.meshgrid(azvec,rangevec,sparse=False, indexing='ij')
y=yy.flatten()
x=xx.flatten() 

pair = ps.pairs2[0]
pair = '20211129_20211205'
for pair in ps.pairs2: #loop through each ifg and save to 
    if not os.path.isdir(ps.intdir + '/' + pair):
        os.system('mkdir ' + ps.intdir+ '/' + pair)
    if not os.path.isfile(ps.intdir + '/' + pair + '/fine_lk.int'):
        print('working on ' + pair)
        
        starttime = timeit.default_timer()
        #Open a file to save stuff to
        out = isceobj.createImage() # Copy the interferogram image from before
        out.dataType = 'CFLOAT'
        out.filename = ps.intdir + '/' + pair + '/fine_lk.int'
        out.width = ps.nxl
        out.length = ps.nyl
        out.dump(out.filename + '.xml') # Write out xml
        fid=open(out.filename,"ab+")
        
        # # open a cor file too
        # outc = isceobj.createImage() # Copy the interferogram image from before
        # outc.dataType = 'FLOAT'
        # outc.filename = ps.intdir + '/' + pair + '/cor_lk.r4'
        # outc.width = ps.nxl
        # outc.length = ps.nyl
        # outc.dump(outc.filename + '.xml') # Write out xml
        # fidc=open(outc.filename,"ab+")
        
        
        # break it into blocks
        for kk in np.arange(0,nblocks):

            start = int(kk*idy)
            stop = start+idy+1

            d2 = pair[9:]
            d = pair[0:8]

            if ps.crop:
                f1 = ps.slcdir +'/'+ d + '/' + d + '.slc.full.crop'    
            else:
                f1 = ps.slcdir +'/'+ d + '/' + d + '.slc.full'              
            slcImage = isceobj.createSlcImage()
            slcImage.load(f1 + '.xml')
            slc1 = slcImage.memMap()[:,:,0][start:stop,:]
            
            if ps.crop:
                f2 = ps.slcdir +'/'+ d2 + '/' + d2 + '.slc.full.crop'    
            else:
                f2 = ps.slcdir +'/'+ d2 + '/' + d2 + '.slc.full'    
            slcImage = isceobj.createSlcImage()
            slcImage.load(f2 + '.xml')
            slc2 = slcImage.memMap()[:,:,0][start:stop,:]
            ifg = np.multiply(slc1,np.conj(slc2))
            
            cohFile = ps.intdir + '/' + pair + '/coh.coh'
            FilterAndCoherence.estCpxCoherence(f1, f2,cohFile, alks=1, rlks=1)
            
            # del(slc1,slc2)
            
            ifg_real = np.real(ifg) * gamma0[start:stop,:]
            ifg_imag = np.imag(ifg) * gamma0[start:stop,:]
            
            # del(ifg)

            ifg_real_filt = cv2.filter2D(ifg_real,-1, win)
            ifg_imag_filt = cv2.filter2D(ifg_imag,-1, win)
            rea_lk = np.reshape((ifg_real_filt)[y,x],(idl,ps.nxl))    
            ima_lk = np.reshape((ifg_imag_filt)[y,x],(idl,ps.nxl))
            
            # del(ifg_imag_filt,ifg_imag,ifg_real_filt,ifg_real)
    
            cpx = ima_lk*1j + rea_lk
            cpx[np.isnan(cpx)] = 0
            fid.write(cpx)
            
            # slc1_F = cv2.filter2D(abs(slc1**2),-1, win) # using win instead of win1 because cpx is an average, not a sum
            # slc2_F = cv2.filter2D(abs(slc2**2),-1, win)
            # denom = np.sqrt(slc1_F[y,x]) * np.sqrt(slc2_F[y,x])
            # cor=(abs(cpx.ravel())/denom).reshape(ps.nyl,ps.nxl)
            # fidc.write(cor)
        
        out.renderHdr()
        out.renderVRT()  
        # outc.renderHdr()
        # outc.renderVRT() 
        fid.close()
        # fidc.close()
    # for pair in ps.pairs2: #loop through each ifg and save to 
        if filterFlag:
            name = ps.intdir + '/' + pair + '/fine_lk.int'
            corname = ps.intdir + '/' + pair + '/cor.r4'
            offilt =  ps.intdir + '/' + pair + '/fine_lk_filt.int'
            # FilterAndCoherence.runFilter(name,offilt,float(filterStrength))
            FilterAndCoherence.estCoherence(name, corname)
            if unwrap:
                unwName = ps.intdir+ '/' + pair + '/filt.unw'
                util.unwrap_snaphu(name,corname,unwName,ps.nyl,ps.nxl)



# a = 1+0j  # angle: 0
# b = 0+1j  # angle: +90
# c=np.multiply(a,np.conj(b)) # a - b = c 
# d =np.angle(c) # a - b =  -90
# If LOS shortened between a and b, then the difference should be positive (uplift)
# If LOS lengthened, then the difference should be negative (subsidence)
# For ifgs, MintPY uses the opposite: ("positive value represents motion away from the satellite."),
#  but for the MintPy time series they use positive is uplift.  
# from osgeo import gdal
# p1='20211129_20211205'
# p2 = '20211205_20211211' # This has the flood
# ds1 = gdal.Open(ps.intdir+'/'+p1+'/coh.coh.vrt')
# preflood = ds1.GetVirtualMemArray()
# ds2 = gdal.Open(ps.intdir+'/'+p2+'/coh.coh.vrt')
# flood = ds2.GetVirtualMemArray()

# diff = preflood[1,:,:]-flood[1,:,:]
# plt.figure()
# plt.imshow(diff[2000:4000,10000:20000],vmin=.1,vmax=.4)