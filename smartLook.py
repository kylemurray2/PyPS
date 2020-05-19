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
#from mroipac.filter.Filter import Filter
params = np.load('params.npy',allow_pickle=True).item()
locals().update(params)

f = tsdir + '/gamma0.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0= intImage.memMap()[:,:,0] 

# Creat window and downlooking vectors
win =np.ones((alks,rlks))
win=win/sum(win.flatten())
rangevec=np.arange(0,nxl) * rlks
azvec=np.arange(0,nyl) * alks
yy,xx= np.meshgrid(azvec,rangevec,sparse=False, indexing='ij')
y=yy.flatten()
x=xx.flatten()

# Perform smart_looks first on gamma0
gam=gamma0.copy() # mmap is readonly, so we need to copy it.
#gam[np.where(gam==0)]=np.nan
gam = cv2.filter2D(gam,-1, win)
gam = np.reshape(gam[y,x],(nyl,nxl))
gam[np.isnan(gam)] = 0
# Save gamma0 file
out = isceobj.createIntImage() # Copy the interferogram image from before
out.dataType = 'FLOAT'
out.filename = tsdir + '/gamma0_lk.int'
out.width = nxl
out.length = nyl
out.dump(tsdir + '/gamma0_lk.int.xml') # Write out xml
gam.tofile(tsdir + '/gamma0_lk.int') # Write file out
out.renderHdr()
out.renderVRT()

np.save('gam.npy',gam)

gamma_thresh = .2
rx=3
ry=3

gausx = np.exp( np.divide( -np.square(np.arange(-rx,rx)), np.square(rx)));
gausy = np.exp( np.divide( -np.square(np.arange(-ry,ry)), np.square(ry)));
gaus = gausx[:, np.newaxis] * gausy[np.newaxis, :]
gaus = gaus-gaus.min()
gaus  = gaus/np.sum(gaus.flatten())

msk_filt = cv2.filter2D(gamma0,-1, win)


for pair in pairs: #loop through each ifg and save to 
    f = intdir + '/' + pair + '/fine.int'
    if not os.path.isfile(f):
        intImage = isceobj.createIntImage()
        intImage.load(f + '.xml')
        ifg_real = np.real(intImage.memMap()[:,:,0] )
        ifg_imag = np.imag(intImage.memMap()[:,:,0] )
        ifg_real=ifg_real.copy() # mmap is readonly, so we need to copy it.
        ifg_imag=ifg_imag.copy()
    #    phs = np.angle(ifg_real+(1j*ifg_imag))
    
        ifg_real_filt0 = cv2.filter2D(ifg_real,-1, win)
        ifg_imag_filt0 = cv2.filter2D(ifg_imag,-1, win)
        ifg_real = ifg_real * gamma0
        ifg_imag = ifg_imag * gamma0
        ifg_real_filt = cv2.filter2D(ifg_real,-1, win)
        ifg_imag_filt = cv2.filter2D(ifg_imag,-1, win)
        
        rea_lk = np.reshape(ifg_real_filt/msk_filt[y,x],(nyl,nxl))
        ima_lk = np.reshape(ifg_imag_filt/msk_filt[y,x],(nyl,nxl))
    #    phs_lk1 = np.arctan2(ima_lk, rea_lk)
        phs_lk1 = (rea_lk+(1j*ima_lk)).astype(np.complex64)
    
        phs_lk1[np.isnan(phs_lk1)]=0
        # DO PS INTERP_________________________________________________
        
    #    # Mask ones where the data is good
    #    mask = np.ones(rea_lk.shape)
    #    mask[np.where(gam < gamma_thresh)]=0
    #   
    ##    mask[np.isnan(mask)]=0 #******************** get rid of the boxes?
    #    
    #    # Zero bad data
    ##    rea_lk[np.where(mask==0)]=0
    ##    ima_lk[np.where(mask==0)]=0
    #
    #    # Smooth everything into zero space
    #    mask_f = cv2.filter2D(mask,-1, gaus)
    #    rea_f = cv2.filter2D(rea_lk,-1, gaus)
    #    ima_f = cv2.filter2D(ima_lk,-1, gaus)
    #    
    #    # Divide by mask. This is how we care for nan values
    #    rea = rea_f/mask_f
    #    ima = ima_f/mask_f
    #    # Add the original data back
    #    #    First make the good data areas 0
    ##    rea[np.where(gam > gamma_thresh)]=0
    ##    ima[np.where(gam > gamma_thresh)]=0
    #    #    Then add the original data back (this has zeros where bad data)
    #    rea += rea_lk
    #    ima += ima_lk
    ##    phs_lk = np.arctan2(ima, rea).astype(np.float32)
    #    phs_lk = (rea+(1j*ima)).astype(np.complex64)
    ##    phs_lk_2 = np.angle(rea+(1j*ima))
    #
    #    phs_lk[np.isnan(phs_lk)]=0
        
    #    p=phs_lk-phs_lk_2
    #    plt.imshow(mask)
    #    plt.figure()
    #    plt.imshow(phs_lk1)
    #    plt.figure()
    #    plt.imshow(phs_lk)
    #    plt.figure()
    #    plt.imshow(phs_lk_interp)
    #    plt.figure()
    #    plt.imshow(phs_lk_2)
    
        # Save downlooked ifg
        out = isceobj.createIntImage() # Copy the interferogram image from before
        out.dataType = 'CFLOAT'
        out.filename = intdir + '/' + pair + '/fine_lk.int'
        out.width = nxl
        out.length = nyl
        out.dump(intdir + '/' + pair + '/fine_lk.int.xml') # Write out xml
        phs_lk1.tofile(intdir + '/' + pair + '/fine_lk.int') # Write file out
        out.renderHdr()
        out.renderVRT()
    
    
    
    
#h = workdir + 'merged/geom_master/hgt.4alks_10rlks.rdr'
#hImg = isceobj.createImage()
#hImg.load(h + '.xml')
#hgt = hImg.memMap()[:,:,0].astype(np.float32)

#a= intImage.memMap()[:,:,0] 
#
#
#
#def runFilter(infile, outfile, filterStrength):
#    # Initialize the flattened interferogram
#    intImage = isceobj.Image.createIntImage()
#    intImage.load( infile + '.xml')
#    intImage.setAccessMode('read')
#    intImage.createImage()
#    
#    # Create the filtered interferogram
#    filtImage = isceobj.createIntImage()
#    filtImage.setFilename(outfile)
#    filtImage.setWidth(intImage.getWidth())
#    filtImage.setAccessMode('write')
#    filtImage.dataType = 'FLOAT'
#    filtImage.length = nyl
#    filtImage.createImage()
#    
#    objFilter = Filter()
#    objFilter.wireInputPort(name='interferogram',object=intImage)
#    objFilter.wireOutputPort(name='filtered interferogram',object=filtImage)
#    objFilter.goldsteinWerner(alpha=filterStrength)
#    intImage.finalizeImage()
#    filtImage.finalizeImage()  
#    
#    
#filterStrength = .8
#for pair in pairs[1:]: #loop through each ifg and save to 
#    infile = intdir + pair + '/fine_lk.int'
#    outfile= intdir + pair + '/filtered_lk.int'
#    runFilter(infile,outfile, filterStrength)
#    
#
#
## Try interpolation instead of filtering
#from scipy.interpolate import griddata
#y1,x1 = np.where(mask==1)
#rl = rea_lk[np.where(mask==1)]
#il = ima_lk[np.where(mask==1)]
#rl[np.isnan(rl)]=0
#il[np.isnan(il)]=0
#y2,x2 = np.meshgrid(np.arange(0,nyl-1),np.arange(0,nxl-1),sparse=False, indexing='ij')
#rea_interp =griddata((y1,x1),rl, (y2,x2), method='linear')
#ima_interp =griddata((y1,x1),il, (y2,x2), method='linear')
#phs_lk_interp = np.arctan2(ima_interp, rea_interp).astype(np.float32)

## Get lon/lat
#f_lon_lk = mergeddir + 'geom_master/lon_lk.rdr'
#f_lat_lk = mergeddir + 'geom_master/lat_lk.rdr'
#
#Image = isceobj.createImage()
#Image.load(f_lon_lk + '.xml')
#lon_ifg = Image.memMap()[ymin:ymax,:,0]
#lon_ifg = lon_ifg.copy().astype(np.float32)
#lon_ifg[lon_ifg==0]=np.nan
#Image.finalizeImage()
#
#Image = isceobj.createImage()
#Image.load(f_lat_lk + '.xml')
#lat_ifg = Image.memMap()[ymin:ymax,:,0]
#lat_ifg = lat_ifg.copy().astype(np.float32)
#lat_ifg[lat_ifg==0]=np.nan
#Image.finalizeImage()
#
#pad=-1
#plt.close()
#plt.rc('font',size=14)
#fig = plt.figure(figsize=(6,6))
#m = Basemap(llcrnrlat=lat_bounds.min()-pad,urcrnrlat=lat_bounds.max()+pad,\
#        llcrnrlon=lon_bounds.min()-pad,urcrnrlon=lon_bounds.max()+pad,resolution='i',epsg=3395)
#m.arcgisimage(service='World_Shaded_Relief',xpixels=1000)
#m.drawstates(linewidth=1.5,zorder=1,color='white')
#m.drawcountries(linewidth=1.5,zorder=1,color='white')
#m.drawparallels(np.arange(np.floor(lat_bounds.min()-pad), np.ceil(lat_bounds.max()+pad), 2), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
#m.drawmeridians(np.arange(np.floor(lon_bounds.min()-pad), np.ceil(lon_bounds.max()+pad),2), linewidth=0,labels=[1,0,0,1])
#cf = m.pcolormesh(lon_ifg,lat_ifg,gam[ymin:ymax,:],shading='flat',cmap=plt.cm.Greys,latlon=True, zorder=8)
#cbar = m.colorbar(cf,location='bottom',pad="10%")
#cbar.set_label('Phase stability')
#plt.show()
#plt.savefig(workdir + 'Figs/gamma0.png',transparent=True,dpi=200 )
