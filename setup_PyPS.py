#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:20:01 2018
  Make dates and pairs dictionaries

@author: kdm95
"""

import numpy as np
import glob
import os
from datetime import date
import isceobj
import matplotlib.pyplot as plt
import makeMap
import cartopy.crs as ccrs
from mroipac.looks.Looks import Looks
from scipy.interpolate import griddata 
import cv2
from scipy import signal
import localParams
import util

def main(plot=True,doDownlook=True,replace=False):
    # plot=True
    # doDownlook=True
    # replace=True
    
    ps = localParams.getLocalParams()
    plt.close('all')
    
    if replace:
        os.system('rm ./merged/geom_reference/*crop*')
        os.system('rm ./merged/geom_reference/*lk*')
        
        
    # Make directories
    if not os.path.isdir(ps.tsdir):
        os.mkdir(ps.tsdir) 
    if not os.path.isdir(ps.workdir + '/Figs'):
        os.mkdir(ps.workdir + '/Figs')
    
    
    geomList = glob.glob(ps.mergeddir + '/geom_reference/*full')
    slcList = glob.glob(ps.slcdir + '/*/*full')
    blList = glob.glob(ps.mergeddir + '/baselines/????????/????????')
    if doDownlook:
        for fname in slcList:
            os.system('fixImageXml.py -i ' + fname + ' -f')
        for fname in geomList:
            os.system('fixImageXml.py -i ' + fname + ' -f')
        # for fname in blList:
        #     os.system('fixImageXml.py -i ' + fname + ' -f')
            
    
    if ps.ifgMode:
        pairs1 = []
        pairs2 = []
        pairs = []
        flist = glob.glob(ps.intdir + '/2*_2*')
        [pairs.append(f[-17:]) for f in flist]
        [pairs1.append(f[-17:-9]) for f in flist]
        [pairs2.append(f[-8:]) for f in flist]
        pairs.sort();pairs1.sort();pairs2.sort()
        dates = np.unique(np.vstack((pairs1,pairs2)))
    else:
        flist = glob.glob(ps.slcdir + '/2*')
        # Convert pairs to dates
        dates = []
        for f in flist:
            dates.append(f[-8:])
        dates.sort()
    
        pairs = [] 
        for ii,d in enumerate(dates[0:-1]):
            pairs.append(dates[ii] + '_' + dates[ii+1])
        
        # Now make pairs2
        pairs2 = []
        for ii,d in enumerate(dates[0:-1]):
            for jj in np.arange(1,ps.skip+1):
                if ii+jj < len(dates):
                    pairs2.append(dates[ii] + '_' + dates[ii+jj])
        
    
    dn = [] 
    dec_year = []
    for d in dates:
        yr = d[0:4]
        mo = d[4:6]
        day = d[6:8]
        dt = date.toordinal(date(int(yr), int(mo), int(day)))
        dn.append(dt)
        d0 = date.toordinal(date(int(yr), 1, 1))
        doy = np.asarray(dt)-d0+1
        dec_year.append(float(yr) + (doy/365.25))
    dn = np.asarray(dn)
    dn0 = dn-dn[0] # make relative to first date
        
    
    
    nd = len(pairs)
    # rename geometry files to add 'full'
    if os.path.isfile('merged/geom_reference/hgt.rdr'):
        os.system('mv merged/geom_reference/hgt.rdr merged/geom_reference/hgt.rdr.full')
        os.system('mv merged/geom_reference/lat.rdr merged/geom_reference/lat.rdr.full')
        os.system('mv merged/geom_reference/lon.rdr merged/geom_reference/lon.rdr.full')
        os.system('mv merged/geom_reference/incLocal.rdr merged/geom_reference/incLocal.rdr.full')
        os.system('mv merged/geom_reference/los.rdr merged/geom_reference/los.rdr.full')
        os.system('mv merged/geom_reference/shadowMask.rdr merged/geom_reference/shadowMask.rdr.full')
    else:
        print('rdr files have already been renamed to full')
    
    # Get width and length
    f_lon = ps.mergeddir + '/geom_reference/lon.rdr.full'
    gImage = isceobj.createIntImage()
    gImage.load(f_lon + '.xml')
    nyf = gImage.length
    nxf = gImage.width
    
    
    if ps.crop:
        ny = ps.cropymax-ps.cropymin
        nx = ps.cropxmax-ps.cropxmin
    else:
        ny = gImage.length
        nx = gImage.width
        ps.cropxmin=0
        ps.cropxmax=nx
        ps.cropymin=0
        ps.cropymax=ny
    
    
    if ps.crop:
        for d in ps.dates:
            infile = ps.slcdir + '/' + d + '/' + d + '.slc.full'
            imgi = isceobj.createSlcImage()
            imgi.load(infile+'.xml')
            
            # Rearrange axes order from small to big 
            slcIm = util.orderAxes(imgi.memMap(),ps.nx,ps.ny)
            slcIm = slcIm[:,ps.cropymin:ps.cropymax,ps.cropxmin:ps.cropxmax]
    
            imgo = imgi.clone()
            imgo.filename = infile+'.crop'
            imgo.width = ps.cropxmax-ps.cropxmin
            imgo.length = ps.cropymax-ps.cropymin
            imgo.dump(imgo.filename+'.xml')
            slcIm.tofile(imgo.filename)
            imgo.finalizeImage()
            del(slcIm)
    
    
    
    file_list = list(['lat','lon','hgt','los','shadowMask','incLocal']) 
    if ps.crop:
        for f in file_list:
            infile = ps.mergeddir + '/geom_reference/' + f + '.rdr.full'
            imgi = isceobj.createImage()
            imgi.load(infile+'.xml')
            # print(imgi.memMap().shape)
            
            # Rearrange axes order from small to big 
            geomIm = util.orderAxes(imgi.memMap(),nx,ny)
            geomIm = geomIm[:,ps.cropymin:ps.cropymax,ps.cropxmin:ps.cropxmax]
    
            imgo = imgi.clone()
            imgo.filename = infile+'.crop'
            imgo.width = ps.cropxmax-ps.cropxmin
            imgo.length = ps.cropymax-ps.cropymin
            imgo.dump(imgo.filename+'.xml')
            geomIm.tofile(imgo.filename)
            imgo.finalizeImage()
            del(geomIm)
            
    if doDownlook:
        def downLook(infile, outfile,alks,rlks):
            inImage = isceobj.createImage()
            inImage.load(infile + '.xml')
            inImage.filename = infile
            lkObj = Looks()
            lkObj.setDownLooks(alks)
            lkObj.setAcrossLooks(rlks)
            lkObj.setInputImage(inImage)
            lkObj.setOutputFilename(outfile)
            lkObj.looks()
        for f in file_list:
            if ps.crop:
                infile = ps.mergeddir + '/geom_reference/' + f + '.rdr.full.crop'
            else:
                infile = ps.mergeddir + '/geom_reference/' + f + '.rdr.full'
            
            outfile = ps.mergeddir + '/geom_reference/' + f + '_lk.rdr'
            
            if not os.path.isfile(outfile):
                downLook(infile, outfile,ps.alks,ps.rlks)
            else:
                print(outfile + ' already exists')
    
    nxl = nx//ps.rlks
    nyl = ny//ps.alks
        
    # Get bounding coordinates (Frame)
    f_lon_lk = ps.mergeddir + '/geom_reference/lon_lk.rdr'
    f_lat_lk = ps.mergeddir + '/geom_reference/lat_lk.rdr'
    f_hgt_lk = ps.mergeddir + '/geom_reference/hgt_lk.rdr'
    f_los_lk = ps.mergeddir + '/geom_reference/los_lk.rdr'
    f_shm_lk = ps.mergeddir + '/geom_reference/shadowMask_lk.rdr'
    f_inc_lk = ps.mergeddir + '/geom_reference/incLocal_lk.rdr'
    
    
    # LON --------------
    Image = isceobj.createImage()
    Image.load(f_lon_lk + '.xml')
    lon_ifg = util.orderAxes(Image.memMap(),nxl,nyl)[0,:,:]
    lon_ifg = lon_ifg.copy().astype(np.float32)
    lon_ifg[lon_ifg==0]=np.nan
    Image.finalizeImage()
    
    
    # LAT --------------
    Image = isceobj.createImage()
    Image.load(f_lat_lk + '.xml')
    lat_ifg =util.orderAxes(Image.memMap(),nxl,nyl)[0,:,:]
    lat_ifg = lat_ifg.copy().astype(np.float32)
    lat_ifg[lat_ifg==0]=np.nan
    Image.finalizeImage()
    
    # HGT --------------
    Image = isceobj.createImage()
    Image.load(f_hgt_lk + '.xml')
    hgt_ifg = util.orderAxes(Image.memMap(),nxl,nyl)[0,:,:]
    hgt_ifg = hgt_ifg.copy().astype(np.float32)
    hgt_ifg[hgt_ifg==0]=np.nan
    Image.finalizeImage()
    
    # LOS --------------
    Image = isceobj.createImage()
    Image.load(f_los_lk + '.xml')
    Image.bands=2
    Image.scheme='BSQ'
    los_ifg = util.orderAxes(Image.memMap(),nxl,nyl)[0,:,:]
    los_ifg = los_ifg.copy()
    az_ifg = util.orderAxes(Image.memMap(),nxl,nyl)[1,:,:]
    az_ifg = az_ifg.copy()
    Image.finalizeImage()
    
    # Write out a new los file
    losOutname = ps.mergeddir + '/geom_reference/los2_lk.rdr'
    fidc=open(losOutname,"wb")
    fidc.write(los_ifg)
    #write out an xml file for it
    out = isceobj.createIntImage() # Copy the interferogram image from before
    out.dataType = 'FLOAT'
    out.bands = 1
    out.filename = losOutname
    out.width = nxl
    out.length = nyl
    out.dump(losOutname + '.xml') # Write out xml
    out.renderHdr()
    out.renderVRT()
    
    
    # Write out a new az file
    azOutname = ps.mergeddir + '/geom_reference/az_lk.rdr'
    fidc=open(azOutname,"wb")
    fidc.write(az_ifg)
    #write out an xml file for it
    out = isceobj.createIntImage() # Copy the interferogram image from before
    out.dataType = 'FLOAT'
    out.bands = 1
    out.filename = azOutname
    out.width = nxl
    out.length = nyl
    out.dump(azOutname + '.xml') # Write out xml
    out.renderHdr()
    out.renderVRT()
    
    # if you want to save these to geom
    los_ifg = los_ifg.copy().astype(np.float32)
    los_ifg[los_ifg==0]=np.nan
    az_ifg = az_ifg.copy().astype(np.float32)
    az_ifg[az_ifg==0]=np.nan
    
    Image = isceobj.createImage()
    Image.load(f_shm_lk + '.xml')
    Image.bands=1
    shm_ifg = util.orderAxes(Image.memMap(),nxl,nyl)[0,:,:]
    shm_ifg = shm_ifg.copy().astype(np.float32)
    shm_ifg[np.isnan(hgt_ifg)]=np.nan
    Image.finalizeImage()
    
    Image = isceobj.createImage()
    Image.load(f_inc_lk + '.xml')
    Image.bands=2
    Image.scheme='BSQ'
    # inc_ifg1 = Image.memMap()[0,:,:] # relative to the local plane of the ground
    inc_ifg = util.orderAxes(Image.memMap(),nxl,nyl)[1,:,:]# relative to surface normal vector (this is the one we want I think)
    inc_ifg = inc_ifg.copy()
    
    # Write out a new inc file
    incOutname = ps.mergeddir + '/geom_reference/inc_lk.rdr'
    fidc=open(incOutname,"wb")
    fidc.write(inc_ifg)
    #write out an xml file for it
    out = isceobj.createIntImage() # Copy the interferogram image from before
    out.dataType = 'FLOAT'
    out.bands = 1
    out.filename = incOutname
    out.width = nxl
    out.length = nyl
    out.dump(incOutname + '.xml') # Write out xml
    out.renderHdr()
    out.renderVRT()
    
    inc_ifg = inc_ifg.copy().astype(np.float32)
    inc_ifg[inc_ifg==0]=np.nan
    Image.finalizeImage()
    
    
    # Get rid of edge artifacts from downlooking
    Q = np.array([[0,0,0],[0,1,0],[0,0,0]])
    lon_ifg = signal.convolve2d(lon_ifg,Q, mode='same')
    lat_ifg = signal.convolve2d(lat_ifg,Q, mode='same')
    hgt_ifg = signal.convolve2d(hgt_ifg,Q, mode='same')
    los_ifg = signal.convolve2d(los_ifg,Q, mode='same')
    shm_ifg = signal.convolve2d(shm_ifg,Q, mode='same')
    inc_ifg = signal.convolve2d(inc_ifg,Q, mode='same')
    
    #Do it again for good measure (could also just make the kernel bigger..)
    lon_ifg = signal.convolve2d(lon_ifg,Q, mode='same')
    lat_ifg = signal.convolve2d(lat_ifg,Q, mode='same')
    hgt_ifg = signal.convolve2d(hgt_ifg,Q, mode='same')
    los_ifg = signal.convolve2d(los_ifg,Q, mode='same')
    shm_ifg = signal.convolve2d(shm_ifg,Q, mode='same')
    inc_ifg = signal.convolve2d(inc_ifg,Q, mode='same')
    
    
    # outputfilename = ps.mergeddir + '/geom_reference/waterMask_lk.rdr.crop'
    # util.getWaterMask(ps.dem, lon_ifg, lat_ifg, outputfilename)
    
    
    if plot:
        cmap = 'Spectral_r'
        fig,ax = plt.subplots(3,2,figsize=(9,9))
        ax[0,0].imshow(lon_ifg,cmap=cmap);ax[0,0].set_title('lon_ifg')
        ax[0,1].imshow(lat_ifg,cmap=cmap);ax[0,1].set_title('lat_ifg')
        ax[1,0].imshow(hgt_ifg,cmap=cmap);ax[1,0].set_title('hgt_ifg')
        ax[1,1].imshow(los_ifg,cmap=cmap);ax[1,1].set_title('los_ifg')
        ax[2,0].imshow(shm_ifg,cmap=cmap);ax[2,0].set_title('shm_ifg')
        ax[2,1].imshow(inc_ifg,cmap=cmap);ax[2,1].set_title('inc_ifg')
        plt.savefig(ps.workdir + '/Figs/geom.svg',transparent=True,dpi=100 )
    
    # Figure out where the nan values begin and end so we can crop them if we want later.
    for l in np.arange(0,nyl):
        ll = lon_ifg[l,:]
        if not np.isnan(ll.max()):
            break
    
    for p in np.arange(l+1,nyl):
        ll = lon_ifg[p,:]
        if np.isnan(ll.max()):
            break
    l+=1
    ymin=l+1
    ymax=p-1
    xmin=0
    xmax=nxl
    ul = (lon_ifg[l+1,1],lat_ifg[l+1,1])
    ur = (lon_ifg[l+1,-2],lat_ifg[l+1,-2])
    ll = (lon_ifg[p-2,1],lat_ifg[p-2,1])
    lr = (lon_ifg[p-2,-2],lat_ifg[p-2,-2])
    lon_bounds = np.array([ul[0],ur[0],ur[0],lr[0],lr[0],ll[0],ll[0],ul[0]])
    lat_bounds = np.array([ul[1],ur[1],ur[1],lr[1],lr[1],ll[1],ll[1],ul[1]])
    
    # Now extrapolate the geom edges out so we can map non-rectangle images
    xx,yy = np.meshgrid(np.arange(0,nxl), np.arange(0,nyl))
    xxValid = xx[~np.isnan(lon_ifg)].astype(np.float32)
    yyValid = yy[~np.isnan(lon_ifg)].astype(np.float32)
    lonValid = lon_ifg[~np.isnan(lon_ifg)].astype(np.float32)
    latValid = lat_ifg[~np.isnan(lon_ifg)].astype(np.float32)
    lonI = griddata((xxValid,yyValid), lonValid , (xx,yy), method='nearest')
    xxValid = xx[~np.isnan(lat_ifg)].astype(np.float32)
    yyValid = yy[~np.isnan(lat_ifg)].astype(np.float32)
    lonValid = lon_ifg[~np.isnan(lat_ifg)].astype(np.float32)
    latValid = lat_ifg[~np.isnan(lat_ifg)].astype(np.float32)
    latI = griddata((xxValid,yyValid), latValid , (xx,yy), method='nearest')
    minlat=latI.min()
    maxlat=latI.max()
    minlon=lonI.min()
    maxlon=lonI.max()
    
    if plot:
        zoomLevel=8
        bg = 'World_Shaded_Relief'
        pad=2
        title = 'Footprint'
        makeMap.mapBackground(bg, minlon, maxlon, minlat, maxlat, pad, zoomLevel, title)
        plt.plot(lon_bounds,lat_bounds,linewidth=2,color='red',zorder=10,transform=ccrs.PlateCarree())
        plt.rc('font',size=14)
        plt.savefig(ps.workdir + '/Figs/areamap.svg',transparent=True,dpi=100 )
    
    
    
    
    ps.lon_ifg = lonI
    ps.lat_ifg = latI
    ps.hgt_ifg = hgt_ifg
    ps.los_ifg = los_ifg
    ps.shm_ifg = shm_ifg
    ps.inc_ifg = inc_ifg
    
    ps.pairs =      pairs
    ps.dates =      dates
    ps.pairs =      pairs
    ps.pairs2 =     pairs2
    ps.dec_year =   dec_year
    ps.dn =         dn
    ps.dn0 =        dn0
    ps.nd =         nd
    
    ps.minlon =     lon_ifg.min()
    ps.maxlon =     lon_ifg.max()
    ps.minlat =     lat_ifg.min()
    ps.maxlat =     lat_ifg.max()
    
    ps.ny =         ny
    ps.nx =         nx
    ps.nxl =        nxl
    ps.nyl =        nyl
    ps.nxf =        nxf
    ps.nyf =        nyf
    ps.lon_bounds = lon_bounds
    ps.lat_bounds = lat_bounds
    ps.ymin =       ymin
    ps.ymax =       ymax
    ps.xmin =       xmin
    ps.xmax =       xmax
    
    # Save the namespace
    np.save('ps.npy',ps)

if __name__ == '__main__':
    main()