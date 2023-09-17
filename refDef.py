#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:35:54 2018

@author: kdm95
"""

import numpy as np
import isceobj
from matplotlib import pyplot as plt
import scipy.signal as signal
from PyPS2 import invertRates,makeMap,util

# xRef=40;yRef=900;disconnected=True;plotStuff=True;doTimeFilt=False;removePlane=False;order=1;skip=1;offset=False;phsElev=True;startID=0;stopID=-1;geoCode=False
def refDef(xRef=False,yRef=False,disconnected=True, plotStuff=True, order=1, skip=5, doTimeFilt=False,offset=False,phsElev=False, startID=0,stopID=-1,geoCode=False):
    '''
    Loads the unwrapped interferograms, does the sbas-like inversion, converts to 
    cm, inverts to find the rates.
    input:
        xRef: reference pixel x-coordinate (radar coordinates)
        yRef: reference pixel y-coordinate (radar coordinates)
        If you don't give values for reference, it will guess the best one. 
        disconnected: True or False.  Make True if there are islands in the image. (Disconnected components)
        plotStuff: True or False if you want to make plot outputs or just run and save the rates.npy file.
        order: order of 2d polynomial to remove. (0 is none, 1 is ramp, 2 is quadratic)
        skip: redundant pairs (1 is a simple sequential chain)
        do timefilt: apply a temporal filter (this doesn't change result of long term rates. recommended to keep False)
        offset: apply an offset (not recommended)
        phsElev: remove a phs-elevation dependence        
        minCor: sets ifgs with median corrlation < minCor to to zero
        startID/stopID: start and stop ids for the dates for the rate estimation (it still returns alld with all dates)
    outputs:
        rates.npy
    '''
      
    ps = np.load('./ps.npy',allow_pickle=True).all()
    msk = np.load('Npy/msk.npy') 
    cor = np.load('Npy/cor.npy')


    pairs = list()  
    for ii,d in enumerate(ps.dates[0:-1]):
        for jj in np.arange(1,skip+1):
            if ii+jj < len(ps.dates):
                pairs.append(ps.dates[ii] + '_' + ps.dates[ii+jj])
    
    # Now make pairs2
    # pairs2Overlap = 5
    # pairs = list()  
    # # pairs2.append(dates[ii] + '_' + dates[0])
    # for ii,d in enumerate(ps.dates[0:-1]):
    #     for jj in np.arange(4,pairs2Overlap+1):
    #         if ii+jj < len(ps.dates):
    #             pairs.append(ps.dates[ii] + '_' + ps.dates[ii+jj])
    
    # order = 2 # Use 0 if you don't want to remove long wavelength function
    
    if not xRef:
        win=80
        Q = np.ones((win,win))
        corF = signal.convolve2d(cor,Q, mode='same')/(win**2)
        yRef,xRef = np.where(corF==np.nanmax(corF)); yRef=yRef[0];xRef=xRef[0]

    print([xRef,yRef])

    if plotStuff:
        plt.close('all')
        fig,ax = plt.subplots(1,3,figsize=(16,4))
        ax[0].imshow(msk);ax[0].set_title('Mask')
        ax[1].imshow(cor);ax[1].set_title('Avg Correlation')
        ax[2].imshow(corF);ax[2].set_title('Filtered Avg Correlation')
        ax[2].scatter(xRef,yRef,color='red')
    
    
    
    stack = []
    for ii,p in enumerate(pairs):
        unw_file = ps.intdir + '/' + p + '/filt.unw'
        unwImage = isceobj.createIntImage()
        unwImage.load(unw_file + '.xml')
        unw = unwImage.memMap()[:,:,0].copy()
        unw  = unw - unw[yRef,xRef]
        stack.append(unw)
    stack = np.asarray(stack,dtype=np.float32)
    
   
    
    # SBAS Inversion to get displacement at each date
    # Make G matrix for dates inversion
    G = np.zeros((len(pairs)+1,len(ps.dates)))# extra row of zeros to make first date zero for reference
    for ii,pair in enumerate(pairs):
        a = ps.dates.index(pair[0:8])
        b = ps.dates.index(pair[9:17])
        G[ii,a] = 1
        G[ii,b] = -1
    G[-1,0]=1
    
    Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
    
    # Do dates inversion
    alld=np.zeros((len(ps.dec_year),ps.nxl*ps.nyl))
    for ii in np.arange(0,ps.nyl-1): #iterate through rows
        tmp = np.zeros((len(pairs)+1,ps.nxl))
        for jj,pair in enumerate(pairs): #loop through each ifg and append to alld 
            tmp[jj,:] = stack[jj,ii,:]
        alld[:,ii*ps.nxl:ps.nxl*ii+ps.nxl] = np.dot(Gg, tmp)
    del(tmp)  
        
    if doTimeFilt:
        alldFilt = util.tsFilt(alld, ps.dec_year, N=5, desiredPeriod = 1)
    
    alld = np.reshape(alld,(len(ps.dates),ps.nyl,ps.nxl))  
    
    
    for ii in range(alld.shape[0]):
        if order > 0:
            alld[ii,:,:] -= util.fitLong(alld[ii,:,:], order,msk)
        if phsElev:
            alld[ii,:,:] -= util.phaseElev(alld[ii,:,:], ps.hgt_ifg,msk,0,ps.nyl,0,ps.nxl)
    
    
    # # CONVERT TO CM 
    alld=alld*ps.lam/(4*np.pi)*100
    
    stacksum = -np.nansum(stack,axis=0)
    
    rates,resstd = invertRates.invertRates(alld[startID:stopID,:,:],ps.dn[startID:stopID], seasonals=False,mcov_flag=False,water_elevation=ps.seaLevel)
    rates = np.asarray(rates,dtype=np.float32)
    resstd = np.asarray(resstd,dtype=np.float32)
    ratesMasked = rates.copy()
    ratesMasked[msk==0] = np.nan
    # plt.figure();plt.plot(ps.dec_year,alld[:,897,46])

    # if disconnected:
    #     #remove mean from each disconnected region
    #     minPix = 1000
    #     labels = util.getConCom(msk,minPix)
    #     if plotStuff:
    #         fig,ax = plt.subplots(2,1,figsize=(5,6))
    #         ax[0].imshow(msk);ax[0].set_title('mask')
    #         ax[1].imshow(labels);ax[1].set_title('connected regions')
    #     for ii in range(int(labels.max())):
    #         if len(rates[labels==ii+1]) < minPix:
    #             rates[labels==ii+1] = np.nan # mask out small islands of data
    #             msk[labels==ii+1] = 0
    #         else:
    #             rates[labels==ii+1]-=np.nanmean(rates[labels==ii+1])
      
    
    if geoCode:
        ratesGeo = util.geocodeKM(rates,method='linear')
        ratesGeo[np.isnan(ratesGeo)] = 0
        np.save('./TS/rates.geo.npy',ratesGeo)
        
        # ratesStdGeo = util.geocodeKM(resstd,method='linear')
        # ratesStdGeo[np.isnan(ratesStdGeo)] = 0
        # np.save('./TS/ratesStd.geo.npy',ratesStdGeo)
        
        mskGeo = util.geocodeKM(msk,method='nearest')
        # mskGeo[np.isnan(mskGeo)] = 0
        # mskGeo[mskGeo<.6] = 0
        # mskGeo[mskGeo>0] = 1
        np.save('./TS/msk.geo.npy',mskGeo)
    
   
    # gamthresh = .5
    # rates[msk == 0 ]=np.nan
    # resstd[msk == 0 ]=np.nan
    # stacksum[msk == 0 ]=np.nan

    
    if offset:
        rates=rates+offset
    
    ssvmin = stacksum[~np.isnan(stacksum)].min()
    ssvmax = stacksum[~np.isnan(stacksum)].max()
    
    vmin,vmax = -3,3
    pad=0


    if plotStuff:
        import cartopy.crs as ccrs

        bg = 'World_Imagery'
        zoomLevel = 13
        title = 'Rates (cm/yr)'
        vmin,vmax = -10,10
        makeMap.mapBackground(bg, ps.minlon, ps.maxlon, ps.minlat, ps.maxlat, zoomLevel, title)
        plt.imshow(ratesGeo,transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax,extent=[ps.minlon, ps.maxlon, ps.minlat, ps.maxlat],zorder=2)
        makeMap.mapImg(rates,ps.lon_ifg,ps.lat_ifg,vmin,vmax,pad,10,'rates (cm/yr)',plotFaults=True)
        makeMap.mapImg(stacksum,ps.lon_ifg,ps.lat_ifg,ssvmin, ssvmax, pad, 10, 'Stack sum (cm)', plotFaults=True)
        fig,ax = plt.subplots(2,1,figsize=(6,8))
        ax[0].imshow(rates,vmin=vmin,vmax=vmax);ax[0].set_title('rates (cm/yr)')
        ax[1].imshow(stacksum,vmin=ssvmin,vmax=ssvmax); ax[1].set_title('stack sum')
    
    
    np.save('rates.npy',rates)
    np.save('resstd.npy',resstd)
    
    return rates,alld

if __name__ == "__main__":
    refDef(xRef=False,yRef=False,disconnected=True, plotStuff=True, order=1, skip=1, doTimeFilt=False,offset=False,phsElev=False, startID=0,stopID=-1,geoCode=False)