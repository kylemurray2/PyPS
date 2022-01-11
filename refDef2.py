#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:35:54 2018

@author: kdm95
"""

import numpy as np
import isceobj
from matplotlib import pyplot as plt
import invertRates 
import scipy.signal as signal
import makeMap
import util

# xRef=False;yRef=False;disconnected=True;plotStuff=True;doTimeFilt=False;removePlane=False;order=1;redundantPairs=True;offset=False;phsElev=False
def refDef(xRef=False,yRef=False,disconnected=True, plotStuff=True, order=1, redundantPairs=True, doTimeFilt=False,offset=False,phsElev=False):
    '''
    Loads the unwrapped interferograms, does the sbas-like inversion, converts to 
    cm, inverts to find the rates.
    input:
        xRef: reference pixel x-coordinate (radar coordinates)
        yRef: reference pixel y-coordinate (radar coordinates)
        If you don't give values for reference, it will guess the best one. 
        disconnected: True or False.  Make True if there are islands in the image. (Disconnected components)
        plotStuff: True or False if you want to make plot outputs or just run and save the rates.npy file.
    outputs:
        rates.npy
    '''
      
    params = np.load('params.npy',allow_pickle=True).item()
    geom = np.load('geom.npy',allow_pickle=True).item()
    msk = np.load('msk.npy') 
    cor = np.load('cor.npy')


    if redundantPairs:
        pairs = params['pairs2']
    else:
        pairs = params['pairs']
        
    # order = 2 # Use 0 if you don't want to remove long wavelength function
    
    if not xRef:
        win=80
        Q = np.ones((win,win))
        corF = signal.convolve2d(cor,Q, mode='same')/(win**2)
        
        if plotStuff:
            plt.close('all')
            fig,ax = plt.subplots(3,1,figsize=(6,8))
            ax[0].imshow(msk);ax[0].set_title('Mask')
            ax[1].imshow(cor);ax[1].set_title('Avg Correlation')
            ax[2].imshow(corF);ax[2].set_title('Filtered Avg Correlation')
        
        yRef,xRef = np.where(corF==np.nanmax(corF)); yRef=yRef[0];xRef=xRef[0]
    
    locals().update(params)
    locals().update(geom)
    
    nxl = params['nxl']
    nyl  = params['nyl']
    
        
    stack = []
    for p in pairs:
        unw_file = params['intdir'] + '/' + p + '/filt.unw'
        unwImage = isceobj.createIntImage()
        unwImage.load(unw_file + '.xml')
        unw = unwImage.memMap()[:,:,0].copy()
        # unw[msk==0]=np.nan


        unw = unwImage.memMap()[:,:,0] - unwImage.memMap()[yRef,xRef,0]
        stack.append(unw)
    stack = np.asarray(stack,dtype=np.float32)
    
    
    # SBAS Inversion to get displacement at each date
    # Make G matrix for dates inversion
    G = np.zeros((len(pairs)+1,len(params['dates'])))# extra row of zeros to make first date zero for reference
    for ii,pair in enumerate(pairs):
        a = params['dates'].index(pair[0:8])
        b = params['dates'].index(pair[9:17])
        G[ii,a] = 1
        G[ii,b] = -1
    G[-1,0]=1
    
    Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
    
    # Do dates inversion
    alld=np.zeros((len(params['dec_year']),nxl*nyl))
    for ii in np.arange(0,nyl-1): #iterate through rows
        tmp = np.zeros((len(pairs)+1,nxl))
        for jj,pair in enumerate(pairs): #loop through each ifg and append to alld 
            tmp[jj,:] = stack[jj,ii,:]
        alld[:,ii*nxl:nxl*ii+nxl] = np.dot(Gg, tmp)
    del(tmp)  
        
    
    if doTimeFilt:
        # First, design the Buterworth filter
        N  = 5    # Filter order
        ''' Wn is the Cutoff frequency between 0 and 1.  0 is infinitely smooth and 1 is the original. 
            this is the frequency multiplied by the nyquist rate. 
            if we have 25 samples per year, then the nyquist rate would be ~12hz. So if we make Wn=.5
            we will have filtered to 6hz (letting signals with wavelengths of 2 months or longer).
            If we make wn=1/12 then we will filter to 1hz (letting only signals with wavelengths of 1 year).
        '''
        dec_year = np.asarray(params['dec_year'])
        samplesPerYear = len(params['dn']) / (dec_year.max()-dec_year.min())
        nyquistRate = samplesPerYear/2 #this is the highest freq we can resolve with our sampling rate
        desiredPeriod = 1 # signals with a period of 1 year or longer
        Wn = 1/(desiredPeriod * nyquistRate)
        B, A = signal.butter(N, Wn, output='ba')
        
        alld = signal.filtfilt(B,A, alld,axis=0)
        alld[alld==0]=np.nan

 
    # alldPlane = []
    # X,Y = np.meshgrid(range(nxl),range(nyl))
    # X=X[msk==1];Y=Y[msk==1]
    # for ii in range(alld.shape[0]):
    #     G  = np.array([np.ones((len(X.flatten()),)), X.flatten(), Y.flatten()]).T
    #     Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
    #     mod   = np.dot(Gg,alld[ii,:][msk.ravel()==1])
    #     synth = mod[0] + mod[1] * X.flatten() + mod[2] * Y.flatten()
    #     alldPlane.append(alld[ii,:][msk.ravel()==1] - synth)  
    
    # alldPlane = np.asarray(alldPlane,dtype= np.float32)     
    # for ii in range(alld.shape[0]):
    #     alld[ii,:][msk.ravel()==1] = alldPlane[ii,:]
    # del(alldPlane)
    
    alld = np.reshape(alld,(len(params['dates']),nyl,nxl))  
    
    for ii in range(alld.shape[0]):
        if order > 0:
            alld[ii,:,:] -= util.fitLong(alld[ii,:,:], order,msk)
        if phsElev:
            alld[ii,:,:] -= util.phaseElev(alld[ii,:,:], geom['hgt_ifg'],msk,0,nyl,0,nxl)
    
    
    # # CONVERT TO CM 
    alld=alld*params['lam']/(4*np.pi)*100
    
    stacksum = -np.nansum(stack,axis=0)
    
    rates,resstd = invertRates.invertRates(alld,params,params['dn'], seasonals=False,mcov_flag=False,water_elevation=params['seaLevel'])
    rates = np.asarray(rates,dtype=np.float32)
    resstd = np.asarray(resstd,dtype=np.float32)
    # rates[hgt_ifg<seaLevel] = np.nan
    # resstd[hgt_ifg<seaLevel] = np.nan
    
    # gamthresh = .5
    rates[msk == 0 ]=np.nan
    # stacksum[msk == 0 ]=np.nan
    
    if disconnected:
        #remove mean from each disconnected region
        minPix = 1000
        labels = util.getConCom(msk,minPix)
        if plotStuff:
            fig,ax = plt.subplots(2,1,figsize=(5,6))
            ax[0].imshow(msk);ax[0].set_title('mask')
            ax[1].imshow(labels);ax[1].set_title('connected regions')
        for ii in range(int(labels.max())):
            if len(rates[labels==ii+1]) < minPix:
                rates[labels==ii+1] = np.nan # mask out small islands of data
            else:
                rates[labels==ii+1]-=np.nanmean(rates[labels==ii+1])
    
    
    if offset:
        rates=rates+offset
    
    ssvmin = stacksum[~np.isnan(stacksum)].min()
    ssvmax = stacksum[~np.isnan(stacksum)].max()
    
    vmin,vmax = -5,5
    pad=0
    
    if plotStuff:
        makeMap.mapImg(rates,geom['lon_ifg'],geom['lat_ifg'],vmin,vmax,pad,10,'rates (cm/yr)',plotFaults=True)
        makeMap.mapImg(stacksum,geom['lon_ifg'],geom['lat_ifg'],ssvmin, ssvmax, pad, 10, 'Stack sum (cm)', plotFaults=True)
        fig,ax = plt.subplots(2,1,figsize=(6,8))
        ax[0].imshow(rates,vmin=vmin,vmax=vmax);ax[0].set_title('rates (cm/yr)')
        ax[1].imshow(stacksum,vmin=ssvmin,vmax=ssvmax); ax[1].set_title('stack sum')
    
    np.save('rates2.npy',rates)
    return rates,alld

if __name__ == "__main__":
    refDef(xRef=False,yRef=False,disconnected=True,plotStuff=False,doTimeFilt=False,removePlane=True,offset=False,phsElev=True)


# 
