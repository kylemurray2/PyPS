#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:39:57 2020
    
    This script checks for any dates in the time series that are abnormally 
    noisy.  This is based on
    
        -variance of each unwrapped ifg (usually dominated by atmosphere)
        -correlation of each ifg (could be related to ground surface properties)
        
    This also writes a file called msk.npy which is based on gamma0 and average correlation 
    
@author: km
"""

import numpy as np
import isceobj
from matplotlib import pyplot as plt
import makeMap
from scipy.interpolate import griddata
import glob
import os
from datetime import date
import util
from scipy import signal

plt.close('all')

def weeding(mincor=0.7,gamThresh=0.7,plotStuff=False,makeChanges=False,pairs2Overlap=1):

    # plotStuff = True
    # makeChanges = False
    # mincor = .7
    # gamThresh = .55
    
    params = np.load('params.npy',allow_pickle=True).item()
    geom = np.load('geom.npy',allow_pickle=True).item()
    locals().update(params)
    locals().update(geom)

    # MASKING______________________________
    gam = np.load('gam.npy')
    
    gamGeo = util.geocodeKM(gam)
    gamGeo[np.isnan(gamGeo)] = 0
    np.save('./TS/gam.geo.npy',gamGeo)
    
    gam[gam<.3] = np.nan

    gamFlat = gam.flatten()
    
    X,Y = np.meshgrid(range(params['nxl']),range(params['nyl']))
    
    pairs3 = list()  
    # pairs2.append(params['dates'][ii] + '_' + params['dates'][0])
    for ii,d in enumerate(params['dates'][0:-1]):
        for jj in np.arange(1,pairs2Overlap+1):
            if ii+jj < len(params['dates']):
                pairs3.append(params['dates'][ii] + '_' + params['dates'][ii+jj])
    
    
    stack = []
    for p in pairs3:
        unw_file = params['intdir'] + '/' + p + '/filt.unw'
        unwImage = isceobj.createIntImage()
        unwImage.load(unw_file + '.xml')
        unw = unwImage.memMap()[:,:,0] #- unwImage.memMap()[ymin:ymax,xmin:xmax,0][r,c]
        unw = unw.copy()
        # unw[np.isnan(gam)] = np.nan
        stack.append(unw)
    stack = np.asarray(stack,dtype=np.float32)
    
    stackTimeMean = np.nanmean(stack,axis=0)
    stackTimeVar = np.nanvar(stack,axis=0)
    
    # plt.figure();plt.imshow(stackTimeMean);plt.title('stack time mean')
    # plt.figure();plt.imshow(stackTimeVar);plt.title('stack time var')
    
    corStack = []
    for p in pairs3:
        cor_file = params['intdir'] + '/' + p + '/cor2.r4'
        corImage = isceobj.createIntImage()
        corImage.load(cor_file + '.xml')
        cor = corImage.memMap()[:,:,0]
        cor = cor.copy()
        # cor[np.isnan(gam)] = np.nan
        corStack.append(cor)
    corStack = np.asarray(corStack,dtype=np.float32)[:,:,:]
    
    connStack = []
    for p in pairs3:
        conn_file = params['intdir'] + '/' + p + '/filt.unw.conncomp'
        connImage = isceobj.createIntImage()
        connImage.load(conn_file + '.xml')
        conn = connImage.memMap()[:,0,:]
        # conn = conn.copy()
        connStack.append(conn)
    connStack = np.asarray(connStack,dtype=np.float32)[:,:,:]
    
    
    # average cor value for each pair
    corAvg = []
    ifgVar = []
    for ii in np.arange(0,len(pairs3)):
        corAvg.append(np.nanmedian(corStack[ii,:,:]))
        iv = stack[ii,:,:]
        iv[np.isnan(corStack[ii,:,:])] =np.nan
        ifgVar.append(np.nanvar(iv))
    
    corAvg = np.asarray(corAvg,dtype=np.float32)
    ifgVar = np.asarray(ifgVar,dtype=np.float32)
    
    
    
    if plotStuff:
        fig,ax = plt.subplots(2,1,figsize=(8,5))
        ax[0].plot(corAvg);ax[0].set_xlabel('time index');ax[0].set_ylabel('Correlation')
        # ax[1].plot(dec_year[1:],corAvg);ax[1].set_xlabel('Year');ax[1].set_ylabel('Correlation')
        
        fig,ax = plt.subplots(2,1,figsize=(8,5))
        ax[0].plot(ifgVar);ax[0].set_xlabel('time index');ax[0].set_ylabel('IFG variance')
        # ax[1].plot(dec_year[1:],ifgVar);ax[1].set_xlabel('Year');ax[1].set_ylabel('IFG variance')
    
    # fig,ax = plt.subplots(1,figsize=(8,3))
    # ax.plot(dec_year[1:],corAvg,'.');ax.set_xlabel('time index');ax.set_ylabel('Correlation')
    # plt.savefig('Figs/corTS.svg')
    # plt.figure()
    # plt.plot(np.ravel(gam)[::10],np.ravel(np.nanmean(corStack,axis=0))[::10],'.',markersize = 1)
    # plt.xlabel('Gamma 0');plt.ylabel('Average Correlation');plt.show()
    
    corAvgMap = np.nanmean(corStack,axis=0)
    corAvgMap[corAvgMap==0]=np.nan
    Q = np.array([[0,0,0],[0,1,0],[0,0,0]])
    corAvgMap = signal.convolve2d(corAvgMap,Q, mode='same')
    corAvgMap[np.isnan(corAvgMap)]=0
    
    corVar = np.nanvar(corStack,axis=0)
    
    if plotStuff:
        fig,ax = plt.subplots(2,1,figsize=(8,10))
        ax[0].imshow(corAvgMap);ax[0].set_title('Average Correlation')
        ax[1].imshow(corVar);ax[1].set_title('Correlation Variance')
    np.save('cor.npy',corAvgMap)
    np.save('corVar.npy',corVar)
    
    corGeo = util.geocodeKM(corAvgMap)
    corGeo[np.isnan(corGeo)] = 0
    np.save('./TS/cor.geo.npy',corGeo)

    corVarGeo = util.geocodeKM(corVar)
    corVarGeo[np.isnan(corVarGeo)] = 0
    np.save('./TS/corVar.geo.npy',corVarGeo)
    
    
    # corAvgMap = corAvgMap/np.nanmax(corAvgMap)
    
    # a = np.zeros(gam.shape)
    # a[np.where((corAvgMap<0.6)&(corVar>.04))] = 1
    # plt.figure();plt.imshow(a)
    
    # plt.figure()
    # plt.plot(np.ravel(np.nanmean(corStack,axis=0))[::10],np.ravel(np.nanvar(corStack,axis=0))[::10],'.',markersize = 1)
    # plt.xlabel('Average Correlation');plt.ylabel('Correlation variance');plt.show()
    
    # Find the bad dates
    # gamThresh = np.nanmedian(gam) - 2*np.nanstd(gam)
    medianCorStack = np.nanmedian(corAvgMap) 
    print('\nThe median correlation for entire stack is ' + str(round(medianCorStack,2)))
    
    corThresh = np.nanmedian(corAvgMap) - np.nanstd(corAvgMap)
    ifgVarThresh = np.nanmedian(ifgVar) + np.nanstd(ifgVar)
    # badPairs = np.where((corAvg<corThresh) | (ifgVar>ifgVarThresh))[0]
    badPairs = np.where(corAvg<corThresh)[0]

    # We'll assume it's a bad dates if it appears in multiple bad pairs.
    possibleBadDates = []
    for b in badPairs:
        possibleBadDates.append(pairs3[b][0:8])
        possibleBadDates.append(pairs3[b][9:])
    
    badDates = []
    for b in possibleBadDates:
        if possibleBadDates.count(b) > 1:
            badDates.append(b)
    badDates = np.unique(badDates)
    
    
    
    # This loop looks at ifgs associated with each date and finds the minimum ifg var for each date.
    #   This is a way to find which dates are noisy as opposed to individual ifgs. 
    dateVar = []
    for ii in np.arange(0,len(params['dn'])):
        dt = params['dates'][ii]
        # first find all of the ifgs that have that date. This is generalized in case there are redundant pairs.
        dtPairs = []
        for jj,p in enumerate(pairs3):
            if p[0:8] == dt or p[9:] == dt:
                dtPairs.append(jj)
        pVars = []
        for kk in dtPairs:
            iv = stack[kk,:,:]
            iv[np.isnan(corStack[kk,:,:])] =np.nan
            pVars.append(np.nanvar(iv))
        dateVar.append(np.nanmin(pVars))
    dateVar = np.asarray(dateVar,dtype=np.float32)
    
    connSum = np.sum(connStack,axis=0)
    
    gamMsk = np.ones(gam.shape)
    gamMsk[gam<gamThresh] = 0
    
    if plotStuff:
        plt.figure();plt.plot(dateVar);plt.xlabel('time index');plt.ylabel('Date variance (average of associated ifgs')
        plt.figure();plt.imshow(gam,vmin=.45,vmax=.5);plt.title('Gamma0')
        plt.figure();plt.imshow(connSum,cmap='jet_r');plt.title('Number of images with connected components')
    #if there are > 90% images with conncomp, then don't mask it, otherwise mask it. 
    
    msk = np.ones(gam.shape)
    msk[connSum < round(.9*params['nd'])] = 0
    
    # # Make the mask msk
    # msk = np.ones(gam.shape)
    msk[gam<gamThresh] = 0
    msk[corAvgMap<mincor] = 0
    # msk[np.isnan(gam)] = 0
    np.save('msk.npy',msk)
    
    
    gamGeo= util.geocodeKM(gam,method='linear')
    np.save('./TS/gam.geo.npy',gamGeo)
    connSumGeo= util.geocodeKM(connSum,method='linear')
    np.save('./TS/connSum.geo.npy',connSumGeo)

    mskGeo = np.ones(gamGeo.shape)
    mskGeo[connSumGeo < round(.9*params['nd'])] = 0
    mskGeo[gamGeo<gamThresh] = 0
    mskGeo[corGeo<mincor] = 0
    np.save('./TS/msk.geo.npy',mskGeo)
    
    if plotStuff:
        fig,ax = plt.subplots(2,1,figsize=(8,10))
        ax[0].imshow(gamMsk,vmin=.45,vmax=.5);ax[0].set_title('Gamma0 Mask')
        ax[1].imshow(msk);ax[1].set_title('mask')
    
    print('\n The bad dates might be: \n')
    print(badDates)
    
    # if medianCorStack < .7:
    #     if pairs2Overlap >1:
    #         pairs2Overlap-=1
    #         print('Cor too low. Rerunning with lower skip. Skip= ' + str(pairs2Overlap))
    #         weeding(mincor=mincor,gamThresh=gamThresh,plotStuff=False,makeChanges=False,pairs2Overlap=pairs2Overlap)
    
    if makeChanges == True:    
        val = input("Do you want to move these dates and redifine params? [y/n]: ")
         
        if val =='y':
            print('ok, moved directories, and reassigned param variables...')
            print('rerun smartLooks.py and runsnaphu.py')
            
            if not os.path.isdir('backup'):
                os.system('mkdir backup')
                os.system('cp ./params.npy backup/')
            
            for b in badDates:
                os.system('mv ' + slcdir + '/' + b + ' ' + slcdir + '/_' + b)
              
            
            datesNew = dates[dates!=badDates]
            
            
            # Redefine dates, pairs
            skip = 1
            dat = list()
            for f in flist:
                dat.append(f[-8:])
            dat.sort()
            pairs1=list()
            pairs2=list()
            pairs = list()
            for ii,d in enumerate(dat):
                for jj in np.arange(1,skip+1):
                    try:
                        pairs.append(dat[ii] + '_' + dat[ii+jj])
                    except:
                        pass
            
            dn2 = list()  
            dec_year = list()
            for d in dat:
                yr = d[0:4]
                mo = d[4:6]
                day = d[6:8]
                dt = date.toordinal(date(int(yr), int(mo), int(day)))
                dn2.append(dt)
                d0 = date.toordinal(date(int(yr), 1, 1))
                doy = np.asarray(dt)-d0+1
                dec_year.append(float(yr) + (doy/365.25))
            dn2 = np.asarray(dn2)
            dn20 = dn2-dn2[0] # make relative to first date
            
            
            # Save arrays and variables to a dictionary 'params'
            params['dates'] =        dat
            params['pairs'] =        pairs
            params['dec_year'] =     dec_year
            params['dn'] =           dn2
            params['dn0'] =          d0
            
            np.save('params.npy',params)
    if plotStuff:
        plt.show()
        
        
if __name__ == '__main__':
    mincor=0.7
    gamThresh=0.7
    plotStuff=True
    makeChanges=False
    weeding(mincor=mincor,gamThresh=gamThresh,plotStuff=plotStuff,makeChanges=makeChanges,pairs2Overlap=1)
