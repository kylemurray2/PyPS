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
from PyPS2 import util

def weeding(mincor=0.7,gamThresh=0.7,varMax =.05, connCompCompleteness = 0.9,plotStuff=False,makeChanges=False,pairs2Overlap=1,overwriteGeo=False):
    '''
    plotStuff = True
    makeChanges = False
    mincor = .5
    gamThresh = .5
    varMax =  .05
    overwriteGeo=False
    pairs2Overlap=1
    connCompCompleteness = 0.9 # at least 90% of the ifgs must have a connected component at the given pixel, or it is masked to nan
    '''
    plt.close('all')

    
    ps = np.load('./ps.npy',allow_pickle=True).all()
    gam = np.load('Npy/gam.npy')
    
    
    if overwriteGeo:
        gamGeo = util.geocodeKM(gam)
        gamGeo[np.isnan(gamGeo)] = 0
        np.save('./TS/gam.geo.npy',gamGeo)
    

    gamFlat = gam.flatten()
    
    X,Y = np.meshgrid(range(ps.nxl),range(ps.nyl))
    
    pairs3 = list()  
    for ii,d in enumerate(ps.dates[0:-1]):
        for jj in np.arange(1,pairs2Overlap+1):
            if ii+jj < len(ps.dates):
                pairs3.append(ps.dates[ii] + '_' + ps.dates[ii+jj])
    
    
    stack = []
    for p in pairs3:
        unw_file = ps.intdir + '/' + p + '/filt.unw'
        unwImage = isceobj.createIntImage()
        unwImage.load(unw_file + '.xml')
        unw = unwImage.memMap()[:,:,0] #- unwImage.memMap()[ymin:ymax,xmin:xmax,0][r,c]
        unw = unw.copy()
        # unw[np.isnan(gam)] = np.nan
        stack.append(unw)
    stack = np.asarray(stack,dtype=np.float32)
    
    # stackTimeMean = np.nanmean(stack,axis=0)
    # stackTimeVar  = np.nanvar(stack,axis=0)
    # plt.figure();plt.imshow(stackTimeMean);plt.title('stack time mean')
    # plt.figure();plt.imshow(stackTimeVar);plt.title('stack time var')
    
    corStack = []
    for p in pairs3:
        cor_file = ps.intdir + '/' + p + '/cor.r4'
        corImage = isceobj.createIntImage()
        corImage.load(cor_file + '.xml')
        cor = corImage.memMap()[:,:,0]
        cor = cor.copy()
        # cor[np.isnan(gam)] = np.nan
        corStack.append(cor)
    corStack = np.asarray(corStack,dtype=np.float32)[:,:,:]
    
    connStack = []
    for p in pairs3:
        conn_file = ps.intdir + '/' + p + '/filt.unw.conncomp'
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
    
    
    

    corAvgMap = np.nanmean(corStack,axis=0)   
    corVar = np.nanvar(corStack,axis=0)


    np.save('Npy/cor.npy',corAvgMap)
    np.save('Npy/corVar.npy',corVar)
    
    if overwriteGeo:
        corGeo = util.geocodeKM(corAvgMap)
        corGeo[np.isnan(corGeo)] = 0
        np.save('./TS/cor.geo.npy',corGeo)

        corVarGeo = util.geocodeKM(corVar)
        corVarGeo[np.isnan(corVarGeo)] = 0
        np.save('./TS/corVar.geo.npy',corVarGeo)
    
    
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
    dateCor = []
    for ii in np.arange(0,len(ps.dn)):
        dt = ps.dates[ii]
        # first find all of the ifgs that have that date. This is generalized in case there are redundant pairs.
        dtPairs = []
        for jj,p in enumerate(pairs3):
            if p[0:8] == dt or p[9:] == dt:
                dtPairs.append(jj)
        pVars = []
        pAvgs = []
        for kk in dtPairs:
            iv = stack[kk,:,:]
            iv[np.isnan(corStack[kk,:,:])] =np.nan
            pVars.append(np.nanvar(iv))
            pAvgs.append(np.nanmedian(corStack[kk,:,:]))
            

        dateVar.append(np.nanmin(pVars))
        dateCor.append(np.nanmax(pAvgs))
    dateVar = np.asarray(dateVar,dtype=np.float32)
    dateCor = np.asarray(dateCor,dtype=np.float32)

    connSum = np.sum(connStack,axis=0)
    np.save('Npy/connSum.npy',connSum)
    
    
    # Make masks based on 4 criteria
    gamMsk = np.ones(gam.shape)
    gamMsk[gam<gamThresh] = 0
    connMsk = np.ones(gam.shape)
    connMsk[connSum<round(connCompCompleteness*ps.nd)] = 0
    corMsk = np.ones(gam.shape)
    corMsk[corAvgMap<mincor] = 0
    varMsk = np.ones(gam.shape)
    varMsk[corVar>varMax] = 0
    
    # # Make the final msk
    msk = np.ones(gam.shape)
    msk[gamMsk==0]  = 0
    msk[connMsk==0] = 0
    msk[corMsk==0]  = 0
    msk[varMsk==0]  = 0
    
    mskSum = gamMsk+connMsk+corMsk+varMsk
    
    np.save('Npy/msk.npy',msk)
    
    if overwriteGeo:
        gamGeo= util.geocodeKM(gam,method='linear')
        np.save('./TS/gam.geo.npy',gamGeo)
        
        mskGeo= util.geocodeKM(msk,method='nearest')
        np.save('./TS/msk.geo.npy',gamGeo)
        
        # connSumGeo= util.geocodeKM(connSum,method='linear')
        # np.save('./TS/connSum.geo.npy',connSumGeo)
        # mskGeo = np.ones(gamGeo.shape)
        # mskGeo[connSumGeo < round(connCompCompleteness*ps.nd)] = 0
        # mskGeo[gamGeo<gamThresh] = 0
        # mskGeo[corGeo<mincor] = 0
        # np.save('./TS/msk.geo.npy',mskGeo)
    
        
    if plotStuff:
        fig,ax = plt.subplots(3,1,figsize=(8,5))
        ax[0].plot(ps.dec_year[1:],corAvg);ax[0].set_xlabel('Year');ax[0].set_ylabel('IFG median coherence')
        ax[1].plot(corAvg);ax[1].set_xlabel('time index');ax[1].set_ylabel('IFG median coherence')
        ax[2].plot(dateCor);ax[2].set_xlabel('time index');ax[2].set_ylabel('Date avg cor (average of associated ifgs')

        fig,ax = plt.subplots(3,1,figsize=(8,5))
        ax[0].plot(ps.dec_year[1:],ifgVar);ax[0].set_xlabel('time index');ax[0].set_ylabel('IFG coherence variance')
        ax[1].plot(ifgVar);ax[1].set_xlabel('Year');ax[1].set_ylabel('IFG coherence variance')
        ax[2].plot(dateVar);ax[2].set_xlabel('time index');ax[2].set_ylabel('Date variance (average of associated ifgs')
    
        fig,ax = plt.subplots(2,2,figsize=(12,10))
        ax[0,0].imshow(gam,vmin=.45,vmax=.55,cmap='magma');ax[0,0].set_title('Gamma0')
        ax[0,1].imshow(connSum,cmap='jet_r'); ax[0,1].set_title('Number of images with connected components')
        ax[1,0].imshow(corAvgMap,cmap='magma');ax[1,0].set_title('Average Correlation')
        ax[1,1].imshow(corVar,cmap='magma');ax[1,1].set_title('Correlation Variance')
        
        fig,ax = plt.subplots(2,2,figsize=(12,10))
        ax[0,0].imshow(gamMsk,cmap='magma');ax[0,0].set_title('Gamma0 Mask')
        ax[0,1].imshow(connMsk,cmap='jet_r'); ax[0,1].set_title('Connected comp. mask')
        ax[1,0].imshow(corMsk,cmap='magma');ax[1,0].set_title('Average Correlation Mask')
        ax[1,1].imshow(varMsk,cmap='magma');ax[1,1].set_title('Correlation Variance Mask')
        
        plt.figure()
        plt.plot(np.ravel(corAvgMap)[::10],np.ravel(corVar)[::10],'.',markersize = 1)
        plt.plot([mincor,mincor,mincor,1,1,1,1,mincor],[0,varMax,varMax,varMax,varMax,0,0,0])
        plt.xlabel('Average Correlation');plt.ylabel('Correlation variance');plt.show()

        plt.figure()
        plt.imshow(msk,cmap='magma')
        plt.title('Final Mask')
        plt.show()

        plt.figure()
        plt.imshow(mskSum,cmap='magma')
        plt.title('Sum of individual Masks')
        plt.show()

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
                os.system('mv ' + ps.slcdir + '/' + b + ' ' + ps.slcdir + '/_' + b)
              
            
            datesNew = ps.dates[ps.dates!=badDates]
            
            
            # Redefine dates, pairs
            skip = 1
            dat = []
            for f in flist:
                dat.append(f[-8:])
            dat.sort()
            pairs1=[]
            pairs2=[]
            pairs =[]
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
            ps.dates    =        dat
            ps.pairs    =        pairs
            ps.dec_year =     dec_year
            ps.dn       =           dn2
            ps.dn0      =          d0
            
            np.save('params.npy',params)
    if plotStuff:
        plt.show()
        
        
if __name__ == '__main__':
    plotStuff = True
    makeChanges = False
    mincor = .5
    gamThresh = .5
    varMax =  .05
    pairs2Overlap=1
    overwriteGeo=False
    weeding(mincor,gamThresh,plotStuff,makeChanges,pairs2Overlap,overwriteGeo)
