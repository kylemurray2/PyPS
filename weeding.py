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

makeChanges = False
mincor = .45
gamThresh = .45


plt.close('all')
params = np.load('params.npy',allow_pickle=True).item()
geom = np.load('geom.npy',allow_pickle=True).item()

locals().update(params)
locals().update(geom)

nxl = params['nxl']
nyl = params['nyl']


# MASKING______________________________
gam = np.load('gam.npy')
gam[gam==0] = np.nan
gamFlat = gam.flatten()

r,c = 115,1665
X,Y = np.meshgrid(range(nxl),range(nyl))

stack = []
for p in params['pairs']:
    unw_file = params['intdir'] + '/' + p + '/filt.unw'
    unwImage = isceobj.createIntImage()
    unwImage.load(unw_file + '.xml')
    unw = unwImage.memMap()[:,:,0] #- unwImage.memMap()[ymin:ymax,xmin:xmax,0][r,c]
    unw = unw.copy()
    unw[np.isnan(gam)] = np.nan
    stack.append(unw)
stack = np.asarray(stack,dtype=np.float32)

stackTimeMean = np.nanmean(stack,axis=0)
stackTimeVar = np.nanvar(stack,axis=0)

# plt.figure();plt.imshow(stackTimeMean);plt.title('stack time mean')
# plt.figure();plt.imshow(stackTimeVar);plt.title('stack time var')

corStack = []
for p in params['pairs']:
    cor_file = params['intdir'] + '/' + p + '/cor.r4'
    corImage = isceobj.createIntImage()
    corImage.load(cor_file + '.xml')
    cor = corImage.memMap()[:,:,0]
    cor = cor.copy()
    cor[np.isnan(gam)] = np.nan
    corStack.append(cor)
corStack = np.asarray(corStack,dtype=np.float32)[:,:,:]

# average cor value for each pair
corAvg = []
ifgVar = []
for ii in np.arange(0,len(pairs)):
    corAvg.append(np.nanmean(corStack[ii,:,:]))
    iv = stack[ii,:,:]
    iv[np.isnan(corStack[ii,:,:])] =np.nan
    ifgVar.append(np.nanvar(iv))

corAvg = np.asarray(corAvg,dtype=np.float32)
ifgVar = np.asarray(ifgVar,dtype=np.float32)

plt.figure();plt.plot(corAvg);plt.xlabel('time index');plt.ylabel('Correlation')
plt.figure();plt.plot(ifgVar);plt.xlabel('time index');plt.ylabel('IFG variance')


# plt.figure()
# plt.plot(np.ravel(gam)[::10],np.ravel(np.nanmean(corStack,axis=0))[::10],'.',markersize = 1)
# plt.xlabel('Gamma 0');plt.ylabel('Average Correlation');plt.show()

corAvgMap = np.nanmean(corStack,axis=0)
corVar = np.nanvar(corStack,axis=0)
plt.figure();plt.imshow(corAvgMap);plt.title('Average Correlation')
plt.figure();plt.imshow(corVar);plt.title('Correlation Variance')
np.save('cor.npy',corAvgMap)
# a = np.zeros(gam.shape)
# a[np.where((corAvgMap<0.6)&(corVar>.04))] = 1
# plt.figure();plt.imshow(a)

# plt.figure()
# plt.plot(np.ravel(np.nanmean(corStack,axis=0))[::10],np.ravel(np.nanvar(corStack,axis=0))[::10],'.',markersize = 1)
# plt.xlabel('Average Correlation');plt.ylabel('Correlation variance');plt.show()

# Find the bad dates
# gamThresh = np.nanmedian(gam) - 2*np.nanstd(gam)
corThresh = .4#np.nanmean(corAvgMap) -2*np.nanstd(corAvgMap)
ifgVarThresh = np.nanmean(ifgVar) + np.nanstd(ifgVar)
badPairs = np.where((corAvg<corThresh) | (ifgVar>ifgVarThresh))[0]

# We'll assume it's a bad dates if it appears in multiple bad pairs.
possibleBadDates = []
for b in badPairs:
    possibleBadDates.append(pairs[b][0:8])
    possibleBadDates.append(pairs[b][9:])

badDates = []
for b in possibleBadDates:
    if possibleBadDates.count(b) > 1:
        badDates.append(b)
badDates = np.unique(badDates)



# This loop looks at ifgs associated with each date and finds the minimum ifg var for each date.
#   This is a way to find which dates are noisy as opposed to individual ifgs. 
dateVar = []
for ii in np.arange(0,len(dn)):
    dt = dates[ii]
    # first find all of the ifgs that have that date. This is generalized in case there are redundant pairs.
    dtPairs = []
    for jj,p in enumerate(pairs):
        if p[0:8] == dt or p[9:] == dt:
            dtPairs.append(jj)
    pVars = []
    for kk in dtPairs:
        iv = stack[kk,:,:]
        iv[np.isnan(corStack[kk,:,:])] =np.nan
        pVars.append(np.nanvar(iv))
    dateVar.append(np.nanmin(pVars))
dateVar = np.asarray(dateVar,dtype=np.float32)
plt.figure();plt.plot(dateVar);plt.xlabel('time index');plt.ylabel('Date variance (average of associated ifgs')

plt.figure();plt.imshow(gam,vmin=.45,vmax=.5)
plt.figure();plt.imshow(np.nanmean(corStack,axis=0))

# Make the mask msk
msk = np.ones(gam.shape)
msk[gam<gamThresh] = 0
msk[np.nanmean(corStack,axis=0)<mincor] = 0
msk[np.isnan(gam)] = 0
plt.figure();plt.imshow(msk);plt.title('mask')
np.save('msk.npy',msk)

print('\n The bad dates might be: \n')
print(badDates)

plt.figure();plt.imshow(msk)

if makeChanges == True:    
    val = input("Do you want to move these dates and redifine params? [y/n]: ")
     
    if val =='y':
        print('ok, moved directories, and reassigned param variables...')
        print('rerun smartLooks.py and runsnaphu.py')

        for b in badDates:
            os.system('mv ' + slcdir + '/' + b + ' ' + slcdir + '/_' + b)
          
        # Now update the params.npy file to exclude the bad ifgs
        flist = glob.glob(slcdir + '/2*')
        
        # Redefine dates, pairs
        skip = 1
        dates = list()
        for f in flist:
            dates.append(f[-8:])
        dates.sort()
        #dates = np.unique(np.asarray(dates,dtype = str))
        pairs1=list()
        pairs2=list()
        pairs = list()
        for ii,d in enumerate(dates):
            for jj in np.arange(1,skip+1):
                try:
                    pairs.append(dates[ii] + '_' + dates[ii+jj])
                except:
                    pass
        
        dn = list()  
        dec_year = list()
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
        
        
        # Save arrays and variables to a dictionary 'params'
        params['dates'] =        dates
        params['pairs'] =        pairs
        params['dec_year'] =     dec_year
        params['dn'] =           dn
        params['dn0'] =          dn0
        
        np.save('params.npy',params)