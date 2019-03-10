#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Kyle Murray
Thu Oct 18 18:20:23 2018
Description:
    

"""

import numpy as np
import pickle
import isceobj
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import scipy.stats as st

with open(tsdir + '/params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks = pickle.load(f)
    

phs_ifg_stack = list()
for pair in pairs:
    ifgfile = intdir + pair + '/fine_lk.int'
    ifgImage = isceobj.createIntImage()
    ifgImage.load(ifgfile + '.xml')
    phs_ifg_stack.append(ifgImage.memMap()[:,:,0]) 
      


plt.close('all')
#ny,nx = lon_ifg.shape
pair_id = 25 # 24  date has the earthquake, 20 has the storm
#num_avg = 16 # number of pairs to boost up the atmosphere
plot_flag =1
#ymin,ymax=600,2000
#xmin,xmax=1475,4800
ymin,ymax=0,nyl
xmin,xmax=0,nxl
#ifg = phs_ifg_stack[pair_id][ymin:ymax,xmin:xmax]# - np.nanmean(phs_ifg_stack[pair_id][ymin:ymax,xmin:xmax])
ifg = phs_ifg_stack[pair_id]
vmin=np.nanmedian(ifg) - 2* np.nanstd(ifg)
vmax=np.nanmedian(ifg) + 2* np.nanstd(ifg)

for num_avg in np.arange(10,11):
    # Do real part first*******************************************************
    r=list()
    l=list()
    # Stack of pairs made with date two of ifg (right hand side)
    r.append(np.real(phs_ifg_stack[pair_id+1][ymin:ymax,xmin:xmax]))
    for ii in np.arange(1,num_avg):
        r.append( (r[ii-1]) + np.real(phs_ifg_stack[ii+pair_id+1][ymin:ymax,xmin:xmax]) )
    r = np.asarray(r)
    
    # Stack of pairs made with date one of ifg (left hand side)
    l.append(np.real(phs_ifg_stack[pair_id-1][ymin:ymax,xmin:xmax]))
    for ii in np.arange(1,num_avg):
        l.append( (l[ii-1]) + np.real(phs_ifg_stack[pair_id-1-ii][ymin:ymax,xmin:xmax]) )
    
    l = np.asarray(l)
    a5 = (1/num_avg) * np.sum(l,axis=0) # a5 has 8X atm of date5
    a6 = (1/num_avg) * np.sum(r,axis=0) # a6 has 8X atm of date6 
    ifg_correct_real = np.real(ifg)+a5+a6
    
    # Do imaginary part second*************************************************
    r=list()
    l=list()
    # Stack of pairs made with date two of ifg (right hand side)
    r.append(np.imag(phs_ifg_stack[pair_id+1][ymin:ymax,xmin:xmax]))
    for ii in np.arange(1,num_avg):
        r.append( (r[ii-1]) + np.imag(phs_ifg_stack[ii+pair_id+1][ymin:ymax,xmin:xmax]) )
    
    r = np.asarray(r)
    # Stack of pairs made with date one of ifg (left hand side)
    l.append(np.imag(phs_ifg_stack[pair_id-1][ymin:ymax,xmin:xmax]))
    for ii in np.arange(1,num_avg):
        l.append( (l[ii-1]) + np.imag(phs_ifg_stack[pair_id-1-ii][ymin:ymax,xmin:xmax]) )

    l = np.asarray(l)
    a5 = (1/num_avg) * np.sum(l,axis=0) # a5 has 8X atm of date5
    a6 = (1/num_avg) * np.sum(r,axis=0) # a6 has 8X atm of date6 
    ifg_correct_imag = np.imag(ifg)+a5+a6
    
    # Add real and imaginary corrections **************************************
    ifg_correct_stack = ifg_correct_real + (1j*ifg_correct_imag)
    ifg_correct_stack_phs =  np.arctan2(ifg_correct_real, ifg_correct_imag).astype(np.float32)

if plot_flag:
    fig = plt.figure()
    ax = fig.add_subplot(141); plt.title('IFG')
    plt.imshow(ifg,cmap=plt.cm.RdBu)
    ax = fig.add_subplot(142); plt.title('d1 atm')
    plt.imshow(a5,cmap=plt.cm.RdBu)
    ax = fig.add_subplot(143); plt.title('d2 atm')
    plt.imshow(a6,cmap=plt.cm.RdBu)
    ax = fig.add_subplot(144); plt.title('Stacked')
    plt.imshow(ifg_correct_stack_phs)

ifg_correct_stack[np.isnan(ifg_correct_stack)]=0
out = ifgImage.clone() # Copy the interferogram image from before
out.filename = intdir + pairs[pair_id] + '/fine_atm.int'
out.dump(out.filename + '.xml') # Write out xml
ifg_correct_stack.tofile(out.filename) # Write file out

plt.figure()
plt.plot(rms1)
plt.xlabel('number of images used to stack atm')
plt.ylabel('RMS of image')

## Now compare with taking median/or mean before and after ifg
#phs_stack_cum = list()
#phs_stack_cum.append(np.zeros((ymax-ymin,xmax-xmin)))
#for ii in np.arange(1,nd):
#    phs_stack_cum.append(phs_stack_cum[ii-1]+phs_ifg_stack[ii][ymin:ymax,xmin:xmax])
#phs_stack_cum = np.asarray(phs_stack_cum)
#
#
#ts = phs_stack_cum[:,560,560]
#plt.figure()
#plt.plot(ts,'.')
#plt.xlabel('Time')
#plt.ylabel('Displacement (cm)')
#
#
#median_before = np.nanmean(phs_stack_cum[1:pair_id-1,:,:],axis=0)
#median_after =np.nanmean(phs_stack_cum[pair_id+1:,:,:],axis=0)
#ifg_corrected= median_after-median_before
#
#fig = plt.figure()
#ax = fig.add_subplot(141); plt.title('IFG')
#plt.imshow(phs_ifg_stack[pair_id,ymin:ymax,xmin:xmax],cmap=plt.cm.RdBu,vmin=-10,vmax=10)
#ax = fig.add_subplot(142); plt.title('median before')
#plt.imshow(median_before,cmap=plt.cm.RdBu,vmin=-10,vmax=10)
#ax = fig.add_subplot(143); plt.title('median after')
#plt.imshow(median_after,cmap=plt.cm.RdBu)
#ax = fig.add_subplot(144); plt.title('corrected')
#plt.imshow(ifg_corrected,cmap=plt.cm.RdBu,vmin=vmin,vmax=vmax)
#
#
## See if math worked
#bb = (1/3) * (gac_c_stack[1] +2*gac_c_stack[2] +3*gac_c_stack[3] +3*gac_c_stack[4] +2*gac_c_stack[5] +gac_c_stack[6])
#plt.figure()
#plt.imshow(bb[ymin:ymax,xmin:xmax],cmap=plt.cm.RdBu,vmin=-10,vmax=10)
#
#plt.show()
#    
