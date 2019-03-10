#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Kyle Murray
Thu Oct 18 18:20:23 2018
Description:
    

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import scipy.stats as st


with open('./TS/params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks = pickle.load(f)
    
phs_ifg_stack = np.load('./TS/phs_ifg_stack.npy')
    

def rms(img):
    return np.sqrt(np.nanmean(img.flatten()**2))
plt.close('all')
#ny,nx = lon_ifg.shape
pair_id = 25 # 24  date has the earthquake, 20 has the storm
num_avg = 16 # number of pairs to boost up the atmosphere
plot_flag =1
ymin,ymax=600,2000
xmin,xmax=1475,4800
# Make a gaussian signal to add to an ifg
#rx = 20
#ry = 20
#rx2 = np.floor(rx*3)
#ry2 = np.floor(ry*3)
#gausx = np.exp( np.divide( -np.square(np.arange(-rx2,rx2)), np.square(rx)));
#gausy = np.exp( np.divide( -np.square(np.arange(-ry2,ry2)), np.square(ry)));
#gaus = gausx[:, np.newaxis] * gausy[np.newaxis, :]
#
ifg = phs_ifg_stack[pair_id][ymin:ymax,xmin:xmax]# - np.nanmean(phs_ifg_stack[pair_id][ymin:ymax,xmin:xmax])

rms1 = list()
rms2 = list()
rms1.append(rms(ifg))

vmin=np.nanmedian(ifg) - 2* np.nanstd(ifg)
vmax=np.nanmedian(ifg) + 2* np.nanstd(ifg)

#
#        vmin=np.nanmedian(ifg_correct_stack) - 2* np.nanstd(ifg_correct_stack)
#        vmax=np.nanmedian(ifg_correct_stack) + 2* np.nanstd(ifg_correct_stack)
#        plt.imshow(np.flipud(ifg_correct_stack),cmap=plt.cm.RdBu,vmin=vmin,vmax=vmax)


vmin = -10
vmax = 10

for num_avg in np.arange(20,21):
    r=list()
    l=list()
    
    # Stack of pairs made with date two of ifg (right hand side)
    r.append(phs_ifg_stack[pair_id+1][ymin:ymax,xmin:xmax])
    for ii in np.arange(1,num_avg):
        r.append( (r[ii-1]) + phs_ifg_stack[ii+pair_id+1][ymin:ymax,xmin:xmax] )
    r = np.asarray(r)
    
    # Stack of pairs made with date one of ifg (left hand side)
    l.append(phs_ifg_stack[pair_id-1][ymin:ymax,xmin:xmax])
    for ii in np.arange(1,num_avg):
        l.append( (l[ii-1]) + phs_ifg_stack[pair_id-1-ii][ymin:ymax,xmin:xmax] )
    l = np.asarray(l)
    
    a5 = (1/num_avg) * np.sum(l,axis=0) # a5 has 8X atm of date5
    a6 = (1/num_avg) * np.sum(r,axis=0) # a6 has 8X atm of date6 
    ifg_correct_stack = ifg+a5+a6
    ifg -= np.nanmean(ifg)
    a5 -= np.nanmean(a5)
    a6 -= np.nanmean(a6)
    if plot_flag:
        fig = plt.figure()
        ax = fig.add_subplot(221); plt.title('IFG')
        plt.imshow(ifg,cmap=plt.cm.RdBu,vmin=vmin,vmax=vmax)
        ax = fig.add_subplot(222); plt.title('d1 atm')
        plt.imshow(a5,cmap=plt.cm.RdBu,vmin=vmin,vmax=vmax)
        ax = fig.add_subplot(223); plt.title('d2 atm')
        plt.imshow(a6,cmap=plt.cm.RdBu,vmin=vmin,vmax=vmax)
        ax = fig.add_subplot(224); plt.title('Stacked')
#        vmin=np.nanmedian(ifg_correct_stack) - 1* np.nanstd(ifg_correct_stack)
#        vmax=np.nanmedian(ifg_correct_stack) + 1* np.nanstd(ifg_correct_stack)
        plt.imshow(ifg_correct_stack,cmap=plt.cm.RdBu,vmin=vmin,vmax=vmax)

#    # Now only take negative values from d1 and postitive from d2
#    a5[np.where(a5<0)] = 0
#    a6[np.where(a6>0)] = 0
#    ifg_gac_correct = ifg_gac+a5+a6 # add them to the ifg
#    
#    if plot_flag:
#        fig = plt.figure()
#        ax = fig.add_subplot(141); plt.title('IFG')
#        plt.imshow(ifg_gac,cmap=plt.cm.RdBu,vmin=-10,vmax=10)
#        ax = fig.add_subplot(142); plt.title('d1 atm zeroed')
#        plt.imshow(a5,cmap=plt.cm.RdBu,vmin=-10,vmax=10)
#        ax = fig.add_subplot(143); plt.title('d2 atm zeroed')
#        plt.imshow(a6,cmap=plt.cm.RdBu,vmin=-10,vmax=10)
#        ax = fig.add_subplot(144); plt.title('Stacked zeroed')
#        plt.imshow(ifg_gac_correct,cmap=plt.cm.RdBu,vmin=-10,vmax=10)


#    print('RMS corrected with zeroed atm: ' + str(rms(ifg_gac_correct)))
    
    rms1.append(rms(ifg_correct_stack))
print('RMS original: ' + str(rms(ifg)))
print('RMS corrected with stacking atm: ' + str(rms(ifg_correct_stack)))
rms1 = np.asarray(rms1)

plt.figure()
plt.plot(rms1)
plt.xlabel('number of images used to stack atm')
plt.ylabel('RMS of image')
#plt.legend(['stack','zeroed stack'])

#fig = plt.figure()
#ax = fig.add_subplot(121)
#plt.imshow(phs_ifg_stack[24][ymin:ymax,xmin:xmax],cmap=plt.cm.RdBu,vmin=-10,vmax=10)
#plt.title('Original IFG')
#ax = fig.add_subplot(122)
#plt.imshow(ifg_correct_stack,cmap=plt.cm.RdBu,vmin=-3,vmax=3)
#plt.title('Corrected IFG')

#ifg_correct_stack -= np.nanmean(ifg_correct_stack)

#vmin=-8
#vmax=8
#
#if plot_flag:
#    # Plot ifg
#    pad=0
#    plt.rc('font',size=12)
#    fig = plt.figure(figsize=(16,7))
#    ax = fig.add_subplot(121)
#    ax.set_title('Interferogram')
#    m = Basemap(projection='cyl',\
#            llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
#            llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='l')
#    m.drawstates(linewidth=0.5,zorder=6)
#    m.drawparallels(np.arange(np.floor(minlat), np.ceil(maxlat), 2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])  # set linwidth to zero so there is no grid
#    m.drawmeridians(np.arange(np.floor(minlon), np.ceil(maxlon),2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])
#    m.drawmapboundary(fill_color='white',zorder=0)  # this will be the background color (oceans)
#    cf = m.pcolormesh(lon_ifg[ymin:ymax,xmin:xmax],lat_ifg[ymin:ymax,xmin:xmax], ifg_gac,vmin=vmin,vmax=vmax, cmap=plt.cm.RdBu, shading='flat',latlon=True, zorder=8)
#    cbar = m.colorbar(cf,location='bottom',pad="10%")
#    cbar.set_label('cm')
#    
#    # Plot nexrad
#    ax = fig.add_subplot(122)
#    ax.set_title('Corrected IFG')
#    m = Basemap(projection='cyl',\
#            llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
#            llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='l')
#    m.drawstates(linewidth=0.5,zorder=6)
#    m.drawparallels(np.arange(np.floor(minlat), np.ceil(maxlat), 2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])  # set linwidth to zero so there is no grid
#    m.drawmeridians(np.arange(np.floor(minlon), np.ceil(maxlon),2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])
#    m.drawmapboundary(fill_color='white',zorder=0)  # this will be the background color (oceans)
#    cf = m.pcolormesh(lon_ifg[ymin:ymax,xmin:xmax],lat_ifg[ymin:ymax,xmin:xmax],ifg_correct_stack, vmin=vmin,vmax=vmax,cmap=plt.cm.RdBu,shading='flat',latlon=True, zorder=8)
#    cbar = m.colorbar(cf,location='bottom',pad="10%")
#    cbar.set_label('cm')




# Now compare with taking median/or mean before and after ifg
phs_stack_cum = list()
phs_stack_cum.append(np.zeros((ymax-ymin,xmax-xmin)))
for ii in np.arange(1,nd):
    phs_stack_cum.append(phs_stack_cum[ii-1]+phs_ifg_stack[ii][ymin:ymax,xmin:xmax])
phs_stack_cum = np.asarray(phs_stack_cum)


ts = phs_stack_cum[:,560,560]
plt.figure()
plt.plot(ts,'.')
plt.xlabel('Time')
plt.ylabel('Displacement (cm)')


median_before = np.nanmean(phs_stack_cum[1:pair_id-1,:,:],axis=0)
median_after =np.nanmean(phs_stack_cum[pair_id+1:,:,:],axis=0)
ifg_corrected= median_after-median_before

fig = plt.figure()
ax = fig.add_subplot(221); plt.title('IFG')
plt.imshow(phs_ifg_stack[pair_id,ymin:ymax,xmin:xmax],cmap=plt.cm.RdBu,vmin=-10,vmax=10)
ax = fig.add_subplot(222); plt.title('median before')
plt.imshow(median_before,cmap=plt.cm.RdBu,vmin=-10,vmax=10)
ax = fig.add_subplot(223); plt.title('median after')
plt.imshow(median_after,cmap=plt.cm.RdBu)
ax = fig.add_subplot(224); plt.title('corrected')
plt.imshow(ifg_corrected,cmap=plt.cm.RdBu,vmin=vmin,vmax=vmax)


# See if math worked
#bb = (1/3) * (gac_c_stack[1] +2*gac_c_stack[2] +3*gac_c_stack[3] +3*gac_c_stack[4] +2*gac_c_stack[5] +gac_c_stack[6])
#plt.figure()
#plt.imshow(bb[ymin:ymax,xmin:xmax],cmap=plt.cm.RdBu,vmin=-10,vmax=10)
#
#plt.show()
    
