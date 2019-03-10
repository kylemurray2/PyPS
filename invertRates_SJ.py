#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:45:20 2018

@author: kdm95
"""

import numpy as np
import isceobj
import pickle
import os
from datetime import date
from mpl_toolkits.basemap import Basemap
from KyPy import savitzky_golay
from matplotlib import pyplot as plt

with open('./TS/params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    pairs,dates,dec_year,dn,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,lon_bounds,lat_bounds,ymin,ymax,alks,rlks = pickle.load(f)
skip=3
# Run refDef.py before this to make the alld_flat_topo.npy file
alld_flat_topo=np.load('alld_flat_topo.npy')
pairs = np.load('pairs_cut.npy')
#crop_mask = np.load('crop_mask.npy')
ymin=390
ymax=3470
xmin=28
xmax=3940
nxl = xmax-xmin
nyl = ymax-ymin
#alld_flat_topo  = alld

# Convert pairs to dates
dates = list()
for p in pairs:
    dates.append(p[0:8])
    dates.append(p[9:])
dates = np.unique(np.asarray(dates,dtype = str))

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

## Example pixel for check
example_ts = list([[1213,2379],[1211,2708]])
example_ts = list([[763,2053],[772,2099]])

d1=0
for ii,point in enumerate(example_ts):
    c=point[0];r=point[1]
    idx = ((r)*nxl)+c #finds the index of flattened array based on row col in image
    y=-alld_flat_topo[:,idx]*lam/(4*np.pi)*100
    G = np.vstack([dn0[d1:], np.ones((len(dn0[d1:]),1)).flatten()]).T
    Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
    mod    = np.dot(Gg, y[d1:])
    y2 = mod[0]*dn0[d1:] + mod[1]
    plt.figure()
    plt.plot(dec_year,y[d1:],'.')
    plt.plot(dec_year[d1:],y2[d1:])
    plt.xlabel('Year'); plt.ylabel('Displacement (cm)')
    plt.title(str(np.round((mod[0]*365), decimals=2)) + ' cm/yr in LOS')
#    plt.savefig(workdir + 'Figs/timeseries' + str(ii) + '.svg',transparent=True,dpi=200)


G = np.vstack([dn0, np.ones((len(dn0),1)).flatten()]).T
Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
mod   = np.dot(Gg, alld_flat_topo)
rates = np.reshape(mod[0,:],(nyl,nxl))*lam/(4*np.pi)*100*365 # cm/yr
rates = rates.astype(np.float32)
r_nomsk = rates
plt.figure();plt.imshow(r_nomsk)

#ra = np.zeros((rates.shape))
#ra[364:3060,1:2437]=rates[364:3060,1:2437]
#offs  = np.reshape(mod[1,:],(nyl, nxl))
#synth  = np.dot(G,mod);
#res    = (alld_flat_topo-synth)*lam/(4*np.pi)*100 # cm
#resstd = np.std(res,axis=0)
#resstd = np.reshape(resstd,(nyl, nxl))


# MASKING______________________________
# Load gamma0_lk
f = tsdir + '/gamma0_lk.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0_lk= intImage.memMap()[ymin:ymax,xmin:xmax,0] 
gamma0_lk=gamma0_lk.copy()
# Load height file
h = workdir + '/merged/geom_master/hgt_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
hgt = hImg.memMap()[ymin:ymax,xmin:xmax,0].astype(np.float32)
# elevations at 4 of the main lakes that we'll mask out

# Mask the rates matrix
#gamma0_lk*=crop_mask
gamma0_lk[np.isnan(gamma0_lk)]=0
rates[np.where(gamma0_lk<.1)]=np.nan #masks the low coherence areas
rates[np.where( (hgt<-103) ) ]=np.nan # masks the water
#rates=np.fliplr(rates)

#rates[np.isnan(rates)]=0


# Save rates
fname = tsdir + '/rates_flat.unw'
out = isceobj.createIntImage() # Copy the interferogram image from before
out.dataType = 'FLOAT'
out.filename = fname
out.width = nxl
out.length = nyl
out.dump(fname + '.xml') # Write out xml
rates.tofile(out.filename) # Write file out
out.renderHdr()
out.renderVRT()

# GEOCODE
#cmd = 'geocodeIsce.py -f ' + tsdir + 'rates_flat.unw -d ' + workdir + 'DEM/demLat_N33_N35_Lon_W119_W116.dem -m ' + workdir + 'master/ -s ' + workdir + 'pawns/20150514 -r ' + str(rlks) + ' -a ' + str(alks) + ' -b "'" 33 35 -118 -116"'" '
#os.system(cmd)


h = workdir + '/merged/geom_master/lon_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
lo = hImg.memMap()[ymin:ymax,xmin:xmax,0].astype(np.float32)
lo[lo==0]=np.nan
h = workdir + '/merged/geom_master/lat_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
la = hImg.memMap()[ymin:ymax,xmin:xmax,0].astype(np.float32)
la[la==0]=np.nan

for l in np.arange(0,nyl):
    ll = lo[l,:]
    if not np.isnan(ll.max()):
        break

for p in np.arange(l+1,nyl):
    ll = lo[p,:]
    if np.isnan(ll.max()):
        break
l+=1
ul = (lo[l,0],la[l,0])
ur = (lo[l,-1],la[l,-1])
ll = (lo[p-1,0],la[p-1,0])
lr = (lo[p-1,-1],la[p-1,-1])

lons = np.array([ul[0],ur[0],ur[0],lr[0],lr[0],ll[0],ll[0],ul[0]])
lats = np.array([ul[1],ur[1],ur[1],lr[1],lr[1],ll[1],ll[1],ul[1]])
pad=0
minlat=lats.min()
maxlat=lats.max()
minlon=lons.min()
maxlon=lons.max()




# Get rid of bad overlaps area
nprof = 10
def improfile(z, x0, y0, x1, y1):
    """
    Get a profile
    Captures 1d profile values from point A to B specified as indices. 
    Inputs: im, x0, y0, x1, y1
    Outputs: z (vector of values along the profile) 
    """
    length = int(np.hypot(x1-x0, y1-y0))
    x, y = np.linspace(x0, x1, length,dtype=int), np.linspace(y0, y1, length,dtype=int)
    zi=np.empty((nprof,len(x)))

    for jj in np.arange(0,nprof):
        zi[jj,:]=z[y-jj,x+jj]
#        plt.plot(x+jj,y-jj)
    
    # Extract the values along the line
#    zi = z[y,x]
    return zi
#
#
#x0,y0 = 2597,60
#x1,y1 = 3740,2140

# Plot rate map
plt.rc('font',size=12)
plt.figure(figsize=(12,12))
m = Basemap(epsg=3395, llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
        llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='l')
m.drawparallels(np.arange(np.floor(minlat-pad), np.ceil(maxlat+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
m.drawmeridians(np.arange(np.floor(minlon-pad), np.ceil(maxlon+pad),1), linewidth=0,labels=[1,0,0,1])
m.arcgisimage(service='World_Shaded_Relief',xpixels=1500)
cf = m.pcolormesh(lo,la,rates,shading='flat',latlon=True, cmap=plt.cm.Spectral_r,zorder=8,vmin=-1,vmax=1)
m.readshapefile('/data/kdm95/LA/qfaults/qfaults_la', 'qfaults_la',zorder=30)

m.plot(lo_p1,la_p1,color='red',zorder=40,latlon=True)
m.plot(lo_p2,la_p2,color='red',zorder=40,latlon=True)
m.plot(lo_p3,la_p3,color='red',zorder=40,latlon=True)
m.plot(lo_p4,la_p4,color='red',zorder=40,latlon=True)

cbar = m.colorbar(cf,location='bottom',pad="10%")
cbar.set_label('cm')
plt.show()
#plt.savefig('Figs/rate_map.png', transparent=True, dpi=500)

lonpt, latpt = m(16558,67529,inverse=True)

fault_listx = list([161895,129528,122949,115581,113213,51636,40058])
fault_listy = list([135211,120474,117580,115475,113633,86528,81528])
dst=list()
lonpt, latpt = m(fault_listx[0],fault_listy[0],inverse=True)

for ii in np.arange(1,len(fault_listx)):
    lonpt2, latpt2 = m(fault_listx[ii],fault_listy[ii],inverse=True)
    dst.append(np.sqrt(np.square(lonpt-lonpt2)+np.square(latpt-latpt2)))
   
# PROFILE 1 san jacinto
x0,y0 = 3345,1563
x1,y1 = 3606,1683
# Convert to lat lon
lo_p1=(lo[y0,x0],lo[y1,x1])
la_p1=(la[y0,x0],la[y1,x1])
#find total distance
prof_dist = np.sqrt(np.square(lo_p1[0]-lo_p1[1]) + np.square(la_p1[0]-la_p1[1])) *111
# Plot profiles
plt.figure(figsize=(6,4))
zi=improfile(rates, x0, y0, x1, y1)
zii = np.nanmedian(zi,axis=0)
prof_vec = np.linspace(0,prof_dist,len(zii))
plt.plot(prof_vec,zii*10,'k.')
plt.vlines(5.8,ymin=-35,ymax=20,color='red')
plt.vlines(11.1,ymin=-35,ymax=20, color='red')
#plt.ylim([-1,.5])
plt.xlabel('Profile Distance (km)')
plt.ylabel('LOS Rate (mm/yr)')
#plt.savefig(workdir + 'Figs/prof1.svg',transparent=True,dpi=100)

# PROFILE 2 east thing
x0,y0 = 433,863
x1,y1 = 554,1095
# Convert to lat lon
lo_p2=(lo[y0,x0],lo[y1,x1])
la_p2=(la[y0,x0],la[y1,x1])
#find total distance
prof_dist = np.sqrt(np.square(lo_p2[0]-lo_p2[1]) + np.square(la_p2[0]-la_p2[1])) *111
plt.figure(figsize=(6,4))
zi=improfile(r, x0, y0, x1, y1)
zii = np.nanmedian(zi,axis=0)
prof_vec = np.linspace(0,prof_dist,len(zii))
plt.plot(prof_vec,zii*10,'k.')
plt.vlines(2.3,ymin=-15,ymax=12,color='red')
plt.vlines(4.5,ymin=-15,ymax=12,color='red')
#plt.ylim([-1,.5])
plt.xlabel('Profile Distance (km)')
plt.ylabel('LOS Rate (mm/yr)')
#plt.savefig(workdir + 'Figs/prof2.svg',transparent=True,dpi=100)

# PROFILE3 Big profile
x0,y0 = 1838,38
x1,y1 = 6262,1877
# Convert to lat lon
lo_p3=(lo[y0,x0],lo[y1,x1])
la_p3=(la[y0,x0],la[y1,x1])
#find total distance
prof_dist = np.sqrt(np.square(lo_p3[0]-lo_p3[1]) + np.square(la_p3[0]-la_p3[1])) *111
plt.figure(figsize=(14,4))
zi=improfile(r, x0, y0, x1, y1)
zii = np.nanmedian(zi,axis=0)
prof_vec = np.linspace(0,prof_dist,len(zii))
plt.plot(prof_vec,zii*10,'k.')
plt.vlines(np.asarray(dst)*111,ymin=-22,ymax=30,color='red')
plt.vlines(83,ymin=-22,ymax=30, linestyle='dashdot',color='red')
#plt.ylim([-1,.5])
plt.xlabel('Profile Distance (km)')
plt.ylabel('LOS Rate (mm/yr)')
#plt.savefig(workdir + 'Figs/prof3.svg',transparent=True,dpi=100)


# PROFILE 4 Mountain subsidence northeast
x0,y0 = 630,36
x1,y1 = 912,198
# Convert to lat lon
lo_p4=(lo[y0,x0],lo[y1,x1])
la_p4=(la[y0,x0],la[y1,x1])
#find total distance
prof_dist = np.sqrt(np.square(lo_p4[0]-lo_p4[1]) + np.square(la_p4[0]-la_p4[1])) *111
fig, ax1 = plt.subplots(figsize=(6,4))
zi=improfile(-r, x0, y0, x1, y1)
zii = np.nanmedian(zi,axis=0)
prof_vec = np.linspace(0,prof_dist,len(zii))
ax1.plot(prof_vec,zii*10,'k.')
ax1.vlines(8.4,ymin=10*zii.min(),ymax=10*zii.max(),color='red')
#plt.ylim([-1,.5])
ax1.set_xlabel('Profile Distance (km)')
ax1.tick_params(axis='y')
ax1.set_ylabel('LOS Rate (mm/yr)')
# topo profile
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
zi_topo = improfile(topo, x0, y0, x1, y1)
zii_topo = np.nanmedian(zi_topo,axis=0)
ax2.plot(prof_vec,zii_topo)
ax2.vlines(8.4,ymin=zii_topo.min(),ymax=zii_topo.max(),color='red')
#plt.ylim([-1,.5])
ax2.set_xlabel('Profile Distance (km)')
ax2.set_ylabel('Elevation (m)',color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
plt.savefig(workdir + 'Figs/prof4_topo.svg',transparent=True,dpi=100)


