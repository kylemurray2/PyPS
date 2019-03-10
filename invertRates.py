#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:45:20 2018

@author: kdm95
"""

import numpy as np
import isceobj
#import os
#from datetime import date
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
from skimage.measure import profile_line as pl
import fitSine

params = np.load('params.npy').item()
locals().update(params)
geom = np.load('geom.npy').item()

skip=3
# Run refDef.py before this to make the alld_flat_topo.npy file
alld_flat_topo=np.load('alld_flat_topo.npy')
pairs = np.load('pairs_cut.npy')
#crop_mask = np.load('crop_mask.npy')
ymin=406
ymax=3415
xmin=2
xmax=3966
#ymin=85
#ymax=3555
#xmin=45
#xmax=1720
nxl = xmax-xmin
nyl = ymax-ymin

# Get the geom files
hgt = geom['hgt_ifg'][ymin:ymax,xmin:xmax]
la = geom['lat_ifg'][ymin:ymax,xmin:xmax]
lo = geom['lon_ifg'][ymin:ymax,xmin:xmax]
gam = np.load('gam.npy')[ymin:ymax,xmin:xmax]

## Example pixel for check
example_ts = list([[113,279],[111,208]])
#example_ts = list([[3707,808],[772,2099]])
#example_ts = list([[3560,1512],[2808,1064],[1192,1504],[1008,448]])
dn0 = dn - dn[0]
d1=0
period = 365.25
#c,r = 3707,808
for ii,point in enumerate(example_ts):
    c=point[0];r=point[1]
    idx = ((r)*nxl)+c #finds the index of flattened array based on row col in image
    y=-alld_flat_topo[:,idx]*lam/(4*np.pi)*100
    phase,amplitude,bias,slope = fitSine.fitSine1d(dn0,y,period)
    yEst = amplitude*np.sin(dn0*(1/period)*2*np.pi + phase * (np.pi/180.0)) + slope*dn0 + bias
    plt.figure()
    plt.plot(dec_year,y[d1:],'.')
    plt.plot(dec_year[d1:],yEst[d1:])
    plt.xlabel('Year'); plt.ylabel('Displacement (cm)')
    plt.title(str(np.round((slope*365), decimals=2)) + ' cm/yr in LOS')
    plt.show()
#    plt.savefig(workdir + 'Figs/timeseries' + str(ii) + '.svg',transparent=True,dpi=200)

# Invert for seasonal plus long term rates
phases,amplitudes,biases,slopes = fitSine.fitSine(dn0,alld_flat_topo,period)

c,r = 100,100
idx = ((r)*nxl)+c #finds the index of flattened array based on row col in image
y=-alld_flat_topo[:,idx]#*lam/(4*np.pi)*100
yEst = amplitudes[idx]*np.sin(dn0*(1/period)*2*np.pi + phases[idx] * (np.pi/180.0)) + slopes[idx]*dn0 + biases[idx]
plt.figure()
plt.plot(dec_year,y[d1:],'.')
plt.plot(dec_year[d1:],yEst[d1:])
plt.xlabel('Year'); plt.ylabel('Displacement (cm)')
plt.title(str(np.round((slopes[idx]*365), decimals=2)) + ' cm/yr in LOS')
plt.show()

#G = np.vstack([dn0, np.ones((len(dn0),1)).flatten()]).T
#Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
#mod   = np.dot(Gg, alld_flat_topo)
#rates = np.reshape(mod[0,:],(nyl,nxl))*lam/(4*np.pi)*100*365 # cm/yr

rates = np.reshape(slopes,(nyl,nxl)).astype(np.float32)*lam/(4*np.pi)*100*365
r_nomsk = rates
amps = np.reshape(amplitudes,(nyl,nxl)).astype(np.float32)*lam/(4*np.pi)*100
a_nomsk = amps
plt.figure();plt.imshow(np.flipud(r_nomsk),vmin=-2,vmax=2)
plt.figure();plt.imshow(np.flipud(a_nomsk),vmin=0,vmax=2)


#offs  = np.reshape(mod[1,:],(nyl, nxl))
synth  = np.dot(G,mod);
res    = (alld_flat_topo-synth)*lam/(4*np.pi)*100 # cm
resstd = np.std(res,axis=0)
resstd = np.reshape(resstd,(nyl, nxl))


# Mask the rates matrix
gam[np.isnan(gam)]=0
rates[np.where(gam<.2)]=np.nan #masks the low coherence areas
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


# Plot rate map
plt.rc('font',size=12)
plt.figure(figsize=(12,12))
m = Basemap(epsg=3395, llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
        llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='l')
m.drawparallels(np.arange(np.floor(minlat-pad), np.ceil(maxlat+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
m.drawmeridians(np.arange(np.floor(minlon-pad), np.ceil(maxlon+pad),1), linewidth=0,labels=[1,0,0,1])
m.arcgisimage(service='World_Shaded_Relief',xpixels=1500)
cf = m.pcolormesh(lo,la,rates,shading='flat',latlon=True, cmap=plt.cm.Spectral_r,zorder=8,vmin=-10,vmax=10)
m.readshapefile('/data/kdm95/qfaults/qfaults_la', 'qfaults_la',zorder=30)
#m.plot(lo_p1,la_p1,color='red',zorder=40,latlon=True)
cbar = m.colorbar(cf,location='bottom',pad="10%")
cbar.set_label('cm')
plt.show()
#plt.savefig('Figs/rate_map.png', transparent=True, dpi=500)
#
# Plot rate std
plt.rc('font',size=12)
plt.figure(figsize=(12,12))
m = Basemap(epsg=3395, llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
        llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='l')
m.drawparallels(np.arange(np.floor(minlat-pad), np.ceil(maxlat+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
m.drawmeridians(np.arange(np.floor(minlon-pad), np.ceil(maxlon+pad),1), linewidth=0,labels=[1,0,0,1])
m.arcgisimage(service='World_Shaded_Relief',xpixels=1500)
cf = m.pcolormesh(lo,la,resstd,shading='flat',latlon=True, cmap=plt.cm.Spectral_r,zorder=8,vmin=0,vmax=8)
m.readshapefile('/data/kdm95/qfaults/qfaults_la', 'qfaults_la',zorder=30)
#m.plot(lo_p1,la_p1,color='red',zorder=40,latlon=True)
cbar = m.colorbar(cf,location='bottom',pad="10%")
cbar.set_label('cm')
plt.show()

#lonpt, latpt = m(16558,67529,inverse=True)
#
#fault_listx = list([161895,129528,122949,115581,113213,51636,40058])
#fault_listy = list([135211,120474,117580,115475,113633,86528,81528])
#dst=list()
#lonpt, latpt = m(fault_listx[0],fault_listy[0],inverse=True)
#
#for ii in np.arange(1,len(fault_listx)):
#    lonpt2, latpt2 = m(fault_listx[ii],fault_listy[ii],inverse=True)
#    dst.append(np.sqrt(np.square(lonpt-lonpt2)+np.square(latpt-latpt2)))




# Profiles
# Example profile: pl(image, src, dst, linewidth=1, order=1, mode='constant', cval=0.0)

# PROFILE 1 
x0,y0 = 164,2412
x1,y1 = 364,2604
# Convert to lat lon
lo_p4=(lo[y0,x0],lo[y1,x1])
la_p4=(la[y0,x0],la[y1,x1])
#find total distance
prof_dist = np.sqrt(np.square(lo_p4[0]-lo_p4[1]) + np.square(la_p4[0]-la_p4[1])) *111
fig, ax1 = plt.subplots(figsize=(6,4))
zii = pl(rates,(y0,x0),(y1,x1),linewidth=2)
prof_vec = np.linspace(0,prof_dist,len(zii))
ax1.plot(prof_vec,zii*10,'k.')
#ax1.vlines(8.4,ymin=10*zii.min(),ymax=10*zii.max(),color='red')
#plt.ylim([-1,.5])
ax1.set_xlabel('Profile Distance (km)')
ax1.tick_params(axis='y')
ax1.set_ylabel('LOS Rate (mm/yr)')
# topo profile
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
zii_topo = pl(hgt,(y0,x0),(y1,x1),linewidth=2)
ax2.plot(prof_vec,zii_topo)
#ax2.vlines(8.4,ymin=zii_topo.min(),ymax=zii_topo.max(),color='red')
#plt.ylim([-1,.5])
ax2.set_xlabel('Profile Distance (km)')
ax2.set_ylabel('Elevation (m)',color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
#plt.savefig(workdir + 'Figs/prof4_topo.svg',transparent=True,dpi=100)
