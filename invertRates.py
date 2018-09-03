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
from matplotlib import pyplot as plt

with open(tsdir + '/params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks = pickle.load(f)

# Run refDef.py before this to make the alld_flat.pkl file
#with open(tsdir + 'alld_flat.pkl','rb')as f:
#    alld_flat = pickle.load(f)
    
#pairs=pairs[:-45]    
# Convert pairs to dates
dn = list()
dn.append( date.toordinal(date(int(pairs[0][9:13]), int(pairs[0][13:15]), int(pairs[0][15:]))) )
dec_year = list()
# Do the first date in the decimal year array
pair = pairs[0]
yr = pair[9:13]
mo = pair[13:15]
day = pair[15:]
dt = date.toordinal(date(int(yr), int(mo), int(day)))
d0 = date.toordinal(date(int(yr), 1, 1))
doy = np.asarray(dt)-d0+1
dec_year.append(float(yr) + (doy/365.25))

for pair in pairs:
    yr = pair[9:13]
    mo = pair[13:15]
    day = pair[15:]
    dt = date.toordinal(date(int(yr), int(mo), int(day)))
    dn.append(dt)
    d0 = date.toordinal(date(int(yr), 1, 1))
    doy = np.asarray(dt)-d0+1
    dec_year.append(float(yr) + (doy/365.25))
dn = np.asarray(dn)
dn0 = dn-dn[0] # make relative to first date

# Example pixel for check
# LOST HILLS OIL
r=2389
c=1595
# MAIN SUBSIDENCE PEAK
r=2199
c=1227
# UPLIFT AREA NEAR LOST HILLS
r=2647
c=1359
idx = ((r-1)*nxl)+c #finds the index of flattened array based on row col in image
# Example pixel for check
y=alld_flat_topo[:,idx]
G = np.vstack([dn0, np.ones((len(dn0),1)).flatten()]).T
Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
mod    = np.dot(Gg, y)
y2 = mod[0]*dn0 + mod[1]

plt.figure()
plt.plot(dec_year,y,'.')
plt.plot(dec_year,y2)
plt.xlabel('Year')
plt.ylabel('Displacement (cm)')
##


G = np.vstack([dn0, np.ones((len(dn0),1)).flatten()]).T
Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
mod    = np.dot(Gg, alld_flat_topo[:,:])
rates   = np.reshape(mod[0,:],(nyl, nxl))*lam/(4*np.pi)*100*365 # cm/yr
rates = rates.astype(np.float32)
#ra = np.zeros((rates.shape))
#ra[364:3060,1:2437]=rates[364:3060,1:2437]
#offs  = np.reshape(mod[1,:],(nyl, nxl))
#synth  = np.dot(G,mod);
#res    = (alld_flat_topo-synth)*lam/(4*np.pi)*100 # cm
#resstd = np.std(res,axis=0)
#resstd = np.reshape(resstd,(nyl, nxl))


# MASKING______________________________
# Load gamma0_lk
f = tsdir + 'gamma0_lk.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0_lk= intImage.memMap()[:,:,0] 
gamma0_lk=gamma0_lk.copy()
# Load height file
h = workdir + 'merged/geom_master/hgt_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
hgt = hImg.memMap()[:,:,0].astype(np.float32)
# elevations at 4 of the main lakes that we'll mask out

# Mask the rates matrix
gamma0_lk*=crop_mask
gamma0_lk[np.isnan(gamma0_lk)]=0
rates[np.where(gamma0_lk<.3)]=np.nan #masks the low coherence areas
rates[np.where( (hgt<.1) ) ]=np.nan # masks the lakes
#rates=np.fliplr(rates)

#rates[np.isnan(rates)]=0
#rates=np.flipud(rates)
#plt.imshow(rates)
#plt.imshow(offs)

# Save rates
fname = tsdir + 'rates_flat.unw'
out = isceobj.createIntImage() # Copy the interferogram image from before
out.dataType = 'FLOAT'
out.filename = fname
out.width = nxl
out.length = nyl
#out.bands=1
out.dump(fname + '.xml') # Write out xml
rates.tofile(out.filename) # Write file out
out.renderHdr()
out.renderVRT()

# GEOCODE
#cmd = 'geocodeIsce.py -f ' + tsdir + 'rates_flat.unw -d ' + workdir + 'DEM/demLat_N33_N35_Lon_W119_W116.dem -m ' + workdir + 'master/ -s ' + workdir + 'pawns/20150514 -r ' + str(rlks) + ' -a ' + str(alks) + ' -b "'" 33 35 -118 -116"'" '
#os.system(cmd)

r=-rates
#r[r<5]=np.nan

h = workdir + 'merged/geom_master/lon_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
lo = hImg.memMap()[:,:,0].astype(np.float32)

h = workdir + 'merged/geom_master/lat_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
la = hImg.memMap()[:,:,0].astype(np.float32)

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
pad=.5
minlat=lats.min()
maxlat=lats.max()
minlon=lons.min()
maxlon=lons.max()

# Plot IFG
plt.rc('font',size=12)
plt.figure(figsize=(5,6))
m = Basemap(epsg=3395, llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
        llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='i')
m.drawstates(linewidth=0.5,zorder=6,color='white')
#m.drawcoastlines(linewidth=0.8,zorder=6,color='white')
m.drawcountries(linewidth=0.8,zorder=6,color='white')
#m.fillcontinents(color='white', lake_color='lightblue', zorder=1)  # set zorder=0 so it plots on the bottom
#m.drawmapboundary(fill_color='lightblue',zorder=0)  # this will be the background color (oceans)
m.drawparallels(np.arange(np.floor(minlat-pad), np.ceil(maxlat+pad), 1), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
m.drawmeridians(np.arange(np.floor(minlon-pad), np.ceil(maxlon+pad),1), linewidth=0,labels=[1,0,0,1])
#m.arcgisimage(service='ESRI_Imagery_World_2D',xpixels=2000)
m.arcgisimage(service='World_Shaded_Relief',xpixels=2000)
cf = m.pcolormesh(lo,la,r,shading='flat',latlon=True, zorder=8,vmin=0,vmax=25)
cbar = m.colorbar(cf,location='bottom',pad="10%")
cbar.set_label('cm')
plt.show()
#plt.savefig('sub.png', transparent=True, dpi=500)
