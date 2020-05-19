#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:20:01 2018
  Make dates and pairs dictionaries

@author: kdm95
"""
import numpy as np
import glob
import os
from datetime import date
import isceobj
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mroipac.looks.Looks import Looks

#<><><><><><><><><><><><>Set these variables><><><><><><><><><><><><><><><
# Define area of interest
#bbox = list([35.8, 36.9, -120.3, -118.7]) #minlat,maxlat,minlon,maxlon
#maxlat = bbox[0]; minlat = bbox[1]; minlon = bbox[2]; maxlon = bbox[3]
workdir = os.getcwd() # Use current directory as working directory
 # working directory (should be where merged is)
skip = 2
alks = int(6) # number of looks in azimuth
rlks = int(3) # number of looks in range
ifg_mode = False
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><


lam = 0.23 # 0.056 for c-band
mergeddir=workdir + '/merged'
intdir = mergeddir + '/interferograms'
tsdir = workdir + '/TS'
slcdir = mergeddir + '/SLC'
# Make directories
if not os.path.isdir(tsdir):
    os.mkdir(tsdir) 
if not os.path.isdir(workdir + '/Figs'):
    os.mkdir(workdir + '/Figs')

if ifg_mode:
    pairs1=list()
    pairs2=list()
    pairs = list()
    flist = glob.glob(intdir + '/2*_2*')
    [pairs.append(f[-17:]) for f in flist]
    [pairs1.append(f[-17:-9]) for f in flist]
    [pairs2.append(f[-8:]) for f in flist]
    pairs.sort();pairs1.sort();pairs2.sort()
    dates = np.unique(np.vstack((pairs1,pairs2)))
else:
    flist = glob.glob(slcdir + '/2*')
    # Convert pairs to dates
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
nd = len(pairs)

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
       
# rename geometry files to add 'full'
os.system('mv merged/geom_master/hgt.rdr.full merged/geom_master/hgt.rdr')
os.system('mv merged/geom_master/lat.rdr.full merged/geom_master/lat.rdr')
os.system('mv merged/geom_master/lon.rdr.full merged/geom_master/lon.rdr')
os.system('mv merged/geom_master/incLocal.rdr.full merged/geom_master/incLocal.rdr')
os.system('mv merged/geom_master/los.rdr.full merged/geom_master/los.rdr')
os.system('mv merged/geom_master/shadowMask.rdr.full merged/geom_master/shadowMask.rdr')

# Get width and length
f_lon = mergeddir + '/geom_master/lon.rdr.vrt'
f_lon = mergeddir + '/geom_master/lon.rdr'
gImage = isceobj.createIntImage()
gImage.load(f_lon + '.xml')
ny = gImage.length
nx = gImage.width
nxl = int(np.floor(nx/rlks))
nyl = int(np.floor(ny/alks))

def downLook(infile, outfile,alks,rlks):
    inImage = isceobj.createImage()
    inImage.load(infile + '.xml')
    inImage.filename = infile

    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inImage)
    lkObj.setOutputFilename(outfile)
    lkObj.looks()

file_list = list(['lat','lon','hgt','los'])#,'incLocal','shadowMask'])
for f in file_list:
    infile = mergeddir + '/geom_master/' + f + '.rdr'
    outfile = mergeddir + '/geom_master/' + f + '_lk.rdr'
    downLook(infile, outfile,alks,rlks)
    
# Get bounding coordinates (Frame)
f_lon_lk = mergeddir + '/geom_master/lon_lk.rdr'
f_lat_lk = mergeddir + '/geom_master/lat_lk.rdr'
f_hgt_lk = mergeddir + '/geom_master/hgt_lk.rdr'

Image = isceobj.createImage()
Image.load(f_lon_lk + '.xml')
lon_ifg = Image.memMap()[:,:,0]
lon_ifg = lon_ifg.copy().astype(np.float32)
lon_ifg[lon_ifg==0]=np.nan
Image.finalizeImage()

Image = isceobj.createImage()
Image.load(f_lat_lk + '.xml')
lat_ifg = Image.memMap()[:,:,0]
lat_ifg = lat_ifg.copy().astype(np.float32)
lat_ifg[lat_ifg==0]=np.nan
Image.finalizeImage()

Image = isceobj.createImage()
Image.load(f_hgt_lk + '.xml')
hgt_ifg = Image.memMap()[:,:,0]
hgt_ifg = hgt_ifg.copy().astype(np.float32)
hgt_ifg[hgt_ifg==0]=np.nan
Image.finalizeImage()

geom = {}
geom['lon_ifg'] = lon_ifg
geom['lat_ifg'] = lat_ifg
geom['hgt_ifg'] = hgt_ifg
np.save('geom.npy',geom)

for l in np.arange(0,nyl):
    ll = lon_ifg[l,:]
    if not np.isnan(ll.max()):
        break

for p in np.arange(l+1,nyl):
    ll = lon_ifg[p,:]
    if np.isnan(ll.max()):
        break
l+=1

ymin=l+1
ymax=p-1
xmin=0
xmax=nxl

ul = (lon_ifg[l+1,1],lat_ifg[l+1,1])
ur = (lon_ifg[l+1,-2],lat_ifg[l+1,-2])
ll = (lon_ifg[p-2,1],lat_ifg[p-2,1])
lr = (lon_ifg[p-2,-2],lat_ifg[p-2,-2])

lon_bounds = np.array([ul[0],ur[0],ur[0],lr[0],lr[0],ll[0],ll[0],ul[0]])
lat_bounds = np.array([ul[1],ur[1],ur[1],lr[1],lr[1],ll[1],ll[1],ul[1]])


pad=2
plt.close()
plt.rc('font',size=14)
fig = plt.figure(figsize=(6,6))
m = Basemap(llcrnrlat=lat_bounds.min()-pad,urcrnrlat=lat_bounds.max()+pad,\
        llcrnrlon=lon_bounds.min()-pad,urcrnrlon=lon_bounds.max()+pad,resolution='i',epsg=3395)
m.arcgisimage(service='World_Shaded_Relief',xpixels=1000)
m.drawstates(linewidth=1.5,zorder=1,color='white')
m.drawcountries(linewidth=1.5,zorder=1,color='white')
m.drawparallels(np.arange(np.floor(lat_bounds.min()-pad), np.ceil(lat_bounds.max()+pad), 2), linewidth=0, labels=[1,0,0,1])  # set linwidth to zero so there is no grid
m.drawmeridians(np.arange(np.floor(lon_bounds.min()-pad), np.ceil(lon_bounds.max()+pad),2), linewidth=0,labels=[1,0,0,1])
m.plot(lon_bounds,lat_bounds,linewidth=2,latlon=True,color='red',zorder=10)
plt.title('Extent of stack')
plt.show()
plt.savefig(workdir + '/Figs/areamap.svg',transparent=True,dpi=100 )

mergeddir =workdir + '/merged'
intdir = mergeddir + '/interferograms'
tsdir = workdir + '/TS'

plt.imshow(hgt_ifg)

# Save arrays and variables to a dictionary 'params'
params = dict()
params['pairs'] =        pairs
params['dates'] =        dates
params['pairs'] =        pairs
params['dec_year'] =     dec_year
params['dn'] =           dn
params['dn0'] =          dn0
params['nd'] =           nd
params['lam'] =          lam
params['workdir'] =      workdir
params['intdir'] =       intdir
params['tsdir'] =        tsdir
params['ny'] =           ny
params['nx'] =           nx
params['nxl'] =          nxl
params['nyl'] =          nyl
params['lon_bounds'] =   lon_bounds
params['lat_bounds'] =   lat_bounds
params['ymin'] =         ymin
params['ymax'] =         ymax
params['xmin'] =         xmin
params['xmax'] =         xmax
params['alks'] =         alks
params['rlks'] =         rlks
params['mergeddir'] =    mergeddir
params['intdir'] =       intdir
params['tsdir'] =        tsdir
params['slcdir'] =        slcdir

# Save the dictionary
np.save('params.npy',params)

# To load the dictionary later, do this:
# params = np.load('params.npy').item()
# locals().update(params) this parses all variables from the dict to local variables
