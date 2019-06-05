#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:55:27 2019

Setup UAVsar stack
1. download the files.. they have to be a 'stack' of SLCs. https://uavsar.jpl.nasa.gov/cgi-bin/data.pl
2. put them in the working directory (we'll move them later)
3. set the number of looks below
4. open a '.ann' file and figure out what the slc dimensions are and define ny and nx below
5. Run this script and all the other ones should work like sentinel.

@author: kdm95
"""

import numpy as np
from matplotlib import pyplot as plt
import glob
import os
from datetime import date
import isceobj
from mpl_toolkits.basemap import Basemap
from mroipac.looks.Looks import Looks


workdir = os.getcwd() # Use current directory as working directory
 # working directory (should be where merged is)
skip = 1
alks = int(4) # number of looks in azimuth
rlks = int(4) # number of looks in range
seaLevel = -200
#1x4 looks
nx = 4781
ny = 8333 

ifg_mode = False
nxl = int(np.floor(nx/rlks))
nyl = int(np.floor(ny/alks))



lam = .23 # 0.056 for c-band.. 0.23 for l-band (?)
mergeddir=workdir + '/merged'
intdir = mergeddir + '/interferograms'
tsdir = workdir + '/TS'
slcdir = mergeddir + '/SLC'


# Make directories and move files there
if not os.path.isdir(mergeddir):
    os.mkdir(mergeddir) 
if not os.path.isdir(mergeddir+'/geom_master'):
    os.mkdir(mergeddir+'/geom_master')     
    os.system('mv *.lkv merged/geom_master/')
    os.system('mv *.llh merged/geom_master/')
    os.system('mv *.dop merged/geom_master/')
if not os.path.isdir(intdir):
    os.mkdir(intdir) 
if not os.path.isdir(slcdir):
    os.mkdir(slcdir) 
    os.system('mv *.slc merged/SLC/')
    os.system('mv *.ann merged/SLC/')
if not os.path.isdir(tsdir):
    os.mkdir(tsdir) 
if not os.path.isdir(workdir + '/Figs'):
    os.mkdir(workdir + '/Figs')

llhFile = glob.glob(mergeddir + '/geom_master/*.llh')[0]

slcFileList = glob.glob(slcdir + '/*.slc')

dates = []
dn = []
dec_year = []
for f in slcFileList:
    f1 = f[-60:]
    yr = '20' + f1[30:32]
    mo = f1[32:34]
    da = f1[34:36]
    dates.append(yr+mo+da)
    dt = date.toordinal(date(int(yr), int(mo), int(da)))
    dn.append(dt)
    d0 = date.toordinal(date(int(yr), 1, 1))
    doy = np.asarray(dt)-d0+1
    dec_year.append(float(yr) + (doy/365.25))
    if not os.path.isdir(slcdir + '/' + dates[-1]):
        os.mkdir(slcdir + '/' + dates[-1])
        slc = np.fromfile(f,dtype=np.complex64)
        slc = np.reshape(slc,(ny,nx))
        im = isceobj.createImage()# Copy the interferogram image from before
        im.filename = slcdir + '/' + dates[-1] + '/' + dates[-1] + '.slc.full'
        im.width = nx
        im.length = ny
        im.dataType = 'CFLOAT'
        im.dump(im.filename + '.xml') # Write out xml
        slc.tofile(im.filename) # Write file out
        im.finalizeImage()

dates.sort()
dates = np.asarray(dates)
dn = np.asarray(dn)
dn0 = dn-dn[0] # make relative to first date

pairs1=list()
pairs2=list()
pairs = list()
for ii,d in enumerate(dates):
    for jj in np.arange(1,skip+1):
        try:
            pairs.append(dates[ii] + '_' + dates[ii+jj])
        except:
            pass

# Get geom info
llh = np.fromfile(llhFile,dtype=np.float32)
laIds = np.arange(0,len(llh),3)
loIds = np.arange(1,len(llh),3)
hgIds = np.arange(2,len(llh),3)
lon_ifg = llh[loIds]
lat_ifg = llh[laIds]
hgt_ifg = llh[hgIds]
lon_ifg = np.reshape(lon_ifg,(ny,nx))
lat_ifg = np.reshape(lat_ifg,(ny,nx))
hgt_ifg = np.reshape(hgt_ifg,(ny,nx))

# write out the geom files
im = isceobj.createImage()# Copy the interferogram image from before
im.filename = mergeddir + '/geom_master/lon.rdr.full'
im.width = nx
im.length = ny
im.dataType = 'FLOAT'
im.dump(im.filename + '.xml') # Write out xml
lon_ifg.tofile(im.filename) # Write file out
im.finalizeImage()

# write out the geom files
im = isceobj.createImage()# Copy the interferogram image from before
im.filename = mergeddir + '/geom_master/lat.rdr.full'
im.width = nx
im.length = ny
im.dataType = 'FLOAT'
im.dump(im.filename + '.xml') # Write out xml
lat_ifg.tofile(im.filename) # Write file out
im.finalizeImage()

# write out the geom files
im = isceobj.createImage()# Copy the interferogram image from before
im.filename = mergeddir + '/geom_master/hgt.rdr.full'
im.width = nx
im.length = ny
im.dataType = 'FLOAT'
im.dump(im.filename + '.xml') # Write out xml
hgt_ifg.tofile(im.filename) # Write file out
im.finalizeImage()


# Downlook geom files
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

file_list = list(['lat','lon','hgt'])#,'incLocal','shadowMask'])
for f in file_list:
    infile = mergeddir + '/geom_master/' + f + '.rdr.full'
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


params = dict()
params['pairs'] =        pairs
params['dates'] =        dates
params['pairs'] =        pairs
params['dec_year'] =     dec_year
params['dn'] =           dn
params['dn0'] =          dn0
params['seaLevel'] = seaLevel
nd = len(pairs)
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
