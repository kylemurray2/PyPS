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
import makeMap
import cartopy.crs as ccrs
from mroipac.looks.Looks import Looks
from scipy.interpolate import griddata 
import cv2
from scipy import signal

#<><><><><><><><><><><><>Set these variables><><><><><><><><><><><><><><><
# Define area of interest
#bbox = list([35.8, 36.9, -120.3, -118.7]) #minlat,maxlat,minlon,maxlon
#maxlat = bbox[0]; minlat = bbox[1]; minlon = bbox[2]; maxlon = bbox[3]
# workdir = os.getcwd() # Use current directory as working directory
#  # working directory (should be where merged is)
# skip = 1
# alks = int(3) # number of looks in azimuth
# rlks = int(8) # number of looks in range
# seaLevel = -200
# ifg_mode = False
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
import localParams
workdir, skip, alks, rlks, seaLevel, ifg_mode,crop,cropymin,cropymax,cropxmin,cropxmax = localParams.getLocalParams()

doDownlook = True
lam = 0.056 # 0.056 for c-band
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
    
        
nd = len(pairs)
# rename geometry files to add 'full'
os.system('mv merged/geom_master/hgt.rdr merged/geom_master/hgt.rdr.full')
os.system('mv merged/geom_master/lat.rdr merged/geom_master/lat.rdr.full')
os.system('mv merged/geom_master/lon.rdr merged/geom_master/lon.rdr.full')
os.system('mv merged/geom_master/incLocal.rdr merged/geom_master/incLocal.rdr.full')
os.system('mv merged/geom_master/los.rdr merged/geom_master/los.rdr.full')
os.system('mv merged/geom_master/shadowMask.rdr merged/geom_master/shadowMask.rdr.full')

# Get width and length
f_lon = mergeddir + '/geom_master/lon.rdr.full.vrt'

f_lon = mergeddir + '/geom_master/lon.rdr.full'
gImage = isceobj.createIntImage()
gImage.load(f_lon + '.xml')

if crop:
    ny = cropymax-cropymin
    nx = cropxmax-cropxmin
else:
    ny = gImage.length
    nx = gImage.width

if crop:
    for d in dates:
        f = slcdir +'/'+ d + '/' + d + '.slc.full'
        if not os.path.isfile(f + '.crop'):

    #        os.system('fixImageXml.py -i ' + f + ' -f')
            slcImage = isceobj.createSlcImage()
            slcImage.load(f + '.xml')
            slc1 = slcImage.memMap()[cropymin:cropymax,cropxmin:cropxmax,0]
            
            slcImagec = isceobj.createSlcImage()
            
            slcImagec.filename = f+'.crop'
            slcImagec.width =cropxmax-cropxmin
            slcImagec.length =cropymax-cropymin
            slcImagec.dump(slcImagec.filename + '.xml') # Write out xml
            
            slc1.tofile(slcImagec.filename) # Write file out
            slcImagec.finalizeImage()

file_list = list(['lat','lon','hgt'])#,'incLocal','shadowMask'])

if crop:
    for f in file_list:
        infile = mergeddir + '/geom_master/' + f + '.rdr.full'
        imgi = isceobj.createImage()
        imgi.load(infile+'.xml')
        geomIm = imgi.memMap()[cropymin:cropymax,cropxmin:cropxmax,0]
        imgo = isceobj.createImage()
        imgo.load(infile+'.xml')
        imgo.filename = infile+'.crop'
        imgo.width = cropxmax-cropxmin
        imgo.length =cropymax-cropymin
        imgo.dump(imgo.filename+'.xml')
        
        geomIm.tofile(imgo.filename)


if doDownlook:
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
    for f in file_list:
        if crop:
            infile = mergeddir + '/geom_master/' + f + '.rdr.full.crop'
        else:
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

nyl,nxl = lon_ifg.shape

# Geniusly get rid of edge artifacts from downlooking
Q = np.array([[0,0,0],[0,1,0],[0,0,0]])
lon_ifg = signal.convolve2d(lon_ifg,Q, mode='same')
lat_ifg = signal.convolve2d(lat_ifg,Q, mode='same')
hgt_ifg = signal.convolve2d(hgt_ifg,Q, mode='same')

# Figure out where the nan values begin and end so we can crop them if we want later.
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

# Now extrapolate the geom edges out so we can map non-rectangle images
xx,yy = np.meshgrid(np.arange(0,nxl), np.arange(0,nyl))
xxValid = xx[~np.isnan(lon_ifg)].astype(np.float32)
yyValid = yy[~np.isnan(lon_ifg)].astype(np.float32)
lonValid = lon_ifg[~np.isnan(lon_ifg)].astype(np.float32)
latValid = lat_ifg[~np.isnan(lon_ifg)].astype(np.float32)
lonI = griddata((xxValid,yyValid), lonValid , (xx,yy), method='nearest')
xxValid = xx[~np.isnan(lat_ifg)].astype(np.float32)
yyValid = yy[~np.isnan(lat_ifg)].astype(np.float32)
lonValid = lon_ifg[~np.isnan(lat_ifg)].astype(np.float32)
latValid = lat_ifg[~np.isnan(lat_ifg)].astype(np.float32)
latI = griddata((xxValid,yyValid), latValid , (xx,yy), method='nearest')
minlat=latI.min()
maxlat=latI.max()
minlon=lonI.min()
maxlon=lonI.max()

bg = 'World_Shaded_Relief'
pad=2
makeMap.mapBackground(bg,minlon,maxlon,minlat,maxlat,1,8,'Footprint',borders=False)
plt.plot(lon_bounds,lat_bounds,linewidth=2,color='red',zorder=10,transform=ccrs.PlateCarree())
plt.rc('font',size=14)
plt.savefig(workdir + '/Figs/areamap.svg',transparent=True,dpi=100 )


mergeddir =workdir + '/merged'
intdir = mergeddir + '/interferograms'
tsdir = workdir + '/TS'

geom = {}
geom['lon_ifg'] = lonI
geom['lat_ifg'] = latI
geom['hgt_ifg'] = hgt_ifg
np.save('geom.npy',geom)

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
params['seaLevel'] =          seaLevel
params['crop'] =        crop
params['workdir'] =      workdir
params['intdir'] =       intdir
params['tsdir'] =        tsdir
params['mergeddir'] =    mergeddir

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
params['intdir'] =       intdir
params['tsdir'] =        tsdir
params['slcdir'] =        slcdir
params['cropymin'] =     cropymin
params['cropymax'] =     cropymax
params['cropxmin'] =     cropxmin
params['cropxmax'] =     cropxmax


# Save the dictionary
np.save('params.npy',params)

# To load the dictionary later, do this:
# params = np.load('params.npy').item()
# locals().update(params) this parses all variables from the dict to local variables
