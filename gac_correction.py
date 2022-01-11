#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 12:37:39 2018

@author: kdm95
"""
# Global Imports
import numpy as np
import isceobj
from osgeo import gdal
import pickle
import os
import cv2
from matplotlib import pyplot as plt
import scipy.spatial.qhull as qhull
from scipy.interpolate import griddata
import util
import requests
import pandas as pd
# Local Imports
#from APS_tools import ifg

# Load stuff
params = np.load('params.npy',allow_pickle=True).item()
locals().update(params)
geom = np.load('geom.npy',allow_pickle=True).item()
locals().update(geom)

dataDir = workdir + '/GACOS/'

do_wrapped =2 # 1 if you want to apply correction to wrapped data. 0 for unwrapped. 

minlat=np.floor(lat_ifg.min())
maxlat=np.ceil(lat_ifg.max())
minlon=np.floor(lon_ifg.min())
maxlon=np.ceil(lon_ifg.max())

path = os.getcwd().split('/')[-2]
frame= os.getcwd().split('/')[-1]

def getTime(path,frame):
    
    '''
     Figure out what time the aquisition was
    '''
    start='2020-05-01T00:00:00Z'
    end='2021-06-01T00:00:00Z'
    asfUrl = 'https://api.daac.asf.alaska.edu/services/search/param?platform=SENTINEL-1&processinglevel=SLC&output=CSV'
    call = asfUrl + '&relativeOrbit=' + path + '&frame=' + frame + '&start=' + start + '&end=' + end
    # Here we'll make a request to ASF API and then save the output info to .CSV file
    if not os.path.isfile('out.csv'):
        r =requests.get(call,timeout=100)
        with open('out.csv','w') as j:
            j.write(r.text)
    # Open the CSV file and get the URL and File names
    hour = pd.read_csv('out.csv')["Start Time"][0][11:13]
    minute = pd.read_csv('out.csv')["Start Time"][0][14:16]
    return int(hour),int(minute)
   
hour,minute = getTime(path,frame)
timee =hour+round(minute/60,2)

# email = 'murray8@hawaii.edu'
email = 'bXVycmF5OEBoYXdhaWkuZWR1'
#Request gacos from API?
url = 'http://www.gacos.net/result.php?flag=MQ==&email='+email+'&'
boundURL = 'S='+str(minlat)+'&N='+str(maxlat)+'&W='+str(minlon)+'&E='+str(maxlon)+'&'
timeURL = 'time_of_day='+str(timee)+'&type=2&date='

dstr = dates[21]
for d in dates[22:]:
    dstr+='-'
    dstr+=d
    
requestURL = url + boundURL + timeURL + dstr
# r =requests.get(requestURL,timeout=100)


print(requestURL)



# Read one of the rsc files
fname = dataDir + dates[0] +'.ztd.tif'
data = gdal.Open(fname, gdal.GA_ReadOnly)
nxg = int(data.RasterXSize)
nyg = int(data.RasterYSize )
geoTransform = data.GetGeoTransform()
lon_gac_min = geoTransform[0]
lat_gac_max = geoTransform[3]
lon_gac_max = lon_gac_min + geoTransform[1] * nxg
lat_gac_min = lat_gac_max + geoTransform[5] * nyg
print([lon_gac_min, lat_gac_max, lon_gac_max, lat_gac_min])


gac_lon_vec = np.linspace(lon_gac_min, lon_gac_max, int(nxg))
gac_lat_vec = np.linspace(lat_gac_min, lat_gac_max, int(nyg))

gac_lat,gac_lon = np.meshgrid(gac_lat_vec, gac_lon_vec, sparse=False, indexing='ij')
gac_lat = np.flipud(gac_lat)

#Aminlat = 35.9
#Amaxlat = 36.8
#Aminlon = -96.97
#Amaxlon = -96.0
#gid1,gid2 = np.where((gac_lon>Aminlon) & (gac_lon<Amaxlon) & (gac_lat>Aminlat) & (gac_lat<Amaxlat))
#
#gac_lon=gac_lon[gid1.min():gid1.max(),gid2.min():gid2.max()]
#gac_lat=gac_lat[gid1.min():gid1.max(),gid2.min():gid2.max()]
#
#id1,id2 = np.where((lon_ifg>Aminlon) & (lon_ifg<Amaxlon) & (lat_ifg>Aminlat) & (lat_ifg<Amaxlat))
#lon_ifg = lon_ifg[id1.min():id1.max(),id2.min():id2.max()]
#lat_ifg = lat_ifg[id1.min():id1.max(),id2.min():id2.max()]

if not os.path.isfile(workdir +  'GACOS/gac_stack.npy'):
    # Make some functions for the grid interpolation
    def interp_weights(xy, uv,d=2):
        tri = qhull.Delaunay(xy)
        simplex = tri.find_simplex(uv)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uv - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    
    def interpolate(values, vtx, wts):
        return np.einsum('nj,nj->n', np.take(values, vtx), wts)
    
    # Get weights for interpolation (this avoids redundant operations in loop)
    vtx, wts = interp_weights(np.asarray((gac_lon.flatten(),gac_lat.flatten())).T, np.asarray((lon_ifg.flatten(),lat_ifg.flatten())).T)
    
    gac_stack = list()
    # Loop through and grid each gacos image
    for ii in np.arange(0,nd):
        print('gridding gacos to ' + pairs[ii])
        date1 = dates[ii];
        date2 =dates[ii+1]
        gf1 = dataDir + date1 + '.ztd.tif'
        gf2 = dataDir + date2 + '.ztd.tif' 
        gactmp1 = gdal.Open(gf1, gdal.GA_ReadOnly)
        gac1 = gactmp1.GetRasterBand(1).ReadAsArray()
        gactmp2 = gdal.Open(gf2, gdal.GA_ReadOnly)
        gac2 = gactmp2.GetRasterBand(1).ReadAsArray()
        gac = gac2-gac1
        gac = np.asarray(gac, dtype=np.float32)
        
        gac_grid2 =griddata((gac_lon.flatten(),gac_lat.flatten()),gac2.flatten(), (lon_ifg,lat_ifg), method='linear')
        gac_grid=interpolate(gac, vtx, wts)
        gac_grid[gac_grid==0]=np.nan
        gac_grid = np.reshape(gac_grid,lon_ifg.shape)
        gac_grid=np.asarray(gac_grid,dtype=np.float32)
        gac_grid-=np.nanmean(gac_grid)
        gac_stack.append(gac_grid)
    
    np.save(dataDir + 'gac_stack.npy', gac_stack)
    gac_coords = [];gac_coords.append(lon_ifg);gac_coords.append(lat_ifg)
    np.save(dataDir + 'gac_coords.npy', gac_coords)

else:
    print(dataDir + 'gac_stack.npy already exists. Loading it...')
    gac_stack = np.load(dataDir + 'gac_stack.npy')


if do_wrapped==1:
    # Load ifg and correct for wrapped data
    gamma_thresh = .2
    rx=2
    ry=2
    gausx = np.exp( np.divide( -np.square(np.arange(-rx,rx)), np.square(rx)));
    gausy = np.exp( np.divide( -np.square(np.arange(-ry,ry)), np.square(ry)));
    gaus = gausx[:, np.newaxis] * gausy[np.newaxis, :]
    gaus = gaus-gaus.min()
    gaus  = gaus/np.sum(gaus.flatten())
    for ii,pair in enumerate(pairs):
        phs_c = (np.zeros((nyl,nxl))*1j).astype(np.complex64)
        f = intdir + pair + '/fine_lk.int'
        Image = isceobj.createIntImage()
        Image.load(f + '.xml')
        phs_ifg = Image.memMap()[ymin:ymax,:,0]
        Image.finalizeImage()    
        gac = ((gac_stack[ii])/np.cos(los_ifg)) #meters
        gac_complex = np.exp( ((gac *np.pi *4)/lam)*1j ) # complex number
        phs_c[ymin:ymax,:] = phs_ifg * np.conj(gac_complex)
        phs_c[np.isnan(phs_c)]=0
        phs_c = np.asarray(phs_c,dtype=np.complex64)
        

        # DO ps interp________________________________________________________
        rea_lk = np.real(phs_c)
        ima_lk = np.imag(phs_c)
        # Mask ones where the data is good
        mask = np.ones(rea_lk.shape)
        mask[np.where(gamma0_lk < gamma_thresh)]=0
        
        # Smooth everything into zero space
        mask_f = cv2.filter2D(mask,-1, gaus)
        rea_f = cv2.filter2D(rea_lk,-1, gaus)
        ima_f = cv2.filter2D(ima_lk,-1, gaus)
        
        # Divide by mask. This is how we care for nan values
        rea = rea_f/mask_f
        ima = ima_f/mask_f
        rea += rea_lk
        ima += ima_lk
        phs_lk = (rea+(1j*ima)).astype(np.complex64)
        phs_lk[np.isnan(phs_lk)]=0
       #_____________________________________________________________

        out = Image.clone() # Copy the interferogram image from before
        out.filename = intdir + pair + '/fine_lk_gac.int'
        out.dump( intdir + pair + '/fine_lk_gac.int.xml') # Write out xml
        phs_lk.tofile(out.filename) # Write file out
        out.finalizeImage()


if do_wrapped==0:
    # Load ifg and correct
    for ii,pair in enumerate(pairs):
        phs_c = np.zeros((nyl,nxl))
        f = intdir + pair + '/fine_lk.unw'
        Image = isceobj.createIntImage()
        Image.load(f + '.xml')
        phs_ifg = Image.memMap()[ymin:ymax,:,0]
        phs_ifg = phs_ifg.copy().astype(np.float32)*lam/(4*np.pi)*100
        phs_ifg[phs_ifg==0]=np.nan
        Image.finalizeImage()
        plt.imshow(phs_ifg)
#        phs_ifg-=np.nanmedian(phs_ifg)
        gac = ((gac_stack[ii]*100)/np.cos(np.deg2rad(los_ifg)))
#        gac-=np.nanmedian(gac)
        phs_c[ymin:ymax,:] = phs_ifg-gac
        phs_c-=np.nanmedian(phs_c)
        phs_c[np.isnan(phs_c)]=0
        phs_c = np.asarray(phs_c,dtype=np.float32)
        out = Image.clone() # Copy the interferogram image from before
        out.filename = intdir + pair + '/fine_lk_gac.unw'
        out.dump( intdir + pair + '/fine_lk_gac.unw.xml') # Write out xml
        phs_c.tofile(out.filename) # Write file out
        out.finalizeImage()


hgt = hgt_ifg

# ymin2=84
# ymax2=2575
# xmin=35
# xmax=6320
# crop_mask = np.zeros(hgt.shape)
# crop_mask[ymin2:ymax2,xmin:xmax] =1

# Load example
# Load phs
idx = 31
pair =pairs[idx]
f = intdir + pair + '/fine_lk.unw'
Image = isceobj.createImage()
Image.load(f + '.xml')
phs_ifg = Image.memMap()[ymin:ymax,:,0]
phs_ifg = phs_ifg.copy().astype(np.float32)*lam/(4*np.pi)*100
Image.finalizeImage()
# phs_ifg*=crop_mask
phs_ifg[phs_ifg==0]=np.nan
phs_ifg[np.where( (hgt<.1) ) ]=np.nan # masks the water
phs_ifg-=np.nanmedian(phs_ifg)

# Load phs corrected
f = intdir + pair + '/fine_lk_gac.unw'
Image = isceobj.createImage()
Image.load(f + '.xml')
phs_c = Image.memMap()[ymin:ymax,:,0]
phs_c = phs_c.copy().astype(np.float32)
# phs_c*=crop_mask
phs_c[phs_c==0]=np.nan
phs_c[np.where( (hgt<.1) ) ]=np.nan # masks the water

# Load gac model
gac_mod = (gac_stack[idx]*100)/np.cos(np.deg2rad(los_ifg))
# gac_mod*=crop_mask
gac_mod[gac_mod==0]=np.nan
gac_mod[np.where( (hgt<.1) ) ]=np.nan # masks the water
gac_mod-=np.nanmedian(gac_mod)

# Plot example gacos model and ifg 
vmin,vmax=-5,5
fig = plt.figure(figsize=(8,10))
ax =  fig.add_subplot(4,1,1);plt.imshow(phs_ifg,vmin=vmin,vmax=vmax)
ax.set_title('ifg')
ax = fig.add_subplot(4,1,2);plt.imshow(gac_mod,vmin=vmin,vmax=vmax)
ax.set_title('model')
ax =  fig.add_subplot(4,1,3);plt.imshow(phs_c,vmin=vmin,vmax=vmax)
ax.set_title('corrected ifg')

pad=0


mapImg(phs_ifg, lon_ifg, lat_ifg, vmin, vmax, pad,10, 'Original IFG (cm)')
mapImg(gac_mod, lon_ifg, lat_ifg, vmin, vmax, pad,10, 'Original IFG (cm)')
mapImg(phs_c, lon_ifg, lat_ifg, vmin, vmax, pad,10, 'Original IFG (cm)')

# plt.savefig(workdir + 'Figs/GACOS_correction.png',transparent=True,dpi=300 )

## Load example
## Load phs
#idx = 31
#pair =pairs[idx]
#f = intdir + pair + '/fine_lk.unw'
#Image = isceobj.createImage()
#Image.load(f + '.xml')
#phs_ifg = Image.memMap()[:,:,0]
#phs_ifg = phs_ifg.copy().astype(np.float32)*lam/(4*np.pi)*100
#Image.finalizeImage()
#phs_ifg*=crop_mask
#phs_ifg[phs_ifg==0]=np.nan
#phs_ifg[np.where( (hgt<.1) ) ]=np.nan # masks the water
#phs_ifg-=np.nanmean(phs_ifg)
#
## Load phs corrected
#f = intdir + pair + '/fine_lk_gac.int'
#Image = isceobj.createImage()
#Image.load(f + '.xml')
#phs_c = Image.memMap()[:,:,0]
#phs_c = phs_c.copy().astype(np.float32)
#phs_c*=crop_mask
#phs_c[phs_c==0]=np.nan
#phs_c[np.where( (hgt<.1) ) ]=np.nan # masks the water

_,_,_,_,phs_ifg_S_bins, phs_ifg_S_bins_std, phs_ifg_dist_bins,_ = structure_function.struct_fun(phs_ifg,250,400,0)
phs_ci = 1.96*(np.divide(phs_ifg_S_bins_std,np.sqrt( 250)))

_,_,_,_,gac_ifg_S_bins, gac_ifg_S_bins_std, gac_ifg_dist_bins,_ = structure_function.struct_fun(phs_c,250,400,0)
gac_ci = 1.96*(np.divide(gac_ifg_S_bins_std,np.sqrt( 250)))

plt.figure()
plt.plot(phs_ifg_dist_bins[:26],phs_ifg_S_bins[:26])
plt.fill_between(phs_ifg_dist_bins[:26],phs_ifg_S_bins[:26]-phs_ci[:26], phs_ifg_S_bins[:26]+phs_ci[:26],
                     alpha=0.3, linewidth=1, linestyle='dashed', antialiased=True)
plt.plot(gac_ifg_dist_bins[:26],gac_ifg_S_bins[:26])
plt.fill_between(gac_ifg_dist_bins[:26],gac_ifg_S_bins[:26]-gac_ci[:26], gac_ifg_S_bins[:26]+gac_ci[:26],
                     alpha=0.3, linewidth=1, linestyle='dashed', antialiased=True)
plt.title('Structure Function ' + pair)
plt.ylabel('RMS of phase difference between pixels (cm)')
plt.xlabel('Distance between pixels(km)')
plt.legend(['Original IFG','Corrected IFG'],loc='upper left')
# plt.savefig(workdir + 'Figs/GACOS_structfun.svg',transparent=True,dpi=100 )

