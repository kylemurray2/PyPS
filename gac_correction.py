#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 12:37:39 2018

@author: kdm95
"""
# Global Imports
import numpy as np
import isceobj
import pickle
import os
import cv2
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
import scipy.spatial.qhull as qhull
import glob
import makeMap

# Local Imports
from Statistics import structure_function
#from APS_tools import ifg


with open(tsdir + 'params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,lon_bounds,lat_bounds,ymin,ymax,alks,rlks = pickle.load(f)   

do_wrapped = 2 # 1 if you want to apply correction to wrapped data. 0 for unwrapped. 

lon_gac_min=-119
lon_gac_max=-116
lat_gac_min=33
lat_gac_max=35

xdim_gac = 3601
ydim_gac = 2401
gac_lon_vec = np.linspace(lon_gac_min, lon_gac_max, xdim_gac)
gac_lat_vec = np.linspace(lat_gac_min, lat_gac_max, ydim_gac)

gac_lat,gac_lon = np.meshgrid(gac_lat_vec, gac_lon_vec, sparse=False, indexing='ij')
gac_lat = np.flipud(gac_lat)

mergeddir=workdir + 'merged/'
f_lat = mergeddir + 'geom_master/lat_lk.rdr'
f_lon = mergeddir + 'geom_master/lon_lk.rdr'
f_los = mergeddir + 'geom_master/los_lk.rdr'


Image = isceobj.createImage()
Image.load(f_lon + '.xml')
lon_ifg = Image.memMap()[ymin:ymax,:,0]
lon_ifg = lon_ifg.copy().astype(np.float32)
lon_ifg[lon_ifg==0]=np.nan
Image.finalizeImage()

Image = isceobj.createImage()
Image.load(f_lat + '.xml')
lat_ifg = Image.memMap()[ymin:ymax,:,0]
lat_ifg = lat_ifg.copy().astype(np.float32)
lat_ifg[lat_ifg==0]=np.nan
Image.finalizeImage()

Image = isceobj.createImage()
Image.load(f_los + '.xml')
los_ifg = Image.memMap()[ymin:ymax,0,:]
los_ifg = los_ifg.copy().astype(np.float32)
los_ifg[los_ifg==0]=np.nan
Image.finalizeImage()
los_ifg = np.deg2rad(los_ifg) # inc angle in radians from earth looking to platform wrt vertical

f = tsdir + 'gamma0_lk.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0_lk= intImage.memMap()[ymin:ymax,:,0]

dates=list()
flist = glob.glob(intdir + '2*_2*')
[dates.append(f[-17:-9]) for f in flist]
dates.append(flist[-1][-8:])
dates.sort()

if not os.path.isfile(workdir +  'GACOS/gac_stack.pkl'):
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
        gf1 = workdir + 'GACOS/' + date1 + '.ztd'
        gf2 = workdir + 'GACOS/' + date2 + '.ztd' 
        gac1 = np.fromfile(gf1,dtype=np.float32)
        gac2 = np.fromfile(gf2,dtype=np.float32)
        gac = gac2-gac1
        gac = np.asarray(gac, dtype=np.float32)
    #    gac_grid =griddata((gac_lon.flatten(),gac_lat.flatten()),gac.flatten(), (lon_ifg,lat_ifg), method='linear')
        gac_grid=interpolate(gac, vtx, wts)
        gac_grid[gac_grid==0]=np.nan
        gac_grid = np.reshape(gac_grid,(len(np.arange(ymin,ymax)),nxl))
        gac_grid=np.asarray(gac_grid,dtype=np.float32)
        gac_grid-=np.nanmean(gac_grid)
        gac_stack.append(gac_grid)
else:
    print(workdir +  'GACOS/gac_stack.pkl already exists. Loading it...')
    with open(workdir +  'GACOS/gac_stack.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        gac_stack = pickle.load(f) 
with open(workdir +  'GACOS/gac_stack.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(gac_stack, f)

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
        gac = ((gac_stack[ii]*100)/np.cos(los_ifg))
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

h = workdir + 'merged/geom_master/hgt_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
hgt = hImg.memMap()[ymin:ymax,:,0].astype(np.float32)

ymin2=84
ymax2=2575
xmin=35
xmax=6320
crop_mask = np.zeros(hgt.shape)
crop_mask[ymin2:ymax2,xmin:xmax] =1

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
phs_ifg*=crop_mask
phs_ifg[phs_ifg==0]=np.nan
phs_ifg[np.where( (hgt<.1) ) ]=np.nan # masks the water
phs_ifg-=np.nanmedian(phs_ifg)

# Load phs corrected
f = intdir + pair + '/fine_lk_gac.unw'
Image = isceobj.createImage()
Image.load(f + '.xml')
phs_c = Image.memMap()[ymin:ymax,:,0]
phs_c = phs_c.copy().astype(np.float32)
phs_c*=crop_mask
phs_c[phs_c==0]=np.nan
phs_c[np.where( (hgt<.1) ) ]=np.nan # masks the water

# Load gac model
gac_mod = (gac_stack[idx]*100)/np.cos(los_ifg)
gac_mod*=crop_mask
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


# MAKE MAPS
pad=0
makeMap.makeImg(phs_ifg,lon_ifg,lat_ifg,vmin,vmax,pad,'Original IFG (cm)')
makeMap.makeImg(gac_mod,lon_ifg,lat_ifg,vmin,vmax,pad,'Modeled Tropospheric delay (cm)')
makeMap.makeImg(phs_c,lon_ifg,lat_ifg,vmin,vmax,pad,'Corrected IFG (cm)')

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
plt.savefig(workdir + 'Figs/GACOS_structfun.svg',transparent=True,dpi=100 )



