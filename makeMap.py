#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:51:21 2020


Map an IFG or other gridded data

@author: km
"""
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.img_tiles as cimgt
from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def mapImg(img, lons, lats, vmin, vmax, pad, title):

    minlat=lats.min()
    maxlat=lats.max()
    minlon=lons.min()
    maxlon=lons.max()
    
    url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg'
    image = cimgt.GoogleTiles(url=url)
    data_crs = ccrs.PlateCarree()
    fig = plt.figure()
    ax = plt.axes(projection=data_crs)
    img_handle = plt.pcolormesh(lons, lats, img, transform=data_crs)
    
    lon_range = (pad+maxlon) - (minlon-pad)
    lat_range = (pad+maxlat) - (minlat-pad)
    rangeMin = np.min(np.array([lon_range,lat_range]))
    tick_increment = round(rangeMin/4,1)
    
    ax.set_xticks(np.arange(np.floor(minlon-pad),np.ceil(maxlon+pad),tick_increment), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.floor(minlat-pad),np.ceil(maxlat+pad),tick_increment), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.add_image(image,8) #zoom level
    plt.colorbar(img_handle,fraction=0.03, pad=0.09,orientation='horizontal')
    plt.title(title)
    plt.show()
    
    
def mapImg3(img1,img2,img3, lons, lats, vmin, vmax, pad, title1,title2,title3):

    minlat=lats.min()
    maxlat=lats.max()
    minlon=lons.min()
    maxlon=lons.max()
    
    lon_range = (pad+maxlon) - (minlon-pad)
    lat_range = (pad+maxlat) - (minlat-pad)
    rangeMin = np.min(np.array([lon_range,lat_range]))
    tick_increment = round(rangeMin/4,1)
    
    url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg'
    image = cimgt.GoogleTiles(url=url)
    data_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10,8))
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    
    # IMG 1
    ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
    img_handle = plt.pcolormesh(lons, lats, img1, transform=data_crs)
    ax1.set_xticks(np.arange(np.floor(minlon-pad),np.ceil(maxlon+pad),tick_increment), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(np.floor(minlat-pad),np.ceil(maxlat+pad),tick_increment), crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.add_image(image,8) #zoom level
    plt.colorbar(img_handle,fraction=0.03, pad=0.09,orientation='horizontal')
    plt.title(title1)
    
    # IMG 2
    ax2 = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
    img_handle = plt.pcolormesh(lons, lats, img1, transform=data_crs)
    ax2.set_xticks(np.arange(np.floor(minlon-pad),np.ceil(maxlon+pad),tick_increment), crs=ccrs.PlateCarree())
    ax2.set_yticks(np.arange(np.floor(minlat-pad),np.ceil(maxlat+pad),tick_increment), crs=ccrs.PlateCarree())
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    ax2.add_image(image,8) #zoom level
    plt.colorbar(img_handle,fraction=0.03, pad=0.09,orientation='horizontal')
    plt.title(title2)
    
    # IMG 3
    ax3 = plt.subplot(1, 3, 33, projection=ccrs.PlateCarree())
    img_handle = plt.pcolormesh(lons, lats, img1, transform=data_crs)
    ax3.set_xticks(np.arange(np.floor(minlon-pad),np.ceil(maxlon+pad),tick_increment), crs=ccrs.PlateCarree())
    ax3.set_yticks(np.arange(np.floor(minlat-pad),np.ceil(maxlat+pad),tick_increment), crs=ccrs.PlateCarree())
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)
    ax3.add_image(image,8) #zoom level
    plt.colorbar(img_handle,fraction=0.03, pad=0.09,orientation='horizontal')
    plt.title(title1)
    
    
    plt.show()
    
    
def mapBackground(bg, minlon, maxlon, minlat, maxlat, pad, zoomLevel, title, borders=True):
  
    '''
    Makes a background map that you can then plot stuff over (footprints, scatterplot, etc.)
    bg: background type from the ARCGIS database choose from the following:
        NatGeo_world_Map
        USA_Topo_Maps
        World_Imagery
        World_Physical_Map
        World_Shaded_Relief 
        World_Street_Map 
        World_Terrain_Base 
        World_Topo_Map 
        Specialty/DeLorme_World_Base_Map
        Specialty/World_Navigation_Charts
        Canvas/World_Dark_Gray_Base 
        Canvas/World_Dark_Gray_Reference
        Canvas/World_Light_Gray_Base 
        Canvas/World_Light_Gray_Reference
        Elevation/World_Hillshade_Dark 
        Elevation/World_Hillshade 
        Ocean/World_Ocean_Base 
        Ocean/World_Ocean_Reference 
        Polar/Antarctic_Imagery
        Polar/Arctic_Imagery
        Polar/Arctic_Ocean_Base
        Polar/Arctic_Ocean_Reference
        Reference/World_Boundaries_and_Places_Alternate
        Reference/World_Boundaries_and_Places
        Reference/World_Reference_Overlay
        Reference/World_Transportation 
        WorldElevation3D/Terrain3D
        WorldElevation3D/TopoBathy3D
    '''
    
    url = 'https://server.arcgisonline.com/ArcGIS/rest/services/' + bg + '/MapServer/tile/{z}/{y}/{x}.jpg'
    image = cimgt.GoogleTiles(url=url)
    data_crs = ccrs.PlateCarree()
    fig =  plt.figure(figsize=(6,6))
    ax = plt.axes(projection=data_crs)
    ax.set_extent([minlon-pad,maxlon+pad,minlat-pad,maxlat+pad], crs=ccrs.PlateCarree())

    if borders:
        ax.add_feature(cfeature.BORDERS,linewidth=1,color='white')
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.LAKES)
    # ax.add_feature(cfeature.RIVERS)
    
    lon_range = (pad+maxlon) - (minlon-pad)
    lat_range = (pad+maxlat) - (minlat-pad)
    rangeMin = np.min(np.array([lon_range,lat_range]))
    tick_increment = round(rangeMin/4,1)
    
    ax.set_xticks(np.arange(np.floor(minlon-pad),np.ceil(maxlon+pad),tick_increment), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.floor(minlat-pad),np.ceil(maxlat+pad),tick_increment), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.add_image(image, zoomLevel) #zoom level
    plt.title(title)
    plt.show()