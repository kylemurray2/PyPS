#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:54:12 2020

@author: kdm95
"""
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve


# phase elevation model
def phaseElev(img, hgt,msk, ymin, ymax, xmin, xmax):
#     img[np.isnan(img)] = 0
    
#     hgt[np.isnan(hgt)] = 0
    p = img[ymin:ymax, xmin:xmax].copy()
    z = hgt[ymin:ymax, xmin:xmax].copy()
    p[msk[ymin:ymax, xmin:xmax]==0] = 0
    z[msk[ymin:ymax, xmin:xmax]==0] = 0
    G = np.vstack([z.ravel(), np.ones((len(z.ravel()),1)).flatten()]).T
    Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
    moda = np.dot(Gg,p.ravel())
    phs_model = moda[0] * hgt.ravel() + moda[1]
    phs_model = phs_model.reshape(img.shape)
    return phs_model

def ll2px(lon, lat, lon_ifgm,lat_ifgm,mapPoints=True):
    if mapPoints:
        lon, lat = m(lon,lat,inverse=True)
    comb = abs(lat-lat_ifgm) + abs(lon-lon_ifgm) 
    y,x = np.where(comb == comb.min())
    return x,y


def px2ll(x, y, lon_ifgm,lat_ifgm):
    lon = lon_ifg[y,x]
    lat = lat_ifg[y,x]
    return lon,lat

def fitLong(image,order):
    
    kernel = Gaussian2DKernel(x_stddev=1) # For smoothing and nan fill
    image = convolve(image,kernel)
    image[np.isnan(image)] = 0
    ny,nx = image.shape
    X,Y = np.meshgrid(range(nx),range(ny))
    X1,Y1 = X.ravel(),Y.ravel()
    
    if order==1:
        G  = np.array([np.ones((len(X1),)), X1, Y1]).T
        Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
        mod   = np.dot(Gg,image.ravel())
        synth = mod[0] + mod[1] * X1 + mod[2] * Y1
        synth = synth.reshape(ny,nx)
            
    if order==2:
        G  = np.array([np.ones((len(X1),)), X1, Y1, X1**2, Y1**2]).T
        Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
        mod   = np.dot(Gg,image.ravel())
        synth = mod[0] + mod[1] * X1 + mod[2] * Y1 + mod[3] * X1**2 + mod[4] * Y1**2 
        synth = synth.reshape(ny,nx)

    if order==3:
        G  = np.array([np.ones((len(X1),)), X1, Y1, X1**2, Y1**2,X1**3, Y1**3]).T
        Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
        mod   = np.dot(Gg,image.ravel())
        synth = mod[0] + mod[1] * X1 + mod[2] * Y1 + mod[3] * X1**2 + mod[4] * Y1**2 + mod[5] * X1**3 + mod[6] * Y1**3
        synth = synth.reshape(ny,nx)

    if order==4:
        G  = np.array([np.ones((len(X1),)), X1, Y1, X1**2, Y1**2,X1**3, Y1**3,X1**4, Y1**4]).T
        Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
        mod   = np.dot(Gg,image.ravel())
        synth = mod[0] + mod[1] * X1 + mod[2] * Y1 + mod[3] * X1**2 + mod[4] * Y1**2 + mod[5] * X1**3 + mod[6] * Y1**3 + mod[7] * X1**4 + mod[8] * Y1**4
        synth = synth.reshape(ny,nx)


    return synth
