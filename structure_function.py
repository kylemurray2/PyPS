#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:28:10 2018
@author: kdm95 
args: 
    ifg_data:       2D vector with unwrapped phase values or other pixel values
    ny, nx:         dimensions of image
    tot:            Total number of differences to make
    lengthscale:    Maximum distance between pixels being differenced
    plot_flag:      0 or 1 (off or on) to make plots or not.
    binwidth:       Bindwidth for differencing distances
    fun:            function to fit to the variogram. Can be 'spherical' or 'exp'.
"""

# Global imports
import numpy as np
import matplotlib.pyplot as plt
import scipy
# Local Imports


def struct_fun(data, ny,nx, tot=600, lengthscale=600, plot_flag=0, binwidth=20, fun=None):
    '''
    Main function to calculate structure function from a unwrapped ifg matrix (data)
    
    '''
    
#    ny,nx = lon_ifg.shape
    
    xx = np.arange(0,nx);yy=np.arange(0,ny)
    X,Y = np.meshgrid(xx,yy, sparse=False, indexing='ij')
    
    xd,yd = np.meshgrid([0,1,2,5,10,15,20,25,35,(lengthscale-binwidth),lengthscale],[-lengthscale, (-lengthscale+binwidth),-35,-25,-20,-15,-10,-5,-4,-2,-1,0,1,2,4,5,10,15,20,25,35,(lengthscale-binwidth),lengthscale], sparse=False, indexing='ij')  #dense sampling near origin

    tx    =np.floor(np.random.randint(1,lengthscale,size=tot))
    ty    =np.floor(np.random.randint(1,lengthscale,size=tot))
    ty[::2] = -ty[::2] # Make half of points negative; start stop step
    q=np.matrix([tx,ty]).T
    
    # Remove duplicates
    jnk,ids = np.unique(q,axis=0,return_index=True)
    tx = tx[ids]
    tx = np.asarray([*map(int, tx)])
    ty = ty[ids]
    ty = np.asarray([*map(int, ty)])
    
    #***add on dense grid from above;
    tx = np.append(tx, xd.flatten())
    ty = np.append(ty, yd.flatten())
    
    #***remove duplicates
#    a=np.array((tx,ty))
#    ix = np.unique(a,return_index=True, axis=1);
#    tx       = tx[ix[1]];
#    ty       = ty[ix[1]];

    aty = abs(ty) # used for the negative offsets
    S = np.empty([len(tx)])
#    S2 = np.empty([len(tx)])
    allnxy = np.empty([len(tx)])
    iters = np.arange(0,len(tx))
    
    for ii in iters:
        i=int(ii)
        if ty[ii] >= 0: 
            A = data[1 : ny-ty[ii] , tx[ii] : nx-1 ]
            B = data[ty[i] : ny-1 , 1 : nx-tx[i] ];
        else:
            A = data[aty[ii] : ny-1 , tx[ii] : nx-1]
            B = data[1 : ny-aty[ii] , 1 : nx-tx[ii]]
    
        C = A-B # All differences
        C2 = np.square(C)
        
        S[ii] = np.nanmean(C2)       
#        S2[ii] = np.nanstd(C2)

        allnxy[ii] = len(C2);
    dists = np.sqrt(np.square(tx) + np.square(ty))
    
#    S[np.isnan(S)]=0
    bins = np.arange(0,dists.max(),binwidth,dtype=int)
    S_bins=list()
#    S2_bins=list()
    Ws = list()
    dist_bins=list()
    for ii,bin_min in enumerate(bins):
        bin_ids = np.where((dists< (bin_min+binwidth)) & (dists>bin_min))
        w = allnxy[bin_ids] #these are the weights for the weighted average
        if len(w)==0:
            S_bins.append(np.nan)  
#            S2_bins.append(np.nan)
            dist_bins.append(np.nan)
        elif len(w)==1:
            S_bins.append(S[bin_ids[0]])  
#            S2_bins.append(S2[bin_ids[0]])  
            dist_bins.append(np.nan)
        else:
            S_bins.append(np.average(S[bin_ids],axis=0,weights=w))  
#            S2_bins.append(np.average(S2[bin_ids],axis=0,weights=w))  
            Ws.append(len(w))
            dist_bins.append(np.nanmean(dists[bin_ids]))
    
    if plot_flag:
        fig = plt.figure(figsize=(14,10))
        # Plot IFG
        ax = fig.add_subplot(221)
        ax.set_title("Image")
        cf = plt.imshow(data)
        #cmap=plt.cm.Spectral.reversed()
        plt.colorbar(cf)
        
        ax = fig.add_subplot(222)
        ax.set_title("sqrt(S) vs. position")
        cf = plt.scatter(tx,ty,c=np.sqrt(S))
        plt.scatter(-tx,-ty,c=np.sqrt(S))
        plt.ylabel('north')
        plt.xlabel('east')
        plt.colorbar(cf)
        
        ax = fig.add_subplot(212)
        ax.set_title("S vs. distance, colored by num points")
        cf = plt.scatter(dists[1:],np.sqrt(S[1:]),c=allnxy[1:])
        plt.ylabel('sqrt(S), units of cm')
        plt.xlabel('distance(km)')
        plt.colorbar(cf)
        plt.show()
        
        
    # Fit a log function to the binned data   
#    S_bins = np.asarray(S_bins)
#    S_bins[np.where(np.isnan(S_bins))]=0
    xd = np.asarray(dist_bins)
    oh=np.asarray(S_bins,dtype=np.float32)/2
#    oh[np.isnan(oh)]=0
    yd = np.sqrt(oh)
#    yd_std = np.sqrt(S2_bins) 
    yd[np.isnan(yd)]=0
#    yd_std[np.isnan(yd_std)]=0

    
    # Fit exponential function to structure function
    # y = A*log(Bx)
    if fun=='exp':
        def fit_log(x,a,b,c):
            '''
            Spherical model of the semivariogram
            '''
            return a*np.log(b*x)+c
    
        popt, pcov = scipy.optimize.curve_fit(fit_log,xd,yd)
        sf_fit = fit_log(xd, *popt)
        
        
    elif fun=='spherical': 
        def spherical(x, a, b ):
            '''
            Spherical model of the semivariogram
            '''
            return b*( 1.5*x/a - 0.5*(x/a)**3.0 )
        
        popt, pcov = scipy.optimize.curve_fit(spherical,xd,yd)
        sf_fit = spherical(xd, *popt)
    
    else:
        print('No function specified. Can be spherical or exp.')
        sf_fit=0
    
    S2=0
    yd_std=0
    return np.sqrt(S/2), S2, dists, allnxy, yd, yd_std, xd, sf_fit