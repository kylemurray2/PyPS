#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Kyle Murray
Tue Jan  1 12:52:56 2019
Description:
    
Does the cluster hgt correction

Inputs:
    image: Input image
    lon
    lat
    hgt
    nx
    ny
    nk: Number of K clusters
    df: decimate_factor - downsamples data to run faster. Use higher # for bigger images
    
    
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.interpolate import griddata
import cv2 as cv
from scipy.stats import kde
from scipy.optimize import curve_fit


def hgtClust(phs_ifg,msk_ifg,msk,lon_ifg,lat_ifg,hgt_ifg,minlon,maxlon,minlat,maxlat,nx,ny,pair,ws,nk=8,df=10,doPowerLaw=False,plotflag=True):
    """ Linear hgt correction """
    
    gaus_size = ws  #gaussian size for smoothing parameters

    mf = msk_ifg.flatten()
    minhgt = -5
    
    # Flatten all the arrays
    lats = lat_ifg.flatten()
    lons = lon_ifg.flatten()
    pc   = phs_ifg.flatten()
    z    = hgt_ifg.flatten()
    
    # Take subset of good pixels from each array
    zz = z[mf==1]
    lats = lats[mf==1]
    lons = lons[mf==1]
    phsy = pc[mf==1] # this is just the unflattened version of pc

   
    # If there are any nans in the heights, then make them -100
    zz[np.isnan(zz)] = -100
    
    # do kmeans clusters
#    iii = np.arange(0,len(zz),1)
    zphs = np.array([zz,phsy,lons,lats]).T
    whitened = whiten(zphs)   # whiten the data so that each dimension contributes equally to kmeans  
    codebook,_ = kmeans(whitened, nk) # Find the cluster centroids. This is saved in 'codebook' nk X nDim
    phs_km,_ = vq(whitened,codebook) #  Vector Quantization. phs_km has the cluster number for each index

    # set parameters for Support Vector Machine (SVM) to remove outliers
    from sklearn import svm
    outliers_fraction = .1 #
    nu_estimate = 0.95 * outliers_fraction + 0.05
    auto_detection = svm.OneClassSVM(kernel='rbf', gamma=0.01, degree=3, nu=nu_estimate)
    
    def pl(hq,Kq,h0q,alphaq,phs0q):
        return Kq*(h0q-hq)**alphaq+phs0q

    def powerLawPE(hgtq, phsq,plotFlag=False):
        phsq = phsq.ravel()
        hgtq = hgtq.ravel()
        maxh=hgtq.max()
        idx=np.where(hgtq==maxh)[0][0]
        phs0=phsq[idx]
        minBounds = [-np.inf, maxh, 1, -np.inf]
        maxBounds = [np.inf, 7000, 5, np.inf]
        initialGuess = [1e-3, maxh, 1.3, phs0]
        popt, pcov = curve_fit(pl, hgtq, phsq,p0=initialGuess,bounds=(minBounds,maxBounds))
        
        if plotFlag:
            plt.figure()
            plt.plot(hgtq, phsq, 'b.', label='data')
            plt.plot(np.sort(hgtq), pl(np.sort(hgtq), popt[0],popt[1],popt[2],popt[3]), 'r-')
        
        return popt, pcov
    
    
    modsPL = []
    modsL = []
    if plotflag:
        fig, ax = plt.subplots(1,3, figsize=(13,4))
    for jj in np.arange(0,nk): # loop through nk clusters
        zr1 = zz[phs_km == jj] # phs_km == jj pulls out all zz associated with cluster jj
        phsr1 = phsy[phs_km == jj]
    
        # Take only every df points (decimating)
        zr = zr1[np.arange(0,len(zr1),df)] # df is the step size to downsample
        phsr = phsr1[np.arange(0,len(phsr1),df)]
        
        # Remove outliers with SVM
        zc = np.array([zr,phsr]).T
        auto_detection.fit(zc)
        evaluation = auto_detection.predict(zc)
        zr = zr[evaluation==1]
        phsr = phsr[evaluation==1]
        
        
        # Power Law phase-elevation model
#        mod1,cov = powerLawPE(zr,phsr,plotFlag=False)
#        modsPL.append(mod1) 
#        phsPL = pl(np.sort(zr),mod1[0],mod1[1],mod1[2],mod1[3])

        # Linear phase-elevation model
        G = np.vstack([zr, np.ones((len(zr),1)).flatten()]).T
        Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
        mod = np.dot(Gg,phsr)
        phsL = mod[0] * zr + mod[1]
        modsL.append(np.dot(Gg, phsr))
    
    
        if plotflag:
            ax[1].scatter(zr,phsr, s=1,rasterized=True)
            ax[1].plot(zr,phsL,'k',linewidth=2)
            ax[1].plot(zr,phsL,linewidth=1)
            ax[1].set_xlabel('Elevation (m)');
            ax[2].scatter(zr,phsr, s=1,rasterized=True)
#            ax[2].plot(np.sort(zr),phsPL,'k',linewidth=2)
#            ax[2].plot(np.sort(zr),phsPL,linewidth=1)

    if doPowerLaw:
        # Do a power law correction for the uniform case
        modo,cov = powerLawPE(zz,phsy,plotFlag=False)
        phso = pl(np.sort(zz),modo[0],modo[1],modo[2],modo[3])
        phs_m_uniform = griddata((lons,lats), phso , (lon_ifg,lat_ifg), method='nearest')
        phs_c_uniform = phs_ifg - phs_m_uniform
        if plotflag:
            ax[0].scatter(zz,phsy,s=.1,color='black',rasterized=True)
            ax[0].plot(np.sort(zz),phso,linewidth=2,color=plt.cm.tab10(0))
            ax[0].set_ylabel('Phase (cm)')
            plt.savefig('PhaseElevation/Figures/clustered_phselev_scatter_' + str(nk) + 'K' + pair + '.svg',transparent=True,dpi=300)

    else:
        G = np.vstack([zz, np.ones((len(zz),1)).flatten()]).T
        Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
        moda = np.dot(Gg,phsy.ravel())
        phso = moda[0] * zz + moda[1]
        phs_m_uniform = griddata((lons,lats), phso , (lon_ifg,lat_ifg), method='nearest')
        phs_c_uniform = phs_ifg - phs_m_uniform
        if plotflag:
            ax[0].scatter(zz,phsy,s=.1,color='black',rasterized=True)
            ax[0].plot(zz,phso,linewidth=2,color=plt.cm.tab10(0))
            ax[0].set_ylabel('Phase (cm)')
            plt.savefig('PhaseElevation/Figures/clustered_phselev_scatter_' + str(nk) + 'K' + pair + '.svg',transparent=True,dpi=300)
    
    
    lons_clust=np.empty((1,len(phs_km)),dtype=np.float32)[0,:]
    lats_clust=np.empty((1,len(phs_km)),dtype=np.float32)[0,:]
    phs_clust=np.empty((1,len(phs_km)),dtype=np.float32)[0,:]
    hgts_clust =np.empty((1,len(phs_km)),dtype=np.float32)[0,:]
    
    for kk in np.arange(0,nk):
        lons_clust[phs_km == kk] = lons[phs_km == kk]
        lats_clust[phs_km == kk] = lats[phs_km == kk]
        phs_clust[phs_km == kk]  = phsy[phs_km==kk]
        hgts_clust[phs_km == kk] = phs_km[phs_km==kk]
        
    phase_map = griddata((lons_clust,lats_clust), phs_clust , (lon_ifg,lat_ifg), method='nearest')
    clust_map = griddata((lons_clust,lats_clust), hgts_clust, (lon_ifg,lat_ifg), method='nearest')
    

    if doPowerLaw:
        a = np.empty(clust_map.shape)
        b = np.empty(clust_map.shape)
        c = np.empty(clust_map.shape)
        d = np.empty(clust_map.shape)

        for kk in np.arange(0,nk):
            a[clust_map==kk] = modsPL[kk][0]
            b[clust_map==kk] = modsPL[kk][1]
            c[clust_map==kk] = modsPL[kk][2]
            d[clust_map==kk] = modsPL[kk][3]
        
        a2 = cv.GaussianBlur(a,(gaus_size,gaus_size),0)
        b2 = cv.GaussianBlur(b,(gaus_size,gaus_size),0)
        c2 = cv.GaussianBlur(c,(gaus_size,gaus_size),0)
        d2 = cv.GaussianBlur(d,(gaus_size,gaus_size),0)
        hgt_m = pl(hgt_ifg,a2,b2,c2,d2) # Use the interpolated parameters in the powerlaw equation
            
    else:  # Do the normal linear model 
        slopes = np.empty(clust_map.shape)
        yints = np.empty(clust_map.shape)
        for kk in np.arange(0,nk):
            slopes[clust_map==kk] = modsL[kk][0]
            yints[clust_map==kk] = modsL[kk][1]
        slopes2 = cv.GaussianBlur(slopes,(gaus_size,gaus_size),0)
        yints2 = cv.GaussianBlur(yints,(gaus_size,gaus_size),0)
        hgt_m = hgt_ifg * slopes2 + yints2
    
    lo = lon_ifg.flatten()
    la = lat_ifg.flatten()
    
    hgt_m_full = griddata((lo[~np.isnan(z)],la[~np.isnan(z)]), hgt_m.flatten()[~np.isnan(z)] , (lon_ifg,lat_ifg), method='nearest')
    phs_c = phs_ifg - hgt_m_full
    
    if plotflag:
        hgt_ifg[np.isnan(hgt_ifg)]=-20
        phase_map[hgt_ifg<minhgt] = np.nan
        hgt_m[hgt_ifg<minhgt] = np.nan
        phs_c[hgt_ifg<minhgt] = np.nan
        
        phs_m_uniform[hgt_ifg<minhgt] = np.nan
        phs_c_uniform[hgt_ifg<minhgt] = np.nan
        
        if doPowerLaw:
            slopes2 = a2
            
        slopes2[hgt_ifg<minhgt] = np.nan    
        yints2[hgt_ifg<minhgt] = np.nan           
        hgt_ifg[hgt_ifg<minhgt] = np.nan


        phase_map[hgt_ifg<minhgt]=np.nan
        
        pm = phase_map.copy()
        pm[msk==0] = np.nan
        hm = hgt_m.copy()
        hm[msk==0] = np.nan
        pcm = phs_c.copy()
        pcm[msk==0] = np.nan
        
        hmu = phs_m_uniform.copy()
        hmu[msk==0] = np.nan
        pcmu = phs_c_uniform.copy()
        pcmu[msk==0] = np.nan
        
        # make correction maps
        plt.rc('font',size=12);
        fig = plt.figure(figsize=(10,8))
        vmin = -12
        vmax = 12
        c = plt.cm.viridis
        pad=0
        ax = fig.add_subplot(131)
        ax.set_title("Original IFG")
        m = Basemap(projection='merc',\
                llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
                llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='i',epsg=3395)
        m.arcgisimage(service='World_Shaded_Relief',xpixels=600)
        m.drawstates(linewidth=1.5,zorder=1,color='white')
        m.drawparallels(np.arange(np.floor(minlat), np.ceil(maxlat), 2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])  # set linwidth to zero so there is no grid
        m.drawmeridians(np.arange(np.floor(minlon), np.ceil(maxlon),2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])
        cf = m.pcolormesh(lon_ifg,lat_ifg,pm-np.nanmedian(pm),vmin=vmin,vmax=vmax,latlon=True,zorder=3,rasterized=True)
        cbar = m.colorbar(cf,location='bottom',pad="10%")
        cbar.set_label('cm')    
        
        ax = fig.add_subplot(132)
        hgt_m[hgt_ifg<minhgt]=np.nan
#        hgt_m[msk_ifg==0]=np.nan
        ax.set_title("Uniform model")
        m = Basemap(projection='merc',\
                llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
                llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='i',epsg=3395)
        m.arcgisimage(service='World_Shaded_Relief',xpixels=600)
        m.drawstates(linewidth=1.5,zorder=1,color='white')
        m.drawparallels(np.arange(np.floor(minlat), np.ceil(maxlat), 2), linewidth=.3,dashes=[5,15], labels=[0,0,0,0])  # set linwidth to zero so there is no grid
        m.drawmeridians(np.arange(np.floor(minlon), np.ceil(maxlon),2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])
        cf = m.pcolormesh(lon_ifg,lat_ifg,hm-np.nanmedian(hm),shading='flat',vmin=vmin,vmax=vmax,latlon=True,zorder=3,rasterized=True)
        cbar = m.colorbar(cf,location='bottom',pad="10%")
        cbar.set_label('cm')     
        
        ax = fig.add_subplot(133)
        phs_c[hgt_ifg<0]=np.nan
#        phs_c[msk_ifg==0]=np.nan
        ax.set_title("Corrected phase")
        m = Basemap(projection='merc',\
                llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
                llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='i',epsg=3395)
        m.arcgisimage(service='World_Shaded_Relief',xpixels=600)
        m.drawstates(linewidth=1.5,zorder=1,color='white')
        m.drawparallels(np.arange(np.floor(minlat), np.ceil(maxlat), 2), linewidth=.3,dashes=[5,15], labels=[0,0,0,0])  # set linwidth to zero so there is no grid
        m.drawmeridians(np.arange(np.floor(minlon), np.ceil(maxlon),2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])
        cf = m.pcolormesh(lon_ifg,lat_ifg,pcm-np.nanmedian(pcm),shading='flat',vmin=vmin,vmax=vmax,cmap=c,latlon=True,zorder=3,rasterized=True)
        cbar = m.colorbar(cf,location='bottom',pad="10%")
        cbar.set_label('cm')     
        
        if doPowerLaw:
            plt.savefig('PhaseElevation/Figures/Uniform_correctionmaps_PL_' + str(nk) + 'K' + pair + '.svg')
        else:
            plt.savefig('PhaseElevation/Figures/correctionmaps' + str(nk) + 'K' + pair + '.svg')


        hgt_ifg[hgt_ifg<-5] = np.nan
        slopes2[hgt_ifg<-5] = np.nan

# Plot cluster,slope, and DEM
        plt.rc('font',size=12);
        fig = plt.figure(figsize=(10,8))
        c = plt.cm.viridis
        pad=0
        ax = fig.add_subplot(141)
        ax.set_title("Elevation")
        m = Basemap(projection='merc',\
                llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
                llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='i',epsg=3395)
        m.arcgisimage(service='World_Shaded_Relief',xpixels=600)
        m.drawstates(linewidth=1.5,zorder=1,color='white')
        m.drawparallels(np.arange(np.floor(minlat), np.ceil(maxlat), 2), linewidth=.3,dashes=[5,15], labels=[0,0,0,0])  # set linwidth to zero so there is no grid
        m.drawmeridians(np.arange(np.floor(minlon), np.ceil(maxlon),2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])
        cf = m.pcolormesh(lon_ifg,lat_ifg,hgt_ifg,shading='flat',cmap=c,latlon=True,zorder=3,vmin=10,vmax=3000,rasterized=True)
        cbar = m.colorbar(cf,location='bottom',pad="10%")
        cbar.set_label('m')
    
        
        ax = fig.add_subplot(142)
        ax.set_title("Clusters")
        # Plot cluster map
        m = Basemap(projection='merc',\
                llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
                llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='i',epsg=3395)
        m.arcgisimage(service='World_Shaded_Relief',xpixels=600)
        m.drawstates(linewidth=1.5,zorder=1,color='white')
        m.drawparallels(np.arange(np.floor(minlat), np.ceil(maxlat), 2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])  # set linwidth to zero so there is no grid
        m.drawmeridians(np.arange(np.floor(minlon), np.ceil(maxlon),2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])
        for kk in np.arange(0,nk): 
              m.scatter(lons[phs_km ==kk], lats[phs_km == kk], s=2,cmap=plt.cm.viridis_r, latlon=True,zorder=12,rasterized=True)
        cbar = m.colorbar(cf,location='bottom',pad="10%")
        cbar.set_label('m')
        
        
        ax = fig.add_subplot(143)
        ax.set_title("Phase/elevation dependence (slope)")
        m = Basemap(projection='merc',\
                llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
                llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='i',epsg=3395)
        m.arcgisimage(service='World_Shaded_Relief',xpixels=600)
        m.drawstates(linewidth=1.5,zorder=1,color='white')
        m.drawparallels(np.arange(np.floor(minlat), np.ceil(maxlat), 2), linewidth=.3,dashes=[5,15], labels=[0,0,0,0])  # set linwidth to zero so there is no grid
        m.drawmeridians(np.arange(np.floor(minlon), np.ceil(maxlon),2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])
        cf = m.pcolormesh(lon_ifg,lat_ifg,slopes2*10,cmap=plt.cm.viridis_r,latlon=True,zorder=3,rasterized=True)
        cbar = m.colorbar(cf,location='bottom',pad="10%")
        cbar.set_label('mm/m')    
        
        ax = fig.add_subplot(144)
        ax.set_title("Bias (Y-intercept)")
        m = Basemap(projection='merc',\
                llcrnrlat=minlat-pad,urcrnrlat=maxlat+pad,\
                llcrnrlon=minlon-pad,urcrnrlon=maxlon+pad,resolution='i',epsg=3395)
        m.arcgisimage(service='World_Shaded_Relief',xpixels=600)
        m.drawstates(linewidth=1.5,zorder=1,color='white')
        m.drawparallels(np.arange(np.floor(minlat), np.ceil(maxlat), 2), linewidth=.3,dashes=[5,15], labels=[0,0,0,0])  # set linwidth to zero so there is no grid
        m.drawmeridians(np.arange(np.floor(minlon), np.ceil(maxlon),2), linewidth=.3,dashes=[5,15], labels=[1,0,0,1])
        cf = m.pcolormesh(lon_ifg,lat_ifg,yints2,cmap=plt.cm.viridis_r,latlon=True,zorder=3,rasterized=True)
        cbar = m.colorbar(cf,location='bottom',pad="10%")
        cbar.set_label('cm')  
        
        if doPowerLaw:
            plt.savefig('PhaseElevation/Figures/clusters_slopes_maps_PL_' + str(nk) + 'K' + pair + '.svg')
        else:
            plt.savefig('PhaseElevation/Figures/clusters_slopes_maps' + str(nk) + 'K' + pair + '.svg')
        plt.show()
        
    return phs_c, hgt_m, phs_c_uniform, phs_m_uniform  #return the corrected image
