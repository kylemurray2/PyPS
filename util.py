#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:54:12 2020

@author: kdm95
"""
import numpy as np

def getTime(path,frame):
    '''
     Figure out what time the aquisition was
    '''
    import os
    import requests
    import pandas as pd
    path = str(path)
    frame = str(frame)
    # path = os.getcwd().split('/')[-2]
    # frame= os.getcwd().split('/')[-1]

    start='2014-05-01T00:00:00Z'
    end='2099-06-01T00:00:00Z'
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
    
    Lon = pd.read_csv('out.csv')["Near Start Lon"][0]
    Lat = pd.read_csv('out.csv')["Near Start Lat"][0]
    return int(hour),int(minute), Lon, Lat
   



def reGridStack(inputImageStack,outputX,outputY):
    '''
    inputImageStack: (m-rows,n-colums,k-stack)
    outputX: output coordinates (probably lon_ifg) (m-out,n-out)
    outputY: output coordiantes (probably lat_ifg) (m-out,n-out)
    refer to https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    '''
    import scipy.interpolate as spint
    import scipy.spatial.qhull as qhull
    import itertools
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
    
    m,n,k = inputImageStack.shape
    mi,ni = outputX.shape    
    [Y,X]   =np.meshgrid(np.linspace(0,1,n),np.linspace(0,2,m))
    [Yi,Xi] =np.meshgrid(np.linspace(0,1,ni),np.linspace(0,2,mi))
    
    xy=np.zeros([X.shape[0]*X.shape[1],2])
    xy[:,0]=Y.ravel()
    xy[:,1]=X.ravel()
    uv=np.zeros([Xi.shape[0]*Xi.shape[1],2])
    uv[:,0]=Yi.ravel()
    uv[:,1]=Xi.ravel()
    
    #Computed once and for all !
    vtx, wts = interp_weights(xy, uv)
    outputStack = np.zeros((mi,ni,k))

    for kk in np.arange(k):
        values=inputImageStack[:,:,kk]
        valuesi=interpolate(values.ravel(), vtx, wts)
        valuesi=valuesi.reshape(Xi.shape[0],Xi.shape[1])
        outputStack[:,:,kk] = valuesi
        
    return outputStack
    

def improfile(z, x0, y0, x1, y1):
    """
    Get a profile
    Captures 1d profile values from point A to B specified as indices. 
    Inputs: im, x0, y0, x1, y1
    Outputs: z (vector of values along the profile) 
    """
    length = int(np.hypot(x1-x0, y1-y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    
    # Extract the values along the line
    zi = z[y.astype(int), x.astype(int)]
    return zi


def ll2pixel(lon_ifg,lat_ifg,lon,lat):
    """
    Output the pixels (radar coords) given the lat/lon matrices and arrays of 
    lat/lon points.
    output: y,x
    """
    x_pts = list()
    y_pts = list()


    
    if np.isscalar(lon):
        if np.nanmean(lon_ifg) * lon <0:
            print('WARNING: you may need to subtract 360')
            
        a = abs(lat_ifg-lat)
        b = abs(lon_ifg-lon)
        c = a+b
        y,x = np.where(c==c.min()) # y is rows, x is columns
        
        if not np.isscalar(x):
            x=x[0];y=y[0]
        
        x_pts.append(x)
        y_pts.append(y)
    
    else:
        if np.nanmean(lon_ifg) * lon[0] <0:
            print('WARNING: you may need to subtract 360')
        for ii in np.arange(0,len(lat)):
            a = abs(lat_ifg-lat[ii])
            b = abs(lon_ifg-lon[ii])
            c = a+b
            y,x = np.where(c==c.min()) # y is rows, x is columns
            
            if not np.isscalar(x):
                x=x[0];y=y[0]
            
            x_pts.append(x)
            y_pts.append(y)
    return y_pts,x_pts 


# phase elevation model
def phaseElev(img, hgt,msk, ymin, ymax, xmin, xmax,makePlot=False):
    '''
    Take the ifg or rate map and the dem and mask and outputs phs/elev dependence.
    Use ymin, xmin, etc. if you want to only use a subset of the image.
      otherwise, put the image len/width for those values.
    '''
#     img[np.isnan(img)] = 0
    
#     hgt[np.isnan(hgt)] = 0
    p = img[ymin:ymax, xmin:xmax].copy()
    z = hgt[ymin:ymax, xmin:xmax].copy()
    if makePlot:
        from matplotlib import pyplot as plt
        plt.figure();plt.scatter(z.ravel(),p.ravel(),.1)
        plt.title('Phase-elevation dependence')
        plt.xlabel('Elevation (m)')
        plt.ylabel('Phase (rad)')
        
    p[msk[ymin:ymax, xmin:xmax]==0] = 0
    z[msk[ymin:ymax, xmin:xmax]==0] = 0
    G = np.vstack([z.ravel(), np.ones((len(z.ravel()),1)).flatten()]).T
    Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
    moda = np.dot(Gg,p.ravel())
    phs_model = moda[0] * hgt.ravel() + moda[1]
    phs_model = phs_model.reshape(img.shape)
    return phs_model


def px2ll(x, y, lon_ifgm,lat_ifgm):
    lon = lon_ifgm[y,x]
    lat = lat_ifgm[y,x]
    return lon,lat


def fitLong(image,order,mask):
    from astropy.convolution import Gaussian2DKernel,convolve
    kernel = Gaussian2DKernel(x_stddev=1) # For smoothing and nan fill
    image = convolve(image,kernel)
    image[np.isnan(image)] = 0
    image[mask==0] = 0
    ny,nx = image.shape
    X,Y = np.meshgrid(range(nx),range(ny))
    X1,Y1 = X.ravel(),Y.ravel()
    
    if order==1: # Plane
        G  = np.array([np.ones((len(X1),)), X1, Y1]).T
        Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
        mod   = np.dot(Gg,image.ravel())
        synth = mod[0] + mod[1] * X1 + mod[2] * Y1
        synth = synth.reshape(ny,nx)
            
    if order==2: # Quadratic
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


def json2bbox(file):
    '''
    Takes a json file (geojson) and returns the bounds of the associated rectangle
    You can make a geojson file here: http://geojson.io/
    (minlon, minlat, maxlon, maxlat). This is the format for stac
    '''
    import json
    import numpy as np
    f = open(file,)
    coords = json.loads(f.read())['features'][0]['geometry']['coordinates'][0]
    lons=[]
    lats=[]
    
    for coord in coords:
        lons.append(coord[0])
        lats.append(coord[1])
        
    bbox = [np.min(lons), np.min(lats), np.max(lons), np.max(lats)]    
    return bbox,lons,lats



def struct_fun(data, ny,nx, tot=600, lengthscale=600, plot_flag=0, binwidth=20, fun=None):
    '''
    Main function to calculate structure function from a unwrapped ifg matrix (data)
    
    '''
    import matplotlib.pyplot as plt
    import scipy
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


def estimate_dem_error(ts0, G0, tbase, date_flag=None, phase_velocity=False):
    """Estimate DEM error with least square optimization.
    Parameters: ts0            - 2D np.array in size of (numDate, numPixel), original displacement time-series
                G0             - 2D np.array in size of (numDate, numParam), design matrix in [G_geom, G_defo]
                tbase          - 2D np.array in size of (numDate, 1), temporal baseline
                date_flag      - 1D np.array in bool data type, mark the date used in the estimation
                phase_velocity - bool, use phase history or phase velocity for minimization
    Returns:    delta_z        - 2D np.array in size of (1,       numPixel) estimated DEM residual
                ts_cor         - 2D np.array in size of (numDate, numPixel),
                                    corrected timeseries = tsOrig - delta_z_phase
                ts_res         - 2D np.array in size of (numDate, numPixel),
                                    residual timeseries = tsOrig - delta_z_phase - defModel
    Example:    delta_z, ts_cor, ts_res = estimate_dem_error(ts, G, tbase, date_flag)
    """
    import scipy
    if len(ts0.shape) == 1:
        ts0 = ts0.reshape(-1, 1)
    if date_flag is None:
        date_flag = np.ones(ts0.shape[0], np.bool_)

    # Prepare Design matrix G and observations ts for inversion
    G = G0[date_flag, :]
    ts = ts0[date_flag, :]
    if phase_velocity:
        tbase = tbase[date_flag, :]
        G = np.diff(G, axis=0) / np.diff(tbase, axis=0)
        ts = np.diff(ts, axis=0) / np.diff(tbase, axis=0)

    # Inverse using L-2 norm to get unknown parameters X
    # X = [delta_z, constC, vel, acc, deltaAcc, ..., step1, step2, ...]
    # equivalent to X = np.dot(np.dot(np.linalg.inv(np.dot(G.T, G)), G.T), ts)
    #               X = np.dot(np.linalg.pinv(G), ts)
    X = scipy.linalg.lstsq(G, ts, cond=1e-15)[0]

    # Prepare Outputs
    delta_z = X[0, :]
    ts_cor = ts0 - np.dot(G0[:, 0].reshape(-1, 1), delta_z.reshape(1, -1))
    ts_res = ts0 - np.dot(G0, X)

    # for debug
    debug_mode = False
    if debug_mode:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))
        ts_all = np.hstack((ts0, ts_res, ts_cor))
        ymin = np.min(ts_all)
        ymax = np.max(ts_all)
        ax1.plot(ts0, '.');           ax1.set_ylim((ymin, ymax)); ax1.set_title('Original  Timeseries')
        ax2.plot(ts_cor, '.');        ax2.set_ylim((ymin, ymax)); ax2.set_title('Corrected Timeseries')
        ax3.plot(ts_res, '.');        ax3.set_ylim((ymin, ymax)); ax3.set_title('Fitting Residual')
        ax4.plot(ts_cor-ts_res, '.'); ax4.set_ylim((ymin, ymax)); ax4.set_title('Fitted Deformation Model')
        plt.show()

    return delta_z, ts_cor, ts_res


def viewIFGstack(flip=True,chain=True):
    ''' look at all the ifgs with napari'''
    
    import numpy as np
    import isceobj
    import napari
    
    ps = np.load('./ps.npy',allow_pickle=True).all()
    
    if chain:
        pairs = ps.pairs
    else:
        pairs = ps.pairs2
    
    stack = np.zeros((len(pairs),ps.nyl,ps.nxl))
    for ii in range(len(pairs)):
        p = pairs[ii]
        f = './merged/interferograms/' + p + '/fine_lk_filt.int'
        intImage = isceobj.createIntImage()
        intImage.load(f + '.xml')
        ifg = intImage.memMap()[:,:,0] 
        ifgc = np.angle(ifg)
        if flip:
            stack[ii,:,:] = np.flipud(ifgc)
        else:
            stack[ii,:,:] = ifgc
    viewer = napari.view_image(stack,colormap='RdYlBu')


def viewUNWstack(flip=True,chain=True):
    ''' look at all the ifgs with napari'''
    
    import numpy as np
    import isceobj
    import napari
    
    ps = np.load('./ps.npy',allow_pickle=True).all()
    gam = np.load('./Npy/gam.npy')
    if chain:
        pairs = ps.pairs
    else:
        pairs = ps.pairs2
    
    
    
    stack = np.zeros((len(pairs),ps.nyl,ps.nxl))
    for ii in range(len(pairs)):
        p = pairs[ii]
        f = './merged/interferograms/' + p + '/filt.unw'
        intImage = isceobj.createImage()
        intImage.dataType='FLOAT'
        intImage.load(f + '.xml')
        unw = intImage.memMap()[:,:,0]
        unw = unw.copy()
        unw[gam==0] = 0
        if flip:
            stack[ii,:,:] = np.flipud(unw)
        else:
            stack[ii,:,:] = unw
    viewer = napari.view_image(stack,colormap='RdYlBu')

    # fig,ax = plt.subplots(4,8)
    # kk=0
    # for a in ax.ravel():
    #     a.imshow(stack[kk,:,:])
    #     a.axes.xaxis.set_visible(False)
    #     a.axes.yaxis.set_visible(False)
    # plt.tight_layout()


def viewCORstack(flip=True,chain=True):
    ''' look at all the ifgs with napari'''
    
    import numpy as np
    import isceobj
    import napari
    
    ps = np.load('./ps.npy',allow_pickle=True).all()
    gam = np.load('./Npy/gam.npy')
    
    if chain:
        pairs = ps.pairs
    else:
        pairs = ps.pairs2
        
    stack = np.zeros((len(pairs),ps.nyl,ps.nxl))
    for ii in range(len(pairs)):
        p = pairs[ii]
        f = './merged/interferograms/' + p + '/cor.r4'
        intImage = isceobj.createImage()
        intImage.dataType='FLOAT'
        intImage.load(f + '.xml')
        unw = intImage.memMap()[:,:,0]
        unw = unw.copy()
        unw[gam==0] = 0
        if flip:
            stack[ii,:,:] = np.flipud(unw)
        else:
            stack[ii,:,:] = unw
    viewer = napari.view_image(stack[0,:,:],colormap='jet')
    
    
def getUNW(pair):
    import isceobj
    gam = np.load('Npy/gam.npy')
    f = './merged/interferograms/' + pair + '/filt.unw'
    intImage = isceobj.createImage()
    intImage.dataType='FLOAT'
    intImage.load(f + '.xml')
    unw = intImage.memMap()[:,:,0]
    unw = unw.copy()
    unw[gam==0] = np.nan
    return unw

def getConCom(msk, minimumPixelsInRegion=1000):
    '''
    Takes a binary input (like a mask) as input and outputs labels for
    regions greater than the given minimum pixels.
    '''
    
    import cv2
    ratesu8 = (msk*255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(ratesu8)
    
    npix = []
    for ii in range(num_labels):
        npix.append(len(np.where(labels==ii)[0]))
    npix = np.asarray(npix)
    # concom = np.where(npix>minimumPixelsInRegion)[0]
    
    newLabels = np.zeros(msk.shape)
    
    for ii in range(len(npix)):
        lab = npix[ii]
        newLabels[labels==lab] = ii
        
    return labels



def coregister(img1,img2):
    """
    Coregister two images
    
    inputs
        img1: reference image you want to align img2 to
        img2: image you want to be aligned 
    outputs:
        img2_coreg: the coregistered version of img2
        H: the homography matrix
        
    Based on this tutorial:
        https://www.sicara.fr/blog/2019-07-16-image-registration-deep-learning
    
    The KAZE algorithm is written up here (AKAZE is a faster version of that):
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.304.4980&rep=rep1&type=pdf
        
    Written: 4/20/2022 @ 4:20:69

    """
    
    from matplotlib import pyplot as plt
    import cv2 as cv
    
    # img1 = cv.imread('/home/km/Pictures/Murray_175px.jpg', cv.IMREAD_GRAYSCALE)  # referenceImage
    # img2 = cv.imread('/home/km/Pictures/Murray2px.jpg', cv.IMREAD_GRAYSCALE)  # sensedImage
    
    img1_8 = cv.normalize(src=img1, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    img2_8 = cv.normalize(src=img2, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    
    
    # Initiate AKAZE detector
    akaze = cv.AKAZE_create()
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = akaze.detectAndCompute(img1_8, None)
    kp2, des2 = akaze.detectAndCompute(img2_8, None)
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
            
    # Draw matches
    img3 = cv.drawMatchesKnn(img1_8,kp1,img2_8,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('matches.jpg', img3)
    
    # Select good matched keypoints
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
    
    # Compute homography
    H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC,5.0) 
    
    # Warp image
    # warped_image = cv.warpPerspective(img2_8, H, (img2_8.shape[1], img2_8.shape[0]))
    img2_coreg = cv.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    
    fig,ax = plt.subplots(2,2)
    ax[0,0].imshow(img1);ax[0,0].set_title('img1')
    ax[0,1].imshow(img2);ax[0,1].set_title('img2')
    ax[1,0].imshow(img2_coreg);ax[1,0].set_title('img2_coreg')
    ax[1,1].imshow(img1-img2_coreg,vmin=-100,vmax=100,cmap='RdBu_r');ax[1,1].set_title('img1 - img2_coreg')
    
    meanRes = np.nanmean(abs(img1-img2_coreg))
    print('Mean Residual: ' + str(meanRes))

    return img2_coreg, H
    
def show(img,title=None,cmap='magma'):
    """
    just plots the image so you don't have to type as much. For quickly viewing.
    """
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(img,cmap=cmap)
    plt.show()
    if title:
        plt.title(title)


def gaussian_kernel(Sx, Sy, sig_x, sig_y):
    if np.mod(Sx,2) == 0:
        Sx = Sx + 1

    if np.mod(Sy,2) ==0:
            Sy = Sy + 1

    x,y = np.meshgrid(np.arange(Sx),np.arange(Sy))
    x = x + 1
    y = y + 1
    x0 = (Sx+1)/2
    y0 = (Sy+1)/2
    fx = ((x-x0)**2.)/(2.*sig_x**2.)
    fy = ((y-y0)**2.)/(2.*sig_y**2.)
    k = np.exp(-1.0*(fx+fy))
    a = 1./np.sum(k)
    k = a*k
    return k

def convolve(data, kernel):
    import cv2
    R = cv2.filter2D(data.real,-1,kernel)
    Im = cv2.filter2D(data.imag,-1,kernel)

    return R + 1J*Im


def butter(img,wavelength,nyq_freq=0.5,order=2):
    from scipy import signal
    
    #wavelength = 150  #Get rid of stuff happening over these spatial scales or longer
    #nyq_freq = .5 # 1 sample/pixel /2
    #order = 2
    cutoff_frequency = 1/(wavelength*2)
    
    def butterLow(cutoff, critical, order):
        normal_cutoff = float(cutoff) / critical
        b, a = signal.butter(order, normal_cutoff, btype='lowpass')
        return b, a
    
    def butterFilter(data, cutoff_freq, nyq_freq, order):
        b, a = butterLow(cutoff_freq, nyq_freq, order)
        y = signal.filtfilt(b, a, data)
        return y
    
    filt = butterFilter(img, cutoff_frequency, nyq_freq, order)
    
    return filt

def write_xml(filename,width,length,bands,dataType,scheme):
    import isceobj
    img=isceobj.createImage()
    img.setFilename(filename)
    img.setWidth(width)
    img.setLength(length)
    img.setAccessMode('Read')
    img.bands=bands
    img.dataType=dataType
    img.scheme = scheme
    img.renderHdr()
    img.renderVRT()
    return

def writeISCEimg(img,outName,nBands,width,length,dtype):
    '''
    quick way to write a file with an xml file. Automatically checks datatype
    img: image to write to file with an xml
    outname: name of output file not including the .xml extension
    dtype: 'Float' or 'Complex'
    ''' 
    import isceobj
    fidc=open(outName,"wb")
    fidc.write(img)
    #write out an xml file for it
    out = isceobj.createIntImage() # Copy the interferogram image from before
    
    if dtype=='Float':
        img = np.asarray(img,dtype=np.float32)
        out.dataType = 'FLOAT'
    elif dtype=='Complex':
        img = np.asarray(img,dtype=np.complex64)
        out.dataType = 'CFLOAT'
    
    out.bands = 1
    out.filename = outName
    out.width = width
    out.length = length
    out.dump(outName + '.xml') # Write out xml
    out.renderHdr()
    out.renderVRT()


def filtAndCoherence(infileIFG,filtFileOut,corFileOut,filterStrength):
    '''
    Runs filtering and coherence estimation
    '''
    
    import FilterAndCoherence as fc
    if filterStrength <= 0:
        print('Skipping filtering because filterStrength is 0')
    else:
        fc.runFilter(infileIFG, filtFileOut, filterStrength)    
    
    fc.estCoherence(filtFileOut, corFileOut)

def unwrap_snaphu(intfile,corfile,unwfile,length, width,rlks,alks):
    '''
    Inputs:
        intfile
        corfile
        length,width: length and width of intfile
        rlks,alks: give rlks and alks just to record it in the xml file
    Outputs:
        unwfile: writes unw image to this file

    '''
    
    from contrib.Snaphu.Snaphu import Snaphu

    altitude = 800000.0
    earthRadius = 6371000.0
    wavelength = 0.056
    defomax = 4.0
    maxComponents = 20
    
    snp = Snaphu()
    snp.setInitOnly(False)
    snp.setInput(intfile)
    snp.setOutput(unwfile)
    snp.setWidth(width)
    snp.setCostMode('SMOOTH')
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(corfile)
    snp.setInitMethod('MST')
    snp.dumpConnectedComponents(True)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(rlks)
    snp.setAzimuthLooks(alks)
    snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()
    write_xml(unwfile, width, length, 2 , "FLOAT",'BIL')
    return

def getWaterMask(DEMfilename, lon_ifg, lat_ifg, outputfilename):
    import createWaterMask as wm
    bbox = wm.dem2bbox(DEMfilename)
    geo_file = wm.download_waterMask(bbox, DEMfilename, fill_value=-1)
    # wm.geo2radar(geo_file, outputfilename, lat_ifg, lon_ifg)
    return geo_file
    
def tsFilt(alld, dec_year, N=5, desiredPeriod = 1):
    '''
    Temporal filter
    Inputs:
        alld: len(time) X n X m matrix
        N: Filter order
        desiredPeriod: roughly the cutoff period in years. (Anything shorter 
            than this value will be filtered out).
    Output:
        alldFilt
   
    Wn is the Cutoff frequency between 0 and 1.  0 is infinitely smooth and 1 is the original. 
        this is the frequency multiplied by the nyquist rate. 
        if we have 25 samples per year, then the nyquist rate would be ~12hz. So if we make Wn=.5
        we will have filtered to 6hz (letting signals with wavelengths of 2 months or longer).
        If we make wn=1/12 then we will filter to 1hz (letting only signals with wavelengths of 1 year).
    '''
    import scipy.signal as signal

    dec_year = np.asarray(dec_year)
    samplesPerYear = len(dec_year) / (dec_year.max()-dec_year.min())
    nyquistRate = samplesPerYear/2 #this is the highest freq we can resolve with our sampling rate
    Wn = 1/(desiredPeriod * nyquistRate)
    B, A = signal.butter(N, Wn, output='ba')
    
    alldFilt = signal.filtfilt(B,A, alld,axis=0)
    alldFilt[alldFilt==0]=np.nan
    
    return alldFilt

def tsFilt1d(ts, dec_year, N=5, desiredPeriod = 1):
    '''
    Temporal filter
    Inputs:
        ts: 1d time series corresponding to dec_year dates
        N: Filter order
        desiredPeriod: roughly the cutoff period in years. (Anything shorter 
            than this value will be filtered out).
    Output:
        alldFilt
   
    Wn is the Cutoff frequency between 0 and 1.  0 is infinitely smooth and 1 is the original. 
        this is the frequency multiplied by the nyquist rate. 
        if we have 25 samples per year, then the nyquist rate would be ~12hz. So if we make Wn=.5
        we will have filtered to 6hz (letting signals with wavelengths of 2 months or longer).
        If we make wn=1/12 then we will filter to 1hz (letting only signals with wavelengths of 1 year).
    '''
    import scipy.signal as signal

    dec_year = np.asarray(dec_year)
    samplesPerYear = len(dec_year) / (dec_year.max()-dec_year.min())
    nyquistRate = samplesPerYear/2 #this is the highest freq we can resolve with our sampling rate
    Wn = 1/(desiredPeriod * nyquistRate)
    B, A = signal.butter(N, Wn, output='ba')
    
    tsF = signal.filtfilt(B,A, ts,axis=0)
    tsF[tsF==0]=np.nan
    
    return tsF


def getLOSvec(psi,theta):
    '''
    Given the flight azimuth and the los azimuth angle, return LOS vector
    theta should be the angle from ground normal
    psi should be the angle from EAST (X) of the flight direction angle. (not the look direction)
    
    In ISCE:
    psi is from the second band in the los.rdr file (and also in the az_lk.rdr file from PyPS)
    theta is from the second band of incLocal.rdr file (which is the angle from the surface normal vector)
    
    '''
    psi+=90 # Add 90 degrees to get the look direction from the flight direction
    losA=np.zeros((3,))
    losA[0] =  np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(psi))
    losA[1] =  np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(psi))
    losA[2] =  np.cos(np.deg2rad(theta))
    return losA
    
    
def invertVertHor(asc,des,psi_a,theta_a,psi_d,theta_d,smooth):
    '''
    damped least squares inversion of ascending/descending LOS data
       returns horizontal (East-west) and Vertical displacements
    IN:
        asc: value of ascending ifg at a pixel
        des: value of descending ifg at a pixel
        psi_a/d: azimuth direction of asc/des 
        theta_a/d: incidence angle of asc/des
        smooth: damping factor 
    OUT:
        vertHor: east-west and vertical deformation 
    '''
    
    losA = getLOSvec(psi_a,theta_a)
    losD = getLOSvec(psi_d,theta_d)

    o = np.array([asc,des]).T
    o = np.concatenate((o,np.array([0,0])),axis=0)
    # Define unit basis vectors
    mx = np.array([1,0,0])
    my = np.array([0,1,0])
    mz = np.array([0,0,1])
    
    A = np.array([[np.dot(losA,mx), np.dot(losA,mz)],
                  [np.dot(losD,mx), np.dot(losD,mz)],
                  [smooth,          0              ],
                  [0,               smooth         ]])
    
    # A = np.array([[np.dot(losA,mx), np.dot(losA,my), np.dot(losA,mz)],
    #               [np.dot(losD,mx),np.dot(losD,my),np.dot(losD,mz)],
    #               [smooth,              0,         0              ],
    #               [0,              smooth,          0              ],
    #               [0,              0,              smooth         ]])
    
    Aa = np.dot( np.linalg.inv(np.dot(A.T,A)), A.T)
    vertHor = np.dot(Aa,o)
    return vertHor

def geocode(filename):
    ''' Geocodes the filename and outputs geocoded image in that directory 
        with .geo. I think filename can be a list of multiple names.
    '''
    
    import geocodeIsce
    import glob
    # setupParams = np.load('setupParams.npy',allow_pickle=True).item()
    
    ps = np.load('./ps.npy',allow_pickle=True).all()
    
    minlon = ps.lon_ifg.min()
    maxlon = ps.lon_ifg.max()
    minlat = ps.lat_ifg.min()
    maxlat = ps.lat_ifg.max()
    dem = glob.glob('./DEM/*wgs84')[0]
    bbox1 = [minlat,maxlat, minlon,maxlon]
    
    class inpsArgs():
        prodlist = filename
        bbox = bbox1
        demfilename = dem
        reference = ps.workdir + '/reference'
        secondary = ps.workdir + '/reference'
        numberRangeLooks = ps.rlks
        numberAzimuthLooks = ps.alks
        
    geocodeIsce.runGeocode(inpsArgs, inpsArgs.prodlist, inpsArgs.bbox, inpsArgs.demfilename, is_offset_mode=False)

def geocodeKM(img,method='linear'):
    
    '''
    This is actually a geocode hack. but better to use the geocode function above
    '''
    
    from scipy.interpolate import griddata 

    ps = np.load('./ps.npy',allow_pickle=True).all()
    
    minlon = ps.lon_ifg.min()
    maxlon = ps.lon_ifg.max()
    minlat = ps.lat_ifg.min()
    maxlat = ps.lat_ifg.max()
    
    ny = int(ps.nyl*2)
    nx = int(ny*1.3)

    xx = np.linspace(minlon,maxlon,nx)
    yy = np.linspace(minlat,maxlat,ny)
    XX,YY = np.meshgrid(xx,yy)
    imgRegrid = griddata((ps.lon_ifg.ravel(),ps.lat_ifg.ravel()), img.ravel(), (XX,YY), method=method)
    imgRegrid = np.flipud(imgRegrid)
    return imgRegrid

def orderAxes(inputArray,nx,ny):
    '''  Rearrange axes order from small to big '''
    imShape = np.asarray(inputArray.shape)
    smaA = np.where(imShape==imShape.min())[0][0]
    inputArray = np.moveaxis(inputArray,smaA,0)
    imShape = np.asarray(inputArray.shape)
    bigA = np.where(imShape==imShape.max())[0][0]
    if nx>ny:
        inputArray = np.moveaxis(inputArray,bigA,2)
    else:
        inputArray = np.moveaxis(inputArray,bigA,1)
    return inputArray