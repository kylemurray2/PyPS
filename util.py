#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:54:12 2020

@author: kdm95
"""
import numpy as np
# from astropy.convolution import Gaussian2DKernel
# from astropy.convolution import convolve


def getTime(path,frame):
    '''
     Figure out what time the aquisition was
    '''
    
    # path = os.getcwd().split('/')[-2]
    # frame= os.getcwd().split('/')[-1]

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
    
    Lon = pd.read_csv('out.csv')["Near Start Lon"][0]
    Lat = pd.read_csv('out.csv')["Near Start Lat"][0]
    return int(hour),int(minute), Lon, Lat
   

def reGrid(inputImageStack,outputX,outputY):
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
    zi = z[y.astype(np.int), x.astype(np.int)]
    return zi


def ll2pixel(lon_mat,lat_mat,lon_pts,lat_pts):
    """
    Output the pixels (radar coords) given the lat/lon matrices and arrays of 
    lat/lon points.
    """
    x_pts = list()
    y_pts = list()
    for ii in np.arange(0,len(lat_pts)):
        a = abs(lat_mat-lat_pts[ii])
        b = abs(lon_mat-lon_pts[ii])
        c = a+b
        y,x = np.where(c==c.min()) # y is rows, x is columns
        x_pts.append(x)
        y_pts.append(y)
    return x_pts,y_pts 

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


def px2ll(x, y, lon_ifgm,lat_ifgm):
    lon = lon_ifgm[y,x]
    lat = lat_ifgm[y,x]
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