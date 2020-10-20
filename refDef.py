#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:35:54 2018

@author: kdm95
"""
import numpy as np
import isceobj
from matplotlib import pyplot as plt
import invertRates 
import scipy.signal as signal
import makeMap

params = np.load('params.npy',allow_pickle=True).item()
geom = np.load('geom.npy',allow_pickle=True).item()
locals().update(params)
locals().update(geom)

removePlane = True
maxElev = 400

nxl = params['nxl']
nyl  = params['ymax']-params['ymin']

hgt_ifg = hgt_ifg[ymin:ymax,:]
lon_ifg = lon_ifg[ymin:ymax,:]
lat_ifg = lat_ifg[ymin:ymax,:]
# MASKING______________________________
gam = np.load('gam.npy')[ymin:ymax,xmin:xmax]
gamFlat = gam.flatten()

plt.imshow(gam)
r,c = 896,760
X,Y = np.meshgrid(range(nxl),range(nyl))

stack = []
#start = 35
#end = 77
##params['decYearCut'] = params['dec_year'][start:end-1]
##np.save('params.npy',params)
#params['pairs'] = params['pairs'][start:end-1]
#params['dates'] = params['dates'][start:end]
#params['dn']= params['dn'][start:end]

for p in params['pairs']:
    unw_file = params['intdir'] + '/' + p + '/fine_lk.unw'
    unwImage = isceobj.createIntImage()
    unwImage.load(unw_file + '.xml')
    unw = unwImage.memMap()[ymin:ymax,xmin:xmax,0] - unwImage.memMap()[r,c,0]
    stack.append(unw)
stack = np.asarray(stack,dtype=np.float32)

plt.imshow(stack[100,:,:])
# SBAS Inversion to get displacement at each date
## Make G matrix for dates inversion
G = np.zeros((len(params['dates']),len(params['dates'])))
for ii,pair in enumerate(params['pairs']):
    a = params['dates'].index(pair[0:8])
    b = params['dates'].index(pair[9:17])
    G[ii,a] = -1
    G[ii,b] = 1
G[-1,0]=1

Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
N = np.dot(G,Gg)
R = np.dot(Gg,G)

# Do dates inversion
alld=np.zeros((len(params['dates']),nxl*nyl))
for ii in np.arange(0,nyl-1): #iterate through rows
    tmp = np.zeros((len(params['dates']),nxl))
    for jj,pair in enumerate(params['pairs']): #loop through each ifg and append to alld 
        tmp[jj,:] = stack[jj,ii,:]
    alld[:,ii*nxl:nxl*ii+nxl] = np.dot(Gg, tmp)
del(tmp)  

alldPlane = []
if removePlane:
    for ii in range(alld.shape[0]):
        G  = np.array([np.ones((len(X.flatten()),)), X.flatten(), Y.flatten()]).T
        Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
        mod   = np.dot(Gg,alld[ii,:])
        synth = mod[0] + mod[1] * X.flatten() + mod[2] * Y.flatten()
        alldPlane.append(alld[ii,:] - synth)  
    
    alldPlane = np.asarray(alldPlane,dtype= np.float32)     
    del(alld)
    alld = alldPlane
    
alld = np.reshape(alld,(len(params['dates']),nyl,nxl))  

# First, design the Buterworth filter
N  = 5    # Filter order
''' Wn is the Cutoff frequency between 0 and 1.  0 is infinitely smooth and 1 is the original. 
    this is the frequency multiplied by the nyquist rate. 
    if we have 25 samples per year, then the nyquist rate would be ~12hz. So if we make Wn=.5
    we will have filtered to 6hz (letting signals with wavelengths of 2 months or longer).
    If we make wn=1/12 then we will filter to 1hz (letting only signals with wavelengths of 1 year).
'''
dec_year = np.asarray(dec_year)
samplesPerYear = len(dn) / (dec_year.max()-dec_year.min())
nyquistRate = samplesPerYear/2 #this is the highest freq we can resolve with our sampling rate
desiredPeriod = 1 # signals with a period of 1 year or longer
Wn = 1/(desiredPeriod * nyquistRate)
B, A = signal.butter(N, Wn, output='ba')

alld_filt = signal.filtfilt(B,A, alld,axis=0)
alld_filt[alld_filt==0]=np.nan
alld[alld==0]=np.nan

plt.figure()
plt.plot(dec_year,alld[:, 215,215],'.')
plt.plot(dec_year,alld_filt[:, 215,215])
plt.title('Example time series and filtered time series for stdimg estimation')

std_img = np.nanstd(alld_filt,axis=0) # Temporal std
std_img = np.reshape(std_img,(nyl,nxl))
std_img[std_img==0]=np.nan

std_thresh = np.nanmedian(std_img)
gamma0_thresh =  np.nanmedian(gamFlat)

fig,ax = plt.subplots(2,1)
ax[0].hist(std_img[~np.isnan(std_img)], 40, edgecolor='black', linewidth=.2)
ax[0].axvline(std_thresh,color='red')
ax[0].set_title('Non Deforming Region Metric')
ax[0].set_xlabel('Large values may be deforming')    
ax[1].hist( gamFlat[~np.isnan(gamFlat)], 40, edgecolor='black', linewidth=.2)
ax[1].axvline(gamma0_thresh,color='red')
ax[1].set_title('Phase stability histogram')
ax[1].set_xlabel('Phase stability (1 is good, 0 is bad)')
msk = np.zeros(gamFlat.shape)



msk[np.where( (std_img.flatten() < std_thresh) & (gamFlat > gamma0_thresh) & (hgt_ifg.flatten() > seaLevel) & (hgt_ifg.flatten() < maxElev))] = 1
msk = np.reshape(msk,std_img.shape)
plt.figure();plt.imshow(msk);plt.title('Mask')

  
alld_flat=np.empty(alld.shape)
for ii in np.arange(0,len(alld[:,0])):
    a = alld[ii,:,:]
    alld_flat[ii,:,:] = alld[ii,:,:] - np.nanmedian(a[msk==1])

varMap = np.var(alld,axis=0)  
varMapFlat = np.var(alld_flat,axis=0)

fig,ax = plt.subplots(2,1)
ax[0].imshow(varMap,vmin=2,vmax=25)
ax[1].imshow(varMapFlat,vmin=2,vmax=25)

  
columnProf = 1250 # plt a profile along this column. 40 goes through part of MX city
plt.figure()
for ii in np.arange(0,len(alld[:,0]),1):
    plt.plot(np.reshape(alld[ii,:],(nyl,nxl))[:,columnProf]) 
    plt.title('not flattened')

plt.figure()
for ii in np.arange(0,len(alld[:,0]),1):
    plt.plot(np.reshape(alld_flat[ii,:],(nyl,nxl))[:,columnProf]) 
    plt.title('flattened')

data = -alld_flat.astype(np.float32).reshape((nd+1,-1)) # Make subsidence negative
np.save('alld.npy', data)

rates,resstd = invertRates.invertRates(data,params,params['dn'], seasonals=False,mcov_flag=False,water_elevation=seaLevel)
np.save('rates.npy',rates)


rate_dict = {}
rate_dict['rates'] = rates
rate_dict['resstd'] = resstd

rates = rate_dict['rates']
resstd = rate_dict['resstd']

rates = np.asarray(rates,dtype=np.float32)
resstd = np.asarray(resstd,dtype=np.float32)

rates[hgt_ifg<seaLevel] = np.nan
resstd[hgt_ifg<seaLevel] = np.nan

gamthresh = 0
rates[gam < gamthresh]=np.nan
resstd[gam < gamthresh]=np.nan


minlat=lat_ifg.min()
maxlat=lat_ifg.max()
minlon=lon_ifg.min()
maxlon=lon_ifg.max()

# Plot rate map
vmin=-2.2
vmax=2
pad=.5



makeMap.makeImg(rates,lon_ifg,lat_ifg,vmin,vmax,pad,'rates (cm)')
makeMap.makeImg(resstd,lon_ifg,lat_ifg,0,8,pad, 'res std (cm)')
makeMap.makeImg(gam,lon_ifg,lat_ifg,.4,.9,pad,'gamma0')


#    axrates.scatter(c,r,14,color='white')
#    axrates.text(c,r,str(ii+1),color='white')
## Save rates
#fname = tsdir + '/rates_flat.unw'
#out = isceobj.createIntImage() # Copy the interferogram image from before
#out.dataType = 'FLOAT'
#out.filename = fname
#out.width = nxl
#out.length = nyl
#out.dump(fname + '.xml') # Write out xml
#rates.tofile(out.filename) # Write file out
#out.renderHdr()
#out.renderVRT()