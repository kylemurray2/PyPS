#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:24:42 2021

@author: km
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:33:24 2020

@author: km
"""

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
plt.close('all')
params = np.load('params.npy',allow_pickle=True).item()
geom = np.load('geom.npy',allow_pickle=True).item()
msk = np.load('msk.npy')

locals().update(params)
locals().update(geom)

nxl = params['nxl']
nyl  = params['nyl']

# MASKING______________________________
gam = np.load('gam.npy')
gamFlat = gam.flatten()

plt.imshow(gam)
r,c = 1330,530 #goma
X,Y = np.meshgrid(range(nxl),range(nyl))

stack = []
for p in params['pairs']:
    unw_file = params['intdir'] + '/' + p + '/filt.unw'
    unwImage = isceobj.createIntImage()
    unwImage.load(unw_file + '.xml')
    unw = unwImage.memMap()[:,:,0] - np.nanmean(unwImage.memMap()[:,:,0][np.where(msk==1)])
    # unw = unwImage.memMap()[:,:,0] - unwImage.memMap()[r,c,0]
    stack.append(unw)
# stack.append(np.zeros(unw.shape))
stack = np.asarray(stack,dtype=np.float32)

stackTimeSum = np.nansum(stack,axis=0)
plt.figure();plt.imshow(stackTimeSum)
stackTimeMean = np.nanmean(stack,axis=0)

# SBAS Inversion to get displacement at each date
## Make G matrix for dates inversion
G = np.zeros((len(params['dates']),len(params['dates'])))
for ii,pair in enumerate(params['pairs']):
    a = params['dates'].index(pair[0:8])
    b = params['dates'].index(pair[9:17])
    G[ii,a] = 1
    G[ii,b] = -1
G[-1,0]=1

Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)

# Do dates inversion
alld=np.zeros((len(dec_year),nxl*nyl))
for ii in np.arange(0,nyl-1): #iterate through rows
    tmp = np.zeros((len(dec_year),nxl))
    for jj,pair in enumerate(params['pairs']): #loop through each ifg and append to alld 
        tmp[jj,:] = stack[jj,ii,:]
    alld[:,ii*nxl:nxl*ii+nxl] = np.dot(Gg, tmp)
del(tmp)  
    
alld = np.reshape(alld,(len(params['dates']),nyl,nxl))  
plt.figure();plt.plot(dec_year,  alld[:,300,300],'.');plt.ylabel('cm'); plt.title('Goma') #Goma


maxElev = 5000

# # CONVERT TO CM 
alld=alld*lam/(4*np.pi)*100
plt.figure();plt.imshow(msk);plt.title('Mask')

stacksum = np.nansum(stack,axis=0)

rates,resstd = invertRates.invertRates(alld,params,params['dn'], seasonals=False,mcov_flag=False,water_elevation=seaLevel)
rates = np.asarray(rates,dtype=np.float32)
resstd = np.asarray(resstd,dtype=np.float32)
rates[hgt_ifg<seaLevel] = np.nan
resstd[hgt_ifg<seaLevel] = np.nan

# gamthresh = .5
rates[msk == 0 ]=np.nan
plt.figure();plt.imshow(rates,vmin=-4,vmax=4)
stacksum[msk == 0 ]=np.nan
plt.figure();plt.imshow(stacksum,vmin=-40,vmax=40)

# resstd[gam < gamthresh]=np.nan
plt.figure();plt.plot(dec_year,  alld[:,300,300],'.');plt.ylabel('cm'); plt.title('Goma') #Goma
# plt.figure();plt.plot(dec_year, alld[:,931,1347],'.');plt.ylabel('cm'); plt.title('Kigali') # 
# plt.figure();plt.plot(dec_year, alld[:,723,72],'.');plt.ylabel('cm'); plt.title('Bakavu') # 
# plt.figure();plt.plot(dec_year, alld[:,204,794],'.');plt.ylabel('cm'); plt.title('Ngozi') # 
# plt.figure();plt.plot(dec_year, alld[:,1568,528],'.');plt.ylabel('cm'); plt.title('Volcano') # 
# plt.figure();plt.plot(dec_year, alld[:,873,812],'.');plt.ylabel('cm'); plt.title('middle random') # 
# plt.figure();plt.plot(dec_year, alld[:,389,233],'.');plt.ylabel('cm'); plt.title('Rugombo') # 
# plt.figure();plt.plot(dec_year, alld[:,84,1295],'.');plt.ylabel('cm'); plt.title('upper right') # 

# plt.figure();plt.plot(dec_year,  alld[:,1170,485],'.');plt.ylabel('cm'); plt.title('Lake') # Lake
# plt.figure();plt.plot(dec_year,  alld[:,800,801],'.');plt.ylabel('cm'); plt.title('middle') # middle

# plt.figure();plt.plot(dec_year,  alld[:,1230,525],'.');plt.ylabel('cm'); plt.title('Goma')

# plt.figure();plt.plot(dec_year,  alld[:,1350,708],'.');plt.ylabel('cm'); plt.title('mikeno') #mikeno
# plt.figure();plt.plot(dec_year,  alld[:,1337,585],'.');plt.ylabel('cm'); plt.title('nyiranango') #nyiranango
# plt.figure();plt.plot(dec_year,  alld[:,1422,558],'.');plt.ylabel('cm'); plt.title('nyamuragira') #nyamuragira


# ts = alld[:,2550:2650,40:60]
# ts = np.nanmean(ts.reshape(124,-1),axis=1)
# plt.figure();plt.plot(dec_year,  ts,'.');plt.ylabel('cm')
# plt.figure();plt.plot(dec_year,  alld[:,2601+5,52+5],'.');plt.ylabel('cm')

# xs = np.array([841,750,750])
# ys = np.array([1283,750,500])

# fig,ax = plt.subplots(2,1)
# ax[0].plot(dec_year, alld[:,ys[0],xs[0]]);plt.ylabel('cm')
# ax[0].plot(dec_year, alld[:,ys[1],xs[1]]);plt.ylabel('cm')
# ax[0].plot(dec_year, alld[:,ys[2],xs[2]]);plt.ylabel('cm')
# ax[0].plot(dec_year, alld[:,ys[3],xs[3]]);plt.ylabel('cm')
# ax[0].plot(dec_year, alld[:,ys[4],xs[4]]);plt.ylabel('cm')
# ax[0].plot(dec_year, alld[:,ys[5],xs[5]]);plt.ylabel('cm')
# ax[0].plot(dec_year, alld[:,ys[6],xs[6]]);plt.ylabel('cm')
# ax[0].plot(dec_year, alld[:,ys[7],xs[7]],'.');plt.ylabel('cm')
# ax[0].set_ylabel('Displacement (cm)')
# ax[0].legend(['1','2','3','4','5','6'])
# ax[1].plot(dec_year,corMeans);ax[1].set_xlabel('Years');ax[1].set_ylabel('Coherence')

# plt.figure()
# plt.imshow(rates,vmin=-5,vmax=5)
# for jj in range(len(xs)):
#     plt.scatter(xs[jj],ys[jj],s=2,color='red')
#     plt.text(xs[jj],ys[jj],str(jj+1),fontsize=10)

# rates[msk==0]=np.nan
# stacksum[msk==0]=np.nan
# stacksum-=np.nanmean(stacksum)
# Plot rate map

vmin=-3
vmax=3
pad=0
makeMap.mapImg(rates,lon_ifg,lat_ifg,vmin,vmax,pad,8,'rates (cm)')

# plt.figure();plt.plot(np.ravel(rates)[::10],np.ravel(np.nanmean(corStack,axis=0))[::10],'.',markersize = 1)

# makeMap.mapImg(resstd,lon_ifg,lat_ifg,0,8,pad, 'res std (cm)')
# makeMap.mapImg(gam,lon_ifg,lat_ifg,.4,.9,pad,'gamma0')


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