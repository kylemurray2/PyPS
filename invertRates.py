#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:45:20 2018

@author: kdm95
"""

import numpy as np
import isceobj
from matplotlib import pyplot as plt
from skimage.measure import profile_line as pl
import fitSine

#********************************
# Set these paramaters
#********************************
#seasonal = False
#mcov_flag = True
#water_elevation = -103
#********************************

def invertRates(data,dn,seasonals=False,mcov_flag=False,water_elevation=-103,uncertainties=False):
    ''' data is a  stack of inverted displacements shape=[ny,nx]
        you can invert for a sinusoidal fit at each pixel with seasonals = True
        mcov_flag is the model covariance and works only with seasonals=False for now.
        Water elevation is usually not zero (relative to wgs84 ellipsoid).'''

    ps = np.load('./ps.npy',allow_pickle=True).all()
    
    data = data.astype(np.float32).reshape((len(dn),-1))     
    dn0 = dn -dn[0]
    period = 365.25
    rate_uncertainty = []

    if seasonals:
        if mcov_flag:
            mcov_flag=False
            print('model covariance only works with seasonals=False for now')
            
        # Invert for seasonal plus long term rates
        phases,amplitudes,biases,slopes = fitSine.fitSine(dn0,data,period)
        rates = np.reshape(slopes,(ps.nyl,ps.nxl)).astype(np.float32)*365
        amps = np.reshape(amplitudes,(ps.nyl,ps.nxl)).astype(np.float32)
        return rates, amps
    
    elif uncertainties:
        G = np.vstack([dn0, np.ones((len(dn0),1)).flatten()]).T
        mod = []
        rates = []
        resstd = []
        for ii in range(data.shape[1]):
            W = np.diag(1/uncertainties[:,ii])
            Gw = np.dot(W,G)
            dw = np.dot(W,data[:,ii])
            mod.append(np.dot( np.linalg.inv(np.dot(Gw.T,Gw)), np.dot(Gw.T,dw)))
            rates.append( mod[ii][0] *365 ) # cm/yr
            #offs  = np.reshape(mod[1,:],(ps.nyl, ps.nxl))
            synth  = np.dot(G,mod[ii]);
            res    = (data[:,ii]-synth)
            rs = np.std(res,axis=0)
            resstd.append(rs)
        
        rates = np.reshape(np.asarray(rates),(ps.nyl,ps.nxl))
        resstd = np.reshape(np.asarray(resstd),(ps.nyl,ps.nxl))
        
        return rates,resstd

    else:
        G = np.vstack([dn0, np.ones((len(dn0),1)).flatten()]).T
        Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
        mod   = np.dot(Gg, data)
        rates = np.reshape(mod[0,:],(ps.nyl,ps.nxl))*365 # cm/yr
        #offs  = np.reshape(mod[1,:],(ps.nyl, ps.nxl))
        synth  = np.dot(G,mod);
        res    = (data-synth)#*lam/(4*np.pi)*100 # cm


        resstd = np.std(res,axis=0)
        resstd = np.reshape(resstd,(ps.nyl, ps.nxl))
        
        if mcov_flag:
            for ii in np.arange(0,len(data[0,:])):
                co=np.cov(data[:,ii]);
                mcov=np.diag(np.dot(Gg,np.dot(co,Gg.T)));
                rate_uncertainty.append(1.96*mcov[0]**.5)
            rate_uncertainty = np.asarray(rate_uncertainty,dtype=np.float32)
            rate_uncertainty = np.reshape(rate_uncertainty,(ps.nyl,ps.nxl))
            rate_uncertainty= rate_uncertainty*365 #cm/yr

        return rates,resstd#,worst,worstVal



## Save rates
#fname = tsdir + '/rates_flat.unw'
#out = isceobj.createIntImage() # Copy the interferogram image from before
#out.dataType = 'FLOAT'
#out.filename = fname
#out.width = ps.nxl
#out.length = ps.nyl
#out.dump(fname + '.xml') # Write out xml
#rates.tofile(out.filename) # Write file out
#out.renderHdr()
#out.renderVRT()
#
## GEOCODE
##cmd = 'geocodeIsce.py -f ' + tsdir + 'rates_flat.unw -d ' + workdir + 'DEM/demLat_N33_N35_Lon_W119_W116.dem -m ' + workdir + 'master/ -s ' + workdir + 'pawns/20150514 -r ' + str(rlks) + ' -a ' + str(alks) + ' -b "'" 33 35 -118 -116"'" '
##os.system(cmd)

