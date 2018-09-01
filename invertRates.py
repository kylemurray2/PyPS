#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:45:20 2018

@author: kdm95
"""

import numpy as np
import isceobj
import pickle
import os
from datetime import date
from matplotlib import pyplot as plt

#with open(tsdir + '/params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#    pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks = pickle.load(f)

# Run refDef.py before this to make the alld_flat.pkl file
#with open(tsdir + '/alld_flat.pkl','rb')as f:
#    alld_flat = pickle.load(f)
# Convert pairs to dates
dn = list()
dn.append( date.toordinal(date(int(pairs[0][9:13]), int(pairs[0][13:15]), int(pairs[0][15:]))) )
for pair in pairs:
    yr = pair[9:13]
    mo = pair[13:15]
    day = pair[15:]
    dn.append(date.toordinal(date(int(yr), int(mo), int(day))))

dn = np.asarray(dn)
dn0 = dn-dn[0] # make relative to first date
# Example pixel for check
r=1508
c=5273
r=1682
c=1520



G = np.vstack([dn0, np.ones((len(dn0),1)).flatten()]).T
Gg = np.dot( np.linalg.inv(np.dot(G.T,G)), G.T)
mod    = np.dot(Gg, alld_flat_topo)
rates   = np.reshape(mod[0,:],(nyl, nxl))*lam/(4*np.pi)*100*365 # cm/yr
rates = rates.astype(np.float32)
#offs  = np.reshape(mod[1,:],(nyl, nxl))
#synth  = np.dot(G,mod);
#res    = (alld_flat_topo-synth)*lam/(4*np.pi)*100 # cm
#resstd = np.std(res,axis=0)
#resstd = np.reshape(resstd,(nyl, nxl))


# MASKING______________________________
# Load gamma0_lk
f = tsdir + 'gamma0_lk.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0_lk= intImage.memMap()[:,:,0] 
# Load height file
h = workdir + 'merged/geom_master/hgt_lk.rdr'
hImg = isceobj.createImage()
hImg.load(h + '.xml')
hgt = hImg.memMap()[:,:,0].astype(np.float32)
# elevations at 4 of the main lakes that we'll mask out

# Mask the rates matrix
#rates[np.where(gamma0_lk<.3)]=np.nan #masks the low coherence areas
rates[np.where( (hgt<.1) ) ]=np.nan # masks the lakes
rates=np.fliplr(rates)
#rates[np.isnan(rates)]=0
#rates=np.flipud(rates)
#plt.imshow(rates)
#plt.imshow(offs)

# Save rates
fname = tsdir + 'rates_flat.unw'
out = isceobj.createIntImage() # Copy the interferogram image from before
out.dataType = 'FLOAT'
out.filename = fname
out.width = nxl
out.length = nyl
#out.bands=1
out.dump(fname + '.xml') # Write out xml
rates.tofile(out.filename) # Write file out
out.renderHdr()
out.renderVRT()

# GEOCODE
cmd = 'geocodeIsce.py -f ' + tsdir + 'rates_flat.unw -d ' + workdir + 'DEM/demLat_N33_N35_Lon_W119_W116.dem -m ' + workdir + 'master/ -s ' + workdir + 'pawns/20150514 -r ' + str(rlks) + ' -a ' + str(alks) + ' -b "'" 33 35 -118 -116"'" '
#os.system(cmd)
