#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:54:40 2019
VV VH analysis for optimizing time series
@author: kdm95
"""

import numpy as np
import isceobj
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import glob
import cv2
import os
#from mroipac.filter.Filter import Filter
params = np.load('params.npy').item()
locals().update(params)
#params['slcdir'] = '/data/kdm95/Delta/p42/merged/SLC_VV'
#np.save('params.npy',params)
nxl= params['nxl']
nyl = params['nyl']
tsdir = params['tsdir']
dates = params['dates']

# Load the gamma0 file
f = params['tsdir'] + '/gamma0.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0= intImage.memMap()[:,:,0] 

vhdir = '/data/kdm95/Delta/p42/merged/SLC_VH'
vvdir = '/data/kdm95/Delta/p42/merged/SLC_VV'
#
## Run this to fix the file paths in the xml files if you changed the dir names (you probably did)
#for f in glob.glob(vhdir + '/*'):
#    os.system('fixImageXml.py -i ' + f + '/*.full -f' )
#for f in glob.glob(vvdir + '/*'):
#    os.system('fixImageXml.py -i ' + f + '/*.full -f' )
    
#ii=20 # start at date 20 just because
ymin=5000
ymax=7000
xmin=44000
xmax=46000


gamImage = isceobj.createSlcImage()
gamImage.load(tsdir + '/gamma0.int.xml')
gam = gamImage.memMap()[ymin:ymax,xmin:xmax,0]

vh_list = glob.glob(vhdir + '/*')
dates_vh = []
for d in vh_list:
    dates_vh.append(d[-8:])
    
dates_vh.sort()

vvvh_stack = np.empty((ymax-ymin,xmax-xmin,len(dates_vh)),dtype=np.complex64)
#vvvh_stack = np.empty((ny,nx,len(dates_vh)),dtype=np.complex64)

#vvvh_stack_amp = np.empty((ymax-ymin,xmax-xmin,len(dates_vh)))

for ii in np.arange(0,len(dates_vh)-1):
    d1 = dates_vh[ii]
    d2 = dates_vh[ii+1]
    out = params['workdir'] + '/vvvh/' + d1 + '_' + d2 + '_diff.int'
    slc1 = vvdir +'/'+ d1 + '/' + d1 + '.slc.full'
    slc2 = vvdir +'/'+ d2 + '/' + d2 + '.slc.full'
    slc3 = vhdir +'/'+ d1 + '/' + d1 + '.slc.full'
    slc4 = vhdir +'/'+ d2 + '/' + d2 + '.slc.full'
#    os.system("imageMath.py -e='a*conj(b)' -t cfloat -o tmp --a=" + slc1 + " --b=" + slc4)
    os.system("imageMath.py -e='a*conj(b)*conj(c)*f' -t cfloat -o tmp --a=" + slc1 + " --b=" + slc2 + " --c=" + slc3 + " --f=" +  slc4)
    os.system("imageMath.py -e='a/abs(a)*sqrt(sqrt(abs(a)))' -t cfloat -o " + out + " --a=tmp")


for ii in np.arange(0,len(dates_vh)-1):
    d1 = dates_vh[ii]
    d2 = dates_vh[ii+1]
    f = params['workdir'] + '/vvvh/' + d1 + '_' + d2 + '_diff.int'
    intImage = isceobj.createIntImage()
    intImage.load(f + '.xml')
    vvvh_stack[:,:,ii] = intImage.memMap()[0,ymin:ymax,xmin:xmax] 
    
cpxm0 = np.divide( np.nanmean(vvvh_stack,axis=2), np.nanmean(abs(vvvh_stack),axis=2) )
out = isceobj.createIntImage() # Copy the interferogram image from before
out.filename = params['tsdir'] + '/vvvh_dif'
out.width = xmax-xmin
out.length = ymax-ymin
out.dump(out.filename + '.xml') # Write out xml
cpxm0.tofile(out.filename) # Write file out