#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:25:53 2022

@author: km
"""

import numpy as np
import os
import glob
import datetime
import isceobj
from osgeo import gdal
import argparse
from PyPS2 import util
from matplotlib import pyplot as plt
import unwrap_fringe as uf
import argparse
import scipy.spatial
from datetime import date
import FilterAndCoherence
import looks


filterFlag      = True
unwrap          = False # Usually better to leave False and use runSnaphu.py for more options and outputs
filterStrength  = '.6'
fixImage        = False  #Do this in case you renamed any of the directories or moved the SLCs since they were made

ps = np.load('./ps.npy',allow_pickle=True).all()


# For unwrapping
inps = argparse.Namespace()
inps.method = 'snaphu'
inps.xmlFile = None
# For downlooking
inps.rglooks = ps.rlks
inps.azlooks = ps.alks

gam = np.ones((ps.nyl,ps.nxl))
np.save('./Npy/gam.npy',gam)
# Make the ifgs
if not os.path.isdir(ps.intdir):
    os.mkdir(ps.intdir)
    
    
for pair in ps.pairs2:
    pairDir = ps.intdir + '/' + pair
    if not os.path.isdir(pairDir):
        os.mkdir(pairDir)
    if not os.path.isfile(pairDir + '/fine.int'):
        ifgOutName = pairDir + '/fine.int'
        ifgLkName = pairDir + '/fine_lk.int'
        print('making ' + pair)
        d1 = pair.split('_')[0]
        d2 = pair.split('_')[1]
        
        if ps.crop:
            fn_slc1 = ps.slcdir +'/'+ d1 + '/' + d1 + '.slc.full.crop.vrt'
            fn_slc2 = ps.slcdir +'/'+ d2 + '/' + d2 + '.slc.full.crop.vrt'
        else:
            fn_slc1 = ps.slcdir +'/'+ d1 + '/' + d1 + '.slc.full.vrt'
            fn_slc2 = ps.slcdir +'/'+ d2 + '/' + d2 + '.slc.full.vrt'
    
        ds1 = gdal.Open(fn_slc1)
        ds2 = gdal.Open(fn_slc2)
        
        slc1 = ds1.GetVirtualMemArray()
        slc2 = ds2.GetVirtualMemArray()
        
        ifg = np.multiply(slc1,np.conj(slc2))
        
        out = isceobj.createImage() # Copy the interferogram image from before
        out.dataType = 'CFLOAT'
        out.filename = ifgOutName
        out.width = ifg.shape[1]
        out.length = ifg.shape[0]
        out.dump(out.filename + '.xml') # Write out xml
        fid=open(out.filename,"wb+")
        fid.write(ifg)
        out.renderHdr()
        out.renderVRT()  
        fid.close()
        
        # Downlook
        inps.infile = ifgOutName
        inps.outfile = ifgLkName
        looks.main(inps)
        
        # Filter and coherence
        corname = pairDir + '/filt_lk.cor'
        offilt =  pairDir + '/filt_lk.int'
        if not os.path.isfile(corname):
            print('\n making ' + pair)
            FilterAndCoherence.runFilter(ifgLkName,offilt,float(filterStrength))
            FilterAndCoherence.estCoherence(offilt, corname)
    else:
        print(pair + ' fine.int already exists') 
        
