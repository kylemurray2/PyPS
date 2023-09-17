#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:24:52 2023

Simple script to make an interferogram

Example

    fn_slc1 = '/d/HI/S1/Asc/Oahu/merged/SLC/20160914/20160914.slc.full.crop.vrt'
    fn_slc2 = '/d/HI/S1/Asc/Oahu/merged/SLC/20160926/20160926.slc.full.crop.vrt'
    ifgOutName = '/d/HI/S1/Asc/Oahu/merged/SLC/20160914_20160926.ifg'
    ifg = makeIfg(fn_slc1,fn_slc2,ifgOutName)
    plt.figure();plt.imshow(np.angle(ifg))

@author: km
"""

import numpy as np
from osgeo import gdal
import isceobj

def makeIfg(slc1_fn,slc2_fn,ifg_fn):
    ds1 = gdal.Open(slc1_fn)
    ds2 = gdal.Open(slc2_fn)
    slc1 = ds1.GetVirtualMemArray()
    slc2 = ds2.GetVirtualMemArray()
    ifg = np.multiply(slc1,np.conj(slc2))
    
    out = isceobj.createIntImage() # Copy the interferogram image from before
    out.dataType = 'CFLOAT'
    out.filename = ifg_fn
    out.width = ifg.shape[1]
    out.length = ifg.shape[0]
    out.dump(out.filename + '.xml') # Write out xml
    fid=open(out.filename,"wb+")
    fid.write(ifg)
    out.renderHdr()
    out.renderVRT()  
    fid.close()
    
    return ifg


