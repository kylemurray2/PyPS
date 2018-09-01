#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:50:42 2018
downlook the geometry files
@author: kdm95
"""
import isceobj
from mroipac.looks.Looks import Looks

with open(tsdir + 'params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks = pickle.load(f)


mergeddir=workdir + 'merged/'


def downLook(infile, outfile,alks,rlks):
    inImage = isceobj.createImage()
    inImage.load(infile + '.xml')
    inImage.filename = infile

    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inImage)
    lkObj.setOutputFilename(outfile)
    lkObj.looks()


file_list = list(['lat','lon','hgt','los','incLocal'])
for f in file_list:
    infile = mergeddir + 'geom_master/' + f + '.rdr.full'
    outfile = mergeddir + 'geom_master/' + f + '_lk.rdr'
    downLook(infile, outfile,alks,rlks)