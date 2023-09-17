#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:01:24 2018
run snaphu
@author: kdm95
"""

import numpy as np
import isce.components.isceobj as isceobj
import os
import glob
from time import sleep

ps = np.load('./ps.npy',allow_pickle=True).all()

geocode = False
nproc='16'
ntilerow='2'
ntilecol='2'
rowovrlp='250'
colovrlp='250'
maxComps = 20
initMethod = 'MCF'
defoMax = 0

ps.intdir = ps.workdir + '/Fringe2/PS_DS/sequential1'

fSizes = []
for ii,p in enumerate(ps.pairs): 
    if os.path.isfile(ps.intdir + '/' + p + '/' + 'filt_lk.int'):       
        if os.path.getsize(ps.intdir + '/' + p + '/' + 'fine_lk.int')==0:
            print('WARNING: ' + ps.intdir + '/' + p + ' File size too small. May be corrupt.' )
            # os.system('rm -r ' + ps.intdir + '/' + p )
            
        else:
            fSizes.append(os.path.getsize(ps.intdir + '/' + p + '/' + 'fine_lk.int'))
            # os.system('rm -r ' + ps.intdir + '/' + p )
    else:
        print(p + '/' + 'filt_lk.int does not exist')
medSize = np.nanmedian(fSizes)
sleep(5)



for ii in range(len(ps.pairs)):
    pair = ps.pairs[ii]
    infile = ps.intdir+ '/' + pair+'/filt_lk2.int'
    corfile = ps.intdir+ '/' + pair+'/filt_lk2.cor'
    outfile = ps.intdir+ '/' + pair+'/filt_lk2.unw'
    conncompOut = ps.intdir+ '/' + pair+'/filt_lk2.unw.conncomp'
    waterMask = ps.mergeddir + '/geom_reference/waterMask_lk.rdr'
    if not os.path.isfile(outfile):
        print('unwrapping ' + pair)
        os.system('rm snaphu_tiles*')
        # The command line way doesn't work right now, so we'll use the config file
        #    cmd = '/home/insar/BIN/LIN/snaphu ' +  infile + ' ' + str(nxl) + ' -o ' + outfile + ' --mcf ' + ' -s --tile 30 30 80 80 --dumpall --nproc ' + nproc 
        #    os.system(cmd)
        
        # Write out the xml file for the unwrapped ifg
        out1 = isceobj.createIntImage() # Copy the interferogram image from before
        out1.scheme =  'BIP' #'BIP'/ 'BIL' / 'BSQ' 
        out1.dataType = 'FLOAT'
        out1.filename = outfile
        out1.width = ps.nxl
        out1.length = ps.nyl
        out1.dump(outfile + '.xml') # Write out xml
        out1.renderHdr()
        out1.renderVRT()
        out1.finalizeImage()

        intImage = isceobj.createIntImage()
        intImage.load(infile + '.xml')
        nxl2= intImage.width
        # intImage.close()

        # Write xml for conncomp files
        out = isceobj.createImage() # Copy the interferogram image from before
        out.accessMode = 'READ'
        out.byteOrder = 'l'
        out.dataType = 'BYTE'
        out.family = 'image'
        out.filename = conncompOut
        out.bands = 1
        out.scheme =  'BIL' #'BIP'/ 'BIL' / 'BSQ' 
        out.width = ps.nxl
        out.length = ps.nyl
        out.dump(conncompOut + '.xml') # Write out xml
        out.renderHdr()
        out.renderVRT()
        out.finalizeImage()
        
     # Write out a config file
        config_file_name = ps.intdir + '/' +  pair + '/snaphu.conf'
        f = ps.intdir + '/' +  pair + '/snaphu_config'
        conf=list()
        conf.append('# Input                                                           \n')
        conf.append('INFILE ' + infile                                              + '\n')
        conf.append('# Input file line length                                          \n')
        conf.append('LINELENGTH '  +  str(nxl2)                                     + '\n')
        conf.append('MAXNCOMPS '  +  str(maxComps)                                     + '\n')
        conf.append('INITMETHOD '  +  str(initMethod)                               + '\n')
        conf.append('DEFOMAX_CYCLE '  +  str(defoMax)                               + '\n')
        conf.append('                                                                  \n')
        conf.append('# Output file name                                                \n')
        conf.append('OUTFILE ' + outfile                                            + '\n')
        conf.append('                                                                  \n')
        conf.append('# Correlation file name                                           \n')
        conf.append('CORRFILE  '    +  corfile                                      + '\n')
        conf.append('BYTEMASKFILE  '    +  waterMask                                + '\n')

        conf.append('                                                                  \n')
        conf.append('# Statistical-cost mode (TOPO, DEFO, SMOOTH, or NOSTATCOSTS)      \n')
        conf.append('STATCOSTMODE    SMOOTH                                            \n')
        conf.append('                                                                  \n')
        conf.append('INFILEFORMAT            COMPLEX_DATA                              \n')
        conf.append('#UNWRAPPEDINFILEFORMAT  COMPLEX_DATA                              \n')
        conf.append('OUTFILEFORMAT           FLOAT_DATA                                \n')
        conf.append('CORRFILEFORMAT          FLOAT_DATA                                \n')
        conf.append('                                                                  \n')
        conf.append('NTILEROW ' + ntilerow                                          + '\n')
        conf.append('NTILECOL ' + ntilecol                                          + '\n')
        conf.append('# Maximum number of child processes to start for parallel tile    \n')
        conf.append('# unwrapping.                                                     \n')
        conf.append('NPROC  '          +     nproc                                  + '\n')
        conf.append('ROWOVRLP ' + rowovrlp + '                                         \n')
        conf.append('COLOVRLP ' + colovrlp + '                                         \n')
        conf.append('RMTMPTILE TRUE                                                    \n')
        with open(config_file_name,'w') as f:
            [f.writelines(c) for c in conf]
        if ntilerow=='1' and ntilecol=='1': # Only use the -S flag if we are using tiles
            command = 'snaphu --mcf -g ' + conncompOut + ' -f ' + config_file_name 
        else:
            command = 'snaphu --mcf -g ' + conncompOut + ' -S -f ' + config_file_name 
        os.system(command)
    else:
        print(outfile + ' already exists.')


if geocode==True:
    '''
    this only works for noncropped imagery
    '''
    setupParams = np.load('setupParams.npy',allow_pickle=True).item()
    DEM = setupParams['DEM']
    bounds = setupParams['bounds']
    mstr = ps.workdir + '/master'
    
    for pair in ps.pairs:
        file = ps.intdir + '/' + pair + '/filt.unw'
        if pair==ps.pairs[0]:
            pwn=mstr
        else:
            pwn = ps.workdir
        
        command = 'geocodeIsce.py -a ' + str(ps.alks) + ' -r ' +str(ps.rlks) + ' -d ' + DEM + ' -m ' + mstr + ' -f ' +file + ' -b '  + bounds + ' -s ' + mstr
        os.system(command)