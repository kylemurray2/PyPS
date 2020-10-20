#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:01:24 2018
run snaphu
@author: kdm95
"""

import numpy as np
import isceobj
import os
import glob

params = np.load('params.npy',allow_pickle=True).item()
locals().update(params)

geocode = False
nproc='20'
ntilerow='4'
ntilecol='2 '
gamma0_file = params['tsdir'] + '/gamma0_lk.int'
pair=params['pairs'][0]

for pair in params['pairs']:
    infile = params['intdir']+ '/' + pair+'/filt.int'
    corfile = params['intdir']+ '/' + pair+'/cor_lk.r4'
    outfile = params['intdir']+ '/' + pair+'/fine.unw'
    if not os.path.isfile(outfile):
        print('unwrapping ' + pair)
        # The command line way doesn't work right now, so we'll use the config file
        #    cmd = '/home/insar/BIN/LIN/snaphu ' +  infile + ' ' + str(nxl) + ' -o ' + outfile + ' --mcf ' + ' -s --tile 30 30 80 80 --dumpall --nproc ' + nproc 
        #    os.system(cmd)
        
        # Write out the xml file for the unwrapped ifg
        out = isceobj.createIntImage() # Copy the interferogram image from before
        out.scheme =  'BIP' #'BIP'/ 'BIL' / 'BSQ' 
        out.dataType = 'FLOAT'
        out.filename = outfile
        out.width = params['nxl']
        out.length = params['nyl']
        out.dump(outfile + '.xml') # Write out xml
        out.renderHdr()
        out.renderVRT()
        
     # Write out a config file
        config_file_name = params['intdir'] + '/' +  pair + '/snaphu.conf'
        f = params['intdir'] + '/' +  pair + '/snaphu_config'
        conf=list()
        conf.append('# Input                                                           \n')
        conf.append('INFILE ' + infile                                              + '\n')
        conf.append('# Input file line length                                          \n')
        conf.append('LINELENGTH '  +  str(params['nxl'])                            + '\n')
        conf.append('                                                                  \n')
        conf.append('# Output file name                                                \n')
        conf.append('OUTFILE ' + outfile                                            + '\n')
        conf.append('                                                                  \n')
        conf.append('# Correlation file name                                           \n')
        conf.append('CORRFILE  '    +  corfile                                  + '\n')
        conf.append('                                                                  \n')
        conf.append('# Statistical-cost mode (TOPO, DEFO, SMOOTH, or NOSTATCOSTS)      \n')
        conf.append('STATCOSTMODE    SMOOTH                                            \n')
        conf.append('                                                                  \n')
        conf.append('INFILEFORMAT            COMPLEX_DATA                              \n')
        conf.append('#UNWRAPPEDINFILEFORMAT   COMPLEX_DATA                             \n')
        conf.append('OUTFILEFORMAT           FLOAT_DATA                                \n')
        conf.append('CORRFILEFORMAT          FLOAT_DATA                               \n')
        conf.append('                                                                  \n')
        conf.append('NTILEROW ' + ntilerow                                          + '\n')
        conf.append('NTILECOL ' + ntilecol                                          + '\n')
        conf.append('# Maximum number of child processes to start for parallel tile    \n')
        conf.append('# unwrapping.                                                     \n')
        conf.append('NPROC  '          +     nproc                                  + '\n')
        conf.append('ROWOVRLP 100                                                      \n')
        conf.append('COLOVRLP 100                                                      \n')
        conf.append('RMTMPTILE TRUE                                                    \n')
        with open(config_file_name,'w') as f:
            [f.writelines(c) for c in conf]
        command = 'snaphu -f ' + config_file_name 
        os.system(command)
    else:
        print(outfile + ' already exists.')


if geocode==True:
    setupParams = np.load('setupParams.npy',allow_pickle=True).item()
    DEM = setupParams['DEM']
    bounds = setupParams['bounds']
    mstr = workdir + '/master'
    
    for pair in pairs:
        file = intdir + '/' + pair + '/filt.unw'
        if pair==pairs[0]:
            pwn=mstr
        else:
            pwn = workdir
        
        command = 'geocodeIsce.py -a ' + str(alks) + ' -r ' +str(rlks) + ' -d ' + DEM + ' -m ' + mstr + ' -f ' +file + ' -b '  + bounds + ' -s ' + mstr
        os.system(command)