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
import pickle

with open(tsdir + 'params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks = pickle.load(f)

nproc='28'
ntilerow='30'
ntilecol='30'

for pair in pairs:
    infile = intdir+pair+'/fine_lk.int'
    outfile = intdir+pair+'/fine_lk.unw'
    print('unwrapping ' + pair)
    # The command line way doesn't work right now, so we'll use the config file
    #    cmd = '/home/insar/BIN/LIN/snaphu ' +  infile + ' ' + str(nxl) + ' -o ' + outfile + ' --mcf ' + ' -s --tile 30 30 80 80 --dumpall --nproc ' + nproc 
    #    os.system(cmd)
    
    # Write out the xml file for the unwrapped ifg
    out = isceobj.createIntImage() # Copy the interferogram image from before
    out.scheme =  'BIP' #'BIP'/ 'BIL' / 'BSQ' 
    out.dataType = 'FLOAT'
    out.filename = outfile
    out.width = nxl
    out.length = nyl
    out.dump(outfile + '.xml') # Write out xml
    out.renderHdr()
    out.renderVRT()
    
    
 # Write out a config file
    config_file_name = intdir + pair + '/snaphu.conf'
    f = intdir + pair + '/snaphu_config'
    conf=list()
    conf.append('# Input                                                           \n')
    conf.append('INFILE ' + infile                                              + '\n')
    conf.append('# Input file line length                                          \n')
    conf.append('LINELENGTH '  +  str(nxl)                                      + '\n')
    conf.append('                                                                  \n')
    conf.append('# Output file name                                                \n')
    conf.append('OUTFILE ' + outfile                                            + '\n')
    conf.append('                                                                  \n')
    conf.append('# Correlation file name                                           \n')
    #conf.append('CORRFILE  '      maskfilerlk                                  + '\n')
    conf.append('                                                                  \n')
    conf.append('# Statistical-cost mode (TOPO, DEFO, SMOOTH, or NOSTATCOSTS)      \n')
    conf.append('STATCOSTMODE    SMOOTH                                            \n')
    conf.append('                                                                  \n')
    conf.append('INFILEFORMAT            FLOAT_DATA                                \n')
    conf.append('UNWRAPPEDINFILEFORMAT   FLOAT_DATA                                \n')
    conf.append('OUTFILEFORMAT           FLOAT_DATA                                \n')
    conf.append('CORRFILEFORMAT          FLOAT_DATA                                \n')
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
    command = '/usr/local/GMT5SAR/snaphu/src/snaphu -f ' + config_file_name 
    os.system(command)


