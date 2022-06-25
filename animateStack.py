#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:16:16 2021
    Convert stack to animation
@author: km
"""

import numpy as np
import isceobj
from matplotlib import animation
from matplotlib import pyplot as plt


params = np.load('params.npy',allow_pickle=True).item()
geom = np.load('geom.npy',allow_pickle=True).item()
locals().update(params)
locals().update(geom)
msk = np.load('msk.npy') 
cor = np.load('cor.npy')



stack = np.zeros((len(params['pairs']),params['nyl'],params['nxl']))
for ii in range(len(params['pairs'])):
    p = params['pairs'][ii]
    f = './merged/interferograms/' + p + '/fine_lk_filt.int'
    intImage = isceobj.createIntImage()
    intImage.load(f + '.xml')
    ifg = intImage.memMap()[:,:,0] 
    ifgc = np.angle(ifg)
    ifgc[msk==0]=np.nan

    if flip:
        stack[ii,:,:] = np.flipud(ifgc)
    else:
        stack[ii,:,:] = ifgc


n_observations = 10

fig, ax = plt.subplots(figsize=(10, 5))

def updateMap(i):
    ax.clear()
    ax.imshow(stack[i,:,:],cmap='prism')
    ax.text(20,275,str(dates[i]),color='black')
    


anim = animation.FuncAnimation(fig, updateMap, frames = 10)#nd-1)
anim.save('test.mp4', fps = 10, writer='imagemagick')  
