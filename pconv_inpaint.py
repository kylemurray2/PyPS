#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Kyle Murray
Sun Nov  4 16:38:10 2018
Description:
  
    Partial convolutional Inpainting with Deep Neural Network

"""

import numpy as np
import isceobj
import pickle
from copy import deepcopy
import cv2
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
# Import modules from libs/ directory
from libs.pconv_layer import PConv2D
from libs.util import random_mask
from libs.pconv_model import PConvUnet

tsdir='/data/kdm95/OK2/TS/'

with open(tsdir + 'params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    pairs,nd,lam,workdir,intdir,tsdir,ny,nx,nxl,nyl,alks,rlks = pickle.load(f)

gamma_thresh = .1

# Load mask
f = tsdir + 'gamma0_lk.int'
intImage = isceobj.createIntImage()
intImage.load(f + '.xml')
gamma0_lk= intImage.memMap()[:,:,0] 

pair=pairs[25]

# Load image
ifgfile = intdir + pair + '/fine_lk.unw'
ifgImage = isceobj.createIntImage()
ifgImage.load(ifgfile + '.xml')
img = ifgImage.memMap()[:,:,0]
# Load mask
mask = np.ones(img.shape)
mask[np.where(gamma0_lk < gamma_thresh)]=0


# Crop img and mask to test
img = img[1000:1300,1100:1400]
mask = mask[1000:1300,1100:1400]
shape = img.shape
print(f"Shape of image is: {shape}")

# Image + mask
masked_img = deepcopy(img)
masked_img[mask==0] = np.nan

# Show side by side
_, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(img)
axes[1].imshow(mask)
axes[2].imshow(masked_img)
plt.show()

img = np.reshape(img,(300,300,1))
#img = np.repeat(img,3,axis=2)


# Input images and masks
input_img = Input(shape=shape)
input_mask = Input(shape=shape)
output_img, output_mask1 = PConv2D(8, kernel_size=(7,7), strides=(2,2))([input_img, input_mask])
output_img, output_mask2 = PConv2D(16, kernel_size=(5,5), strides=(2,2))([output_img, output_mask1])
output_img, output_mask3 = PConv2D(32, kernel_size=(5,5), strides=(2,2))([output_img, output_mask2])
output_img, output_mask4 = PConv2D(64, kernel_size=(3,3), strides=(2,2))([output_img, output_mask3])
output_img, output_mask5 = PConv2D(64, kernel_size=(3,3), strides=(2,2))([output_img, output_mask4])
output_img, output_mask6 = PConv2D(64, kernel_size=(3,3), strides=(2,2))([output_img, output_mask5])
output_img, output_mask7 = PConv2D(64, kernel_size=(3,3), strides=(2,2))([output_img, output_mask6])
#output_img, output_mask8 = PConv2D(64, kernel_size=(3,3), strides=(2,2))([output_img, output_mask7])


# Create model
model = Model(
    inputs=[input_img, input_mask], 
    outputs=[
        output_img, output_mask1, output_mask2, 
        output_mask3, output_mask4, output_mask5,
        output_mask6, output_mask7
    ])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Show summary of the model
model.summary()


formatted_img = np.expand_dims(masked_img, 0) 
formatted_img = np.reshape(formatted_img,(1,300,300,1))

formatted_mask = np.expand_dims(mask, 0)
formatted_mask = np.reshape(formatted_mask,(1,300,300,1))

print(f"Original Mask Shape: {formatted_mask.shape} - Max value in mask: {np.max(formatted_mask)}")

output_img, o1, o2, o3, o4, o5, o6, o7 = model.predict([formatted_img, formatted_mask])
_, axes = plt.subplots(2, 4, figsize=(20, 10))
axes[0][0].imshow(o1[0, :,:, 0], cmap = 'gray', vmin=0, vmax=1)
axes[0][1].imshow(o2[0, :,:, 0], cmap = 'gray', vmin=0, vmax=1)
axes[0][2].imshow(o3[0, :,:, 0], cmap = 'gray', vmin=0, vmax=1)
axes[0][3].imshow(o4[0, :,:, 0], cmap = 'gray', vmin=0, vmax=1)
axes[1][0].imshow(o5[0, :,:, 0], cmap = 'gray', vmin=0, vmax=1)
axes[1][1].imshow(o6[0, :,:, 0], cmap = 'gray', vmin=0, vmax=1)
axes[1][2].imshow(o7[0, :,:, 0], cmap = 'gray', vmin=0, vmax=1)
axes[1][3].imshow(o7[0, :,:, 0], cmap = 'gray', vmin=0, vmax=1)
axes[0][0].set_title(f"Shape: {o1.shape}")
axes[0][1].set_title(f"Shape: {o2.shape}")
axes[0][2].set_title(f"Shape: {o3.shape}")
axes[0][3].set_title(f"Shape: {o4.shape}")
axes[1][0].set_title(f"Shape: {o5.shape}")
axes[1][1].set_title(f"Shape: {o6.shape}")
axes[1][2].set_title(f"Shape: {o7.shape}")
axes[1][3].set_title(f"Shape: {o7.shape}")
plt.show()

# Part 3: Implement U-Net architecture
MAX_BATCH_SIZE = int(128)
PConvUnet().summary()
from keras.preprocessing.image import ImageDataGenerator
class DataGenerator(ImageDataGenerator):
    def flow(self, x, *args, **kwargs):
        while True:
            
            # Get augmentend image samples
            ori = next(super().flow(x, *args, **kwargs))

            # Get masks for each image sample
            mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori        

# Create datagen
datagen = DataGenerator(  
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Create generator from numpy arrays
batch = np.stack([img for _ in range(MAX_BATCH_SIZE)], axis=0)
generator = datagen.flow(x=batch, batch_size=4)

# Get samples & Display them
(masked, mask), ori = next(generator)

# Show side by side
_, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(masked[0,:,:,:])
axes[1].imshow(mask[0,:,:,:])
axes[2].imshow(ori[0,:,:,:])


