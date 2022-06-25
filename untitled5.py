#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:25:11 2022

@author: km
"""


import numpy as np
import okada
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import profile_line as pl
import coupledGreens
from util import show 

nu = 0.5 #poisson
alphay =.1 # Multiplied by the dilatation G matrix (0 turns it off)
dr =10 # max distance between points
los1 = 37.5
ngps = 100

xrvec = []
yrvec = []
zrvec = []

plt.close('all')

# Make fake data with Okada fault
x = np.arange(-20,20)
y = np.arange(-20,20)
X,Y = np.meshgrid(x,y)
Xvec = X.ravel()
Yvec = Y.ravel()

npix = len(Xvec)
n = len(x)
xoff, yoff = 0,0
length, width = 15, 8
depth = 5
slip = 4
opening = 0
strike = 45
dip = 90
rake = 0
ue,un,uz = okada.forward(X, Y, xoff, yoff, depth, length, width, slip, opening, strike, dip, rake, nu)


#randomly sampled GPS locations. Find the indices
xid = np.random.randint(0,len(x),ngps)
yid = np.random.randint(0,len(x),ngps)

xi = X[yid,xid]
yi = Y[yid,xid]

xivec = xi.ravel()
yivec = yi.ravel()

ui = ue[yid,xid]
vi = un[yid,xid]
uv = np.concatenate((ui,vi),axis=0)
u0=np.nanmean(ui)
v0=np.nanmean(vi)

# Number of grid cells to solve
n1 = Xvec.shape[0]
n2 = int(2*n1) # for x and y
# n1 = len(xi)
# n2 = int(2*n1) # for x and y

vmin,vmax = None,None
fix,ax = plt.subplots(1,3,figsize=(10,5))
ax[0].pcolormesh(X,Y,ue,vmin=vmin,vmax=vmax);ax[0].set_xlabel('X')
ax[0].scatter(xi,yi,2,color='white')
ax[1].pcolormesh(X,Y,un,vmin=vmin,vmax=vmax);ax[1].set_xlabel('Y')
ax[1].scatter(xi,yi,2,color='white')
ax[2].pcolormesh(X,Y,uz,vmin=vmin,vmax=vmax);ax[2].set_xlabel('Z')
ax[2].scatter(xi,yi,2,color='white')
ax[1].set_title('Okada displacements')
ax[0].set_aspect('equal');ax[1].set_aspect('equal');ax[2].set_aspect('equal')
plt.show()




# Evaluate the three Greens functions q, p, w:
Pin = np.stack((xi.ravel(), yi.ravel()),axis=1) # Location of GPS 
Xin = np.stack((Xvec, Yvec),axis=1) # Location of grid cells
q, p, w = coupledGreens.get_qpw(Xin, Pin, dr, nu)

radii,dx,dy = coupledGreens.get_radius(Xin,Xin)


# Make design matrix Glos1
col1 = np.concatenate((q, w),axis=0)
col2 = np.concatenate((w, p),axis=0)
G = np.concatenate((col1,col2),axis=1)

e = np.eye(G.shape[1])
dime = len(e)
e=e*alphay
Ge = np.concatenate((G,e),axis=0)
Gg = np.dot( np.linalg.inv( np.dot( Ge.T,Ge )), G.T )
forces = np.dot(Gg,uv) # this equals forces*R
fx = forces[0:n1]
fy = forces[n1:n2]

# U = np.zeros(un.shape)
# V = np.zeros(un.shape)
# for ii in range(len(x)):
#     for jj in range(len(x)):
#         q,p,w = coupledGreens.get_qpw(Xin, np.array([X[ii,jj],Y[ii,jj]]), dr, nu)
#         U[ii,jj] =   u0 + np.dot(q,fx) + np.dot(w,fy)
#         V[ii,jj] =   v0 + np.dot(w,fx) + np.dot(p,fy)
        
q2, p2, w2 = coupledGreens.get_qpw(Xin, Xin, dr, nu)
U = np.dot(q2, fx) + np.dot(w2, fy)
V = np.dot(w2, fx) + np.dot(p2, fy)

U = U.reshape(ue.shape)
V = V.reshape(ue.shape)

fix,ax = plt.subplots(1,2,figsize=(7,5))
ax[0].imshow(U,vmin=vmin,vmax=vmax);ax[0].set_xlabel('Recovered X')
ax[0].scatter(xid,yid,2,color='white')
ax[1].imshow(V,vmin=vmin,vmax=vmax);ax[1].set_xlabel('Recovered Y')
ax[1].scatter(xid,yid,2,color='white')
plt.show()


# weave the Xvec and Yvec
positions = np.zeros(forces.shape)
positions[0::2] = Xvec
positions[1::2] = Yvec
data= np.zeros(forces.shape)
data[0::2] = ue.ravel()
data[1::2] = un.ravel()


Gs = np.zeros((n2,6))
for jj  in np.arange(0,n1,2):
    #odd rows:
    Gs[jj,0] = 1
    Gs[jj,2] = positions[jj]
    Gs[jj,3] = positions[jj+1]
    #even rows:
    Gs[jj+1,1] = 1;
    Gs[jj+1,4] = positions[jj]
    Gs[jj+1,5] = positions[jj+1]
    
beta=4
inv1 = []
inv2 = [] 
inv3 = []
ext = []
shear = []
for ii in range(len(Xvec)):
    
    distances_sq =  ((Xvec-Xvec[ii])**2) + ((Yvec-Yvec[ii])**2)  
    
    
    # plt.figure();plt.imshow(distances.reshape(ue.shape))
    # Wv2 = np.zeros(forces.shape)
    # Wv = np.exp(- distances_sq / (2*(beta**2)) )
    # Wv2[0::2] = Wv
    # Wv2[1::2] = Wv
    Wv2 = np.ones(forces.shape)
    W = np.diag(Wv2)
    
    Gw = np.dot(W,Gs)
    dw = np.dot(W,data)
    m2 = np.dot( np.linalg.inv(np.dot(Gw.T,Gw)), np.dot(Gw.T,dw))    
    
    L = np.array([np.array([m2[2],m2[3]]),np.array([m2[4],m2[5]])])
    
    #Calculate the symmetric and anti-symmetric parts (the strain and
    #rotation tensors
    strain = 0.5*(L+L.T)
    # rotation = 0.5*(L-L.T)
    
    #First and second invariants of the strain tensor:


    
    #Calculate invariants using the eigen values (these should be the same as before)
    inv1.append(np.trace(strain))  #dilatation
    
    # inv2.append(0.5* ( strain[0,1]**2 + (strain[0,0]-strain[1,1])**2) ) # matt
    #inv2.append(np.sqrt(0.5* ( strain[0,0]**2 + strain[0,1]**2 + strain[1,0]**2 + strain[1,1]**2 ) )) #unipd
    inv2.append(0.5* ( np.trace(strain)**2 - np.trace(np.dot(strain,strain)) )) #magnitude (Freymueller)
    # inv2.append(np.sqrt(0.5*(    strain[0,0]*strain[1,1] - strain[0,1]**2     )))
    # inv2.append(np.sqrt(strain[0,0]**2 +strain[1,1]**2 + 2*strain[0,1]**2)) #Sandwell 2016
    #Calculate the principle axes.
    w,v = np.linalg.eig(strain) # w: eigenvalues, v: eigenvectors
    shear.append(w.max()-w.min())
    ext.append(w.max())
    

    
          # 

    
inv1 = np.asarray(inv1)
inv1 = inv1.reshape(ue.shape)
inv2 = np.asarray(inv2)
inv2 = inv2.reshape(ue.shape)
shear = np.asarray(shear)
shear = shear.reshape(ue.shape)
ext = np.asarray(ext)
ext = ext.reshape(ue.shape)

fig,ax = plt.subplots(1,4)
ax[0].pcolormesh(X,Y,inv1)
ax[0].set_title('first inv')
ax[1].pcolormesh(X,Y,inv2)
ax[1].set_title('second inv')
ax[2].pcolormesh(X,Y,shear)
ax[2].set_title('shear')
ax[3].pcolormesh(X,Y,ext)
ax[3].set_title('ext')
ax[0].set_aspect('equal');ax[1].set_aspect('equal');ax[2].set_aspect('equal');ax[3].set_aspect('equal')

# Shear: difference between eigenvalues
# Ext: 
# Dilatation: 1st invariant
# Magnitude: 2nd invariant


plt.show()

