#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:39:00 2019
    Make greens functions G matrix for coupled inversions
@author: kdm95
"""
import numpy as np

# Evaluate the three Greens functions q, p, w:
def get_radius(X,P):
    #GET_RADIUS  Returns dx, dy,and radial distances between points in X and P
    # X: rows contain the 2-dimensional coordinates in X
    # P: rows contain the 2-dimensional coordinates in P

    if P.shape[0] == 2:
        n = X.shape[0] # n is number of points in X
        m = 1 # m is number of points in P
        dx = np.zeros((m,n))
        dy = np.zeros((m,n))
        
        for k in np.arange(0,m):
            dx[k,:] = np.array([X[:,0] - np.ones((n,)) * P[0]])
            dy[k,:] = np.array([X[:,1] - np.ones((n,)) * P[1]])   
    else:
        n = X.shape[0] # n is number of points in X
        m = P.shape[0] # m is number of points in P
        dx = np.zeros((m,n))
        dy = np.zeros((m,n))
        for k in np.arange(0,m-1):
            dx[k,:] = np.array([X[:,0] - np.ones((n,)) * P[k,0]])
            dy[k,:] = np.array([X[:,1] - np.ones((n,)) * P[k,1]])
    r = np.sqrt((dx**2 +  dy**2))
    return r.astype(np.float32),dx.astype(np.float32),dy.astype(np.float32)

#r,dx,dy = get_radius(np.array([xi,yi]).T,np.array([xi,yi]).T)

def get_qpw(X, P, dr, nu):
    """
    Compute the Green's functions for all (X;P) combinations
    X should be even grid, and P can be scattered
    """
    r,dx,dy = get_radius(X, P); 
    logr = np.log(r+dr);  # Add fudge dr term
    r2 = np.square((r+dr),dtype=np.float32)
    q = np.multiply((3 - nu), logr,dtype=np.float32)  + np.divide((1+nu)*(dy**2),r2,dtype=np.float32)
    p = np.multiply((3 - nu), logr,dtype=np.float32)  + np.divide((1+nu)*(dx**2),r2,dtype=np.float32)
    w = np.multiply((-(1+nu)*dx), (np.divide(dy,r2)),dtype=np.float32)
    return q,p,w


def get_subpqw(X1,Y1,dim,nu,dr):
    # Now we're going to do the inversion in a moving window
    ii,jj=0,0
    X = X1[ii:ii+dim,jj:jj+dim]
    Y = Y1[ii:ii+dim,jj:jj+dim]
    xi,yi=X.ravel(),Y.ravel()
    n1 = len(xi)
    n2 = int(2*n1)
    # Evaluate the three Greens functions q, p, w:
    Xin = np.stack((xi, yi),axis=1)
    Pin = np.stack((xi, yi),axis=1)
    q, p, w = get_qpw(Xin, Pin, dr, nu)
    return X,Y,q,p,w,n1,n2
    
def make_Glos(q,p,w,LosPath='/data/kdm95/WW/LOS.npy'):
   # this one doesn't solve for the Z component
    los1,los2,los3,los4 = np.load(LosPath)
    # Make design matrix Glos1
    col1 = los1[0]*q + los1[1]*w
    col2 = los1[0]*w + los1[1]*p
#    col3 = np.eye((col1.shape[0]))*los1[2]
    Glos1 = np.concatenate((col1,col2),axis=1)
    
    col1 = los2[0]*q + los2[1]*w
    col2 = los2[0]*w + los2[1]*p
#    col3 = np.eye((col1.shape[0]))*los2[2]
    Glos2 = np.concatenate((col1,col2),axis=1)
    
    col1 = los3[0]*q + los3[1]*w
    col2 = los3[0]*w + los3[1]*p
#    col3 = np.eye((col1.shape[0]))*los3[2]
    Glos3 = np.concatenate((col1,col2),axis=1)
    
    col1 = los4[0]*q + los4[1]*w
    col2 = los4[0]*w + los4[1]*p
#    col3 = np.eye((col1.shape[0]))*los4[2]
    Glos4 = np.concatenate((col1,col2),axis=1)
    return Glos1,Glos2,Glos3,Glos4

def make_GlosZ(q,p,w,LosPath='/data/kdm95/WW/LOS.npy'):
   
    los1,los2,los3,los4 = np.load(LosPath)
    los5 = np.load('lUAV.npy')
    
    col1 = los1[0]*q + los1[1]*w
    col2 = los1[0]*w + los1[1]*p
    col3 = np.ones((col1.shape[0],1))*los1[2]

    Glos1 = np.concatenate((col1,col2,col3),axis=1)
    
    col1 = los2[0]*q + los2[1]*w
    col2 = los2[0]*w + los2[1]*p
    col3 = np.ones((col1.shape[0],1))*los2[2]
    Glos2 = np.concatenate((col1,col2,col3),axis=1)
    
    col1 = los3[0]*q + los3[1]*w
    col2 = los3[0]*w + los3[1]*p
    col3 = np.ones((col1.shape[0],1))*los3[2]
    Glos3 = np.concatenate((col1,col2,col3),axis=1)
    
    col1 = los4[0]*q + los4[1]*w
    col2 = los4[0]*w + los4[1]*p
    col3 = np.ones((col1.shape[0],1))*los4[2]
    Glos4 = np.concatenate((col1,col2,col3),axis=1)
    
    col1 = los5[0]*q + los5[1]*w
    col2 = los5[0]*w + los5[1]*p
    col3 = np.ones((col1.shape[0],1))*los5[2]
    GlosUAV = np.concatenate((col1,col2,col3),axis=1)
    
    return Glos1,Glos2,Glos3,Glos4,GlosUAV

def make_GlosZ2(q,p,w,LosPath='/data/kdm95/WW/LOS.npy'):
   # this is if you're doing the  inversion as a single step rather thatn windowed
   # this one doesn't solve for the Z component
    los1,los2,los3,los4 = np.load(LosPath)
    # Make design matrix Glos1
    col1 = los1[0]*q + los1[1]*w
    col2 = los1[0]*w + los1[1]*p
    col3 = np.eye((col1.shape[0]))*los1[2]
    Glos1 = np.concatenate((col1,col2,col3),axis=1)
    
    col1 = los2[0]*q + los2[1]*w
    col2 = los2[0]*w + los2[1]*p
    col3 = np.eye((col1.shape[0]))*los2[2]
    Glos2 = np.concatenate((col1,col2,col3),axis=1)
    
    col1 = los3[0]*q + los3[1]*w
    col2 = los3[0]*w + los3[1]*p
    col3 = np.eye((col1.shape[0]))*los3[2]
    Glos3 = np.concatenate((col1,col2,col3),axis=1)
    
    col1 = los4[0]*q + los4[1]*w
    col2 = los4[0]*w + los4[1]*p
    col3 = np.eye((col1.shape[0]))*los4[2]
    Glos4 = np.concatenate((col1,col2,col3),axis=1)
    return Glos1,Glos2,Glos3,Glos4


def make_greens(Glos1,Glos2,Glos3,Glos4,alphax,alphay):
    e = np.eye(Glos1.shape[1])
    dime = len(e)
    e[0:int(dime/2),0:int(dime/2)] = e[0:int(dime/2),0:int(dime/2)]*alphax
    e[int(dime/2):,int(dime/2):] = e[int(dime/2):,int(dime/2):]*alphay
    e[:,-1] = 0 # This is so we don't damp Z
    GlosAll = np.concatenate((Glos1,Glos2,Glos3,Glos4),axis=0)
    GlosAlle = np.concatenate((Glos1,Glos2,Glos3,Glos4,e),axis=0)
    Gg = np.dot( np.linalg.inv( np.dot( GlosAlle.T,GlosAlle )), GlosAll.T )
    return GlosAll,GlosAlle,Gg

def make_greensWhole(Glos1,Glos2,Glos3,Glos4,alphax,alphay):
    e = np.eye(Glos1.shape[1])
    dime = len(e)
    e[0:int(dime/3),0:int(dime/3)] = e[0:int(dime/3),0:int(dime/3)]*alphax
    e[int(dime/3):int(dime*(2/3)),int(dime/3):int(dime*(2/3))] =e[int(dime/3):int(dime*(2/3)),int(dime/3):int(dime*(2/3))]*alphay
    e[int(dime*(2/3)):,int(dime*(2/3)):] =e[int(dime*(2/3)):,int(dime*(2/3)):]*0
    GlosAll = np.concatenate((Glos1,Glos2,Glos3,Glos4),axis=0)
    GlosAlle = np.concatenate((Glos1,Glos2,Glos3,Glos4,e),axis=0)
    Gg = np.dot( np.linalg.inv( np.dot( GlosAlle.T,GlosAlle )), GlosAll.T )
    return GlosAll,GlosAlle,Gg


def make_greens2(Glos1,Glos2,alphax,alphay):
    e = np.eye(Glos1.shape[1])
    dime = len(e)
    e[0:int(dime/2),0:int(dime/2)] = e[0:int(dime/2),0:int(dime/2)]*alphax
    e[int(dime/2):,int(dime/2):] = e[int(dime/2):,int(dime/2):]*alphay
    e[:,-1] = 0 # This is so we don't damp Z
    GlosAll = np.concatenate((Glos1,Glos2),axis=0)
    GlosAlle = np.concatenate((Glos1,Glos2,e),axis=0)
    Gg = np.dot( np.linalg.inv( np.dot( GlosAlle.T,GlosAlle )), GlosAll.T )
    return GlosAll,GlosAlle,Gg

def make_greens3(Glos1,Glos2,Glos3,alphax,alphay):
    e = np.eye(Glos1.shape[1])
    dime = len(e)
    e[0:int(dime/2),0:int(dime/2)] = e[0:int(dime/2),0:int(dime/2)]*alphax
    e[int(dime/2):,int(dime/2):] = e[int(dime/2):,int(dime/2):]*alphay
    e[:,-1] = 0 # This is so we don't damp Z
    GlosAll = np.concatenate((Glos1,Glos2,Glos3),axis=0)
    GlosAlle = np.concatenate((Glos1,Glos2,Glos3,e),axis=0)
    Gg = np.dot( np.linalg.inv( np.dot( GlosAlle.T,GlosAlle )), GlosAll.T )
    return GlosAll,GlosAlle,Gg

def make_greens5(Glos1,Glos2,Glos3,Glos4,GlosUAV,alphax,alphay):
    e = np.eye(Glos1.shape[1])
    dime = len(e)
    e[0:int(dime/2),0:int(dime/2)] = e[0:int(dime/2),0:int(dime/2)]*alphax
    e[int(dime/2):,int(dime/2):] = e[int(dime/2):,int(dime/2):]*alphay
    e[:,-1] = 0 # This is so we don't damp Z
    GlosAll = np.concatenate((Glos1,Glos2,Glos3,Glos4,GlosUAV),axis=0)
    GlosAlle = np.concatenate((Glos1,Glos2,Glos3,Glos4,GlosUAV,e),axis=0)
    Gg = np.dot( np.linalg.inv( np.dot( GlosAlle.T,GlosAlle )), GlosAll.T )
    return GlosAll,GlosAlle,Gg