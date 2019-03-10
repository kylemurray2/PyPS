#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:22:43 2019

@author: kdm95
"""

from pylab import *
import numpy as np
from math import atan2
def fitSine(tList,yList,period):
   '''
       period in days
       time list in days
   returns
       phase in degrees
   '''
   b = np.asarray(yList)
#   b = b.T
         
   rows = [ [sin(1/period*2*pi*t), cos(1/period*2*pi*t), 1,t] for t in tList]
   A = matrix(rows)
   (w,residuals,rank,sing_vals) = lstsq(A,b)
   phase = np.arctan2(w[1,:],w[0,:])*180/pi  
   amplitude = np.sqrt((np.square(w[0,:]) + np.square(w[1,:])))
   bias = w[2,:]
   slope = w[3,:]
   return (phase,amplitude,bias,slope)

def fitSine1d(tList,yList,period):
   '''
       period in days
       time list in days
   returns
       phase in degrees
   '''
   b = matrix(yList).T

       
   rows = [ [sin(1/period*2*pi*t), cos(1/period*2*pi*t), 1,t] for t in tList]
   A = matrix(rows)
   (w,residuals,rank,sing_vals) = lstsq(A,b)
   phase = atan2(w[1,0],w[0,0])*180/pi
   amplitude = norm([w[0,0],w[1,0]],2)
   bias = w[2,0]
   slope = w[3,0]
   return (phase,amplitude,bias,slope)
 
    
if __name__=='__main__':
   import random
 
   tList = arange(0.0,1.0,0.001)
   tSamples = arange(0.0,1.0,0.05)
   random.seed(0.0)
   phase = 65
   amplitude = 3
   bias = -0.3
   frequency = 4
   yList = amplitude*sin(tList*frequency*2*pi+phase*pi/180.0)+bias
   ySamples = amplitude*sin(tSamples*frequency*2*pi+phase*pi/180.0)+bias
   yMeasured = [y+random.normalvariate(0,2) for y in ySamples]
   #print yList
   (phaseEst,amplitudeEst,biasEst) = fitSine(tSamples,yMeasured,frequency)
   print ('Phase estimate = %f, Amplitude estimate = %f, Bias estimate = %f'
       % (phaseEst,amplitudeEst,biasEst))
        
   yEst = amplitudeEst*sin(tList*frequency*2*pi+phaseEst*pi/180.0)+biasEst
 
   figure(1)
   plot(tList,yList,'b')
   plot(tSamples,yMeasured,'+r',markersize=12,markeredgewidth=2)
   plot(tList,yEst,'-g')
   xlabel('seconds')
   legend(['True value','Measured values','Estimated value'])
   grid(True)
   show()