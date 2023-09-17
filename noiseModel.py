#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input:
    data time series
    dates
    
Output:
    linear model: rate, intercept
    
    
@author: km
"""

import numpy as np
from scipy import signal,optimize
from scipy.linalg import lstsq
import os
import h5py
from matplotlib import pyplot as plt
from scipy.special import gamma
from scipy.interpolate import interp1d
import scipy
from datetime import date
from sklearn.linear_model import LinearRegression
import math
import argparse

def interpolate_timeseries(time_vec, data_ts, fs):
    """
    Interpolates a time series with unevenly sampled data to an evenly sampled time series.
    Adds white noise to interpolated values

    Args:
        time (np.ndarray): A 1D array of time values.
        data (np.ndarray): A 1D array of data values corresponding to the time values.
        freq (float): Sampling interval for time series data.

    Returns:
        np.ndarray: A 1D array of interpolated data values.
    """
    # Determine the start and end times of the time series
    start_time = np.min(time_vec)
    end_time = np.max(time_vec)
    # Generate the evenly spaced time values
    interp_time = np.arange(start_time, end_time, fs)
    # Interpolate the data values
    f = interp1d(time_vec, data_ts, kind='linear')
    interp_data = f(interp_time)
    
    mask = np.in1d(interp_time,time_vec)
    indices = np.where(~mask)[0]
    
    # Figure out data residual standard deviation so we can add white noise.
    _,_,_,residuals,_ = linear(data_ts,time_vec)
    interp_data[indices] +=np.random.normal(0, np.nanstd(residuals), size=len(indices))
    
    return interp_data,interp_time


def linear(ts,time):
    # Fit a linear function
    A = np.vstack([time, np.ones((len(time), 1)).flatten()]).T
    Aa = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
    mod = np.dot(Aa, ts)
    synth = np.dot(A, mod)
    residuals = (ts-synth)  # *lam/(4*np.pi)*100 # cm
    C_model_white = np.var(residuals) * np.linalg.inv( np.dot( A.T,A) )
    return mod, A, synth, residuals, C_model_white


def sineDesign(t,period=365.25):
   '''
    Input time vector and desired period
    Output design matrix for lsq inversion
   '''
   rows = [np.sin(1/period*2*np.pi*t), np.cos(1/period*2*np.pi*t), np.ones(len(t)),t] 
   A = np.asarray(rows)
   return A.T


def fitSine(A,y):
   (mod,residuals,rank,sing_vals) = lstsq(A,y)
   return mod


def makeSine(mod, time_vec):
    phase = np.arctan2(mod[0],mod[1])*180/np.pi
    amplitude = np.linalg.norm([mod[0],mod[1]],2)
    bias = mod[2]
    slope = mod[3]
    sineSynth = time_vec*slope + amplitude*np.sin((2*np.pi/365)*(time_vec) + phase) + bias
    
    
    
    
    return sineSynth


# Define the power law model
def power_law(f,fs, sig_pl2, k):
    P_0 = (2*(2*np.pi)**k * sig_pl2) / fs**(1-(-k/2))
    return  (P_0 / f**k) 


def getPLcov(ts,k,dT):
    ''' 
    Colored noise uncertainties
    From Langbein 2004
    
    Inputs:
        ts: time series data values
        k: spectral index
        dT: sampling interval
    Outputs:
        E: colored noise covariance matrix
    
    '''
    N = len(ts)
    k+=1e-10 # add a tiny number to stabilize in case k=0
    gs=[]
    iterate = 100        # no longer computable after k=170 (on this computer)
    # what happens to psi as you increase k?
    for ni in range(iterate):
        g = gamma(ni-(-k/2)) / (np.math.factorial(ni) * gamma(k/2))
        gs.append(g)
        
    gs = np.asarray(gs)
    gs[np.isnan(gs)] = 0.0
    gs=gs.T
    g_vec = gs[-1] * np.ones((N,))
    g_vec[0:len(gs)] = gs 
    H = scipy.linalg.toeplitz(g_vec)
    H *= np.tri(*H.shape)
    H  = H* dT**(-k/4)
    E = np.dot(H,H.T)
    E[E<1e-9]=0
    # plt.figure();plt.imshow(E);plt.show()
    return E


def bootstrap_linreg(time_vec, data_ts, num_bootstraps=1000):
    # Initialize arrays to store bootstrap estimates of slope and intercept
    bootstrap_slopes = np.zeros(num_bootstraps)
    bootstrap_intercepts = np.zeros(num_bootstraps)
    # Fit the original data to a linear regression model
    # linreg = LinearRegression().fit(x.reshape(-1, 1), y)
    # Generate bootstrap samples and fit each one to a linear regression model
    for i in range(num_bootstraps):
        # Generate a bootstrap sample by randomly selecting with replacement from the original data
        bootstrap_indices = np.random.choice(len(time_vec), len(time_vec), replace=True)
        bootstrap_x = time_vec[bootstrap_indices]
        bootstrap_y = data_ts[bootstrap_indices]
        # Fit the bootstrap sample to a linear regression model
        # bootstrap_linreg = LinearRegression().fit(bootstrap_x.reshape(-1, 1), bootstrap_y)
        mod, A,synth, residuals,_ = linear(bootstrap_y,bootstrap_x)
        # Store the slope and intercept estimates from the bootstrap
        bootstrap_slopes[i] = mod[0]
        bootstrap_intercepts[i] = mod[1]
    
    # Calculate the mean and standard deviation of the bootstrap estimates of slope and intercept
    mean_slope = np.mean(bootstrap_slopes)
    std_slope = np.std(bootstrap_slopes)
    mean_intercept = np.mean(bootstrap_intercepts)
    std_intercept = np.std(bootstrap_intercepts)
    
    # Calculate the 95% confidence intervals for the slope and intercept estimates
    ci_slope = np.percentile(bootstrap_slopes, [2.5, 97.5])
    ci_intercept = np.percentile(bootstrap_intercepts, [2.5, 97.5])
    
    # Return the original linear regression model, along with the bootstrap estimates and confidence intervals
    return  mean_slope, std_slope, ci_slope, mean_intercept, std_intercept, ci_intercept


def get_uncertainties(residuals,A,fs,plot=False):
    '''
    Parameters
    ----------
    residuals : 
    A : Design matrix for inverse problem
    fs : sampling freq
    plot : True or falst

    Returns
    -------
    m_uncertainty_white : 
    m_uncertainty_color : 
    rmse : 
    '''
    rmse = np.sqrt(np.mean(residuals**2))
    # compute the power spectral density using Welch's method
    f, psd = signal.welch(residuals)
    # Call least_squares with method='lm' to use LAPACK implementation
    init_guess = [.01,1.5]
    res = optimize.least_squares(lambda params: power_law(f[1:],fs, *params) - psd[1:], init_guess, method='lm')
    # output of res.x = [sig_pl2, k, sig_wh2]
    
    if plot:
        # plot the power spectral density and the fitted model
        plt.figure()
        plt.semilogx(f, psd,'.', label='Power Spectral Density')
        plt.semilogx(f, power_law(f,fs, *res.x), 'r--', label='Fitted Model')
        plt.xlabel('Frequency (log scale)')
        plt.ylabel('Power Spectral Density')
        plt.title('Power Spectral Density of Time Series')
        plt.legend()
        plt.show()
    
    sig_pl2 = res.x[0]           # Variance of powerlaw noise
    spectral_index = res.x[1]   # Spectral index for powerlaw noise model (k)
    
    dT = 1 # sampling interval
    Epl = getPLcov(residuals,spectral_index,dT)
    
    I = np.eye(len(residuals))
    C = np.var(residuals)*I + sig_pl2*Epl
    C_model_white = np.var(residuals) * np.linalg.inv( np.dot( A.T,A) ) # same as np.linalg.inv( np.dot( A.T,np.dot(np.linalg.inv(sig_wh2*I),A) ) )
    C_model_color = np.linalg.inv( np.dot( A.T,np.dot(np.linalg.inv(C),A) ) )
    
    m_uncertainty_white = 1.96*np.sqrt(np.diag(C_model_white))
    m_uncertainty_color = 1.96*np.sqrt(np.diag(C_model_color))

    return m_uncertainty_white, m_uncertainty_color, rmse, spectral_index, sig_pl2*Epl

