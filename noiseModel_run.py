#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:06:21 2023

@author: km
"""

from PyPS2 import noiseModel as nm
import numpy as np
import os
import h5py
from matplotlib import pyplot as plt
from PyPS2 import util


periodic = True
fs = 6
plt.close('all')
workDir = os.getcwd()
mintDir = './MintPy/'

ps = np.load('./ps.npy', allow_pickle=True).all()

# ifgramStack.h5
filename = mintDir + 'inputs/ifgramStack.h5'
ds_ifgramStack = h5py.File(filename,'r+')   
pairs = np.asarray(ds_ifgramStack['date'])
ds_ifgramStack.close()

filename = mintDir + 'timeseries.h5'
ds = h5py.File(filename, 'r+')
timeseries = ds['timeseries']
dates = np.asarray(ds['date'])
ts = np.asarray(ds['timeseries'][:, 1400, 1125])
ds.close()


ts_interp,time_interp = nm.interpolate_timeseries(ps.dn0, ts, fs)
# Get decimal years for plotting time series
dec_year = []
yr0 = ps.dates[0][0:4]
dec_year_interp=[]
for dn in time_interp:
    yr = np.floor(dn/365) + int(yr0)
    doy = dn%365
    dec_year_interp.append(float(yr) + (doy/365.25))
dec_year_interp = np.asarray(dec_year_interp,dtype=np.float32)


# Fit the data to a linear regression model using bootstrapping
mean_slope, std_slope, ci_slope, mean_intercept, std_intercept, ci_intercept = nm.bootstrap_linreg(time_interp, ts_interp,1000)
synth_line = mean_slope*time_interp + mean_intercept
ts_detrend = ts_interp - synth_line

freq_cos, freq_sin, amplitude_cos, amplitude_sin, phase_shift_cos, phase_shift_sin = util.fitSine1d(dec_year_interp, ts_detrend)
synth_sine = amplitude_cos * np.cos(2 * np.pi * freq_cos * dec_year_interp + phase_shift_cos) + amplitude_sin * np.sin(2 * np.pi * freq_sin * dec_year_interp + phase_shift_sin)


freq_cos = 1
freq_sin = 2
amplitude_cos = np.mean(abs(ts_detrend))
amplitude_sin = np.mean(abs(ts_detrend))/2
phase_shift_cos = 0.0
phase_shift_sin = 0.0
synth_sine_0 = amplitude_cos * np.cos(2 * np.pi * freq_cos * dec_year_interp + phase_shift_cos) + amplitude_sin * np.sin(2 * np.pi * freq_sin * dec_year_interp + phase_shift_sin)


plt.figure()
plt.plot(dec_year_interp,ts_detrend,'.')
plt.plot(dec_year_interp,synth_sine,label='sin model')
plt.plot(dec_year_interp,synth_sine_0,label='sin 0')
plt.legend()

    
    
    
plt.plot(dec_year_interp,ts_interp,'.')
plt.plot(dec_year_interp,synth_line,'.')
plt.plot(dec_year_interp,ts_detrend,'.')

# plt.plot(ps.dn0,ts,'.')
plt.plot(dec_year_interp,synth)

plt.plot(dec_year_interp,synth_sine)

# # Generate a sequence of random steps
# steps = np.random.randn(len(ts_interp))
# # Construct the random walk series
# ts_interp = np.cumsum(steps)








if periodic:
    # Periodic Fit
    period = 365.25
    A = nm.sineDesign(time_interp,period)
    # Fit the sinusoid with slope and intercept
    mod = nm.fitSine(A,ts_interp)#(phase,amplitude,bias,slope)
    synth = nm.makeSine(mod,time_interp)
    residuals_s = ts_interp-synth
else:
    # Linear Fit
    mod, A,synth, residuals,C_model_white = nm.linear(ts_interp,time_interp)


from scipy.stats import f_oneway
f_statistic, p_value = f_oneway(residuals, residuals_s)

print("F-Statistic:", f_statistic)
print("p-value:", p_value)
alpha = 0.05  # significance level

if p_value < alpha:
    print("The models are significantly different. Model 2 provides a better representation.")
else:
    print("There is no significant difference between the models.")
    
    
    
plt.figure()
plt.plot(dec_year_interp,np.cumsum(abs(residuals)),label='lsq')
plt.plot(dec_year_interp,np.cumsum(abs(residuals_s)),label='sine')
plt.legend()

m_uncertainty_white, m_uncertainty_color, rmse, spectral_index,sig = nm.get_uncertainties(residuals,A,fs,plot=True)

mod2 = [0.0021011/365, 1.35800236e-02]
mle_upper = np.dot(A,mod+ mod2)
mle_lower = np.dot(A,mod- mod2)

synth_upper_wh = np.dot(A, mod+m_uncertainty_white)
synth_lower_wh = np.dot(A, mod-m_uncertainty_white)
synth_upper_pl = np.dot(A, mod+m_uncertainty_color)
synth_lower_pl = np.dot(A, mod-m_uncertainty_color)



plt.figure()
plt.plot(dec_year_interp, ts_interp, '.',color='gray')
plt.plot(dec_year_interp, synth,'black')
plt.plot(dec_year_interp, synth_lower_wh,'g')
plt.plot(dec_year_interp, synth_upper_wh,'g')
plt.plot(dec_year_interp, synth_lower_pl,'--',color='purple')
plt.plot(dec_year_interp, synth_upper_pl,'--',color='purple')
plt.plot(dec_year_interp, mle_upper,'--',color='red')
plt.plot(dec_year_interp, mle_lower,'--',color='red')

plt.legend(['Data','mean rate','white','white','Powerlaw','powerlaw','mle'])
plt.show()

print('Boot strapped Rate uncertainty: ', str(np.round(100000*1.96*std_slope,3)))
print('White noise Rate uncertainty: ', str(np.round(100000*m_uncertainty_white[0],3)))
print('Colored noise Rate uncertainty: ', str(np.round(100000*m_uncertainty_color[0],3)))
print('spectral index: ', str(np.round(spectral_index,3)))
print('rate: ', str(np.round(mod[0],7)))




import numpy as np
from scipy.optimize import minimize

def doMLE(C,r):
    N = len(r)
    ln_det_C = np.log(np.linalg.det(C))
    rtCr = np.dot(r.T, np.dot(np.linalg.inv(C), r))
    Nln2pi = N * np.log(2 * np.pi)
    mle = -0.5 * (ln_det_C + rtCr + Nln2pi)
    return np.log(mle)

r=residuals
# define initial guess for C
C_guess = np.eye(len(r))
C = C_guess
# mle = doMLE(r,C_guess)
# set constraints for C to be positive-definite
# bounds = [(0, None) for _ in range(len(r)**2)]
r.reshape(1,-1)
# optimize the negative log-likelihood function with L-BFGS-B
result = minimize(doMLE, C_guess,args=(r,), method='L-BFGS-B')
# get the optimized value of C
C_opt = result.x.reshape((len(r), len(r)))


def doMLE(C_1d, r):
    C = C_1d.reshape(len(r),len(r))  # Reshape C_1d into a square matrix
    C = np.dot(C.T, C)  # Ensure positive semi-definite
    C += np.eye(len(r)) * 1e-6  # Add small positive value to the diagonal elements
    ln_det_C = np.log(np.linalg.det(C))
    rtCr = np.dot(r.T, np.dot(np.linalg.inv(C), r))
    Nln2pi = N * np.log(2 * np.pi)
    mle = -0.5 * (ln_det_C + rtCr + Nln2pi)
    return -np.sum(np.log(mle))  # Minimize the negative log likelihood, ensure scalar output


C_guess = np.eye(len(r)).reshape(1,-1)  # Initial guess for C_1d
result = minimize(doMLE, C_guess, args=(r,)) # solve for C
# j
estimated_C_1d = result.x
estimated_C = estimated_C_1d.reshape(len(r),len(r))
estimated_C = np.dot(estimated_C.T, estimated_C)  # Ensure positive semi-definite



N = len(residuals)
U = np.linalg.cholesky(C_model_white)
ln_det_I = 0.0
for i in range(0,1):
    ln_det_I -= np.log(U[i,i])
ln_det_I *= 2.0

ln_det_C = np.log(np.linalg.det(C_model_white))
# sigma_eta = 
logL = -0.5 * (N*np.log(2*np.pi) + ln_det_C + 2.0*(N)*np.log(sigma_eta) + N)

#____________________________________________
# Try using Hectorp functions
# t = 
[theta,C_theta,ln_det_C,sigma_eta] = compute_leastsquares(t,H,x,F,samenoise=False)
