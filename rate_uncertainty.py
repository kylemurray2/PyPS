# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:13:50 2023

@author: km
"""

from scipy import signal,optimize
from scipy.signal import freqz
from scipy.signal import periodogram
import os
import isceobj
import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from scipy.special import gamma
import scipy
import fitSine

workDir = os.getcwd()
mintDir = './MintPy/'

ps = np.load('./ps.npy', allow_pickle=True).all()


# ifgramStack.h5
filename = mintDir + 'inputs/ifgramStack.h5'
ds_ifgramStack = h5py.File(filename,'r+')   
pairs = np.asarray(ds_ifgramStack['date'])
ts1 = np.asarray(ds_ifgramStack['unwrapPhase'][:, 900, 2795])
ts2 = np.asarray(ds_ifgramStack['unwrapPhase'][:, 960, 2795])
plt.figure()
plt.plot(ts1,label='1')
plt.plot(ts2,label='2')
plt.legend()
plt.figure()
plt.plot(ts1-ts2,'.');plt.show()
coh = np.asarray(ds_ifgramStack['coherence'][:, 1000, 2000])
np.pdf
ds_ifgramStack.close()

ds = h5py.File(filename, 'r+')
dates = np.asarray(ds['date'])

filename = mintDir + 'timeseries.h5'
ds = h5py.File(filename, 'r+')
timeseries = ds['timeseries']
dates = np.asarray(ds['date'])
ts = np.asarray(ds['timeseries'][:, 1400, 1125])
ds.close()
# plt.figure();plt.imshow(timeseries[5,:,:],'magma')
plt.figure();plt.plot(ts)

# Fit a linear function
G = np.vstack([ps.dn0, np.ones((len(ps.dn0), 1)).flatten()]).T
Gg = np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)
mod = np.dot(Gg, ts)
rate = mod[0]*365  # cm/yr
#offs  = np.reshape(mod[1,:],(ps.nyl, ps.nxl))
synth = np.dot(G, mod)
res = (ts-synth)  # *lam/(4*np.pi)*100 # cm

# Find the uncertainty
co = np.cov(ts)
# mcov=np.diag(np.dot(Gg,np.dot(co,Gg.T)))
mcov = co * np.diag(np.linalg.inv(np.dot(G.T, G)))
# mcov = np.inv( np.dot( np.dot()))

m_uncertainty = 1.96*mcov**.5
synthlow = np.dot(G, mod-m_uncertainty)
synthhigh = np.dot(G, mod+m_uncertainty)



# Fit a periodic funcition
period = 365.25
phase, amp, bias, slope = fitSine.fitSine1d(ps.dn0, ts, period)
sinesynth = ps.dn0*slope + amp*np.sin((2*np.pi/365)*(ps.dn0) + phase) + bias
plt.figure()
plt.plot(ps.dec_year, ts, '.')
plt.plot(ps.dec_year, synth)
plt.plot(ps.dec_year, synthlow)
plt.plot(ps.dec_year, synthhigh)
plt.plot(ps.dec_year, sinesynth)




# Colored noise uncertainties
# From Langbein 2004
d = ts      # Data time series
A= np.vstack([ps.dn0, np.ones((len(ps.dn0), 1)).flatten()]).T # Design matrix
n = 1.33      # Spectral index
N = len(d)

gs=[]
iterate = 150
I = np.eye(len(ts))
# what happens to psi as you increase n?
for ni in range(iterate):
    # g = gamma(I + n/2)/(gamma(n/2) * np.math.factorial(ni))

    g = gamma(ni-(-n/2)) / (np.math.factorial(ni) * gamma(n/2))
    # no longer computable after n=170 (on this computer)
    gs.append(g)
plt.figure();plt.plot(range(len(gs)),gs,'.');plt.ylabel('g');plt.xlabel('N')
plt.title('k=-1; psi goes to 0 inf as');plt.show()
gs = np.asarray(gs)

gs[np.isnan(gs)] = 0.0

g_vec = gs[-1] * np.ones((N,))
g_vec[0:len(gs)] = gs 
H = scipy.linalg.toeplitz(g_vec)
H *= np.tri(*H.shape)
plt.figure();plt.imshow(H);plt.title('Transformation matrix H')
plt.show()

gamma(ni-(n/2)) / (np.math.factorial(ni) * gamma(-n/2))


dT =1# 6 * 24*60*60# 6 days (in seconds) sampling interval (we'll use days) !!!! might need to interpolate TS?????
fs  = 1/dT # sampling frequency in Hz 
H  = H* dT**(-n/4)
E = np.dot(H,H.T)
plt.figure();plt.imshow(E);plt.title('E')
plt.show()


sig_pl2 = 1e-4
# (11) Amplitude of the power law noise in the freq domain
P_0 = (2*(2*np.pi)**-n * sig_pl2) / fs**(1-(n/2))

# compute the power spectral density using Welch's method
f, psd = signal.welch(ts)

# (7) Power law noise is described in the frequency domain:
P_f = P_0/f**n
plt.figure()
plt.semilogx(f, P_f, label='Power Spectral Density of power law model')
plt.semilogx(f, psd, label='Power Spectral Density of time series')

plt.xlabel('Frequency (log scale)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density of Time Series')
plt.legend()
plt.show()




# Define the power law model
def power_law(f, sig_pl2, n,y0):
    P_0 = (2*(2*np.pi)**n * sig_pl2) / fs**(1-(-n/2))
    return  (P_0 / f**n) + y0

# fit the model to the power spectral density

popt, pcov = optimize.curve_fit(power_law, f[1:], psd[1:])
''' output of popt = [sig_pl2, n, sig_wh2] '''

# Call least_squares with method='lm' to use LAPACK implementation
init_guess = popt
res = optimize.least_squares(lambda params: power_law(f[1:], *params) - psd[1:], init_guess, method='lm')

# plot the power spectral density and the fitted model
plt.figure()
plt.semilogx(f[1:], psd[1:],'.', label='Power Spectral Density')
plt.semilogx(f, power_law(f, *res.x), 'r--', label='Fitted Model')
plt.xlabel('Frequency (log scale)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density of Time Series')
plt.legend()
plt.show()

sig_wh2 = popt[2]   # Variance of white noise
sig_pl2 = popt[0]   # Variance of white noise


C = np.nanvar(ts)*I + sig_pl2*E
C_model_white = np.var(ts) * np.linalg.inv( np.dot( A.T,A) ) # same as np.linalg.inv( np.dot( A.T,np.dot(np.linalg.inv(sig_wh2*I),A) ) )
C_model_color = np.linalg.inv( np.dot( A.T,np.dot(np.linalg.inv(C),A) ) )

m_uncertainty_white = m_uncertainty # Defined above during inversion
m_uncertainty_color = 1.96*np.sqrt(C_model_color)


# rate_upper_wh = rate + m_uncertainty_white[0,0]
# rate_lower_wh = rate - m_uncertainty_white[0,0]

synth = np.dot(A, mod)
synth_upper_wh = np.dot(A, mod+m_uncertainty_white)
synth_lower_wh = np.dot(A, mod-m_uncertainty_white)

synth_upper_pl = np.dot(A, mod+np.diag(m_uncertainty_color))
synth_lower_pl = np.dot(A, mod-np.diag(m_uncertainty_color))


plt.figure()
plt.plot(ps.dec_year, ts, '.',color='black')
plt.plot(ps.dec_year, synth,'black')
plt.plot(ps.dec_year, synth_lower_wh,'g')
plt.plot(ps.dec_year, synth_upper_wh,'g')
plt.plot(ps.dec_year, synth_lower_pl,'--',color='gray')
plt.plot(ps.dec_year, synth_upper_pl,'--',color='gray')
plt.legend(['Data','mean rate','white','white','Powerlaw','powerlaw'])
plt.show()



standardError = np.sqrt(np.diag(sig_x2))
from scipy.stats import t
t_value = t.ppf(0.975, N - 3)
ci = t_value * standardError



















# # Code from Williams 2003
# from scipy.special import gamma
# import scipy
# x= ts       # time series of data
# n = len(x)  # number of data
# k = 0 # Spectral index for colored noise model

# # Form the covariance matrix for colored noise
# psis=[]
# iterate = 150
# # what happens to psi as you increase n?
# for ni in range(iterate):
#     psi_n = gamma(ni-(k/2)) / (np.math.factorial(ni) * gamma(-k/2))
#     # no longer computable after n=170 (on this computer)
#     psis.append(psi_n)
# plt.figure();plt.plot(range(len(psis)),psis,'.');plt.ylabel('psi');plt.xlabel('n')
# plt.title('k=-1; psi goes to 0 inf as');plt.show()
# psis = np.asarray(psis)

# psis[np.isnan(psis)] = 0.0

# psi_vec = psis[-1] * np.ones((n,))
# psi_vec[0:len(psis)] = psis 
# T = scipy.linalg.toeplitz(psi_vec)
# T *= np.tri(*T.shape)
# plt.figure();plt.imshow(T);plt.title('Transformation matrix T')

# # Scale T to ensure that the power spectra for k will cross at the consistent freq given sampling interval
# Tdel = 6 * 24*60*60# 6 days (in seconds) sampling interval (we'll use days) !!!! might need to interpolate TS?????
# T  = T* Tdel**(-k/4)
# J_k = np.dot(T,T.T)
# plt.figure();plt.imshow(J_k);plt.title('J')
# plt.show()

# # Equation for the power spectrum (10)
# D_k = 2*(2*np.pi)**k * (24*60*60*365.25)**(k/2)
# b_k = 1 #??? noise amplitude 
# fs  = 1/Tdel # sampling frequency in Hz 
# f = 1 #??? frequency array?
# powerAmp   = ((D_k * b_k**2)/(fs**(k/2 + 1))) * f**k #this is for frequency domain. function of f?

# f_0 = fs**.5 / (2* np.pi* np.sqrt(24*60*60*365.25))


# # uncertainty in slope for any colored noise source (25)
# beta = -(k/2) - 2
# gam = -3-k
# P = np.array([-.0237,-.3881,-2.661,-9.8529,-21.0922,-25.1638,-11.4275,10.7839,20.3377,11.9942])
# N = len(P)
# nu_vec = np.zeros((N))

# for ii in range(len(P)):
#     nu_vec[ii] = (P[ii]*k**(N-ii))
# nu = np.sum(nu_vec)
# sig_r2 = b_k**2 * nu * Tdel**(beta) * n**(gam)




# # Gaussian processes regression
# # Define the kernel function
# lengthScale = 5
# kernel = C(1.0, (1e-3, 1e3)) * RBF(lengthScale, (1, 365))
# data_var = 1.96*.01  # np.sqrt(np.var(ts))
# noise = data_var*np.ones_like(ts)
# # Create a Gaussian Process regressor object
# gp = GaussianProcessRegressor(
#     kernel=kernel, alpha=noise**2, n_restarts_optimizer=10)
# gp.fit(ps.dn0.reshape(-1, 1), ts.reshape(-1, 1))
# # Make predictions for some test data
# X_test = np.linspace(ps.dn0[0], ps.dn0[-1], 500).reshape(-1, 1)
# x_dec = (X_test/365) + ps.dec_year[0]
# y_pred, sigma = gp.predict(X_test, return_std=True)
# # Plot the results
# plt.figure(figsize=(10, 5))
# plt.plot(ps.dec_year, ts, 'ko', label='Observations')
# plt.plot(x_dec, y_pred, 'b-', label='Prediction')
# # plt.fill_between(X_test[:, 0], y_pred[:, 0] - sigma, y_pred[:, 0] + sigma,
# #                  alpha=0.5, color='lightblue', label='Uncertainty')
# plt.fill_between(x_dec.ravel(), y_pred.ravel() - sigma, y_pred.ravel() + sigma,
#                  alpha=0.5, color='lightblue', label='Uncertainty')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()



# Fit a weighted least squares linear function to the ts array
# coh_subset = np.zeros(len(ps.pairs))
# for i in range(len(ps.pairs)):
#     p = ps.pairs[i]
#     if p in 
#     index = ps.pairs2.index(p)
#     coh_subset[i] = coh[index]

# corStack = []
# for p in ps.pairs:
#     cor_file = '/' + p + '/fine_lk.cor'
#     corImage = isceobj.createIntImage()
#     corImage.load(cor_file + '.xml')
#     cor = corImage.memMap()[:,:,0]
#     cor = cor.copy()
#     # cor[np.isnan(gam)] = np.nan
#     corStack.append(cor)
# corStack = np.asarray(corStack,dtype=np.float32)[:,:,:]
# # make the G matrix
# G = np.vstack([ps.dn0, np.ones((len(ps.dn0), 1)).flatten()]).T
# # make the weights matrix based on the inverse of coherence coh 
# W = np.diag(1/coh)
# # make the weighted G matrix
# Gw = np.dot(np.linalg.inv(np.dot(np.dot(G.T, W), G)), np.dot(G.T, W))
# # make the weighted ts array
# ts_w = ts/coh
# # make the weighted model
# mod_w = np.dot(Gw, ts_w)
# # make the weighted synthetic time series
# synth_w = np.dot(G, mod_w)
# # make the weighted residuals
# res_w = (ts_w-synth_w)  # *lam/(4*np.pi)*100 # cm
# # Get the norm of the residuals
# resnorm = np.linalg.norm(res, axis=0)
# s = resnorm/np.sqrt(len(res)) # from page 39 of Aster Parameter Estimation and Inverse Problems
# resstd = np.std(res, axis=0)
# # Do a chi-squared test
# import scipy.stats as stats
# chi2 = np.sum(res**2/s)
# p = 1 - stats.chi2.cdf(chi2, len(res)-2)
# # Plot the time series with error bars using s for each point
# plt.figure()
# plt.plot(ps.dec_year, ts, '.')
# plt.errorbar(ps.dec_year, ts, yerr=s, fmt='o', color='black', ecolor='lightgray', elinewidth=1, capsize=3, capthick=1)  
# plt.plot(ps.dec_year, synth)       
# plt.xlabel('Time')      
# plt.ylabel('Value')
# plt.title('Time series with error bars')
# plt.show()
