#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from scipy.fftpack import fftshift, ifftshift
import sys
import os
import pickle

#%%

job_ID = int(os.environ.get('SLURM_ARRAY_JOB_ID', default=-1))       # job ID
task_ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', default=-1)) # task ID

# T = 1000
wmax = 512
N = 2**25
T = np.pi/wmax * N # T is tmax
eta_conv = 1/T * 5
J = 1
J = J/np.sqrt(2) # choose same convention as stephan
mus = np.concatenate([np.linspace(0,0.01,11), np.linspace(0.01,0.1,21)[1:]])

#if task_ID == -1:
mu = mus[task_ID]

if mu < 0.01:
    temp = 15e-5
else:
    temp = 5e-4
beta = 1/temp

etas = [0,1,0.1]

alpha = 0.1
#%%

def fermi(eps):
    return 1 / ( np.exp(eps * beta) + 1)

def fft_(x):
    return fftshift(scipy.fft.fft(fftshift(x)))

def fft(x):
    return fftshift(scipy.fft.ifft(fftshift(x))) * len(x)

for eta in etas:
    xx = []
    t = np.arange(-N/2, N/2) / N * T
    # t = np.linspace(-T/2, T/2, N)
    w = np.arange(-N/2, N/2)  * 2 * np.pi / T
    # w = np.linspace(-N/2, N/2, N)  * 2 * np.pi / T

    dt = T / N
    dw = 2 * np.pi / T

    # initialize randomly
    np.random.seed(1)
    GRR = - 1j * np.abs(np.random.rand(N))
    GLL = - 1j * np.abs(np.random.rand(N))
    GLR = - 1j * np.abs(np.random.rand(N))

    # initialize once
    nf = fermi(w)
    nf_ = fermi(-w)

    ax = np.newaxis

    exp_eta_t = np.exp( - eta_conv * t)
    exp_eta_t[:N//2] = 0

    # rhoRR = - 1 / np.pi * GRR.imag
    # rhoLL = - 1 / np.pi * GLL.imag
    # rhoLR = 1j / np.pi * GLR.real
    rhoRR = np.ones(N)
    rhoLL = np.ones(N)
    rhoLR = np.zeros(N)
    rhoRR[0] = 0
    rhoLL[0] = 0

    #normalize
    rhoRR /= dw * np.sum(rhoRR, axis=0)
    rhoLL /= dw * np.sum(rhoLL, axis=0)

    # %%


    # perform self-consistency equation
    for i in np.arange(250):
        nRR = dw * fft_(rhoRR * nf)
        nLL = dw * fft_(rhoLL * nf)
        nLR = dw * fft_(rhoLR * nf)

        nRR_ = dw * fft_(rhoRR * nf_)
        nLL_ = dw * fft_(rhoLL * nf_)
        nLR_ = dw * fft_(rhoLR * nf_)

        SRR = 2 * J**2 * (1 - eta)**2 * (-1j) * dt * fft(exp_eta_t * (nRR**3 + nRR_**3))
        SLL = 2 * J**2 * (1 + eta)**2 * (-1j) * dt * fft(exp_eta_t * (nLL**3 + nLL_**3))
        SLR = 2 * J**2 * (1 - eta**2) * (-1j) * dt * fft(exp_eta_t * (nLR**3 + nLR_**3))

        D = (w - SRR) * ( w - SLL) + (1j*mu - SLR)**2

        GLL = (w - SRR) / D
        GRR = (w - SLL) / D
        GLR = - (1j*mu - SLR) / D

        rhoRR_ = -1 / np.pi * GRR.imag
        rhoLL_ = -1 / np.pi * GLL.imag
        rhoLR_ = 1j / np.pi * GLR.real

        if i % 10 == 0:
            x = (np.sum(np.abs(rhoRR_-rhoRR)**2) + np.sum(np.abs(rhoLL_-rhoLL)**2) + np.sum(np.abs(rhoLR_-rhoLR)**2)) / N
            xx.append(x)
            print(i, x)#, 'sum_rule:', dw * np.sum(rhoRR), dw * np.sum(rhoLL))

        rhoRR = rhoRR * (1-alpha) + alpha * rhoRR_
        rhoLL = rhoLL * (1-alpha) + alpha * rhoLL_
        rhoLR = rhoLR * (1-alpha) + alpha * rhoLR_

        # symmetrize
        rhoRR[1:] = 0.5 * (rhoRR[1:] + np.flip(rhoRR[1:]))
        rhoLL[1:] = 0.5 * (rhoLL[1:] + np.flip(rhoLL[1:]))
        rhoLR[1:] = 0.5 * (rhoLR[1:] - np.flip(rhoLR[1:]))

        # normalize
        rhoRR /= dw * np.sum(rhoRR, axis=0)
        rhoLL /= dw * np.sum(rhoLL, axis=0)

        if i % 10 == 0 and x < 1e-8:
            break

    del rhoRR_
    del rhoLL_
    del rhoLR_

    w_re = 4*mu**(2/3)/(2*np.pi)
    sel2 = np.logical_and(t>=0, t*w_re<10*np.pi)
    if mu == 0:
        sel2 = np.logical_and(t>=0, t<np.max(t)/10)
    t = t[sel2]
    GRRg_t = (1/np.sqrt(N) * fft_(-1j * ( 1 - nf) * rhoRR))[sel2]
    GLLg_t = (1/np.sqrt(N) * fft_(-1j * ( 1 - nf) * rhoLL))[sel2]
    GLRg_t = (1/np.sqrt(N) * fft_(-1j * ( 1 - nf) * rhoLR))[sel2]

    sel = np.logical_and(w>=0, w<100*mu**(2/3))
    if mu == 0:
        sel = np.logical_and(w>=0, w<10)
    res = {
        'N': N,
        'wmax': wmax,
        'w': w[sel],
        'T': T,
        'eta_conv': eta_conv,
        'temp': temp,
        'beta': beta,
        'J': J,
        'mu': mu,
        'eta': eta,
        'rhoRR': rhoRR[sel],
        'rhoLL': rhoLL[sel],
        'rhoLR': rhoLR[sel],
        'iterations': i,
        'convergence_x': x,
        'convergence_xx': xx,
        'alpha': alpha,
        't': t,
        'GRRg_t': GRRg_t[::15],
        'GLLg_t': GLLg_t[::15],
        'GLRg_t': GLRg_t[::15]
    }

    f1 = open(f'{task_ID}.pickle', 'ab')
    pickle.dump(res, f1)
    f1.close()

