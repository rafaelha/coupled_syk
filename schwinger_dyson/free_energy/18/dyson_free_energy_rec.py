# %matplotlib qt

import numpy as np
import matplotlib.pyplot as plt
# mpl.use('Agg')
from scipy.fftpack import fft, ifft
from scipy.fftpack import fftfreq
from scipy.fftpack import fftshift, ifftshift
import time
import sys
import pickle
from scipy.stats import linregress

mus = np.linspace(0, 0.15, 64)
etas = [1]

idx = int(sys.argv[1])
mu = mus[idx]

plot = False

lin = np.linspace

for k in np.arange(len(etas)):
    eta = etas[k]

    FF = []
    FFleft = []
    FFright = []
    TT = []
    TTleft = []
    TTright = []

    def handle_close(evt):
        return
        # sys.exit()

    if plot:
        fig = plt.figure('free energy', figsize=(15, 15))
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(2500,100,1100, 545)

        fig.canvas.mpl_connect('close_event', handle_close)

    def ft(t):
        return -0.5*tstep*(ifft(t))*N

    def ift(w):
        return -fft(w)/beta


    def diff(a, b):
        return np.linalg.norm(np.log(np.abs(a[z]))-np.log(np.abs(b[z])))/N

    def freeenergy_old(GLLiw, GRRiw, GLRiw, SLLiw, SRRiw, SLRiw, freqi):
        f = ((1+eta)/(1-eta))**2
        FE = np.log(2.0) + \
            (1.0/2.0)*np.sum(np.log((1.0 + SLLiw * SRRiw/(-1.0j*freqi/tstep)**2 +
                                    (SLLiw + SRRiw)/(-1.0j*freqi/tstep) -
                                    (SLRiw/(freqi/tstep))**2))) \
            + 3.0/4.0 * np.sum(0.5 * (SLLiw * GLLiw) * f
                            + 0.5 * (SRRiw * GRRiw) / f - (SLRiw * GLRiw))
        return -(1/beta) * FE

    def freeenergy(GLLiw, GRRiw, GLRiw, SLLiw, SRRiw, SLRiw, iw, imu):
        FE = np.log(2.0) + \
            1.0/2.0 * np.sum(np.log( (1- SRRiw/iw)*(1-SLLiw/iw) - (imu - SLRiw)*(imu - SLRiw)/(iw*iw*(-1)) )) + \
            3.0/4.0 * np.sum(0.5 * SLLiw * GLLiw + 0.5 * SRRiw * GRRiw - SLRiw * GLRiw)

        return -(1/beta) * FE

    J = 1  # intra-SYK coupling
    N = 2**18  # number of frequency points (points in Patel's code is N/2)
    total_iteration = 900  # number of iterations for each run
    x = 0.03  # mixing of consecutive iterations. (0 means no mixing)

    # Initialize G
    ar = N/2 - 0.5 - np.arange(N)
    GLLt0 = 0.5*np.sign(ar)
    GRRt0 = GLLt0
    GLRt0 = np.ones(N) * 0.005*1.0j
    GLRt0[int(N/4):int(3*N/4)] = -0.005*1.0j

    GLLw = np.zeros(N, dtype='complex')
    GRRw = np.zeros(N, dtype='complex')
    GLRw = np.zeros(N, dtype='complex')


    T = 0.08

    lT = 0.08
    lF = 0.02

    smax = 0.04
    savg = 0.015
    Tmin = 5e-5

    Tend = 0.08

    dT = 0.001

    sign = -1
    TURN = 0

    GO = True

    fstep = 2*np.pi/N
    freq = np.zeros(N-1)
    for i in range(int(N/2)):
        # First all the positive frequency in increasing order
        freq[i] = (i+1)*fstep
    # Then negative frequency in increasing order so as to compactify w on a circle
    for i in range(int(N/2), N-1):
        freq[i] = (i+1-N)*fstep

    temp = np.linspace(0.04,0.00001,200)
    ttemp = np.concatenate([temp, np.flip(temp)])
    for T in ttemp:
        beta = 1/T
        omegacutoff = N*(np.pi)/beta  # Max cut-off frequency,
        # NOTE: this is fixed by inverse temperature beta

        tstep = 2*beta/N
        tau = np.arange(N) * tstep - beta
        z = np.logical_and(tau > 4, tau < 30)
        t = tau[z]


        dLL = np.zeros(total_iteration)
        dRR = np.zeros(total_iteration)
        dLR = np.zeros(total_iteration)

        S = np.zeros(total_iteration)

        # set initial conditions
        GLLt = np.copy(GLLt0)
        GRRt = np.copy(GRRt0)
        GLRt = np.copy(GLRt0)

        for i in np.arange(total_iteration):
            # self consistent iteration
            SRRt = (1-eta)**2 * 2*(J**2)*GRRt*GRRt*GRRt
            SLLt = (1+eta)**2 * 2*(J**2)*GLLt*GLLt*GLLt
            SLRt = (1-eta**2) * 2*(J**2)*GLRt*GLRt*GLRt

            SRRw = ft(SRRt)
            SLLw = ft(SLLt)
            SLRw = ft(SLRt)

            iw = 1.0j*freq[0:N-1:2] / tstep
            imu = 1.0j*mu

            Dw = (iw - SRRw[1:N:2])*(iw - SLLw[1:N:2]) + \
                (imu - SLRw[1:N:2])*(imu - SLRw[1:N:2])
            GLLw[1:N:2] = (iw - SRRw[1:N:2]) / Dw
            GRRw[1:N:2] = (iw - SLLw[1:N:2]) / Dw
            GLRw[1:N:2] = -(imu - SLRw[1:N:2]) / Dw

            # store result of iteration
            GRRtn = ift(GRRw)
            GLLtn = ift(GLLw)
            GLRtn = ift(GLRw)

            # compare new iteration to old one
            dRR[i] = diff(GRRt, GRRtn)
            dLL[i] = diff(GLLt, GLLtn)
            dLR[i] = diff(GLRt, GLRtn)

            # update with newest iteration
            GRRt = (1-x)*GRRt + x*GRRtn
            GLLt = (1-x)*GLLt + x*GLLtn
            GLRt = (1-x)*GLRt + x*GLRtn

            # symmetrize
            GRRt[:N//2] = 0.5 * (GRRt[:N//2] - np.flip(GRRt[N//2:]))
            GRRt[N//2:] = -np.flip(GRRt[:N//2])

            GLLt[:N//2] = 0.5 * (GLLt[:N//2] - np.flip(GLLt[N//2:]))
            GLLt[N//2:] = -np.flip(GLLt[:N//2])

            GLRt[:N//2] = 0.5 * (GLRt[:N//2] + np.flip(GLRt[N//2:]))
            GLRt[N//2:] = np.flip(GLRt[:N//2])

            print(i)

        F = freeenergy(GLLw[1:N:2], GRRw[1:N:2], GLRw[1:N:2],
                    SLLw[1:N:2], SRRw[1:N:2], SLRw[1:N:2], iw, imu)
        FF.append(F)
        TT.append(T)

        if plot:
            plt.figure('free energy')
            plt.clf()
            plt.plot(TT,FF,'.')
            plt.pause(0.01)


    res = {'eta': eta,
            'mu': mu,
            'J': J,
            'FF': FF,
            'TT': TT,
            'total_iteration': total_iteration,
            'x': x}


    f1 = open(str(idx) + '.pickle', 'ab')
    pickle.dump(res, f1)
    f1.close()
