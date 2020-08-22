# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.fftpack import fftfreq
from scipy.fftpack import fftshift, ifftshift
import time
import sys
import pickle


def solveSD(mu, eta, temp = 0.00086603, N=2**17, total_iteration = 4000):
    N = 2**17  # number of frequency points (points in Patel's code is N/2)
    beta = 1/temp  # inverse temperature
    J = 1

    omegacutoff = N*(np.pi)/beta  # Max cut-off frequency,
    # NOTE: this is fixed by inverse temperature beta

    x = 0.03  # mixing of consecutive iterations. (0 means no mixing)

    tstep = 2*beta/N
    fstep = 2*np.pi/N
    freq = np.zeros(N-1)
    for i in range(int(N/2)):
        # First all the positive frequency in increasing order
        freq[i] = (i+1)*fstep
    # Then negative frequency in increasing order so as to compactify w on a circle
    for i in range(int(N/2), N-1):
        freq[i] = (i+1-N)*fstep
    tau = np.arange(N) * tstep - beta
    z = np.logical_and(tau > 4, tau < 30)
    t = tau[z]

    BREAK = False

    # True if exponent is e^{-iwt} in FT from time to frequencies ft(t)
    ft_of_t_neg_iw = False

    def ft(t):
        return -0.5*tstep*(ifft(t))*N

    def ift(w):
        return -fft(w)/beta

    def diff(a, b):
        return np.linalg.norm(np.log(np.abs(a[z]))-np.log(np.abs(b[z])))/N

    def freeenergy(GLLiw, GLRiw, SLLiw, SLRiw, freqi):
        FE = np.log(2.0) \
            + (1.0/2.0)*np.sum(np.log(np.power(1.0-SLLiw/(1.0j*freqi/tstep), 2)
                                    + np.power((1.0j*mu-SLRiw)/(1.0j*freqi/tstep), 2))) \
            + 3.0/4.0 * np.sum((SLLiw * GLLiw) - (SLRiw * GLRiw))
        FE = 0
        # J**2/2 * (1-eta**2) * np.sum(GLRt
        return -(N/beta)*FE

    # Initialize G
    ar = N/2 - 0.5 - np.arange(N)
    GLLt = 0.5*np.sign(ar)

    GRRt = 2*GLLt

    GLLw = ft(GLLt)
    GRRw = ft(GRRt)
    SLLt = np.zeros(N, dtype='complex')
    SLLw = np.zeros(N, dtype='complex')
    SRRt = np.zeros(N, dtype='complex')
    SRRw = np.zeros(N, dtype='complex')

    GLRt = np.ones(N) * 0.005*1.0j
    GLRt[int(N/4):int(3*N/4)] = -0.005*1.0j

    GLRw = ft(GLRt)
    SLRt = np.zeros(N, dtype='complex')
    SLRw = np.zeros(N, dtype='complex')

    S = np.zeros(total_iteration)
    dLL = []
    dRR = []
    dLR = []

    for i in np.arange(total_iteration):
        # self consistent iteration
        SRRt = (1-eta)**2 * 2*(J**2)*GRRt*GRRt*GRRt #changed sign
        SLLt = (1+eta)**2 * 2*(J**2)*GLLt*GLLt*GLLt #changed sign
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

        # compute free energy
        #S[i] = freeenergy(GLLw[1:N:2], GLRw[1:N:2], SLLw[1:N:2], SLRw[1:N:2], freq[0:N-1:2])

        # compare new iteration to old one
        if i%10==0:
            print(i, 'of', total_iteration)
            dRR.append(diff(GRRt, GRRtn))
            dLL.append(diff(GLLt, GLLtn))
            dLR.append(diff(GLRt, GLRtn))

        # update with newest iteration
        GRRt = (1-x)*GRRt + x*GRRtn
        GLLt = (1-x)*GLLt + x*GLLtn
        GLRt = (1-x)*GLRt + x*GLRtn

        GRRt[:N//2] = 0.5 * (GRRt[:N//2] - np.flip(GRRt[N//2:]))
        GRRt[N//2:] = -np.flip(GRRt[:N//2])

        GLLt[:N//2] = 0.5 * (GLLt[:N//2] - np.flip(GLLt[N//2:]))
        GLLt[N//2:] = -np.flip(GLLt[:N//2])

        GLRt[:N//2] = 0.5 * (GLRt[:N//2] + np.flip(GLRt[N//2:]))
        GLRt[N//2:] = np.flip(GLRt[:N//2])

        if len(dLL) > 20 and np.mean(np.abs(np.diff(dLL)[-10:])) < 1e-15:
            break

    dd = 1
    res = {'eta': eta,
        'beta': beta,
        'mu': mu,
        'J': J,
        'tau': tau[N//2:3*N//4:dd],
        'GRRt': GRRt[N//2:3*N//4:dd],
        'GLLt': GLLt[N//2:3*N//4:dd],
        'GLRt': GLRt[N//2:3*N//4:dd],
        'w': freq[0:N-1:2][::dd] / tstep,
        'GRRw': GRRw[1:N:2][::dd],
        'GLLw': GLLw[1:N:2][::dd],
        'GLRw': GLRw[1:N:2][::dd],
        'dRR': dRR,
        'dLL': dLL,
        'dLR': dLR,
        'total_iterations': total_iteration,
        'x': x,
        'N': N}
    return res
#%%
from scipy.fftpack import fftshift as fs
def extract(r):
    return r['tau'], r['GRRt'], r['GLLt'], r['GLRt'],\
         fs(r['w']), fs(r['GRRw']), fs(r['GLLw']), fs(r['GLRw']), r, r['eta'], r['mu']

# %%
eta = 1.2
eta_ = 1.1
mu = 0.1
mu_ = ( abs(1-eta_**2)/abs(1-eta**2) )**(1/4) * mu
# mu_ = mu
res = solveSD(mu, eta, total_iteration=2000)
res_ = solveSD(mu_, eta_, total_iteration=2000)

#%%
plt.figure(1)
plt.clf()
tau, GRRt, GLLt, GLRt, w, GRRw, GLLw, GLRw, data, eta, mu = extract(res)
tau, GRRt_, GLLt_, GLRt_, w, GRRw_, GLLw_, GLRw_, data_, eta_, mu_ = extract(res_)

plt.loglog(tau, np.abs(GRRt_))
plt.plot(tau, np.abs(GRRt * np.sqrt((1-eta)/(1-eta_))), 'r--')
plt.plot(tau,1/np.sqrt(abs(1-eta_)) * 1/(8*np.pi)**(1/4) / np.abs(tau)**(1/2), 'k--')

plt.loglog(tau, np.abs(GLLt_))
plt.plot(tau, np.abs(GLLt * np.sqrt(abs(1+eta)/abs(1+eta_))), 'r--')
plt.plot(tau,1/np.sqrt(abs(1+eta_)) * 1/(8*np.pi)**(1/4) / np.abs(tau)**(1/2), 'k--')

plt.xlabel('$t$')
plt.ylabel('$G$')

#w
plt.figure(2)
plt.clf()
plt.loglog(w, np.abs(GRRw_))
plt.plot(w, np.abs(GRRw * np.sqrt((1-eta)/(1-eta_))), 'r--')
plt.plot(w,abs(1/np.sqrt(abs(1-eta_)) * 1/(8*np.pi)**(1/4) \
    * np.sqrt(np.pi) * (1j*w)**(1/2)), 'k--')

plt.plot(w, np.abs(GLLw_))
plt.plot(w, np.abs(GLLw * np.sqrt(abs(1+eta)/abs(1+eta_))), 'r--')

plt.xlabel('$\omega_n$')
plt.ylabel('$G$')

# plt.loglog(tau, np.abs(GLRt_))
#