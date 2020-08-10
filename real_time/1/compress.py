#%%
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import scipy.fft
from scipy.fftpack import fftshift, ifftshift

def fft_(x):
    return fftshift(scipy.fft.fft(fftshift(x)))

def fft(x):
    return fftshift(scipy.fft.ifft(fftshift(x))) * len(x)

files = glob.glob(f'*.pickle')

for f in files[0:1]:
    f = '296453_0.pickle'
    print(f)
    reader = open(f,'rb')
    data = pickle.load(reader)

    #%%
    mu = data['mu']
    print('mu=',mu)
    # w = data['w']
    N = data['N']
    T = data['T']
    w = np.arange(-N/2, N/2)  * 2 * np.pi / T
    t = np.arange(-N/2, N/2) / N * T
    x = data['convergence_x']
    nf = data['nf']
    wmax = data['wmax']
    eta_conv = data['eta_conv']
    temp = data['temp']
    beta = data['beta']
    J = data['J']
    eta = data['eta']
    i = data['iterations']
    alpha = data['alpha']

    w_re = 4*mu**(2/3)/(2*np.pi)
    sel2 = np.logical_and(t>0, t*w_re<10*np.pi)
    if mu == 0:
        sel2 = np.logical_and(t>0, t<np.max(t)/10)

    rhoRR = data['rhoRR']
    rhoLL = data['rhoLL']
    rhoLR = data['rhoLR']
    GRRg_t = (1/np.sqrt(2*N) * fft_(-1j * ( 1 - nf) * rhoRR))[sel2][::10]
    GLLg_t = (1/np.sqrt(2*N) * fft_(-1j * ( 1 - nf) * rhoLL))[sel2][::10]
    GLRg_t = (1/np.sqrt(2*N) * fft_(-1j * ( 1 - nf) * rhoLR))[sel2][::10]
    t = t[sel2][::10]

    sel = np.logical_and(w>=0, w<100*mu**(2/3))
    if mu == 0:
        sel = np.logical_and(w>=0, w<10)

    w = w[sel]

    rhoRR = data['rhoRR'][sel]
    rhoLL = data['rhoLL'][sel]
    rhoLR = data['rhoLR'][sel]

    print(rhoRR[0:10])


    """
    plt.figure(1)
    plt.clf()
    wmu = w/(mu**(2/3))
    plt.plot(wmu,rhoRR, label=r'$\rho_{RR}$')
    plt.plot(wmu,rhoLL, label=r'$\rho_{LL}$')
    plt.plot(wmu,rhoLR.imag, label=r'$\rho_{LR}$')
    plt.title(f'$J={J}, \mu={mu}, \eta={eta}, \\beta={beta}$')
    plt.legend()
    # l = 60
    # plt.xlim((-l,l))
    plt.xlabel('$\omega$')
    """


    """
    plt.figure('T')
    plt.clf()
    plt.plot(t*w_re, np.abs(GRRg_t), label=r'$|T_{RR}(t)|$')
    plt.plot(t*w_re, np.abs(GLLg_t), label=r'$|T_{LL}(t)|$')
    plt.plot(t*w_re, np.abs(GLRg_t), label=r'$|T_{LR}(t)|$')
    plt.title(f'$\omega_m={str(wmax)}, N={N}$')
    plt.xlabel('$t$')
    plt.legend()
    plt.tight_layout()
    """


    res = {
        'N': N,
        'wmax': wmax,
        'w': w,
        'T': T,
        'eta_conv': eta_conv,
        'temp': temp,
        'beta': beta,
        'J': J,
        'mu': mu,
        'eta': eta,
        'rhoRR': rhoRR,
        'rhoLL': rhoLL,
        'rhoLR': rhoLR,
        'iterations': i,
        'convergence_x': x,
        'alpha': alpha,
        't': t,
        'GRRg_T': GRRg_t,
        'GLLg_T': GLLg_t,
        'GLRg_T': GLRg_t
    }

    f1 = open(f+'2', 'ab')
    pickle.dump(res, f1)
    f1.close()
