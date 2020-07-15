import matplotlib as mpl
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import linregress
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d

plt.ion()

if 'res' not in vars():
    res = []

def load():
    files = glob.glob('*.pickle')

    for f in files:
        reader = open(f,'rb')
        try:
            while True:
                a = pickle.load(reader)
                res.append(a)

        except:
            reader.close()

#########################################################
if len(res)==0:
    print('loading data...')
    load()

r = 60
etas = [1,0.99,0.95]
# etas = [1, 0.99, 0.95]
etas = np.sort(etas)
for i in np.arange(len(etas)):
    eta = etas[i]
    mus = np.linspace(0, 0.05, 64)

    phase = np.ones((len(mus), r))*(-1)
    entr = np.ones((len(mus), r-1))*(-1)

    for x in res:
        if x['eta']==eta:

            T = x['TT']
            F = x['FF']
            m = x['mu']
            turn = len(T)//2

            idx = np.where(m==mus)[0][0]

            t1 = T[:turn]
            f1 = F[:turn]
            t2 = T[turn:]
            f2 = F[turn:]

            phase[idx,:] = np.flip(f1) - f2
            entr[idx,:] = -np.diff(f2)/(t1[9]-t1[10])


    """
    plt.figure()
    cl = np.max(np.abs(phase[phase!=-1]))/100
    plt.pcolormesh(t2,mus,phase, cmap='seismic', vmax=cl, vmin=-cl)
    plt.colorbar()
    plt.xlabel('$T$')
    plt.ylabel('$\mu$')
    plt.title(f'$\eta={eta}$')
    plt.xlim(0,0.01)
    # plt.savefig(f'00_eta_{i}.pdf')
    """
    """
    plt.figure()
    cl = np.max(np.abs(entr[entr!=-1]))/100
    plt.pcolormesh(t2[:-1],mus,entr, cmap='seismic', vmax=cl, vmin=-cl)
    plt.colorbar()
    plt.xlabel('$T$')
    plt.ylabel('$\mu$')
    plt.title(f'Entropy for $\eta={eta}$')
    """




"""
for i in np.arange(len(res)):
    x = res[i]
    T = x['TT']
    F = x['FF']
    plt.figure()
    plt.subplot(131)
    eta = x['eta']
    mu = x['mu']
    plt.title(f'Free energy, $\eta={eta}, \mu={np.round(mu,3)}$')
    plt.plot(T,F)
    plt.subplot(132)
    plt.title(f'Hysteresis $F_L - F_R$')
    plt.plot(T[:turn],F[:turn]-np.flip(F[turn:]))

    plt.subplot(133)
    TR = T[turn:]
    FR = F[turn:]
    plt.plot(TR[:-1], np.diff(FR))
"""

plt.figure()
for k in [0, 10, 20, 30, 40, 50, 60]:
    plt.plot(t2[:-1],entr[k,:], label=f'$\mu={np.round(mus[k],3)}$')
    plt.ylim((0, 0.8))
    plt.xlabel(f'$T$')
    plt.ylabel(f'$S=-\partial F/\partial T$')
    plt.legend()
    plt.title(f'$\eta=1$')
    plt.tight_layout()