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

r = 400
etas = [0, 1, 0.9, 0.95, 0.96, 0.97,0.98,0.99,0.995]
etas = np.sort(etas)
for i in np.arange(len(etas)):
    eta = etas[i]
    mus = np.linspace(0, 0.2, 64)

    phase = np.ones((len(mus), r))*(-1)

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


    plt.figure()
    cl = np.max(np.abs(phase[phase!=-1]))
    plt.pcolormesh(t2,mus,phase, cmap='seismic', vmax=cl, vmin=-cl)
    plt.colorbar()
    plt.xlabel('$T$')
    plt.ylabel('$\mu$')
    plt.title(f'$\eta={eta}$')
    #plt.xlim(0,0.04)
    plt.savefig(f'00_eta_{i}.pdf')