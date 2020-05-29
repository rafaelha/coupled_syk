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


etas = [0, 1, 0.99, 0.98, 1.1, 0.97, 0.95, 0.9, 0.85, 0.8]
etas = np.sort(etas)
for i in np.arange(len(etas)):
    eta = etas[i]
    mus = np.linspace(0,0.2,64)

    r = 500
    phase = np.ones((len(mus), r))*-1

    for x in res:
        if x['eta']==eta:

            T = x['TT']
            F = x['FF']
            turn = x['TURN']
            m = x['mu']

            idx = np.where(m==mus)[0][0]

            t1 = T[:turn]
            f1 = F[:turn]
            t2 = T[turn:]
            f2 = F[turn:]

            t = np.linspace(0.0025,0.079, r)
            i1 = interp1d(t1,f1)
            i2 = interp1d(t2,f2)

            phase[idx,:] = i1(t)-i2(t)


    plt.figure()
    cl = np.max(np.abs(phase[phase!=-1]))
    plt.pcolormesh(t,mus,phase, cmap='seismic', vmax=cl, vmin=-cl)
    plt.colorbar()
    plt.xlabel('$T$')
    plt.ylabel('$\mu$')
    plt.title(f'$\eta={eta}$')
    plt.xlim(0,0.08)
    plt.savefig(f'00_eta_{i}.pdf')