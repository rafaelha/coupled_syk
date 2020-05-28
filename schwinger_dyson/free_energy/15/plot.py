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

eta = 0
mus = np.linspace(0,0.2,64)

r = 200
phase = np.zeros((len(mus), r))

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

        t = np.linspace(0.0007,0.079, r)
        i1 = interp1d(t1,f1)
        i2 = interp1d(t2,f2)

        phase[idx,:] = i1(t)-i2(t)


plt.figure()
cl = np.max(np.abs(phase))
plt.pcolormesh(t,mus,phase, cmap='seismic', vmax=cl, vmin=-cl)
plt.colorbar()
plt.xlabel('$T$')
plt.ylabel('$\mu$')
plt.title(f'$\eta={eta}$')