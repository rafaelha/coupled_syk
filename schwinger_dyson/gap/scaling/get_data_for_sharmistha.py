# %%
from operator import index
import matplotlib as mpl
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import linregress
from scipy.optimize import curve_fit as cf

def fit(tau, G, plot=False, label=''):
    G = np.log(np.abs(G))
    # throw away all values less than 10^-13
    th = -13
    ts = tau[G>th]
    gs = G[G>th]


    if ~np.any(ts):
        ts = tau
        tss = tau
        gs = G
        gss = G
    else:
        tss = ts[ts>ts[-1]/20]
        gss = gs[ts>ts[-1]/20]


    c = linregress(tss, gss)
    if plot:
        plt.plot(ts, gs, label=label)
        plt.plot(tss, c.intercept+tss*c.slope, '--b')
        if label != '':
            plt.legend()
    return c.slope, c.intercept, c.stderr
def fit_all(a, plot=False, d=1.1):
    if plot:
        plt.figure('fit')
        plt.clf()
    a1, a0, aa =  fit(a['tau'], a['GLLt'], plot=plot, label='GLLt')
    b1, b0, bb =  fit(a['tau'], a['GRRt'], plot=plot, label='GRRt')
    c1, c0, ca =  fit(a['tau'], a['GLRt'], plot=plot, label='GLRt')
    if plot:
        plt.title(r'$\mu=$'+str(a['mu'])+r', $\eta=$'+str(a['eta']))
        plt.pause(0.01)

    if max(np.abs(a1/b1),np.abs(b1/a1)) > d \
       or max(np.abs(a1/c1),np.abs(b1/c1)) > d \
       or max(np.abs(c1/b1),np.abs(b1/c1)) > d:
        gap = 0
    else:
        gap = (a1 + b1 + c1) / 3
    return gap


def load():
    files = glob.glob('*.pickle')

    plt.ion()

    res = []
    mus = []
    etas = []
    gaps = []
    Js = []
    #for f in files:
    for f in files:
        reader = open(f,'rb')
        try:
            while True:
                a = pickle.load(reader)
                res.append(a)

                gap = fit_all(a, plot=False)
                gaps.append(gap)
                mus.append(a['mu'])
                etas.append(a['eta'])
                Js.append(a['J'])

        except:
            reader.close()
    return np.array(mus), np.array(etas), np.abs(gaps), res

#########################################################
if 'gaps' not in vars():
    # the function only runs the very first execution of the python script to save you some time (load data only once)
    print('loading data...')
    mus_, etas_, gaps_, result = load()

    mus = np.sort(np.unique(mus_))
    etas = np.sort(np.unique(etas_))

    data = np.zeros((len(mus), len(etas)))
    def indexof(x, x_array):
        return np.where(x==x_array)[0][0]

    for i in np.arange(len(result)):
        data[indexof(mus_[i], mus), indexof(etas_[i],etas)] = gaps_[i]


# mus containes all possible mu values
# etas contains all possible eta values
# data[i,j] contains the gap in a simulation with parameters mus[i] and etas[j]

# plot a single curve for \mu=mus[i]
i = 5
plt.plot(etas, data[i], '.-', label=f'$\mu={mus[i]}$')
plt.legend()
plt.xlabel('$\eta$')
plt.ylabel('$E_{gap}$')
