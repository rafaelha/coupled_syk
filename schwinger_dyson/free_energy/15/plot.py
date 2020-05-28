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

                gap = fit_all(a, plot=True)
                gaps.append(gap)
                mus.append(a['mu'])
                etas.append(a['eta'])
                Js.append(a['J'])

        except:
            reader.close()
    return np.array(mus), np.array(etas), np.abs(gaps)

#########################################################
if 'gaps' not in vars():
    print('loading data...')
    mus, etas, gaps = load()

plt.figure('res')
plt.clf()


mu_unique = np.unique(mus)
eta_unique = np.unique(etas)
n_crit = np.zeros(len(mu_unique))


def func(x, a, b, c):
    return a*np.tanh(c * np.sqrt(b/x - 1))

def find_eta(etas, gaps, mu, plot=False):
    x = etas[gaps>0]
    y = gaps[gaps>0]

    if len(x) == 0:
        return 1

    y = y[x>x[-1]/3*2]
    x = x[x>x[-1]/3*2]

    popt, pcov = cf(func, x, y, p0=[np.max(gaps),x[-1], 1])

    if plot:
        plt.plot(etas, gaps, '.-', label=r'$\mu=$'+str(np.round(mu,2)))
        plt.xlabel('$\eta$')
        plt.ylabel('$E_{gap}$')
        plt.legend()
        xx = np.linspace(x[0], popt[1], 1000)
        plt.plot(xx, func(xx, *popt), 'k')
        #plt.title(r'$\eta_c=$'+str(popt[1]))
        plt.pause(0.01)

    return popt[1]

def plot_eta_crit():
    i = 0
    for mu in mu_unique:
        etas_ = etas[mus == mu]
        gaps_ = gaps[mus == mu]
        idx = np.argsort(etas_)
        gaps_ = gaps_[idx]
        etas_ = etas_[idx]
        plot = False
        if i%10 == 0:
            plot=True
        n_crit[i] = find_eta(etas_, gaps_, mu, plot=plot)
        i += 1

    plt.figure('final-res')
    plt.clf()
    plt.plot(n_crit, mu_unique, '.', label='large-N')

    [eeta, mmu] = np.loadtxt('wormhole_mu_eta.dat')
    plt.plot(eeta, mmu, '.', label='ED')
    plt.xlabel('$\eta_{c}$')
    plt.ylabel('$\mu$')
    plt.legend()


def mu_scaling(eta):
    mus_ = mus[etas == eta]
    gaps_ = gaps[etas == eta]

    idx = np.argsort(mus_)
    mus_ = mus_[idx][1:]
    gaps_ = gaps_[idx][1:]

    mus_ = mus_[gaps_>0]
    gaps_ = gaps_[gaps_>0]

    plt.figure('2')
#plt.clf()
    plt.loglog(mus_, gaps_, '.-')
    plt.title(r'$\eta=$'+str(eta))

    #fit
    """
    mus__ = mus_[mus_<0.1]
    gaps__ = gaps_[mus_<0.1]

    if len(mus__) == 0:
        mus__ = mus_[0:4]
        gaps__ = gaps_[0:4]
    """
    if len(mus_) < 4 or mus_[0]>0.1:
        return 0
    mus__ = mus_[0:4]
    gaps__ = gaps_[0:4]

    print(mus__)

    c = linregress(np.log(mus__), np.log(gaps__))
    xx = np.linspace(mus__[0], mus__[-1], 100)
    plt.loglog(xx, np.exp(c.intercept)*xx**c.slope,'k--')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$E_{gap}$')
    plt.tight_layout()

    return c.slope


plot_eta_crit()
plt.figure('res')
plt.savefig('../figs/eta_gap.pdf')
plt.figure('final-res')
plt.savefig('../figs/comp.pdf')


plt.figure('2')
plt.clf()
v = len(eta_unique)
exponent = np.zeros(v)
for i in np.arange(v):
    eta = eta_unique[i]
    exponent[i] = mu_scaling(eta)

plt.tight_layout()
plt.savefig('../figs/scaling.pdf')

plt.figure('3', figsize=(5,5))
plt.clf()

plt.plot(eta_unique[exponent>0], exponent[exponent>0], '.-')
plt.xlabel(r'$\eta$')
plt.ylabel(r'exponent $\nu$ of $\mu^{\nu}$')
plt.axvline(1)
plt.axhline(2/3)
plt.tight_layout()
plt.savefig('../figs/exponents.pdf')



