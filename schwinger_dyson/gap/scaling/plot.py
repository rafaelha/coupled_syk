# %%
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
    print('loading data...')
    mus, etas, gaps, result = load()

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

    """
    if plot:
        plt.plot(etas, gaps, '.-', label=r'$\mu=$'+str(np.round(mu,2)))
        plt.xlabel('$\eta$')
        plt.ylabel('$E_{gap}$')
        plt.legend()
        xx = np.linspace(x[0], popt[1], 1000)
        plt.plot(xx, func(xx, *popt), 'k')
        #plt.title(r'$\eta_c=$'+str(popt[1]))
        plt.pause(0.01)
    """

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
            plot = False
        n_crit[i] = find_eta(etas_, gaps_, mu, plot=plot)
        i += 1

    plt.figure('final-res')
    plt.clf()
    plt.plot(n_crit, mu_unique, '.', label='large-N')

    # [eeta, mmu] = np.loadtxt('wormhole_mu_eta.dat')
    # plt.plot(eeta, mmu, '.', label='ED')
    # plt.xlabel('$\eta_{c}$')
    # plt.ylabel('$\mu$')
    # plt.legend()


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
# plt.savefig('../figs/eta_gap.pdf')
plt.figure('final-res')
# plt.savefig('../figs/comp.pdf')


"""
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
"""




# %%

i=0
for mu in mu_unique:
    etas_ = etas[mus == mu]
    gaps_ = gaps[mus == mu]
    idx = np.argsort(etas_)
    gaps_ = gaps_[idx]
    etas_ = etas_[idx]
    plot = False
    if i%10 == 0:
        plot=True
        plot = False
    n_crit[i] = find_eta(etas_, gaps_, mu, plot=plot)
    i += 1

plt.figure('final-res')
plt.clf()

etaprime = n_crit[-1]
alpha = mu_unique[-1]

x = np.linspace(1,2,1000)

plt.plot(n_crit, mu_unique, '.', label='large-N')
plt.plot(x, alpha*((1-x**2)/(1-etaprime**2))**(1/1.6))

# [eeta, mmu] = np.loadtxt('wormhole_mu_eta.dat')
# plt.plot(eeta, mmu, '.', label='ED')
# plt.xlabel('$\eta_{c}$')
# plt.ylabel('$\mu$')
# plt.legend()

# %%
def select(eta, mu):
    eta = etas[np.argmin(np.abs(etas-eta))]
    mu = mus[np.argmin(np.abs(mus-mu))]
    r = result[(np.logical_and(etas==eta,mus==mu)).argmax()]

    return r['tau'], r['GRRt'], r['GLLt'], r['GLRt'],\
         r['w'], r['GRRw'], r['GLLw'], r['GLRw'], r, r['eta'], r['mu']


num = 4
eta = 1.5 # use this solution to model the other one
mu = 0.3
eta_ = 1.05 # model this eta
plt.figure(1)
plt.clf()

print('eta:',eta, ', mu: ', mu)
tau, GRRt, GLLt, GLRt, w, GRRw, GLLw, GLRw, data, eta, mu = select(eta,mu)
print('closest eta:',eta, ', closest mu: ', mu)

mu_ = ( abs(1-eta_**2)/abs(1-eta**2) )**(1/4) * mu
# mu_ = mu

print('eta_:',eta_, ', mu_: ', mu_)
tau, GRRt_, GLLt_, GLRt_, w, GRRw_, GLLw_, GLRw_, data_, eta_, mu_ = select(eta_,mu_)
print('closest eta_:',eta_, ', closest mu_: ', mu_)

plt.loglog(tau, np.abs(GRRt_), 'blue', label='$G_{RR}(\\tilde{\eta},\\tilde{\mu})$')
plt.plot(tau, np.abs(GRRt * np.sqrt(abs(1-eta)/abs(1-eta_))), 'r--', label='$G_{RR}(\eta,\mu)\sqrt{\\frac{1-\eta}{1-\\tilde{\eta}}}$')
plt.plot(tau,1/np.sqrt(abs(1-eta_)) * 1/(8*np.pi)**(1/4) / np.abs(tau)**(1/2), 'k--', label='SYK analyical')


plt.plot(tau, np.abs(GLLt_), 'orange', label='$G_{LL}(\\tilde{\eta},\\tilde{\mu})$')
plt.plot(tau, np.abs(GLLt * np.sqrt(abs(1+eta)/abs(1+eta_))), 'g--', label='$G_{LL}(\eta,\mu)\sqrt{\\frac{1+\eta}{1+\\tilde{\eta}}}$')
plt.plot(tau,1/np.sqrt(abs(1+eta_)) * 1/(8*np.pi)**(1/4) / np.abs(tau)**(1/2), 'k--')

# plt.plot(tau, np.abs(GLRt_))
# plt.plot(tau, np.abs(GLRt * ((abs(1-eta**2))/(abs(1-eta_**2)))**(1/4)), 'r--')
plt.xlabel('$t$')
plt.ylabel('$G(\\tau)$')
plt.title(f'$(\eta={np.round(eta,3)},\mu={mu})  - ' + '(\\tilde{\\eta}='+str(np.round(eta_,3))+',\\tilde{\mu}='+str(mu_)+')$')
plt.legend()
plt.savefig(f'{num}a.pdf')
plt.tight_layout()
#w
plt.figure(2)
plt.clf()
plt.loglog(w, np.abs(GRRw_),'blue')
plt.loglog(w, np.abs(GLLw_),'orange')
plt.plot(w, np.abs(GRRw * np.sqrt(abs((1-eta))/abs(1-eta_))), 'r--')
# plt.plot(w,abs(1/np.sqrt(abs(1-eta_)) * 1/(8*np.pi)**(1/4) \
    # * np.sqrt(np.pi) * (1j*w)**(1/2)), 'k--') 
# plt.plot(w, np.abs(GLLw_))
# plt.plot(w, np.real(GLLw_))
# plt.plot(w, np.imag(GLLw_))
plt.plot(w, np.abs(GLLw * np.sqrt(abs(1+eta)/abs(1+eta_))), 'g--')

plt.xlabel('$\omega_n$')
plt.ylabel('$|G(\omega_n)|$')
plt.tight_layout()
plt.savefig(f'{num}b.pdf')

#%%

# plt.plot(w, np.abs(GLLw_))
plt.plot(w, np.real(GLLw_))
plt.plot(w, np.imag(GLLw_))
plt.xlim((-30,30))

#%%

tau, GRRt, GLLt, GLRt, w, GRRw, GLLw, GLRw, data, eta, mu = select(0.3,0.2)
print(eta,mu)

plt.plot(w,np.real(GLLw/GLRw), label='Re $G_{LL}/G_{LR}$')
plt.plot(w,np.imag(GLLw/GLRw), label='Im $G_{LL}/G_{LR}$')
plt.xlim((-5,5))
a = np.max(np.abs(GLLw/GLRw)[np.abs(w)<5])
plt.ylim((-a,a))

plt.plot(w,-w/mu, 'k--', label='$-\omega/\mu$')
plt.legend()
plt.xlabel('$\omega$')
plt.title(f'$\eta={eta}, \mu={mu}$')

plt.figure()
plt.plot(w,GLLw.real, label='Re $G_{LL}$')
plt.plot(w,GLLw.imag, label='Im $G_{LL}$')
plt.ylabel('$G_{LL}(\omega)$')
plt.xlabel('$\omega$')
plt.title(f'$\eta={eta}, \mu={mu}$')
plt.legend()

plt.figure()
plt.plot(w,GLRw.real, label='Re $G_{LR}$')
plt.plot(w,GLRw.imag, label='Im $G_{LR}$')
plt.ylabel('$G_{LR}(\omega)$')
plt.xlabel('$\omega$')
plt.title(f'$\eta={eta}, \mu={mu}$')
plt.legend()
