import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import minimize

N = 32

mus = [0, 0.2, 0.4, 0.6]
etas = np.linspace(0,2,64)

file_or = glob.glob('data/*real.txt')
file_oi = glob.glob('data/*imag.txt')
file_en = glob.glob('data/*energies.txt')


def extract_params(string):
    string = string.strip('data\\')
    string = string.strip('_overlap_imag.txt')
    string = string.strip('_overlap_real.txt')
    string = string.strip('_energies.txt')
    string = string.replace('_','.')
    string = string.replace('eta','n')
    string = string.replace('mu','')

    string = string.split('n')
    N = int(string[0])
    eta = float(string[1])
    mu = float(string[2])

    return N, eta, mu


mus = np.linspace(0,1,50)
etas = np.sort(np.concatenate([np.linspace(0,2,64),[1]]))


a=np.loadtxt(file_en[0],skiprows=4)
n_levels = len(a)

spectrum = np.zeros((n_levels, len(mus), len(etas)))

for f in file_en:
    levels=np.loadtxt(f,skiprows=4)
    N, eta, mu = extract_params(f)
    i_eta = np.where(np.abs(etas-eta)<1e-5)[0][0]
    i_mu = np.where(np.abs(mus-mu)<1e-5)[0][0]

    spectrum[:,i_mu,i_eta] = levels

# %% plot spectrum
for i in np.arange(19):
    plt.figure()
    plt.plot(etas,spectrum[::2,i,:].T,c='black')
    plt.axvline(1,c='black',lw=0.1)
    plt.title(f'$\mu={str(np.round(mus[i],2))}$')
    plt.savefig(f'spectrum_{i}.pdf')

#%% line not straight
i = 18
plt.figure()
plt.plot(etas,spectrum[::2,i,:].T,c='black')
plt.axvline(1,c='black',lw=0.1)
plt.title(f'$\mu={str(np.round(mus[i],2))}$')
plt.plot([0,2],[np.min(spectrum[:,i,0])]*2)
plt.ylim((-3.36,-3.34))
plt.savefig(f'spectrum_{i}_zoom.pdf')

# %%

fidelity = np.zeros((len(mus), len(etas)))
betas = np.zeros((len(mus), len(etas)))
ev_single_syk = np.loadtxt(f'{N}n_ev_syk.txt', skiprows=4)

for f in file_or:
    N, eta, mu = extract_params(f)
    overlap_real = np.loadtxt(f, dtype=float, delimiter=', ', skiprows=4)
    overlap_imag = np.loadtxt(f.replace('real', 'imag'), dtype=float, delimiter=', ', skiprows=4)
    overlap = overlap_real + 1j * overlap_imag
    overlapd = np.diag(np.abs(overlap))
    overlap_wo_diagonal = overlap - np.diag(np.diag(overlap))

    # plt.figure()
    # plt.imshow(np.abs(overlap.imag))
    # plt.imshow(np.abs(overlap_wo_diagonal))
    # plt.xlim((0,100))
    # plt.ylim((0,100))
    # plt.colorbar()
    # plt.title(f'$N={N}, \mu={str(np.round(mus[i],2))}, \eta={str(np.round(eta,2))}$')

    def n_overlap(beta):
        Z = np.sqrt( np.sum( np.exp(-beta * ev_single_syk)))
        return -np.sum(np.exp(-beta * ev_single_syk/2) * overlapd) / Z

    N, eta, mu = extract_params(f)
    i_eta = np.where(np.abs(etas-eta)<1e-5)[0][0]
    i_mu = np.where(np.abs(mus-mu)<1e-5)[0][0]

    res = minimize(n_overlap, 10, bounds=((1e-6,1e6),))
    beta_fit = res.x[0]
    fid = -n_overlap(beta_fit)

    fidelity[i_mu, i_eta] = fid
    betas[i_mu, i_eta] = beta_fit

print('finito')
# %% plot overlap
for i in [0,30,31,32,33,34,35,36,37,38,39]:
    plt.plot(mus,fidelity[:,i],'.-', label=f'$\eta={str(np.round(etas[i],2))}$')
plt.xlim((0,0.35))
plt.legend()
plt.xlabel('$\mu$')
plt.ylabel('Overlap')
plt.savefig('overlap.pdf')

# %% plot mu-beta relation
for i in [0,30,31,32,30,31,32,33,34,35,36,37,38,39]:
    high_overlap = fidelity[:,i]>0.8
    plt.plot(mus[high_overlap],1/betas[high_overlap,i],'.-', label=f'$\eta={str(np.round(etas[i],2))}$')
plt.xlim((0,0.55))
plt.legend(loc='upper right')
plt.xlabel('$\mu$')
plt.ylabel(r'$1/\beta$')
# plt.ylim((0,50))
plt.savefig('mu_beta_relation.pdf')

#%% plot phase diagram
Y, X = np.meshgrid(mus[:-1],etas)
phase = np.diff(fidelity,axis=0)
phase[phase<0] = 0
plt.pcolormesh(X.T,Y.T,phase)
plt.xlim((0,2))
plt.ylim((0,0.5))
plt.xlabel(f'$\eta$')
plt.ylabel(f'$\mu$')


[eeta, mmu] = np.loadtxt('wormhole_mu_eta16.dat')
plt.plot(eeta, mmu, '.', label='ED16')
plt.legend()
plt.savefig('ed16_32.pdf')