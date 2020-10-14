import numpy as np
import matplotlib.pyplot as plt

N = 32

%cd C:\Users\rafae\source\repos\syk_gamma_sparse

overlap_real = np.loadtxt(str(N) + "n_overlap_real.txt", dtype=float, delimiter=', ', skiprows=3)
overlap_imag = np.loadtxt(str(N) + "n_overlap_imag.txt", dtype=float, delimiter=', ', skiprows=3)
ev_syk_cpp = np.loadtxt(str(N) + "n_ev_syk.txt", dtype=float, delimiter=', ', skiprows=3)
HLRgs_real = np.loadtxt(str(N) + "n_HLRgs_real.txt", dtype=float, delimiter=', ', skiprows=3)
HLRgs_imag = np.loadtxt(str(N) + "n_HLRgs_imag.txt", dtype=float, delimiter=', ', skiprows=3)
HLRgs = HLRgs_real + 1j * HLRgs_imag

idx = np.argsort(ev_syk_cpp)
ev_syk_cpp = ev_syk_cpp[idx]

plt.ion()
overlap = overlap_real + 1j * overlap_imag
overlap_wo_diagonal = overlap - np.diag(np.diag(overlap))
plt.imshow(np.abs(overlap_wo_diagonal))
plt.xlabel(r'|n$\rangle$')
plt.ylabel(r'|m$\rangle$')
plt.title(str(N) + " Majoranas, J=1, $\mu=0.5$")
plt.colorbar()

# overlap = np.flip(overlap, axis=0)
# overlap = np.flip(overlap, axis=1)
# plt.imshow(np.abs(overlap[0:20,0:20]))
plt.figure()
plt.imshow(np.abs(overlap))
plt.title(str(N) + " Majoranas, J=1, $\mu=0.2$")
plt.xlabel(r'|n$\rangle$')
plt.ylabel(r'|m$\rangle$')
plt.colorbar()
# plt.figure()
# plt.imshow(basis_overlap)
# plt.xlabel(r'|n$\rangle$')
# plt.ylabel(r'|m$\rangle$')


plt.figure()
# plt.plot(ev_syk, '.')
plt.plot(ev_syk_cpp, '.')
# print(factor)

plt.figure()
plt.plot(np.abs(HLRgs)**2)
plt.title(r'Norm $[(H_L-H_R)|GS\rangle$] = ' + str(np.sum(np.abs(HLRgs)**2)))

# print("Couples GS energy: ", np.min(ev))

#%%
plt.figure('gibbs',figsize=(2.6,2.6))
plt.clf()
dd = np.abs(np.diag(overlap))
data = np.abs(overlap)
mask = 1 - 2*np.diag(np.ones(data.shape[0]))
data_ = data * mask
plt.plot(dd, 'purple', label='ED32')
beta=3

from scipy.optimize import curve_fit

def fit(x,beta):
    return np.exp(-beta/2*x)/ np.sqrt(np.sum(np.exp(-beta*x)))

param, _ = curve_fit(fit, ev_syk_cpp, dd)


plt.plot(fit(ev_syk_cpp,*param), '--', c='red', label='$\exp(-\\beta E_n/2)$')
plt.xlabel(r'|n$\rangle\otimes |n\rangle$')
plt.ylabel('$\left\|\psi_{nn}\\right\|$')
plt.legend()
plt.tight_layout()
# [plt.plot(np.abs(np.diag(overlap,k))*10) for k in range(1,256)];
# [plt.plot(np.abs(np.diag(overlap,-k))*10) for k in range(1,256)];

datamax = np.max(data_.reshape((data_.size,)))
plt.savefig('gibbs-fit.pdf')

# plt.figure()
# plt.imshow(data, vmin=0, vmax=datamax, cmap='plasma')
# plt.colorbar()
# plt.xlabel(r'|n$\rangle$')
# plt.ylabel(r'|m$\rangle$')

#%%
nd = []
[nd.append(np.diag(data,k)) for k in range(1,256)]
[nd.append(np.diag(data,-k)) for k in range(1,256)]

#%%
plt.figure('hist', figsize=(4.4,3.5))
plt.clf()
h = data_.reshape((data.size,))
h = h[h>=0]
plt.hist(h, bins=200,color='k');
plt.ylim((0,1000))
plt.xlabel('$\left\|\psi_{nm}\\right\|$')
plt.ylabel('frequency')
plt.tight_layout()
plt.savefig('hist1.pdf')

plt.figure('hist2', figsize=(2.5,3.5))
plt.clf()
plt.xlabel('$\left\|\psi_{nm}\\right\|$')
plt.ylabel('frequency')
plt.hist(np.diag(data), bins=35, color='purple');
plt.ylim((0,20))
plt.tight_layout()
plt.savefig('hist2.pdf')
