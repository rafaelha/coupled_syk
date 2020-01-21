import numpy as np
import matplotlib.pyplot as plt

N = 32


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
plt.title(str(N) + " Majoranas, J=1, $\mu=0.5$")
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
print(factor)


plt.figure()
plt.plot(np.abs(HLRgs)**2)
plt.title(r'Norm $[(H_L-H_R)|GS\rangle$] = ' + str(np.sum(np.abs(HLRgs)**2)))

print("Couples GS energy: ", np.min(ev))
