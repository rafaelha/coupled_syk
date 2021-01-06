import numpy as np
import random
import scipy.linalg as la
import matplotlib.pyplot as plt
import pickle
import sys

def permutations_4(n, eta):
    random.seed(seed)
    list_4 = []
    for i in range(n//2):
        for j in range(i+1, n//2):
            for k in range(j+1, n//2):
                for l in range(k+1, n//2):
                    factor = random.gauss(0.,1.)  # this creates two identical copies of SYK
                    list_4.append([2*i, 2*j, 2*k, 2*l, factor * (1 + eta) ])
                    #factor = random.gauss(0., 1.) # this creates two uncorrelated copies of SYK
                    list_4.append([2*i+1, 2*j+1, 2*k+1, 2*l+1, factor * (1 - eta)])
    return list_4

##this function considers all even MZMs on one flake, and all odd MZMs on the other
def permutations_2(n):
    list_2 = []
    for i in range(n//2):
        list_2.append([2*i, 2*i+1, 1])   # Constant chemical potential term
    return list_2

def binary(i, n):
    bin = []
    for j in range(n//2):
        bin.append(i//2**j%2)
    return bin

def unbinary(bin):
    a = 0
    for i in range(np.size(bin)):
        a += 2**i * bin[i]
    return a

def normalordering_n(occ, n):
    coeff = 1.
    for i in range(0,n,2):
        for ii in range(i //2 + 1, n//2):
            #print(ii)
            if occ[ii]==1:
                coeff *= -1    
    return coeff    

def normalordering_4(occ, i, j, k, l, n):
    coeff_i = (-1)**sum( occ[i//2+1 : n//2] )
    coeff_j = (-1)**sum( occ[j//2+1 : n//2] )
    coeff_k = (-1)**sum( occ[k//2+1 : n//2] )
    coeff_l = (-1)**sum( occ[l//2+1 : n//2] )
    return coeff_i*coeff_j*coeff_k*coeff_l

def normalordering_2(occ, i, j, n):
    coeff_i = (-1)**sum( occ[i//2+1 : n//2] )
    coeff_j = (-1)**sum( occ[j//2+1 : n//2] )

    return coeff_i*coeff_j

def normalordering_1(occ, i, n):
    #coeff = 1.
    coeff = (-1)**sum( occ[i//2+1 : n//2] )
    
    return coeff

def ED(n, Q, mu, gamma, get_wf, eta): #only valid with correlated disorder!!!
    t2 = permutations_2(n)
    g4 = permutations_4(n, eta)

    dimH = ind[Q]
    #Getting the Hamiltonian matrix - different sectors separately
    H = np.zeros( (dimH, dimH), dtype = float)

    for i in states[Q]:
        bin_i = binary(i, n)

        for t in t2:   #Two-fermion terms
            bin_j = np.copy(bin_i)
            factor = 1
            for a in range(2): #loop over 2 indices of a 2-fermion term
                if (t[a]%2!=0) and (bin_j[t[a]//2]==0):
                    factor *= (- 1.j)
                if (t[a]%2!=0) and (bin_j[t[a]//2]==1):
                    factor *= 1.j
                bin_j[t[a]//2] = 1 - bin_j[t[a]//2] # switch occupation

            j = unbinary(bin_j)
            elem = 1.j * mu * normalordering_2(bin_i, t[0], t[1], n=n) * factor/2
            if np.imag(elem) != 0:
                print('imaginary, error!')

            H[c[Q,i], c[Q,j]] += np.real(elem)

        for g in g4:    #Four-fermion terms
            bin_j = np.copy(bin_i)
            factor = 1
            for a in range(4): #loop over 4 indices of a 4-fermion term 
                if (g[a]%2!=0) and (bin_j[g[a]//2]==0):
                    factor *= (- 1.j)
                if (g[a]%2!=0) and (bin_j[g[a]//2]==1):
                    factor *= 1.j
                bin_j[g[a]//2] = 1 - bin_j[g[a]//2]

            j = unbinary(bin_j)
            elem = gamma * normalordering_4(bin_i, g[0], g[1], g[2], g[3], n=n) * g[4] * factor/4
            if np.imag(elem) != 0:
                print('imaginary, error!')

            H[c[Q,i], c[Q,j]] += np.real(elem)

    #print( str('H constructed, diagonalizing ...') )      
    # Diagonalize H
    if get_wf == True:
        eigvals, eigvect = la.eigh(H)
    if get_wf == False:
        eigvals = la.eigvalsh(H)


    """
    ff = open('WormHole_Data/eigv_N' + str(n) + '_mu' + str(mu) + '_seed' + str(seed) + '_sector' + str(Q) + '.dat','w')
    for eigv in np.sort(eigvals):
        ff.write(str(eigv))
        ff.write('\n')
    ff.close()
    """

    if get_wf == True:
        return eigvals, eigvect
    if get_wf == False:
        return eigvals

# Construct TFD state for the 4-fermion symmetry basis
def TFD_new(n): #Should work for any n%4 = 0

    ####First need to find the non-degenerate states in E0

    #When n =16 or 32: Non-degenerate SYK spectrum. The elements of E0 which are not in E2 are the unique elements in SYK dot 
    if n % 16 == 0: 
        ind_unique = []
        for i in range(len(E0)):
            if E0[i] not in E2:
                ind_unique.append(i)

    #Otherwise: Doubly-degenerate SYK spectrum. The unique elements of E0 are 2-fold degenerate in SYK.
    #Note that when n%8 = 0 (e.g. 24) E0 and E2 are not indentical. Unique levels in E0 correspond to 3-fold degeneracy in E2.
    #In contrast, when n=20,28,36... E0 and E2 are identical. Unique levels in E0 are also unique in E2.
    else:
        ind_unique = [0,0,len(E0)-1,len(E0)-1]  # First and last items are non-degenerate. 
        for i in range(1,len(E0)-1):
            if E0[i] != E0[i-1]: # If unique level in E0, then 3-fold degenerate in E2. Appears twice in SYK dot.
                if E0[i+1] != E0[i]:
                    ind_unique.append(i) #Need to count twice each level
                    ind_unique.append(i)

    #This should be equal to 2**(n//4) if noting went wrong
    print('number of eigenstates in TFD construction:' + str(len(ind_unique))  )
    print(len(E0))
    ind_unique = np.sort(ind_unique)
    E_unique = E0[ind_unique]

    #Shift spectrum
    E_unique = E_unique - E0[0]
    E = E0 - E0[0]

    ###########################################################################
    #Construct |I> -- TFD at infinite temperature. Easy in this basis
    psi0 = np.zeros(ind[0]) #dimension of Q=0 symmetry block
    psi0[0] = 1
    Z0 = 2**(n//4) #Z0 -- for normalization

    #Construct TFD state at temperature beta from the |I> state defined above
    psi_b = np.zeros( (len(beta_list), 2**(n//2-1) ), dtype=float)
    Zbeta = np.zeros( len(beta_list), dtype=float)

    for b in range(len(beta_list)):
        beta = beta_list[b]
        Zbeta[b] = np.sum( np.exp(-beta*E_unique/2) )/Z0 #different normalization here - need to divide by Z_0

    coeffs = np.exp(-np.outer(beta_list,E)/4) * np.dot(np.conjugate(wf0.T), psi0) #Compute coefficients of each eigenstate
    psi_b  = (wf0 @ coeffs.T)/np.sqrt(Zbeta)

    return (psi_b).T

def TFD_new2(n): #Should work for any n%4 = 0
    E = E0 - E0[0]

    #Construct |I> -- TFD at infinite temperature. Easy in this basis
    psi0 = np.zeros(ind[0]) #dimension of Q=0 symmetry block
    psi0[0] = 1

    psi_b = wf0 @ (np.exp(-np.outer(beta_list,E)/4) * np.dot(np.conjugate(wf0.T), psi0)).T #Compute coefficients of each eigenstate
    psi_b = psi_b/np.linalg.norm(psi_b, axis=0)

    return (psi_b).T
##############################################################################
#### MAIN 
##############################################################################
idx = int(sys.argv[1])

n = 16
nstates = 10
etas = np.sort(np.concatenate((np.linspace(0,2,83), np.array([1]))))
eta = etas[idx]
gamma =  np.sqrt(6/(n//2)**3)
mu_range = np.concatenate((np.linspace(0,0.1,101),np.linspace(0.101,1,102)))
#mu_range = np.concatenate((np.linspace(0,0.1,11),np.linspace(0.101,1,12)))
#, 0.005, 0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]) #values for mu/J

beta_list   = np.linspace(0,500,1001) # list of beta J
seed_list = [3]
#seed_list = np.linspace(0,10,11,dtype=int)

overlap     = np.zeros( (len(mu_range), nstates, len(beta_list)) )
max_overlap = np.zeros( (len(mu_range), nstates) )
max_max_overlap = np.zeros( (len(mu_range), nstates) )
max_beta    = np.zeros( len(mu_range) )
max_max_beta    = np.zeros( (len(mu_range), nstates) )
parity = np.array(np.zeros( (len(mu_range), nstates) ), dtype=bool) # True means GS is in even sector
energies = np.zeros( (len(mu_range), nstates) )

#Distribute states in 2 parity sectors
p = 2
c = np.zeros( (p, 2**(n//2)), dtype = int)
ind = np.zeros( p, dtype=np.int )
states = [ [], [], [], [] ]
for i in range(2**(n//2)): #This routine could be made faster?
    for Q in range(p):
        if sum(binary(i, n))%p == Q:
            states[Q].append(i)
            c[Q,i] = ind[Q]
            ind[Q] = ind[Q] + 1
for seed in seed_list:  #Loop over disorder realizations
    for m in range(len(mu_range)): #Loop over mu
        mm = 0
        mu = mu_range[m]
        print('mu:' + str(mu))

        Eeven, wfeven = ED(n, 0, mu, gamma, get_wf=True, eta=eta)
        Eodd, wfodd = ED(n, 1, mu, gamma, get_wf=True, eta=eta)
        #E2 = ED(n, 2, mu, gamma, get_wf=False)
        #E3 = ED(n, 3, mu, gamma, get_wf=False)
        EE = np.concatenate((Eeven, Eodd))
        idd = np.argsort(EE)
        energies[m] = EE[idd[0:nstates]]

        if mu == 0: #Construct TFD
            E0, wf0 = ED(n, 0, mu, gamma, get_wf=True, eta=0)
            tfd = TFD_new2(n)

        for i in np.arange(nstates):
            if idd[i] >= len(Eeven):
                j = idd[i] - len(Eeven)
                wf = wfodd[:,i]
                parity[m,i] = False
                wf = wfodd[:,j]
            else:
                j = idd[i]
                wf = wfodd[:,i]
                parity[m,i] = True
                wf = wfeven[:,j]

            #Compute the overlap between the true ground state wavefunction 
            #and the various thermofield double states
            overlap[m,i,:] = np.abs( np.dot( np.conjugate(tfd), wf ) )
            index_betamax = np.argmax(overlap[m,i,:])

            o = overlap[m,i,index_betamax]
            if o > mm:
                mm = o
                index_betamaxmax = index_betamax
            max_max_overlap[m,i] = o
            #compute index of beta corresponding to the maximal overlap with gs wf
            max_max_beta[m,i] = beta_list[index_betamax]

        max_overlap[m,:] = overlap[m,:,index_betamaxmax] #same temperature for
        # for all states
        max_beta[m] = beta_list[index_betamaxmax]
"""
plt.figure('Res')
plt.plot(mu_range, max_overlap, '.-', label='$\eta=$'+str(eta))
plt.xlabel('$\mu$')
plt.ylabel('Overlap')
plt.legend()
"""
res = {'eta':eta,
       'n':n,
       'mu_range':mu_range,
       'seed_list':seed_list,
       'max_overlap':max_overlap,
       'max_max_overlap':max_max_overlap,
       #'overlap':overlap,
       'max_beta':max_beta,
       'max_max_beta':max_max_beta,
       'beta_list':beta_list,
       'energies':energies,
       'parity':parity}


f1 = open(str(idx) + '_overlap.pickle', 'ab')
pickle.dump(res, f1)
f1.close()
