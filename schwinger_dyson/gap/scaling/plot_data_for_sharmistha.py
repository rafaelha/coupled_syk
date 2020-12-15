
import matplotlib as mpl
import pickle
import numpy as np
import matplotlib.pyplot as plt

# load the data
reader = open('data.pickle','rb')
a = pickle.load(reader)
reader.close()

# extract variables
gaps = a['gaps']
mus = a['mus']
etas = a['etas']
beta = a['beta']
J = a['J']
N = a['N']

# mus containes all possible mu values
# etas contains all possible eta values
# data[i,j] contains the gap in a simulation with parameters mus[i] and etas[j]

# plot a single curve for \mu=mus[i]
i = 5
plt.plot(etas, gaps[i], '.-', label=f'$\mu={mus[i]}$')
plt.legend()
plt.xlabel('$\eta$')
plt.ylabel('$E_{gap}$')