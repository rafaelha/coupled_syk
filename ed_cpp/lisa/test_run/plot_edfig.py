import numpy as np
import matplotlib.pyplot as plt
import glob

N = 32

mus = [0, 0.2, 0.4, 0.6]
etas = np.linspace(0,2,64)

file_or = glob.glob('data/*imag.txt')


def extract_params(string):
    string = file_or[1]
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

