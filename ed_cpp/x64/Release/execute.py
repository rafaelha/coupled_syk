import numpy as np
import sys
import subprocess

idx = int(sys.argv[1])
# idx = 0

mus = np.linspace(0,1,11)
etas = np.linspace(0,2,64)

eta = etas[idx]

for mu in mus:
    print(mu)
    bashCommand = f"ed_cpp.exe {eta} {mu}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()