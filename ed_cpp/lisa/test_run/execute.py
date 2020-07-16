import numpy as np
import sys
import subprocess

# idx = int(sys.argv[1])
# idx = 0

mus = [0, 0.2, 0.4, 0.6]
etas = np.linspace(0,2,64)

for mu in mus:
    for eta in etas:
        print(eta)
        bashCommand = f"./ed_cpp.exe {eta} {mu}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)
