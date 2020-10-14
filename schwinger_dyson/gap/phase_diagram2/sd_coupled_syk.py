import numpy as np
# mpl.use('Agg')
from scipy.fftpack import fft, ifft
from scipy.fftpack import fftfreq
from scipy.fftpack import fftshift, ifftshift
import time
import sys
import pickle

etas = np.linspace(0,1.5,64)
mus = np.linspace(0, 0.3, 21)

temps = np.linspace(0, 0.08, 21)
temps[0] = 0.00086603

idx = int(sys.argv[1])
eta = etas[idx]
for temp in temps:
    for mu in mus:
        J = 1  # intra-SYK coupling
        N = 2**20  # number of frequency points (points in Patel's code is N/2)
        beta = 1/temp  # inverse temperature

        omegacutoff = N*(np.pi)/beta  # Max cut-off frequency,
        # NOTE: this is fixed by inverse temperature beta

        total_iteration = 3000  # number of iterations for each run
        x = 0.05  # mixing of consecutive iterations. (0 means no mixing)

        tstep = 2*beta/N
        fstep = 2*np.pi/N
        freq = np.zeros(N-1)
        for i in range(int(N/2)):
            # First all the positive frequency in increasing order
            freq[i] = (i+1)*fstep
        # Then negative frequency in increasing order so as to compactify w on a circle
        for i in range(int(N/2), N-1):
            freq[i] = (i+1-N)*fstep
    # print(freq)
    # print(np.size(freq))
        tau = np.arange(N) * tstep - beta
    #np.array([(tstep*i-beta) for i in range(1,N+1)])
        z = np.logical_and(tau > 4, tau < 30)
        t = tau[z]

        BREAK = False

        # True if exponent is e^{-iwt} in FT from time to frequencies ft(t)
        ft_of_t_neg_iw = False

        def ft(t):
            return -0.5*tstep*(ifft(t))*N

        def ift(w):
            return -fft(w)/beta

        def diff(a, b):
            return np.linalg.norm(np.log(np.abs(a[z]))-np.log(np.abs(b[z])))/N

        def freeenergy(GLLiw, GLRiw, SLLiw, SLRiw, freqi):
            FE = np.log(2.0) \
                + (1.0/2.0)*np.sum(np.log(np.power(1.0-SLLiw/(1.0j*freqi/tstep), 2)
                                        + np.power((1.0j*mu-SLRiw)/(1.0j*freqi/tstep), 2))) \
                + 3.0/4.0 * np.sum((SLLiw * GLLiw) - (SLRiw * GLRiw))
            FE = 0
            # J**2/2 * (1-eta**2) * np.sum(GLRt
            return -(N/beta)*FE

        # Initialize G
        ar = N/2 - 0.5 - np.arange(N)
        GLLt = 0.5*np.sign(ar)

        GRRt = 2*GLLt

        GLLw = ft(GLLt)
        GRRw = ft(GRRt)
        SLLt = np.zeros(N, dtype='complex')
        SLLw = np.zeros(N, dtype='complex')
        SRRt = np.zeros(N, dtype='complex')
        SRRw = np.zeros(N, dtype='complex')

        GLRt = np.ones(N) * 0.005*1.0j
        GLRt[int(N/4):int(3*N/4)] = -0.005*1.0j

        GLRw = ft(GLRt)
        SLRt = np.zeros(N, dtype='complex')
        SLRw = np.zeros(N, dtype='complex')

        S = np.zeros(total_iteration)
        dLL = []
        dRR = []
        dLR = []

        for i in np.arange(total_iteration):
            print(i)
            # self consistent iteration
            SRRt = (1-eta)**2 * 2*(J**2)*GRRt*GRRt*GRRt #changed sign
            SLLt = (1+eta)**2 * 2*(J**2)*GLLt*GLLt*GLLt #changed sign
            SLRt = (1-eta**2) * 2*(J**2)*GLRt*GLRt*GLRt

            SRRw = ft(SRRt)
            SLLw = ft(SLLt)
            SLRw = ft(SLRt)

            iw = 1.0j*freq[0:N-1:2] / tstep
            imu = 1.0j*mu

            Dw = (iw - SRRw[1:N:2])*(iw - SLLw[1:N:2]) + \
                (imu - SLRw[1:N:2])*(imu - SLRw[1:N:2])
            GLLw[1:N:2] = (iw - SRRw[1:N:2]) / Dw
            GRRw[1:N:2] = (iw - SLLw[1:N:2]) / Dw
            GLRw[1:N:2] = -(imu - SLRw[1:N:2]) / Dw

            # store result of iteration
            GRRtn = ift(GRRw)
            GLLtn = ift(GLLw)
            GLRtn = ift(GLRw)

            # compute free energy
            #S[i] = freeenergy(GLLw[1:N:2], GLRw[1:N:2], SLLw[1:N:2], SLRw[1:N:2], freq[0:N-1:2])

            # compare new iteration to old one
            if i%10==0:
                dRR.append(diff(GRRt, GRRtn))
                dLL.append(diff(GLLt, GLLtn))
                dLR.append(diff(GLRt, GLRtn))

            # update with newest iteration
            GRRt = (1-x)*GRRt + x*GRRtn
            GLLt = (1-x)*GLLt + x*GLLtn
            GLRt = (1-x)*GLRt + x*GLRtn

            GRRt[:N//2] = 0.5 * (GRRt[:N//2] - np.flip(GRRt[N//2:]))
            GRRt[N//2:] = -np.flip(GRRt[:N//2])

            GLLt[:N//2] = 0.5 * (GLLt[:N//2] - np.flip(GLLt[N//2:]))
            GLLt[N//2:] = -np.flip(GLLt[:N//2])

            GLRt[:N//2] = 0.5 * (GLRt[:N//2] + np.flip(GLRt[N//2:]))
            GLRt[N//2:] = np.flip(GLRt[:N//2])

        dd = 8*50
        res = {'eta': eta,
            'beta': beta,
            'mu': mu,
            'J': J,
            'tau': tau[N//2:3*N//4:dd],
            'GRRt': GRRt[N//2:3*N//4:dd],
            'GLLt': GLLt[N//2:3*N//4:dd],
            'GLRt': GLRt[N//2:3*N//4:dd],
            'dRR': dRR,
            'dLL': dLL,
            'dLR': dLR,
            'total_iterations': total_iteration,
            'x': x,
            'N': N}
        f1 = open(str(idx) + '.pickle', 'ab')
        pickle.dump(res, f1)
        f1.close()
