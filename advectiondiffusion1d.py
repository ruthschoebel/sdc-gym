import numpy as np
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
import scipy
import scipy.sparse as sp
from scipy.sparse import identity
import matplotlib.pyplot as plt

import pyfftw

num_nodes=3
dt=0.01
N=128
restol = 1e-6
L=10 
nu=1.
c=1.
dx=L/N
x = np.arange(-L/2,L/2,dx)

def iterate(x, k_impl, k_expl, _u0, N):

    k = k_impl + k_expl
    coll = CollGaussRadau_Right(num_nodes, 0, 1)
    Q = coll.Qmat[1:, 1:]
    C = identity(num_nodes*N) -  dt *  scipy.sparse.kron(Q, sp.diags(k))
    u0 = np.concatenate((_u0, _u0, _u0), axis=None)
    u = np.concatenate((_u0, _u0, _u0), axis=None) 

    print("u", u)
    Qdmat = np.zeros_like(Q)
    np.fill_diagonal(Qdmat, x)



    
    QD_expl = np.zeros(Q.shape)

    for m in range(coll.num_nodes):
        QD_expl[m, 0:m] = coll.delta_m[0:m]


    mat0_i =[]
    mat1_i =[]
    mat2_i =[]

    for i in k_impl:
        #hier wuerde jetzt von RL fuer jedes kappa ein Qd gewaehlt werden
        mat0_i = np.concatenate((mat0_i, x[0]*i), axis=None)  
        mat1_i = np.concatenate((mat1_i, x[1]*i), axis=None)  
        mat2_i = np.concatenate((mat2_i, x[2]*i), axis=None)  


    mat_i = sp.diags(np.concatenate((mat0_i, mat1_i, mat2_i), axis=None)  )


    Pinv_impl = np.linalg.inv(
        np.eye(coll.num_nodes*N) -  dt * ( mat_i   +   scipy.sparse.kron(QD_expl, sp.diags(k_expl))  )    ,
    )


    residual = u0 - C @ u

    done = False
    err = False
    niter = 0
    while not done and not niter >= 50 and not err:
        niter += 1
        u = np.squeeze( np.array( u + Pinv_impl @ (u0 - C @ u) ))
        residual = np.squeeze( np.array( u0 - C @ u ))

        norm_res = np.linalg.norm(residual, np.inf)
        print(norm_res)
        if np.isnan(norm_res) or np.isinf(norm_res):
            niter = 51
            break
        done = norm_res < restol
    print("niter", niter)
    return niter, u

def u_exact(t):

    freq =1
    u = np.arange(-L/2,L/2,dx)
    omega = 2.0 * np.pi * freq
    u = np.sin(omega * (x - c * t)) * np.exp(-t * nu * omega ** 2)

    return u


def main():


    #kappa = - np.power(2*np.pi*np.fft.fftfreq(N, d=dx), 2)


############################################################
    kx = np.arange(-L/2,L/2,dx)
    for i in range(0, len(kx)):
        kx[i] = 2 * np.pi / L * i

    #ddx = kx * 1j
    #lap = -kx ** 2
    ddx = 2*np.pi*np.fft.fftfreq(N, d=dx)
    lap = - np.power(ddx, 2)

    kappa_impl = nu * lap 
    kappa_expl = - c * ddx

    rfft_in = pyfftw.empty_aligned(N, dtype='float64')
    fft_out = pyfftw.empty_aligned(N // 2 + 1, dtype='complex128')
    ifft_in = pyfftw.empty_aligned(N // 2 + 1, dtype='complex128')
    irfft_out = pyfftw.empty_aligned(N, dtype='float64')
    rfft_object = pyfftw.FFTW(rfft_in, fft_out, direction='FFTW_FORWARD')
    irfft_object = pyfftw.FFTW(ifft_in, irfft_out, direction='FFTW_BACKWARD')
############################################################


    u0 = u_exact(0)


    plt.plot(
        x,
        u0,
        label="time 0"
    )


    MIN = [
            0.3203856825077055,
            0.1399680686269595,
            0.3716708461097372,
        ]


    u0hat2 = np.zeros(N)
    u0hat2 = rfft_object(u0)

    u0hat = np.fft.fft(u0)


    niter, u = iterate(MIN, kappa_impl, kappa_expl, u0hat, N)


    u = u.reshape(num_nodes,N)

    u_ = np.zeros_like(u)


    for i in range(len(u)):
        u_[i] = np.fft.ifft(u[i])


    plt.plot(
        x,
        u_[2],
        label="time 1"
    )

    u__ = u_exact(dt)
    plt.plot(
        x,
        u__,
        label="exact"
    )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

