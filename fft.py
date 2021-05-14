import numpy as np
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
import scipy
import scipy.sparse as sp
from scipy.sparse import identity
import matplotlib.pyplot as plt


num_nodes=3
dt=1.
N=1024
restol = 1e-6
 
def iterate(x, k, _u0):

    coll = CollGaussRadau_Right(num_nodes, 0, 1)
    Q = coll.Qmat[1:, 1:]
    C = identity(num_nodes*N) -  dt *  scipy.sparse.kron(Q, sp.diags(k))
    u0 = np.concatenate((_u0, _u0, _u0), axis=None)
    u = np.concatenate((_u0, _u0, _u0), axis=None) 

    Qdmat = np.zeros_like(Q)
    np.fill_diagonal(Qdmat, x)


    mat0 =[]
    mat1 =[]
    mat2 =[]
    for i in k:
        #hier wuerde jetzt von RL fuer jedes kappa ein Qd gewaehlt werden
        mat0 = np.concatenate((mat0, x[0]*i), axis=None)  
        mat1 = np.concatenate((mat1, x[1]*i), axis=None)  
        mat2 = np.concatenate((mat2, x[2]*i), axis=None)  


    mat = sp.diags(np.concatenate((mat0, mat1, mat2), axis=None)  )

    Pinv = np.linalg.inv(
        np.eye(coll.num_nodes*N) -  dt * mat,
    )

    residual = u0 - C @ u

    done = False
    err = False
    niter = 0
    while not done and not niter >= 50 and not err:
        niter += 1
        u = np.squeeze( np.array( u + Pinv @ (u0 - C @ u) ))
        residual = np.squeeze( np.array( u0 - C @ u ))
        norm_res = np.linalg.norm(residual, np.inf)
        if np.isnan(norm_res) or np.isinf(norm_res):
            niter = 51
            break
        done = norm_res < restol
    return niter, u

def main():

    a=1
    L=100

    dx=L/N
    x = np.arange(-L/2,L/2,dx)

    kappa = - np.power(2*np.pi*np.fft.fftfreq(N, d=dx), 2)

    u0=np.zeros_like(x)

    u0[ int((L/2-L/10)/dx):int((L/2+L/10)/dx) ] = 1

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


    u0hat = np.fft.fft(u0)


    niter, u = iterate(MIN, kappa, u0hat)


    u = u.reshape(num_nodes,N)

    u_ = np.zeros_like(u)

    
    for i in range(len(u)):
        u_[i] = np.fft.ifft(u[i])
    #plt.plot(
    #    x,
    #    u_[0]
    #)
    #plt.plot(
    #    x,
    #    u_[1]
    #)
    plt.plot(
        x,
        u_[2],
        label="time 1"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

