from __future__ import division

import numpy as np
import scipy.sparse as sp
from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError


class Heat(ptype):
    """
    Implementing u_t = \lambda u

    Attributes:
        lambda
    """

    def __init__(self, problem_params, dtype_u, dtype_f, t):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
        """
        # these parameters will be used later, so assert their existence
        essential_keys = ['lam', 'nvars']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        self.lam = problem_params['lam']
        self.nvars = problem_params['nvars']

        self.dx = 1.0 / (self.nvars-1)
        self.xvalues = np.array([ i * self.dx for i in range(self.nvars)])
        #print(self.xvalues)
        self.A = self.__get_A(self.nvars, self.dx)

        # nu wird so gewaehlt, dass die rechte Seite nu/(dx^2)*A den Eigenwert lambda hat
        self.nu = -self.lam/  max(abs(np.linalg.eigvals(self.A)))
        self.A *= self.nu
        self.rhs_mat = self.A 

        self.u0 = self.u_exact(t*0)
        self.M= len(t)


    def __get_A(self, N, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            nu (float): diffusion coefficient
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil = [1, -2, 1]
        A = sp.diags(stencil, [-1, 0, 1], shape=(N, N), format='csc')
        A *= 1. / (dx ** 2)
        return A.todense()

    def return_j(self, t=0):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """


        return np.kron(  np.eye(self.M)   , self.A) 

    def eval_f(self, u, t=0):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        #n = u.size / self.nvars

        return np.squeeze( np.array(    np.kron(np.eye(self.M)   , self.A) @ u.flatten() ))

    def eval_j(self, u, t=0):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        #n = u.size / self.nvars
        return np.kron(  np.eye(self.M)   , self.A) 



    def u_exact(self, t ):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """
        exact=[]
        for time in t:
            exact = np.append(exact, np.sin(np.pi * self.xvalues)*np.exp(-time * np.pi**2 * self.nu))
        
        return exact


