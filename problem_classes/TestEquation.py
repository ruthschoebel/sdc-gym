from __future__ import division

import numpy as np
import scipy.sparse as sp
from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

# noinspection PyUnusedLocal
class Test(ptype):
    """
    Implementing u_t = \lambda u

    Attributes:
        lambda
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
        """
        # these parameters will be used later, so assert their existence
        essential_keys = ['lam','nvars']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        if problem_params['nvars'] != 1:
            raise ProblemError('nvars for TestEquation should be 1!')


        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(Test, self).__init__(init=1, dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        self.lam = self.params.lam
        self.rhs_mat = self.lam 


    def eval_f(self, u, t=0):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """


        f = self.lam * u
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """
        raise NotImplementedError('ERROR: problem has to implement solve_system(self, u, t)')

        #me = self.dtype_u(self.init)
        #me.values = spsolve(sp.eye(self.params.nvars, format='csc'), rhs.values)
        #return me


