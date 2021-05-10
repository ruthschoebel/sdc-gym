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

    def __init__(self, problem_params, dtype_u, dtype_f, t):
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
        self.u0 = np.ones(len(t), dtype=dtype_u)
        self.M= len(t)

    def eval_f(self, u, t=0):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        #print(self.lam, u)
        f = self.lam * u
        return f

    def return_j(self, t=0):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """


        return np.eye(self.M)*self.lam

    def eval_j(self, u, t=0):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """


        return np.eye(u.size)*self.lam




