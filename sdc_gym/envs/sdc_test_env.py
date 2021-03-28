
import itertools
import math

import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from problem_classes.TestEquation import Test
from problem_classes.HeatEquation import Heat
from problem_classes.NonlinearEquation import Flame

import scipy.optimize as opt

from pySDC.implementations.problem_classes.HeatEquation_2D_FD_periodic import heat2d_periodic

from pySDC.implementations.datatype_classes.mesh import mesh

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right

class SDC_Full_test_Env(gym.Env):
    action_space = None
    observation_space = None
    num_envs = 1

    def __init__(
            self,
            M=None,
            dt=None,
            restol=None,
            prec=None,
            seed=None,
            lambda_real_interval=[-100, 0],
            lambda_imag_interval=[0, 0],
            lambda_real_interpolation_interval=None,
            norm_factor=1,
            residual_weight=0.5,
            step_penalty=0.1,
            reward_iteration_only=True,
            collect_states=False,
            run_example=0, #0 = TestEquation
            nvars=4,
            dtype=np.float64,
            use_doubles=True
    ):

        self.coll = CollGaussRadau_Right(M, 0, 1)
        self.num_nodes = self.coll.num_nodes
        self.dtype = dtype
        self.dt = dt
        self.Q = self.coll.Qmat[1:, 1:]

        self.lam = None

        self.num_episodes = 0
        self.prec = prec
        self.restol = restol

        self.observation_space = spaces.Box(
            low=-1E10,
            high=+1E10,
            shape=(1,),
            dtype=self.dtype,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(M,),
            dtype=np.float64 if use_doubles else np.float32,
        )
        self.seed(seed)
        


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed


    def step(self, action):



        def rho(x): #Iterationmatrix for SDC and test-equation
            return max(abs(np.linalg.eigvals(self.lam   *  self.dt * np.linalg.inv( np.eye(self.num_nodes)    -  self.dt * self.lam *  np.diag([x[i] for i in range(self.num_nodes)])   ).dot(self.Q    -   np.diag([x[i] for i in range(self.num_nodes)]))    )                   ))

        MIN = [
                0.3203856825077055,
                0.1399680686269595,
                0.3716708461097372,
            ]  

        RL=MIN
        OPT = opt.minimize(rho, MIN, method='Nelder-Mead').x   
        rho_opt = rho(OPT) 
       
        if self.prec is None:

            RL[0]=np.interp(action[0], (-1, 1), (0.28, 0.34)) 
            RL[1]=np.interp(action[1], (-1, 1), (0.1, 0.2)) 
            RL[2]=np.interp(action[2], (-1, 1), (0.35, 0.4))    
            RHO = rho(RL)      


        elif self.prec.lower() == 'min':
            RHO = rho(MIN)


        reward = rho_opt - RHO


        info = {
            's_radius': RHO,
            'lam': self.lam,
            'diag': RL
        }

        return (self.lam, reward, True, info) 

    def reset(self):

        self.lam = (1 * self.np_random.uniform(low=-20.0, high=0.0))


        return self.lam



