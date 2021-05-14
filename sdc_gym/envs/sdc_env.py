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

from scipy.sparse import identity



from problem_classes.TestEquation import Test
from problem_classes.HeatEquation import Heat
from problem_classes.NonlinearEquation import Flame

from pySDC.implementations.collocation_classes.gauss_radau_right import \
    CollGaussRadau_Right


class SDC_Full_Env(gym.Env):
    """This environment implements a full iteration of SDC, i.e. for
    each step we iterate until
        (a) convergence is reached (residual norm is below restol),
        (b) more than `self.max_iters` iterations are done (not converged),
        (c) diverged.
    """
    action_space = None
    observation_space = None
    num_envs = 1
    max_iters = 50


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
            reward_iteration_only=None,
            reward_strategy='iteration_only',
            collect_states=False,
            use_doubles=True,
            do_scale=True,
            example=1, #0 = TestEquation
            nvars=1000,
            dtype=np.complex128, #np.float64,
            model=None,
            params=None
    ):

        self.np_random = None
        self.niter = None
        self.restol = restol
        self.dt = dt
        self.M = M
        self.coll = CollGaussRadau_Right(M, 0, 1)
        self.num_nodes = self.coll.num_nodes
        self.Q = self.coll.Qmat[1:, 1:]
        self.C = None
        self.lam = None
        #self.u0 = np.ones(M, dtype=np.complex128)
        self.old_res = None
        self.prec = prec
        self.initial_residual = None
        self.initial_residual_time = None

        if(example==0):
            print("running TEST-Equation")
            self.nvars=1

        elif(example==1):
            print("running Heat-Equation")
            self.nvars=nvars



        self.model=model
        self.params=params
        self.example=example
        self.dtype=np.complex128
        self.prob= None
        self.linear = False 
        self.lambda_real_interval = lambda_real_interval
        self.lambda_real_interval_reversed = list(
            reversed(lambda_real_interval))
        self.lambda_imag_interval = lambda_imag_interval
        self.lambda_real_interpolation_interval = \
            lambda_real_interpolation_interval

        self.norm_factor = norm_factor
        self.residual_weight = residual_weight
        self.step_penalty = step_penalty
        if reward_iteration_only is None:
            self.reward_strategy = reward_strategy.lower()
        elif reward_iteration_only:
            self.reward_strategy = 'iteration_only'
        else:
            self.reward_strategy = 'residual_change'
        self.collect_states = False #collect_states
        self.do_scale = do_scale

        self.num_episodes = 0
        # self.rewards = []
        # self.episode_rewards = []
        # self.norm_resids = []
        # Setting the spaces: both are continuous, observation box
        # artificially bounded by some large numbers
        # note that because lambda can be complex, U can be complex,
        # i.e. the observation space should be complex
        self.observation_space = spaces.Box(
            low=-1E10,
            high=+1E10,
            shape=(M * self.nvars * 2, self.max_iters) if collect_states else (2, M * self.nvars),
            dtype=np.complex128,
        )
        # I read somewhere that the actions should be scaled to [-1,1],
        # values will be real.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(M,),
            dtype=np.float64 if use_doubles else np.float32,
        )

        self.seed(seed)
        self.state = None
        if collect_states:
            self.old_states = np.zeros((M * 2, self.max_iters),
                                       dtype=np.complex128)



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def set_num_episodes(self, num_episodes):
        self.num_episodes = num_episodes

    def _scale_action(self, action):
        # I read somewhere that the actions should be scaled to [-1,1],
        # scale it back to [0,1] here...
        if self.do_scale:
            scaled_action = np.interp(action, (-1, 1), (0, 1))
        else:
            scaled_action = action
        return scaled_action

    def solve_system(self, u,  rhs, Qdmat):

        Id = sp.eye(self.num_nodes*self.nvars)
        n=1
        while n < 5: #TODO 

            #TODO: check if problem has boundary conditions
            #TODO: node perspective is more efficient for solve
            g = np.squeeze(np.array(u.flatten()  - self.dt * np.kron(Qdmat, np.eye(self.nvars)) @ self.prob.eval_f(u, 0.0)   - rhs.flatten() ))
            


            res = np.linalg.norm(g, np.inf)
            if res < 1e-15:
                break

            #dg = Id - self.dt * scipy.sparse.kron(Qdmat, np.eye(self.nvars)) @   self.prob.eval_j(u, 0.0)
            dg = Id - self.dt * scipy.sparse.kron(Qdmat, np.eye(self.nvars)).multiply(self.prob.eval_j(u, 0.0))


            du = spsolve(dg, g)
            u -= du 
            n += 1

        return np.squeeze(np.array(u.flatten()))  # u #np.squeeze(np.array(u.flatten() +    Pinv @ old_residual.flatten()  ))

    def _get_prec(self, scaled_action):
        """Return a preconditioner based on the `scaled_action`."""
        # Decide which preconditioner to use
        # (depending on self.prec string)... not very elegant
        if self.prec is None:
            Qdmat = np.zeros_like(self.Q)
            np.fill_diagonal(Qdmat, scaled_action)
        elif self.prec.upper() == 'LU':
            QT = self.Q.T
            [_, _, U] = scipy.linalg.lu(QT, overwrite_a=True)
            Qdmat = U.T
        elif self.prec.lower() == 'min':
            Qdmat = np.zeros_like(self.Q)
            if self.M == 7:
                x = [
                    0.15223871397682717,
                    0.12625448001038536,
                    0.08210714764924298,
                    0.03994434742760019,
                    0.1052662547386142,
                    0.14075805578834127,
                    0.15636085758812895
                ]
            elif self.M == 5:
                x = [
                    0.2818591930905709,
                    0.2011358490453793,
                    0.06274536689514164,
                    0.11790265267514095,
                    0.1571629578515223,
                ]
            elif self.M == 4:
                x = [
                    0.3198786751412953,
                    0.08887606314792469,
                    0.1812366328324738,
                    0.23273925017954,
                ]
            elif self.M == 3:
                x = [
                    0.3203856825077055,
                    0.1399680686269595,
                    0.3716708461097372,
                ]
            else:
                # if M is some other number, take zeros. This won't work
                # well, but does not raise an error
                x = np.zeros(self.M)
            np.fill_diagonal(Qdmat, x)
        else:
            raise NotImplementedError()
        return Qdmat

    def _compute_pinv(self, scaled_action):
        # Get Q_delta, based on self.prec (and/or scaled_action)
        Qdmat = self._get_prec(scaled_action=scaled_action)

        # Compute the inverse of P
        Pinv = np.linalg.inv(
            np.eye(self.M*self.nvars) -  self.dt * np.kron(Qdmat, np.eye(self.nvars)) @ self.prob.return_j(0) ,
        )
        return Pinv

    def _compute_residual(self, u):


        Cu     = np.squeeze(u.flatten() - self.dt * np.kron(self.Q, np.eye(self.nvars)) @ self.prob.eval_f(u.flatten()).flatten())  
        residual = np.squeeze(np.array(self.u0.flatten() - Cu.flatten() )) 

        return residual

    def _inf_norm(self, v):
        return np.linalg.norm(v, np.inf)

    def step_old(self, action):
        u, old_residual = self.state


        scaled_action = self.model(self.params, self.lam)
        #scaled_action = self._scale_action(action)

        Qdmat = self._get_prec(scaled_action=scaled_action) #, M=u.size)

        if self.linear:
            Pinv = self._compute_pinv(scaled_action)

        norm_res_old = self._inf_norm(old_residual)

        # Re-use what we already have
        residual = old_residual

        done = False
        err = False
        self.niter = 0
        # Start the loop
        while not done and not self.niter >= self.max_iters:
            self.niter += 1

            # This is the iteration (yes, there is a typo in the slides,
            # this one is correct!)
            rhs  = np.squeeze( np.array(             u.flatten()  - self.dt * np.kron(Qdmat, np.eye(self.nvars)) @ self.prob.eval_f(u) + old_residual.flatten()                        )) 

            if self.linear:

                u = np.squeeze( np.array( u.flatten() + Pinv @ residual.flatten() ))

            else:
                u = np.squeeze( self.solve_system( u, rhs, Qdmat).flatten())   #TODO clean this up and use better data-structures from pySDC
                Cu     =  np.squeeze( u.flatten() - self.dt * np.kron(self.Q, np.eye(self.nvars)) @ self.prob.eval_f(u, 0.0).flatten() ) #u.flatten() 

            # Compute the residual and its norm

            residual = self._compute_residual(u)
            norm_res = self._inf_norm(residual)
            old_residual = residual.copy()
            # stop if something goes wrong
            err = np.isnan(norm_res) or np.isinf(norm_res)
            # so far this seems to be the best setup:
            #   - stop if residual gets larger than the initial one
            #     (not needed, but faster)
            #   - reward = -self.max_iters, if this happens (crucial!)
            if self.collect_states and self.niter < self.max_iters:
                self.old_states[:, self.niter] = np.concatenate((u, residual))
            err = err or norm_res > norm_res_old * 100
            if err:
                reward = -self.step_penalty * (self.max_iters + 1)
                # reward = -(self.max_iters + 1)
                break
            # check for convergence
            done = norm_res < self.restol

        print(self.lam, self.niter)
        #print(self.niter)
        if not err:
            reward = self.reward_func(
                self.initial_residual,
                residual,
                done,
                self.niter,
                scaled_action,
                Pinv,
            )

        done = True

        # self.episode_rewards.append(reward)
        # self.norm_resids.append(norm_res)


        _u   = u.reshape(self.num_nodes, self.nvars)
        _res = residual.reshape(self.num_nodes, self.nvars)


        self.state = (u, residual)

        if self.collect_states and self.niter < 50:
            self.old_states[:, self.niter] = np.concatenate((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ))

        print(self.nvars)
        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.nvars,
        }
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:
            return ((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ), reward, done, info)#(self.state, reward, done, info)


    def step(self, action):
        #u, old_residual = self.state


        a=1
        L=100.
        
        dx=L/self.nvars
        x = np.arange(-L/2.,L/2.,dx)

        kappa = - np.power(2*np.pi*np.fft.fftfreq(self.nvars, d=dx), 2)

        _u0=np.zeros_like(x)

        _u0[ int((L/2.-L/10.)/dx):int((L/2.+L/10.)/dx) ] = 1



        u0hat = np.fft.fft(_u0)

        ###########################################################################
        #u0 = u0hat.copy()
        #u = u0hat.copy()
        iterationen=[]
        j=0
        for i in kappa:
            u0 = [u0hat[j],u0hat[j],u0hat[j]]
            u = [u0hat[j],u0hat[j],u0hat[j]]
            j=j+1
            C = identity(self.num_nodes) -  self.dt *  self.Q * i
            mat0 =[]
            mat1 =[]
            mat2 =[]
            mat0 = np.concatenate((mat0, self.model(self.params, i)[0][0]*i), axis=None)  
            mat1 = np.concatenate((mat1, self.model(self.params, i)[0][1]*i), axis=None)  
            mat2 = np.concatenate((mat2, self.model(self.params, i)[0][2]*i), axis=None)  
            mat = sp.diags(np.concatenate((mat0, mat1, mat2), axis=None)  )

            Pinv = np.linalg.inv(
                np.eye(self.coll.num_nodes) -  self.dt * mat  ,
            )
            #print("mat", mat, C)

            residual = np.squeeze(np.array(u0 - C @ u))
            done = False
            err = False
            self.niter = 0


            while not done and not self.niter >= self.max_iters and not err:
                self.niter += 1

                u =  np.squeeze( np.array(u + Pinv @ np.squeeze( np.array((u0 - C @ u))) ))

                residual =  np.squeeze( np.array(u0 - C @ u)) 
                norm_res = np.linalg.norm(residual, np.inf)
                #print(norm_res)
                if np.isnan(norm_res) or np.isinf(norm_res):
                    self.niter = 51
                    break
                done = norm_res < self.restol
            print(i, self.niter)
            iterationen.append(self.niter)


        reward = -1

        done = True


        self.state = (u, residual)


        info = {
            'residual': norm_res,
            'niter': iterationen,
            'lam': -kappa, #self.lam,
        }

        return (self.state, reward, done, info)

        ###########################################################################




        u0 = np.concatenate((u0hat, u0hat, u0hat), axis=None)
        u = np.concatenate((u0hat, u0hat, u0hat), axis=None) 



        C = identity(self.num_nodes*self.nvars) -  self.dt *  scipy.sparse.kron(self.Q, sp.diags(kappa))

        MIN = [
                0.3203856825077055,
                0.1399680686269595,
                0.3716708461097372,
            ]

        if self.prec is None:
            mat0 =[]
            mat1 =[]
            mat2 =[]
            for i in kappa:
                #mat0 = np.concatenate((mat0, MIN[0]*i), axis=None)  
                #mat1 = np.concatenate((mat1, MIN[1]*i), axis=None)  
                #mat2 = np.concatenate((mat2, MIN[2]*i), axis=None) 
                mat0 = np.concatenate((mat0, self.model(self.params, i)[0][0]*i), axis=None)  
                mat1 = np.concatenate((mat1, self.model(self.params, i)[0][1]*i), axis=None)  
                mat2 = np.concatenate((mat2, self.model(self.params, i)[0][2]*i), axis=None)  

            #mat = scipy.sparse.kron(sp.diags(MIN), sp.diags(kappa)) #sp.diags(np.concatenate((mat0, mat1, mat2), axis=None)  )
            mat = sp.diags(np.concatenate((mat0, mat1, mat2), axis=None)  )

        else:
            scaled_action = self._scale_action(action)
            Qdmat = self._get_prec(scaled_action=scaled_action)         
            mat = scipy.sparse.kron(Qdmat, sp.diags(kappa))


        Pinv = np.linalg.inv(
            np.eye(self.coll.num_nodes*self.nvars) -  self.dt * mat,
        )

        residual = u0 - C @ u


        done = False
        err = False
        self.niter = 0


        while not done and not self.niter >= self.max_iters and not err:
            self.niter += 1
            u = np.squeeze( np.array( u + Pinv @ (u0 - C @ u) ))

            residual = np.squeeze( np.array( u0 - C @ u ))
            norm_res = np.linalg.norm(residual, np.inf)
            #print(norm_res)
            if np.isnan(norm_res) or np.isinf(norm_res):
                self.niter = 51
                break
            done = norm_res < self.restol


        u = u.reshape(self.coll.num_nodes,self.nvars)

        u__ = np.zeros_like(u)

        
        for i in range(len(u)):
            u__[i] = np.fft.ifft(u[i])
        #plt.plot(
        #    x,
        #    u_[0]
        #)
        #plt.plot(
        #    x,
        #    u_[1]
        #)
        #plt.plot(
        #    x,
        #    u__[2],
        #    label="time 1"
        #)
        #plt.legend()
        #plt.show()


        reward = -1

        done = True


        _u   = u.reshape(self.num_nodes, self.nvars)
        _res = residual.reshape(self.num_nodes, self.nvars)


        self.state = (u, residual)

        if self.collect_states and self.niter < 50:
            self.old_states[:, self.niter] = np.concatenate((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ))

        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.nvars, #self.lam,
        }
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:
            return ((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ), reward, done, info)#(self.state, reward, done, info)



    def _reset_vars(self):
        self.num_episodes += 1
        self.niter = 0

        # self.rewards.append(self.episode_rewards)
        # self.episode_rewards = []

    def _generate_lambda(self):
        # Draw a lambda (here: negative real for starters)
        # The number of episodes is always smaller than the number of
        # time steps, keep that in mind for the interpolation
        # hyperparameters.
        if self.lambda_real_interpolation_interval is not None:
            lam_low = np.interp(self.num_episodes,
                                self.lambda_real_interpolation_interval,
                                self.lambda_real_interval_reversed)
        else:
            lam_low = self.lambda_real_interval[0]
        self.lam = (
            1 * self.np_random.uniform(
                low=lam_low,
                high=self.lambda_real_interval[1])
            + 1j * self.np_random.uniform(
                low=self.lambda_imag_interval[0],
                high=self.lambda_imag_interval[1])
        )

    def _compute_system_matrix(self):
        # Compute the system matrix
        self.C = np.eye(self.M) - self.lam * self.dt * self.Q

    def _compute_initial_state(self):
        #self._generate_lambda()
        self._compute_system_matrix()

        # Initial guess u^0 and initial residual for the state
        u = self.prob.u0.copy()
        #u = np.ones(self.M, dtype=np.complex128)
        residual = self._compute_residual(u)
        self.initial_residual = residual

        self.initial_residual_time =    np.array(     [np.linalg.norm(residual[i])  for i in range(self.num_nodes)]   ) 

        return (u, residual)

    def reset(self):

        self._reset_vars()
        self._generate_lambda()
        # ---------------- Now set up the problem ----------------
        # This comes as read-in for the problem class
        problem_params = {}
        problem_params['lam'] = self.lam
        problem_params['nvars'] = self.nvars

        if (self.example==0): #test equation
            problem_class = Test
            self.linear = True
        else: # run heatequation
            problem_class = Heat
            self.linear = True

        self.prob = problem_class(problem_params, dtype_u=self.dtype, dtype_f=self.dtype, t=self.coll.nodes) 

        self.u0 = self.prob.u0.copy()
        (u, residual) = self._compute_initial_state()

        self.state = (u, residual)

        if self.collect_states:
            # Try if this works instead of the line below it.
            # I didn't use it for safety, but it's a bit faster.
            self.old_states[:, 0] = np.concatenate(self.state)
            self.old_states[:, 1:] = 0
            # self.old_states = np.zeros((self.M * 2, self.max_iters),
            #                            dtype=np.complex128)

        if self.collect_states:
            return self.old_states
        else:
            return self.state

    def _reward_iteration_only(self, steps):
        return -steps * self.step_penalty

    def _reward_residual_change(self, old_residual, residual, steps):
        # reward = -self.initial_residual / 100
        # reward = -self._inf_norm(residual)
        reward = abs(
            (math.log(self._inf_norm(old_residual * self.norm_factor))
             - math.log(self._inf_norm(residual * self.norm_factor)))
            / (math.log(self._inf_norm(self.initial_residual
                                       * self.norm_factor))
               - math.log(self.restol * self.norm_factor)),
        )
        reward *= self.residual_weight
        # jede der `self.max_iters` Iterationen wird bestraft
        reward -= steps * self.step_penalty
        return reward

    def _reward_gauss_kernel(self, residual, reached_convergence, steps):
        self.gauss_facts = [1]
        self.gauss_invs = [1 / self.restol]
        norm_res = self._inf_norm(residual)
        gauss_dist = sum(
            (gauss_fact
             * np.exp(-(norm_res * gauss_inv)**2 / 2))
            for (gauss_fact, gauss_inv) in zip(self.gauss_facts,
                                               self.gauss_invs)
        )
        if reached_convergence:
            extra_fact = (self.max_iters + 1 - steps)**2 * 10
        else:
            extra_fact = 1

        reward = gauss_dist * extra_fact
        return reward

    def _reward_fast_convergence(self, residual, reached_convergence, steps):
        norm_res = self._inf_norm(residual)
        if reached_convergence:
            extra_fact = (self.max_iters + 1 - steps)**2 * 10
        else:
            extra_fact = 1

        if norm_res == 0:
            # Smallest double exponent 1e-323's -log is about 744
            reward = 1000
        else:
            reward = -math.log(norm_res)
        reward *= extra_fact
        return reward

    def _reward_smooth_fast_convergence(
            self, residual, reached_convergence, steps):
        norm_res = self._inf_norm(residual)
        if reached_convergence:
            extra_fact = (self.max_iters + 1 - steps)**2 * 10
        else:
            extra_fact = 1

        if norm_res == 0:
            # Smallest double exponent 1e-323's -log is about 744
            reward = 1000
        else:
            reward = -math.log(norm_res)
        if reward > 1:
            reward = 1 + math.log(reward)
        reward *= extra_fact
        return reward

    def _reward_smoother_fast_convergence(
            self, residual, reached_convergence, steps):
        norm_res = self._inf_norm(residual)
        if reached_convergence:
            extra_fact = (self.max_iters + 1 - steps)**2 * 10
        else:
            extra_fact = 1

        if norm_res == 0:
            # Smallest double exponent 1e-323's -log is about 744
            reward = 1000
        else:
            reward = -math.log(norm_res)
        reward *= extra_fact
        if reward > 1:
            reward = 1 + math.log(reward)
        return reward

    def _reward_spectral_radius(self, scaled_action, Pinv):
        Qdmat = self._get_prec(scaled_action)
        mulpinv = Pinv.dot(self.Q - Qdmat)
        eigvals = np.linalg.eigvals(self.lam * self.dt * mulpinv)
        return max(abs(eigvals))

    def reward_func(
            self,
            old_residual,
            residual,
            reached_convergence,
            steps,
            scaled_action,
            Pinv,
    ):
        """Return the reward obtained with the `old_residual` with the
        new `residual`.
        `reached_convergence` indicates whether convergence was reached.
        `steps` indicates how many time steps to penalize.
        `scaled_action` is the action taken.
        `Pinv` is the iteration matrix.
        """
        if self.reward_strategy == 'iteration_only':
            return self._reward_iteration_only(steps)
        elif self.reward_strategy == 'residual_change':
            return self._reward_residual_change(old_residual, residual, steps)
        elif self.reward_strategy == 'gauss_kernel':
            return self._reward_gauss_kernel(
                residual, reached_convergence, steps)
        elif self.reward_strategy == 'fast_convergence':
            return self._reward_fast_convergence(
                residual, reached_convergence, steps)
        elif self.reward_strategy == 'smooth_fast_convergence':
            return self._reward_smooth_fast_convergence(
                residual, reached_convergence, steps)
        elif self.reward_strategy == 'smoother_fast_convergence':
            return self._reward_smoother_fast_convergence(
                residual, reached_convergence, steps)
        elif self.reward_strategy == 'spectral_radius':
            return self._reward_spectral_radius(scaled_action, Pinv)

        raise NotImplementedError(
            f'unknown reward strategy {self.reward_strategy}')

    def plot_rewards(self):
        plt.xlabel('time')
        plt.ylabel('reward/residual norm')

        all_rewards = [reward for ep in self.rewards for reward in ep]
        plt.plot(
            np.arange(len(all_rewards)),
            all_rewards,
            label='individual rewards',
        )

        episode_lengths = (len(ep) for ep in self.rewards)
        episode_ends = list(itertools.accumulate(episode_lengths))
        episode_rewards = [sum(ep) for ep in self.rewards]
        plt.plot(
            episode_ends,
            episode_rewards,
            label='episodic rewards',
            marker='.',
        )

        max_reward = max(map(abs, all_rewards))
        max_norm_resid = max(self.norm_resids)
        plt.plot(
            np.arange(len(self.norm_resids)),
            [r / max_norm_resid * max_reward for r in self.norm_resids],
            label='residual norm (rescaled)',
        )

        plt.legend()
        plt.savefig('rewards.pdf', bbox_inches='tight')
        plt.show()


class SDC_Step_Env(SDC_Full_Env):
    """This environment implements a single iteration of SDC, i.e.
    for each step we just do one iteration and stop if
        (a) convergence is reached (residual norm is below restol),
        (b) more than `self.max_iters` iterations are done (not converged),
        (c) diverged.
    """


    def step(self, action):

        u, old_residual = self.state

        scaled_action = self._scale_action(action)

        Qdmat = self._get_prec(scaled_action=scaled_action)
        #Pinv = self._compute_pinv(scaled_action)

        rhs  = np.squeeze( np.array(             u.flatten()  - self.dt * np.kron(Qdmat, np.eye(self.nvars)) @ self.prob.eval_f(u) + old_residual.flatten()                        )) 

        u = self.solve_system( u, rhs, Qdmat)   #TODO clean this up and use better data-structures from pySDC

        # Do the iteration (note that we already have the residual)
        # u += Pinv @ old_residual

        # The new residual and its norm
        Cu     =  u - self.dt * np.kron(self.Q, np.eye(self.nvars)) @ self.prob.eval_f(u, 0.0) 


        residual = np.squeeze(np.array(self.u0.flatten() - Cu.flatten() ))

        # The new residual and its norm
        # residual = self._compute_residual(u)

        norm_res = self._inf_norm(residual)
        norm_res_old = self._inf_norm(old_residual)

        self.niter += 1

        # Check if something went wrong
        err = np.isnan(norm_res) or np.isinf(norm_res)
        # so far this seems to be the best setup:
        #   - stop if residual gets larger than the initial one
        #     (not needed, but faster)
        #   - reward = -self.max_iters, if this happens (crucial!)
        err = err or norm_res > norm_res_old * 100
        # Stop iterating when converged
        done = norm_res < self.restol

        if not err:
            reward = self.reward_func(
                old_residual,
                residual,
                done,
                self.niter,
                scaled_action,
                Pinv,
            )

        else:
            # return overall reward of -(self.max_iters + 1)
            # (slightly worse than -self.max_iters in the
            # "not converged" scenario)
            # reward = -self.step_penalty * ((self.max_iters + 2) - self.niter)
            reward = -self.step_penalty * (self.max_iters + 1)
            # reward = -(self.max_iters + 1) + self.niter
            # reward = -self.max_iters + self.niter
        # Stop iterating when iteration count is too high or when
        # something bad happened
        done = done or self.niter >= self.max_iters or err
        # self.episode_rewards.append(reward)
        # self.norm_resids.append(norm_res)

        _u   = u.reshape(self.num_nodes, self.nvars)
        _res = residual.reshape(self.num_nodes, self.nvars)


        self.state = (u, residual)
        if self.collect_states and self.niter < self.max_iters:
            self.old_states[:, self.niter] = np.concatenate((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ))#np.concatenate(self.state)

        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.lam,
        }
        #if self.collect_states:
        #    return (self.old_states, reward, done, info)
        #else:
        #    return (self.state, reward, done, info)

        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:

            return ((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ), reward, done, info)
