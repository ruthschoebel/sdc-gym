
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

class SDC_Full_Env(gym.Env):
    """This environment implements a full iteration of SDC, i.e. for
    each step we iterate until
        (a) convergence is reached (residual norm is below restol),
        (b) more than 50 iterations are done (not converged),
        (c) diverged.
    """
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

        if(run_example==0):
            print("running TEST-Equation")
            self.nvars=1

        elif(run_example==1):
            print("running Heat-Equation")
            self.nvars=nvars


            print("Heat")
        elif(run_example==2):
            print("running Fisher")
            self.nvars=nvars



        self.run_example=run_example

        self.u0 = None
        self.np_random = None
        self.niter = None
        self.restol = restol
        self.dt = dt


        self.Q = self.coll.Qmat[1:, 1:]
        self.C = None
        self.lam = None

        self.old_res = None
        self.prec = prec
        self.initial_residual = None        


        self.prob = None
        self.rhs_mat = None


        self.lambda_real_interval = lambda_real_interval
        self.lambda_real_interval_reversed = list(
            reversed(lambda_real_interval))
        self.lambda_imag_interval = lambda_imag_interval
        self.lambda_real_interpolation_interval = \
            lambda_real_interpolation_interval

        self.norm_factor = norm_factor
        self.residual_weight = residual_weight
        self.step_penalty = step_penalty
        self.reward_iteration_only = reward_iteration_only
        self.collect_states = collect_states

        self.num_episodes = 0

        self.observation_space = spaces.Box(
            low=-1E10,
            high=+1E10,
            shape=(M * 2, 50) if collect_states else (2, M),
            dtype=self.dtype,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(M,),
            dtype=np.float64 if use_doubles else np.float32,
        )

        self.seed(seed)
        self.state = None
        if collect_states:
            self.old_states = np.zeros((M * 2, 50), dtype=self.dtype)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def _get_prec(self, scaled_action, M):
        """Return a preconditioner based on the `scaled_action`.
        `M` is the problem size.
        """

        if self.prec is None:
            Qdmat = np.zeros_like(self.Q)
            np.fill_diagonal(Qdmat, scaled_action)
        elif self.prec.upper() == 'LU':
            QT = self.Q.T
            [_, _, U] = scipy.linalg.lu(QT, overwrite_a=True)
            Qdmat = U.T
        elif self.prec.lower() == 'min':
            Qdmat = np.zeros_like(self.Q)
            if M == 7:
                x = [
                    0.15223871397682717,
                    0.12625448001038536,
                    0.08210714764924298,
                    0.03994434742760019,
                    0.1052662547386142,
                    0.14075805578834127,
                    0.15636085758812895
                ]
            elif M == 5:
                x = [
                    0.2818591930905709,
                    0.2011358490453793,
                    0.06274536689514164,
                    0.11790265267514095,
                    0.1571629578515223,
                ]
            elif M == 4:
                x = [
                    0.3198786751412953,
                    0.08887606314792469,
                    0.1812366328324738,
                    0.23273925017954,
                ]
            elif M == 3:
                x = [
                    0.3203856825077055,
                    0.1399680686269595,
                    0.3716708461097372,
                ]
            else:

                x = np.zeros(M)
            np.fill_diagonal(Qdmat, x)
        elif (self.prec == 'optMIN'): 
            Qdmat = np.zeros_like(self.Q)
            if M == 3:
                Qdmat = np.zeros_like(self.Q)
                x0 = [
                        0.3203856825077055,
                        0.1399680686269595,
                        0.3716708461097372,
                    ]
                def rho(x):
                    return max(abs(np.linalg.eigvals(self.lam   *   np.linalg.inv( np.eye(self.num_nodes)    -  self.lam *  np.diag([x[i] for i in range(self.num_nodes)])   ).dot(self.Q    -   np.diag([x[i] for i in range(self.num_nodes)]))    )                   ))
                x = opt.minimize(rho, x0, method='Nelder-Mead').x       
            else:

                x = np.zeros(M)
            np.fill_diagonal(Qdmat, x)
        elif (self.prec == 'trivial'): 
            Qdmat = np.zeros_like(self.Q)
            if M == 3:
                Qdmat = np.zeros_like(self.Q)
                x = [
                        1.0,
                        1.0,
                        1.0,
                    ]
            np.fill_diagonal(Qdmat, x)
        else:
            raise NotImplementedError()
        return Qdmat

    def step(self, action):

        u, old_residual = self.state


        scaled_action = np.interp(action, (-1, 1), (0, 1))


        Qdmat = self._get_prec(scaled_action=scaled_action, M=u.size)

        # Precompute the inverse of P
        Pinv = np.linalg.inv(
            np.eye(self.num_nodes*self.nvars) - self.dt * np.kron(Qdmat, self.rhs_mat),
        )
        norm_res_old = np.linalg.norm(old_residual, np.inf)

        done = False
        err = False
        self.niter = 0
        # Start the loop
        while not done and not self.niter >= 50:
            self.niter += 1

            u =  np.squeeze(np.array(u.flatten() +    Pinv @ (self.u0.flatten() - np.squeeze(np.array(self.C @ u)))) )  #TODO clean this up and use better data-structures from pySDC

            # Compute the residual and its norm
            residual = self.u0 - self.C @ u
            norm_res = np.linalg.norm(residual, np.inf)
            # stop if something goes wrong
            err = np.isnan(norm_res) or np.isinf(norm_res)
            # so far this seems to be the best setup:
            #   - stop if residual gets larger than the initial one
            #     (not needed, but faster)
            #   - reward = -50, if this happens (crucial!)

            _u   = u.reshape(self.num_nodes, self.nvars)
            _res = residual.reshape(self.num_nodes, self.nvars)

            if self.collect_states and self.niter < 50:
                self.old_states[:, self.niter] = np.concatenate((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ) )

            err = err or norm_res > norm_res_old * 100
            if err:
                reward = -self.step_penalty * 51
                # reward = -51
                break
            # check for convergence
            done = norm_res < self.restol

        if not err:
            reward = self.reward_func(
                self.initial_residual,
                residual,
                self.niter,
            )

        done = True
        # self.episode_rewards.append(reward)
        # self.norm_resids.append(norm_res)

        self.state = (u, residual)

        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.lam,
        }
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:

            return ((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ), reward, done, info)

    def reset(self):
        self.num_episodes += 1


        if self.lambda_real_interpolation_interval is not None:
            lam_low = np.interp(self.num_episodes,
                                self.lambda_real_interpolation_interval,
                                self.lambda_real_interval_reversed)
        else:
            lam_low = self.lambda_real_interval[0]


        if (self.dtype==np.float64): #falls float Datentyp keine komplexen lambda
            self.lam = (
                1 * self.np_random.uniform(
                    low=lam_low,
                    high=self.lambda_real_interval[1])
            )
        else:
            self.lam = (
                1 * self.np_random.uniform(
                    low=lam_low,
                    high=self.lambda_real_interval[1])
                + 1j * self.np_random.uniform(
                    low=self.lambda_imag_interval[0],
                    high=self.lambda_imag_interval[1])
            )

        # ---------------- Now set up the problem ----------------
        # This comes as read-in for the problem class
        #problem_params = dict()
        problem_params = {}

        if (self.run_example==0): #test
            problem_params['lam'] = self.lam
            problem_params['nvars'] = self.nvars
            problem_class = Test
            dtype=self.dtype
            self.prob = problem_class(problem_params, dtype_u=dtype, dtype_f=dtype) 
            self.u0 = np.ones(self.num_nodes*self.nvars, dtype=self.dtype)
            u = np.ones(self.num_nodes*self.nvars, dtype=self.dtype)
            #dtype=mesh
        elif(self.run_example==1): # run Heat
            problem_params['lam'] = self.lam
            problem_params['nvars'] = self.nvars
            problem_class = Heat
            dtype=self.dtype
            self.prob = problem_class(problem_params, dtype_u=dtype, dtype_f=dtype, t=self.coll.nodes) 
            self.u0 = self.prob.u_exact(self.coll.nodes*0)
            u = self.prob.u_exact(self.coll.nodes*0)

        elif(self.run_example==2): # run Heat
            problem_params['lam'] = self.lam
            problem_params['nvars'] = self.nvars
            problem_class = Flame
            dtype=self.dtype
            self.prob = problem_class(problem_params, dtype_u=dtype, dtype_f=dtype, t=self.coll.nodes) 
            self.u0 = self.prob.u_exact(self.coll.nodes*0) 
            u = np.ones(self.num_nodes*self.nvars, dtype=self.dtype)




        self.rhs_mat = self.prob.rhs_mat


        # Compute the system matrix 
        self.C = np.eye(self.num_nodes*self.nvars) - self.dt * np.kron(self.Q, self.rhs_mat) 





        Cu     = u - self.dt * np.kron(self.Q, np.eye(self.nvars)) @ self.prob.eval_f(u)  

        residual = np.squeeze(np.array(self.u0.flatten() - Cu.flatten() )) 
        self.initial_residual = residual



        _u   = u.reshape(self.num_nodes, self.nvars)
        _res = residual.reshape(self.num_nodes, self.nvars)


        self.state = (u, residual)  
        if self.collect_states:

            self.old_states = np.zeros((self.num_nodes * 2, 50), dtype=self.dtype)
            self.old_states[:, 0] = np.concatenate((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ) )


        # self.rewards.append(self.episode_rewards)
        # self.episode_rewards = []
        self.niter = 0

        if self.collect_states:
            return self.old_states
        else:

            return (np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ) 


    def reward_func(self, old_residual, residual, steps=1):
        """Return the reward obtained with the `old_residual` with the
        new `residual`.
        `steps` indicates how many time steps to penalize.
        """
        if self.reward_iteration_only:
            return -steps * self.step_penalty

        # reward = -self.initial_residual / 100
        # reward = -np.linalg.norm(residual, np.inf)
        reward = abs(
            (math.log(np.linalg.norm(old_residual * self.norm_factor, np.inf))
             - math.log(np.linalg.norm(residual * self.norm_factor, np.inf)))
            / (math.log(np.linalg.norm(self.initial_residual
                                       * self.norm_factor, np.inf))
               - math.log(self.restol * self.norm_factor)),
        )
        reward *= self.residual_weight
        # jede der 50 Iterationen wird bestraft
        reward -= steps * self.step_penalty
        return reward

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
        (b) more than 50 iterations are done (not converged),
        (c) diverged.
    """
    def solve_system(self, u,  rhs, Qdmat):

        Id = sp.eye(self.num_nodes*self.nvars)
        n=1
        while n < 5:

            #TODO: check if problem has boundary conditions
            #TODO: node perspective is more efficient for solve
            g = np.squeeze(np.array(u.flatten()  - self.dt * np.kron(Qdmat, np.eye(self.nvars)) @ self.prob.eval_f(u, 0.0)   - rhs.flatten() ))
            


            res = np.linalg.norm(g, np.inf)

            if res < 1e-15:
                break

            dg = Id - self.dt * np.kron(Qdmat, np.eye(self.nvars)) @   self.prob.eval_j(u, 0.0)



            du = spsolve(dg, g)
            u -= du 
            n += 1

        return u #np.squeeze(np.array(u.flatten() +    Pinv @ old_residual.flatten()  ))

    def step(self, action):

        u, old_residual = self.state

 
        scaled_action = np.interp(action, (-1, 1), (0, 1))


        Qdmat = self._get_prec(scaled_action=scaled_action, M=self.num_nodes)





        rhs  = np.squeeze( np.array(             u.flatten()  - self.dt * np.kron(Qdmat, np.eye(self.nvars)) @ self.prob.eval_f(u) + old_residual.flatten()                        )) 

        u = self.solve_system( u, rhs, Qdmat)   #TODO clean this up and use better data-structures from pySDC


        #u += Pinv @ old_residual

        # The new residual and its norm
        Cu     =  u - self.dt * np.kron(self.Q, np.eye(self.nvars)) @ self.prob.eval_f(u, 0.0)  #u.flatten() 

        residual = np.squeeze(np.array(self.u0.flatten() - Cu.flatten() ))


        norm_res = np.linalg.norm(residual, np.inf)
        norm_res_old = np.linalg.norm(old_residual, np.inf)

        norm_res_old = np.linalg.norm(old_residual, np.inf)

        self.niter += 1

        # Check if something went wrong
        err = np.isnan(norm_res) or np.isinf(norm_res)


        err = err or norm_res > norm_res_old * 100


        done = norm_res < self.restol or self.niter >= 50 or err


        if not err:
            reward = self.reward_func(old_residual, residual, self.niter)


        else:

            reward = -self.step_penalty * 51



        _u   = u.reshape(self.num_nodes, self.nvars)
        _res = residual.reshape(self.num_nodes, self.nvars)


        self.state = (u, residual)
        if self.collect_states and self.niter < 50:
            self.old_states[:, self.niter] = np.concatenate((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ))


        info = {
            'residual': norm_res,
            'niter': self.niter,
            'lam': self.lam,
        }
        if self.collect_states:
            return (self.old_states, reward, done, info)
        else:

            return ((np.array([np.linalg.norm(_u[i], np.inf)  for i in range(self.num_nodes)]) , np.array([np.linalg.norm(_res[i], np.inf)  for i in range(self.num_nodes)])          ), reward, done, info)



