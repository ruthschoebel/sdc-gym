# sdc-gym

recent changes:
heat equation has the form u_t = nu/(dx^2) A, where A is a Finite Differences discretization and nu is defined by spectral radius(nu/dx^2 A)=lambda

use --train_heat True --test_heat True to train or test heat equation

the test equation u_t = lambda u is used in both cases (training and testing) as default

normally the test equation should be used for training and the heat equation should be used for testing, run for this setup e.g.  

```shell
python rl_playground.py --envname sdc-v1 --num_envs 8 --model_class PPG --activation_fn ReLU --collect_states True --reward_iteration_only False --norm_obs True --tests 100 --steps 1000 --train_heat False --test_heat True
```

in the optimal case the result of RL is between MIN and optMIN, this does not happen yet (see output figure)
