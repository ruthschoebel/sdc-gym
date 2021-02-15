import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt

import sdc_gym
import utils
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--steps',
        type=float,
        default=10000,
        help='Number of action steps to take in the environment.',
    )
    parser.add_argument(
        '--M',
        type=int,
        default=3,
        help=(
            '"Difficulty" of the problem '
            '(proportionally relates to nth-order differential equation)'
            '(choose M = 3 or M = 5 for comparing MIN-preconditioner)'
        ),
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=1.0,
        help=(
            'Size of time step.'
        ),
    )
    parser.add_argument(
        '--reward_iteration_only', #does only affect sdc-v1
        type=int,
        default=True, #use False for the new reward that depends on the residual
        help=(
            'How to reward the agent '
        ),
    )
    parser.add_argument(
        '--restol',
        type=float,
        default=1E-10, #1E-10,
        help=(
            'Residual tolerance; '
            'when residual is below this, stop iterating.'
            'Lower means better solution but more iterations.'
        ),
    )
    parser.add_argument(
        '--envname',
        type=str,
        default='sdc-v1',
        choices=['sdc-v0', 'sdc-v1'],
        help=(
            'Gym environment to use;\n    sdc-v0 – SDC with a full iteration '
            'per step (no intermediate observations),\n    sdc-v1 – SDC with '
            'a single iteration per step (with intermediate observations).'
        ),
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        # default for PPO2 in stable-baselines
        default=25E-05,
        help='Learning rate/step size of the model.',
    )
    parser.add_argument(
        '--rescale_lr',
        type=bool,
        default=True,
        help=(
            'Whether to rescale the learning rate by the number'
            'of environments.'
        ),
    )
    parser.add_argument(
        '--model_class',
        type=str,
        default='PPO2',
        help='Class of model to instantiate.',
    )
    parser.add_argument(
        '--policy_class',
        type=str,
        default='MlpPolicy',
        help='Class of model policy.',
    )
    parser.add_argument(
        '--policy_kwargs',
        type=json.loads,
        default={},
        help=(
            'Keyword arguments for policy creation. '
            'See the documentation for details.'
            'Example for MlpLstmPolicy: '
            '`--policy_kwargs \'{"net_arch": [128, 128, "lstm"]}\'`'
        ),
    )
    parser.add_argument(
        '--tests',
        type=float,
        default=50,
        help='Number of test runs for each preconditioning method.',
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=4,
        help='How many environments to use in parallel.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help=(
            'Base seed for seeding the environments. For multiple '
            'environments, all will have different seeds based on this one.'
        ),
    )
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=0,
        help=(
            'How often to evaluate the model during training, storing the '
            'best performing one. If this is 0, do not evaluate.'
        ),
    )
    return parser.parse_args()


def test_model(model, env, ntests, name):
    """Test the `model` in the Gym `env` `ntests` times.
    `name` is the name for the test run for logging purposes.
    """
    mean_niter = 0
    nsucc = 0
    results = []

    num_envs = env.num_envs
    # Amount of test that will be ran in total
    ntests_total = ntests * num_envs
    a=[]
    for i in range(ntests):
        state = None
        obs = env.reset()
        done = [False for _ in range(num_envs)]
        #print(i)
        while not all(done):
            action, state = model.predict(
                obs,
                state=state,
                mask=done,
                deterministic=True,
            )
            obs, rewards, done, info = env.step(action)
            scaled_action = np.interp(action, (-1, 1), (0, 1))

        if (name=='RL'):
            print(scaled_action)
            a = np.append(a, scaled_action)        


        for (env_, info_) in zip(env.envs, info):
            # We work on the info here because its information is
            # not lost with the automatic env reset from a
            # vectorized environment.
            if info_['niter'] < 50 and info_['residual'] < env_.restol:
                nsucc += 1
                mean_niter += info_['niter']
                # Store each iteration count together with the respective
                # lambda to make nice plots later on
                results.append((info_['lam'].real, info_['niter']))
    if (name=='RL'):
        print(a)
        a=a.reshape(int(a.size/3),3)
        print("Qd_11")
        print("Mittel ", np.mean(a[:,0])) #Mittelwert
        print("Abweichung ", np.std(a[:,0]))  #Standardabweichung
        print("Maximum", np.max(a[:,0]))  #Maximum
        print("Minimum ", np.min(a[:,0]))

        print("Qd_22")
        print("Mittel ", np.mean(a[:,1])) #Mittelwert
        print("Abweichung ", np.std(a[:,1]))  #Standardabweichung
        print("Maximum", np.max(a[:,1]))  #Maximum
        print("Minimum ", np.min(a[:,1]))

        print("Qd_33")
        print("Mittel ", np.mean(a[:,2])) #Mittelwert
        print("Abweichung ", np.std(a[:,2]))  #Standardabweichung
        print("Maximum", np.max(a[:,2]))  #Maximum
        print("Minimum ", np.min(a[:,2]))

    # Write out mean number of iterations (smaller is better) and the
    # success rate (target: 100 %)
    if nsucc > 0:
        mean_niter /= nsucc
    else:
        mean_niter = 666
    print(f'{name}  -- Mean number of iterations and success rate: '
          f'{mean_niter:4.2f}, {nsucc / ntests_total * 100} %')
    return results


def plot_results(results, color, label):
    sorted_results = sorted(results, key=lambda x: x[0])
    plt.plot(
        [i[0] for i in sorted_results],
        [i[1] for i in sorted_results],
        color=color,
        label=label,
    )


def main():
    args = parse_args()
    seed = args.seed
    eval_seed = seed
    if eval_seed is not None:
        eval_seed += args.num_envs

    # ---------------- TRAINING STARTS HERE ----------------

    # Set up gym environment
    env = utils.make_env(args, include_norm=True)
    # Set up model
    model_class = utils.get_model_class(args.model_class)
    policy_class = utils.get_policy_class(args.policy_class, args.model_class)

    policy_kwargs = args.policy_kwargs

    # Learning rate to try for PPO2: 1E-05
    # Learning rate to try for ACKTR: 1E-03
    learning_rate = args.learning_rate
    if args.rescale_lr:
        learning_rate *= args.num_envs

    eval_callback = utils.create_eval_callback(args, learning_rate)

    model_kwargs = {
        'verbose': 1,
        'policy_kwargs': policy_kwargs,
        'tensorboard_log': str(Path(
            f'./sdc_tensorboard/'
            f'{args.model_class.lower()}_{args.policy_class.lower()}/'
        )),
        'learning_rate': learning_rate,
        'seed': seed,
    }

    utils.check_num_envs(args, policy_class)
    utils.maybe_fix_nminibatches(model_kwargs, args, policy_class)

    model = model_class(policy_class, env, **model_kwargs)

    start_time = time.perf_counter()
    # Train the model (need to put at least 100k steps to
    # see something)
    model.learn(total_timesteps=int(args.steps), callback=eval_callback)
    duration = time.perf_counter() - start_time
    print(f'Training took {duration} seconds.')

    fname = Path(f'sdc_model_{args.model_class.lower()}_'
                 f'{args.policy_class.lower()}_{learning_rate}.zip')
    model.save(str(fname))
    # delete trained model to demonstrate loading, not really necessary
    # del model

    # ---------------- TESTING STARTS HERE ----------------

    # Not vectorizing is faster for testing for some reason.
    num_test_envs = args.num_envs if policy_class.recurrent else 1

    ntests = int(args.tests)
    ntests = utils.maybe_fix_ntests(ntests, num_test_envs)

    # Load the trained agent for testing
    # model = model_class.load(fname)

    start_time = time.perf_counter()
    # Test the trained model.
    env = utils.make_env(args, num_envs=num_test_envs, seed=eval_seed)
    results_RL = test_model(model, env, ntests, 'RL')

    # Restart the whole thing, but now using the LU preconditioner (no RL here)
    # LU is serial and the de-facto standard. Beat this (or at least be on par)
    # and we win!
    env = utils.make_env(
        args,
        num_envs=num_test_envs,
        prec='LU',
        seed=eval_seed,
    )
    results_LU = test_model(model, env, ntests, 'LU')

    # Restart the whole thing, but now using the minization preconditioner
    # (no RL here)
    # This minimization approach are just magic numbers we found using
    # indiesolver.com, parallel and proof-of-concept
    env = utils.make_env(
        args,
        num_envs=num_test_envs,
        prec='min',
        seed=eval_seed,
    )
    results_min = test_model(model, env, ntests, 'MIN')
    duration = time.perf_counter() - start_time
    print(f'Testing took {duration} seconds.')

    # Plot all three iteration counts over the lambda values
    plt.xlabel('re(λ)')
    plt.ylabel('iterations')

    plot_results(results_RL, color='b', label='RL')
    plot_results(results_LU, color='r', label='LU')
    plot_results(results_min, color='g', label='MIN')

    plt.legend()

    fig_path = utils.find_free_path('results_{:>04}.pdf')
    plt.savefig(fig_path, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()
