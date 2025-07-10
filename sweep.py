import argparse
import itertools
import os
import numpy as np
from functools import partial
import pickle

from RLBase.Agents.Utils import HyperParameters
from RLBase.Experiments import BaseExperiment, ParallelExperiment
from RLBase.Evaluate import SingleExpAnalyzer
from RLBase.Environments import get_env, ENV_LST
from Configs.loader import load_config, fmt_wrap
from Configs.base_config import AGENT_DICT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, required=True,
                   help='SLURM_ARRAY_TASK_ID')
     # Config file name
    parser.add_argument("--config", type=str, default="base_config", help="path to the experiment config file")
    # Agent type to run
    parser.add_argument("--agent", type=str, default="Random", choices=list(AGENT_DICT.keys()), help="Which agent to run")
    # Environment name
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0", choices=ENV_LST, help="which environment")
    # Add a name tag
    parser.add_argument("--name_tag", type=str, default="", help="name tag for experiment folder")
    # seed for reproducibility
    parser.add_argument('--seed',  type=int, default=1)
    # Number of runs per configuration
    parser.add_argument("--num_runs", type=int, default=3, help="number of runs per each Hyper-Params")
    # Number of total environment steps per run
    parser.add_argument("--total_steps", type=int, default=0, help="number of steps in each run")
    # Episodes per run
    parser.add_argument("--num_episodes", type=int, default=0, help="number of episode in each run")
    # Maximum steps per episode
    parser.add_argument("--episode_max_steps", type=int, default=200, help="maximum number of steps in each episode")
    # Number of parallel environments
    parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments")
    # Number of workers to run parallel for each trial
    parser.add_argument('--num_workers',  type=int, default=1)
    return parser.parse_args()


def main(hp_search_space):
    args = parse_args()   

    # Build list of (actor, critic, rollout) combos
    keys = list(hp_search_space.keys())
    grid = list(itertools.product(*(hp_search_space[k] for k in keys)))
    total = len(grid)
    if args.idx < 0 or args.idx >= total:
        raise IndexError(f"idx must be in [0,{total}); got {args.idx}")

    # pick current combo
    values = grid[args.idx]
    tuning_params = dict(zip(keys, values))

    # load config and experiment factory
    cofig_path = os.path.join('Configs', f'{args.config}.py')
    config = load_config(cofig_path)
    
    runs_dir = f"Runs/Sweep/"
    os.makedirs(runs_dir, exist_ok=True)
    
    env_fn = partial(
        get_env,
        env_name     = args.env,
        num_envs     = args.num_envs,
        max_steps    = args.episode_max_steps,
        env_params   = config.env_params,
        wrapping_lst = config.env_wrapping,
        wrapping_params = config.wrapping_params,
    )
    agent_fn = lambda env: config.AGENT_DICT[args.agent](env)
    
    # get the default params and the tuning
    default_hp = agent_fn(env_fn()).hp
    base_dict = default_hp.to_dict()
    base_dict.update(tuning_params)
    tuning_hp = HyperParameters(**base_dict)
    
    # choose experiment class
    exp_class = BaseExperiment if args.num_envs == 1 else ParallelExperiment
    
    exp_name = f"{args.name_tag}_seed[{args.seed}]" #_{timestamp}
    env_params_str = "_".join(f"{k}-{v}" for k, v in config.env_params.items())  # env param dictionary to str
    wrappers_str = "_".join(fmt_wrap(w, p) for w, p in zip(config.env_wrapping, config.wrapping_params))
    exp_dir = os.path.join(runs_dir, f"{args.env}_{env_params_str}", wrappers_str, args.agent, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # create a directory for this trial
    trial_dir = os.path.join(exp_dir, f'trial_{args.idx}')
    os.makedirs(trial_dir, exist_ok=True)

    # instantiate and run
    experiment = exp_class(env_fn, agent_fn, exp_dir=trial_dir, args=args)
    
    metrics = experiment.multi_run(
        num_runs=args.num_runs,
        num_episodes=args.num_episodes,
        total_steps=args.total_steps,
        num_workers=args.num_workers,
        seed_offset=args.seed,
        dump_metrics=True,
        tuning_hp=tuning_hp,
    )

    # save agent and seeds
    with open(os.path.join(trial_dir, 'agent.txt'), 'w') as f:
        f.write(str(experiment.agent))
    analyzer = SingleExpAnalyzer(metrics=metrics)
    analyzer.save_seeds(save_dir=trial_dir)

    print(f"Done trial {args.idx+1}/{total}: {full_params}")
    


if __name__ == '__main__':
    hp_search_space = {
        'actor_step_size':   [0.0001, 0.001, 0.01],
        'critic_step_size':  [0.0001, 0.001, 0.01],
        'rollout_steps':     [1, 3, 5, 7, 9],
    }
    
    main(hp_search_space)


