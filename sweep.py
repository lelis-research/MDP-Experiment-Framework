import argparse
import itertools
import os
import numpy as np
from functools import partial
import pickle
import json

from RLBase.Trainers import OnlineTrainer
from RLBase.Evaluate import SingleExpAnalyzer
from RLBase.Environments import get_env, ENV_LST
from Configs.loader import load_config, fmt_wrap
from Configs.config_agents_base import AGENT_DICT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, required=True,
                   help='SLURM_ARRAY_TASK_ID')
     # Config file name
    parser.add_argument("--config", type=str, default="config_agents_base", help="path to the experiment config file")
    # Agent type to run
    parser.add_argument("--agent", type=str, default="Random", choices=list(AGENT_DICT.keys()), help="Which agent to run")
    # Environment name
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0", choices=ENV_LST, help="which environment")
    # List of wrappers for the environment
    parser.add_argument("--env_wrapping",   type=json.loads, default="[]", help="list of wrappers")
    # A list of dictionary of the parameters for each wrapper
    parser.add_argument("--wrapping_params", type=json.loads, default="[]", help="list of dictionary represeting the parameters for each wrapper")
    # A dictionary of the environment parameters
    parser.add_argument("--env_params",     type=json.loads, default="{}", help="dictionary of the env parameters")
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
    # Info for agent specification
    parser.add_argument("--info", type=json.loads, help='JSON dict for the default values, e.g. \'{"lr":0.001,"epochs":10}\'')
    # Search space
    parser.add_argument("--hp_search_space", type=json.loads, 
                        help='JSON dict of hyper-params, e.g. \'{"actor_step_size":[0.001,0.0003], "rollout_steps":[1024,2048]}\'')
    return parser.parse_args()


def validate_hp_space(space: dict):
    if not isinstance(space, dict) or not space:
        raise ValueError("hp_search_space must be a non-empty dict")
    for k, v in space.items():
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            raise ValueError(f"hp_search_space['{k}'] must be a non-empty list/tuple; got {v}")
    return space

def main():
    args = parse_args()   
    hp_search_space = validate_hp_space(args.hp_search_space)
        
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
        env_params   = args.env_params,
        wrapping_lst = args.env_wrapping,
        wrapping_params = args.wrapping_params,
    )
   
    params = args.info
    params.update(tuning_params) #update the default params with the tuning params from search space
    print(f"Start trial {args.idx+1}/{total}: {params}")
    
    agent_fn = lambda env: config.AGENT_DICT[args.agent](env, params)
    
    exp_name = f"{args.name_tag}_seed[{args.seed}]" #_{timestamp}
    env_params_str = "_".join(f"{k}-{v}" for k, v in args.env_params.items())  # env param dictionary to str
    wrappers_str = "_".join(fmt_wrap(w, p) for w, p in zip(args.env_wrapping, args.wrapping_params))
    exp_dir = os.path.join(runs_dir, f"{args.env}_{env_params_str}", wrappers_str, args.agent, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # create a directory for this trial
    trial_dir = os.path.join(exp_dir, f'trial_{args.idx}')
    os.makedirs(trial_dir, exist_ok=True)

    # instantiate and run
    experiment = OnlineTrainer(env_fn, agent_fn, exp_dir=trial_dir, args=args)
    
    metrics = experiment.multi_run(
        num_runs=args.num_runs,
        num_episodes=args.num_episodes,
        total_steps=args.total_steps,
        num_workers=args.num_workers,
        seed_offset=args.seed,
        dump_metrics=True,
    )

    # save agent and seeds
    with open(os.path.join(trial_dir, 'agent.txt'), 'w') as f:
        f.write(str(experiment.agent))
    analyzer = SingleExpAnalyzer(metrics=metrics)
    analyzer.plot_combined(save_dir=trial_dir, show_legend=False)
    analyzer.save_seeds(save_dir=trial_dir)
    

    print(f"Done trial {args.idx+1}/{total}: {params}")
    


if __name__ == '__main__':
    main()


