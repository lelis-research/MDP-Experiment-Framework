import os
import datetime
import numpy as np
import optuna
import argparse
import argcomplete
import math
from optuna.samplers import GridSampler, TPESampler
from functools import partial
from optuna.storages import RDBStorage
import json
from RLBase.Agents.Utils import HyperParameters  # For handling hyper-parameter objects
from RLBase.Experiments import BaseExperiment, ParallelExperiment
from RLBase.Evaluate import SingleExpAnalyzer
from RLBase.Environments import get_env, ENV_LST
from Configs.config_agents_base import AGENT_DICT
from Configs.loader import load_config, fmt_wrap

def parse():
    parser = argparse.ArgumentParser()
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
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    # Number of hyper-parameter configurations (trials)
    parser.add_argument("--num_trials", type=int, default=None, help="number of Hyper-Params to try")
    # Number of workers to run parallel for each trial
    parser.add_argument("--num_workers_each_trial", type=int, default=None, help="number of workers per trial (useful for multiple runs for each trial)")
    # Number of parallel jobs
    parser.add_argument("--num_jobs", type=int, default=None, help="number of parallel jobs on optuna")
    # Number of runs per configuration
    parser.add_argument("--num_runs", type=int, default=2, help="number of runs per each Hyper-Params")
    # Number of total environment steps per run
    parser.add_argument("--total_steps", type=int, default=0, help="number of steps in each run")
    # Episodes per run
    parser.add_argument("--num_episodes", type=int, default=0, help="number of episode in each run")
    # Maximum steps per episode
    parser.add_argument("--episode_max_steps", type=int, default=200, help="maximum number of steps in each episode")
    # Number of parallel environments
    parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments")
    # Ratio of last episodes to consider for metric calculation
    parser.add_argument("--metric_ratio", type=float, default=0.5, help="Ratio of the last episode to consider")
    # Exhaustive grid search instead of baysian optimization
    parser.add_argument("--exhaustive", action="store_true", help="If set, run an exhaustive grid search instead of TPE-based tuning")
    # Just create the optuna study 
    parser.add_argument("--just_create_study", action="store_true", help="Just create the optuna study")
    # Info for agent specification
    parser.add_argument("--info", type=json.loads, help='JSON dict, e.g. \'{"lr":0.001,"epochs":10}\'')

    argcomplete.autocomplete(parser)
    return parser.parse_args()

def make_grid(low, high, n):
    # For floats: n values including both endpoints
    return list(np.linspace(low, high, num=n))

def tune_hyperparameters(env_fn, agent_fn, default_hp, hp_search_space, exp_dir, exp_class, 
                         exhaustive=True, ratio=0.5, n_trials=20, num_workers_each_trial=1,
                         num_runs=3, num_episodes=0, total_steps=0, n_jobs=1,
                         seed_offset=1, study_name=None, storage=None,
                         args=None):
    """
    Tune hyperparameters using Optuna.
    
    Args:
        env: Environment instance.
        agent: Agent instance (must have set_hp() method).
        default_hp: Default HyperParameters object.
        hp_search_space: Dictionary of search ranges for hyperparameters.
        exp_dir: Directory to save experiment logs.
        exp_class: Experiment class (e.g., BaseExperiment or ParallelExperiment).
        ratio: Fraction of last episodes used to compute average reward.
        n_trials: Number of Optuna trials.
        num_runs: Number of runs per trial.
        num_episodes: Episodes per run.
        seed_offset: Fixed seed offset for reproducibility.
    
    Returns:
        best_hp: HyperParameters object with best found values.
        study: The Optuna study object.
    """
    # Convert default hyper-parameters to a dictionary for tweaking.
    base_params = default_hp.to_dict()
    
    def objective(trial):
        # Sample new hyperparameters within specified ranges.
        new_params = {}

        if exhaustive:
            for key in hp_search_space:
                new_params[key] = trial.suggest_categorical(key, hp_search_space[key])
            for key, default_value in base_params.items():
                if key not in hp_search_space:
                    new_params[key] = default_value
        else:
            for key, default_value in base_params.items():
                if key in hp_search_space:
                    if isinstance(default_value, float):
                        low, high = hp_search_space[key]
                        new_params[key] = trial.suggest_float(key, low, high)
                    elif isinstance(default_value, int):
                        low, high = hp_search_space[key]
                        new_params[key] = trial.suggest_int(key, low, high)
                    else:
                        new_params[key] = default_value
                else:
                    new_params[key] = default_value

        # Create a unique directory for the trial.
        trial_dir = os.path.join(exp_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        experiment = exp_class(env_fn, agent_fn, exp_dir=trial_dir, args=args)
        
        # Update agent with the sampled hyperparameters.
        tuning_hp = HyperParameters(**new_params)
        
        # Run the experiment for the current trial.
        metrics = experiment.multi_run(num_episodes=num_episodes, num_runs=num_runs,
                                       total_steps=total_steps, num_workers=num_workers_each_trial,
                                       seed_offset=seed_offset, dump_metrics=True, 
                                       tuning_hp=tuning_hp)
        with open(f"{trial_dir}/agent.txt", "w") as file:
            file.write(str(experiment.agent))
            
        # Save seed info.
        analyzer = SingleExpAnalyzer(metrics=metrics)
        analyzer.save_seeds(save_dir=trial_dir)

        # Compute average reward over the last fraction of episodes, per run
        run_avgs = []
        for run in metrics:
            returns = [ep["ep_return"] for ep in run]
            n = len(returns)
            # start index for the last `ratio` fraction
            idx = int(n * ratio)
            # guard in case ratio is very small or runs empty
            idx = max(0, min(idx, n - 1))
            run_avg = np.mean(returns[idx:]) if n > 0 else 0.0
            run_avgs.append(run_avg)
        # overall average across runs
        avg_reward = float(np.mean(run_avgs))
        
        # Return negative reward for minimization.
        return -avg_reward

    
    if exhaustive:
        # for the grid sampler it will exhaust all combinations
        # n_trials = 1 # For compute canada jobs
        n_trials = math.prod(len(v) for v in hp_search_space.values()) # For single job
        sampler = GridSampler(hp_search_space)
    else:
        sampler = TPESampler()
        
    study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs) 

    # Build the best hyper-parameters object from the best trial.
    best_params = study.best_trial.params
    best_hp = HyperParameters(**best_params)
    return best_hp, study

def create_study(hp_search_space, exhaustive=True, study_name=None, storage=None):
    # Create and run the Optuna study.
    if exhaustive:
        sampler = GridSampler(hp_search_space)
    else:
        sampler = TPESampler()
        
    study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name,
            storage=storage,
        )
    return study
    
def main(hp_search_space):
    args = parse()
    config_path = os.path.join("Configs", f"{args.config}.py")
    config = load_config(config_path)
    runs_dir = f"Runs/Tune/"
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
    # Instantiate agent using factory
    agent_fn = lambda env, info: config.AGENT_DICT[args.agent](env, args.info)
    default_hp = agent_fn(env_fn()).hp
       
    # Select the experiment class based on the number of environments.
    if args.num_envs == 1:
        experiment_class = BaseExperiment
    else:
        experiment_class = ParallelExperiment
        
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")    
    exp_name = f"{args.name_tag}_seed[{args.seed}]" #_{timestamp}
    env_params_str = "_".join(f"{k}-{v}" for k, v in config.env_params.items())  # env param dictionary to str
    wrappers_str = "_".join(fmt_wrap(w, p) for w, p in zip(config.env_wrapping, config.wrapping_params))
    exp_dir = os.path.join(runs_dir, f"{args.env}_{env_params_str}", wrappers_str, args.agent, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    db_path = os.path.join(exp_dir, "optuna_study.db")
    storage_url = f"sqlite:///{db_path}?timeout=60"
    storage = RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    
    if args.just_create_study:
        create_study(hp_search_space, exhaustive=args.exhaustive, 
                 study_name=exp_name, storage=storage)
        print("Study successfully created.")
        exit(0)
    
    # Run hyperparameter tuning.
    best_hp, study = tune_hyperparameters(
        env_fn,
        agent_fn,
        default_hp,
        hp_search_space,
        exp_dir=exp_dir,
        exp_class=experiment_class,
        exhaustive=args.exhaustive,
        ratio=args.metric_ratio,
        n_trials=args.num_trials,
        num_workers_each_trial=args.num_workers_each_trial,
        num_runs=args.num_runs,
        num_episodes=args.num_episodes,
        total_steps=args.total_steps,
        n_jobs=args.num_jobs,
        seed_offset=args.seed,
        study_name=exp_name,
        storage=storage,
        args=args,
    )

    print("Best hyperparameters found:") 
    print(best_hp)
    print(f"Study logs saved at: {runs_dir}")

if __name__ == "__main__":
    # Define the search ranges for hyperparameters.
    
    # n = 5
    # hp_search_space = { #             example for the exhaustive case
    #     "actor_step_size": make_grid(0.001, 0.5, n),
    #     "critic_step_size": make_grid(0.001, 0.5, n),
    #     "rollout_steps":    list(range(1, 7)),
    # }
    
    # hp_search_space = { #             example for the non-exhaustive case
    #     "actor_step_size":  (0.001, 0.5),
    #     "critic_step_size": (0.001, 0.5),
    #     "epsilon":          (0.01,  0.5),
    #     "rollout_steps":    (1, 4),
    # }
    
    hp_search_space = { 
        "actor_step_size": [0.0001, 0.001, 0.01],
        "critic_step_size": [0.0001, 0.001, 0.01],
        "rollout_steps":    [1, 3, 5, 7, 9],
    }
    
    main(hp_search_space)