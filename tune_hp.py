import os
import datetime
import numpy as np
import optuna
import argparse
import argcomplete

from RLBase.Agents.Utils import HyperParameters  # For handling hyper-parameter objects
from RLBase.Experiments import BaseExperiment, ParallelExperiment
from RLBase.Evaluate import SingleExpAnalyzer
from RLBase.Environments import get_env, ENV_LST
from config import AGENT_DICT, env_wrapping, wrapping_params, env_params

def parse():
    parser = argparse.ArgumentParser()
    # Agent type to run
    parser.add_argument("--agent", type=str, default="Random", choices=list(AGENT_DICT.keys()), help="Which agent to run")
    # Environment name
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0", choices=ENV_LST, help="which environment")
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    # Number of hyper-parameter configurations (trials)
    parser.add_argument("--num_configs", type=int, default=300, help="number of Hyper-Params to try")
    # Number of runs per configuration
    parser.add_argument("--num_runs", type=int, default=2, help="number of runs per each Hyper-Params")
    # Episodes per run
    parser.add_argument("--num_episodes", type=int, default=200, help="number of episode in each run")
    # Maximum steps per episode
    parser.add_argument("--episode_max_steps", type=int, default=200, help="maximum number of steps in each episode")
    # Number of parallel environments
    parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments")
    # Ratio of last episodes to consider for metric calculation
    parser.add_argument("--metric_ratio", type=float, default=0.5, help="Ratio of the last episode to consider")
    argcomplete.autocomplete(parser)
    return parser.parse_args()

def tune_hyperparameters(env, agent, default_hp, hp_range, exp_dir, exp_class, ratio=0.5, n_trials=20, num_runs=3, num_episodes=50, seed_offset=1, args=None):
    """
    Tune hyperparameters using Optuna.
    
    Args:
        env: Environment instance.
        agent: Agent instance (must have set_hp() method).
        default_hp: Default HyperParameters object.
        hp_range: Dictionary of search ranges for hyperparameters.
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
        for key, default_value in base_params.items():
            if key in hp_range:
                if isinstance(default_value, float):
                    low, high = hp_range[key]
                    new_params[key] = trial.suggest_float(key, low, high)
                elif isinstance(default_value, int):
                    low, high = hp_range[key]
                    new_params[key] = trial.suggest_int(key, low, high)
                else:
                    new_params[key] = default_value
            else:
                new_params[key] = default_value

        # Update agent with the sampled hyperparameters.
        tuned_hp = HyperParameters(**new_params)
        agent.set_hp(tuned_hp)

        # Create a unique directory for the trial.
        trial_dir = os.path.join(exp_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        with open(f"{trial_dir}/agent.txt", "w") as file:
            file.write(str(agent))

        # Run the experiment for the current trial.
        experiment = exp_class(env, agent, exp_dir=trial_dir, args=args)
        metrics = experiment.multi_run(num_episodes=num_episodes, num_runs=num_runs, seed_offset=seed_offset, dump_metrics=True)

        # Save seed info.
        analyzer = SingleExpAnalyzer(metrics=metrics)
        analyzer.save_seeds(save_dir=trial_dir)

        # Compute average reward over the last fraction of episodes across runs.
        rewards = np.array([[episode.get("total_reward") for episode in run] for run in metrics])
        avg_reward_over_runs = np.mean(rewards, axis=0)
        ind = int(len(avg_reward_over_runs) * ratio) - 1
        avg_reward = np.mean(np.mean(rewards, axis=0)[ind:])
        
        # Return negative reward for minimization.
        return -avg_reward

    # Create and run the Optuna study.
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Build the best hyper-parameters object from the best trial.
    best_params = study.best_trial.params
    best_hp = HyperParameters(**best_params)
    return best_hp, study

def main(hp_range):
    args = parse()
    runs_dir = f"Runs/Tune/"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    # Create the environment with specified wrappers.
    env = get_env(
        env_name=args.env,
        num_envs=args.num_envs,
        render_mode=None,
        env_params=env_params,
        max_steps=args.episode_max_steps,
        wrapping_lst=env_wrapping,
        wrapping_params=wrapping_params,
    )

    # Instantiate the agent with its default hyperparameters.
    agent = AGENT_DICT[args.agent](env)
    default_hp = agent.hp

    # Select the experiment class based on the number of environments.
    if args.num_envs == 1:
        experiment_class = BaseExperiment
    else:
        experiment_class = ParallelExperiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.env}_{env_params}_{args.agent}_seed[{args.seed}]_{timestamp}"
    exp_dir = os.path.join(runs_dir, exp_name)

    # Run hyperparameter tuning.
    best_hp, study = tune_hyperparameters(
        env,
        agent,
        default_hp,
        hp_range,
        exp_dir=exp_dir,
        exp_class=experiment_class,
        ratio=args.metric_ratio,
        n_trials=args.num_configs,
        num_runs=args.num_runs,
        num_episodes=args.num_episodes,
        seed_offset=args.seed,
        args=args,
    )

    print("Best hyperparameters found:")
    print(best_hp)
    print(f"Study logs saved at: {runs_dir}")

if __name__ == "__main__":
    # Define the search ranges for hyperparameters.
    hp_range = {
        "actor_step_size": [0.001, 0.5],
        "critic_step_size": [0.001, 0.5],
        "epsilon": [0.01, 0.5],
        "rollout_steps": [1, 7]
    }
    main(hp_range)