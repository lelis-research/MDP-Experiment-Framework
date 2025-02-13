import os
import datetime
import numpy as np
import optuna
import argparse

from Agents.Utils.HyperParams import HyperParameters
from Experiments.BaseExperiment import BaseExperiment
from Experiments.ParallelExperiment import ParallelExperiment
from Evaluate.SingleExpAnalyzer import SingleExpAnalyzer
from Environments.MiniGrid.GetEnvironment import *
from config import AGENT_DICT


def tune_hyperparameters(
    env,
    agent,
    default_hp,
    hp_range,
    exp_dir,
    exp_class,
    ratio=0.5,
    n_trials=20,
    num_runs=3,
    num_episodes=50,
    seed_offset=1
):
    """
    Automatically tune hyperparameters using Optuna.

    Args:
        env: The environment instance.
        agent: An instantiated agent (with `set_hp()` method).
        default_hp (HyperParameters): Default hyperparameters.
        hp_range (dict): Ranges for hyperparameters (e.g., {"step_size": [0.01, 0.5]}).
        n_trials (int): Number of Optuna trials. Defaults to 20.
        num_runs (int): Number of runs per trial. Defaults to 3.
        num_episodes (int): Number of episodes per run. Defaults to 50.
        seed_offset (int): A fixed seed offset for reproducibility. Defaults to 1.

    Returns:
        (best_hp, study, runs_dir):
            best_hp (HyperParameters): HyperParameters object with the best found values.
            study (optuna.study.Study): The Optuna study object containing all trials.
            runs_dir (str): Directory where run logs/metrics were saved.
    """
    # Convert the default hyperparameters into a dict so we can tweak them.
    base_params = default_hp.to_dict()
    
    def objective(trial):
        """
        Objective function for Optuna. Samples hyperparameters within the given ranges,
        sets them on the agent, and then runs an experiment to measure the average reward.
        """
        new_params = {}
        for key, default_value in base_params.items():
            # If the hyperparameter is listed in hp_range, sample it from Optuna
            if key in hp_range:
                if isinstance(default_value, float):
                    low, high = hp_range[key]
                    new_params[key] = trial.suggest_float(key, low, high)
                elif isinstance(default_value, int):
                    low, high = hp_range[key]
                    new_params[key] = trial.suggest_int(key, low, high)
                else:
                    # For non-float/int hyperparameters, keep the default
                    new_params[key] = default_value
            else:
                # If it's not in hp_range, keep the default
                new_params[key] = default_value

        # Create a new HyperParameters instance for this trial and update the agent
        tuned_hp = HyperParameters(**new_params)
        agent.set_hp(tuned_hp)

        # Create a unique logging directory for this trial
        trial_dir = os.path.join(exp_dir, f"trial_{trial.number}_{agent.__class__.__name__}")
        os.makedirs(trial_dir, exist_ok=True)

        # Run the experiment
        experiment = exp_class(env, agent, exp_dir=trial_dir)
        metrics = experiment.multi_run(
            num_episodes=num_episodes,
            num_runs=num_runs,
            seed_offset=seed_offset,
            dump_metrics=True
        )

        # Analyze results
        analyzer = SingleExpAnalyzer(metrics=metrics)
        analyzer.save_seeds(save_dir=trial_dir)

        # Compute the average of the last $ratio episodes' total reward across runs
        rewards = np.array([
            [episode.get("total_reward") for episode in run] 
            for run in metrics
        ])
        avg_reward_over_runs = np.mean(rewards, axis=0)
        ind = int(len(avg_reward_over_runs) * ratio) - 1
        avg_reward = np.mean(np.mean(rewards, axis=0)[ind:])

        # Since we want to maximize reward, but Optuna (in 'minimize' mode) looks
        # for the lowest value, we return the negative of the average reward
        return -avg_reward

    # Create the Optuna study (direction="minimize" because we return negative reward)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Build the best hyperparameters object
    best_params = study.best_trial.params
    best_hp = HyperParameters(**best_params)

    return best_hp, study

def main(default_hp, hp_range):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        default="Random",
        choices=list(AGENT_DICT.keys()),
        help="Which agent to run"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="MiniGrid-Empty-5x5-v0",
        choices=ENV_LST,
        help="which environment"
    )
    parser.add_argument(
        "--num_configs",
        type=int,
        default=300,
        help="number of Hyper-Params to try"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=2,
        help="number of runs per each Hyper-Params"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=200,
        help="number of episode in each run"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="number of parallel environments"
    )
    parser.add_argument(
        "--metric_ratio",
        type=float,
        default=0.5,
        help="Ratio of the last episode to consider"
    )
    args = parser.parse_args()
    runs_dir = f"Runs/Tuning"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Environment creation
    if args.num_envs == 1:
        env = get_single_env(
            env_name=args.env_name,
            render_mode=None,
            max_steps=200,
            wrapping_lst=["ViewSize", "StepReward", "FlattenOnehotObj"],
            wrapping_params=[{"agent_view_size": 3}, {"step_reward": -1}, {},],
        )
        experiment_class = BaseExperiment

    elif args.num_envs > 1:
        env = get_parallel_env(
            env_name=args.env_name,
            num_envs=args.num_envs,
            render_mode=None,
            max_steps=200,
            wrapping_lst=["ViewSize", "StepReward", "FlattenOnehotObj"],
            wrapping_params=[{"agent_view_size": 3}, {"step_reward": -1}, {},],
        )
        experiment_class = ParallelExperiment

    # Instantiate agent with default hyperparameters
    agent = AGENT_DICT[args.agent](env)

    exp_name = f"{agent.__class__.__name__}_{args.env_name}_{timestamp}"
    exp_dir = os.path.join(runs_dir, exp_name)

    # Run tuning
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
        seed_offset=1
    )

    print("Best hyperparameters found:")
    print(best_hp)
    print(f"Study logs saved at: {runs_dir}")

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Default hyperparameters
    default_hp = HyperParameters(
        step_size=0.1,
        gamma=0.99,
        epsilon=0.01,
        n_steps=2
    )

    # Define parameter search ranges
    hp_range = {
        "step_size": [0.001, 0.5],
        "epsilon": [0.01, 0.5],
        "n_steps": [1, 7]
    }

    main(default_hp, hp_range)
    