import os
import datetime
import numpy as np
import optuna

from Agents.Utils.HyperParams import HyperParameters
from Experiments.BaseExperiment import BaseExperiment
from Evaluate.SingleExpAnalyzer import SingleExpAnalyzer


def tune_hyperparameters(
    env,
    agent,
    default_hp,
    hp_range,
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

    # Create a timestamped directory for logging results.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = f"Runs/Tuning/{agent.__class__.__name__}_{timestamp}"
    os.makedirs(runs_dir, exist_ok=True)

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
        trial_dir = os.path.join(runs_dir, f"trial_{trial.number}_{agent.__class__.__name__}")
        os.makedirs(trial_dir, exist_ok=True)

        # Run the experiment
        experiment = BaseExperiment(env, agent, exp_dir=trial_dir)
        metrics = experiment.multi_run(
            num_episodes=num_episodes,
            num_runs=num_runs,
            seed_offset=seed_offset,
            dump_metrics=True
        )

        # Analyze results
        analyzer = SingleExpAnalyzer(metrics)
        analyzer.save_seeds(save_dir=trial_dir)

        # Compute the average of the last 20 episodes' total reward across runs
        rewards = np.array([
            [episode.get("total_reward") for episode in run] 
            for run in metrics
        ])
        avg_reward = np.mean(np.mean(rewards, axis=0)[-20:])

        # Since we want to maximize reward, but Optuna (in 'minimize' mode) looks
        # for the lowest value, we return the negative of the average reward
        return -avg_reward

    # Create the Optuna study (direction="minimize" because we return negative reward)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Build the best hyperparameters object
    best_params = study.best_trial.params
    best_hp = HyperParameters(**best_params)

    return best_hp, study, runs_dir


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    from Environments.MiniGrid.GetEnvironment import get_empty_grid
    from Agents.TabularAgent.QLearningAgent import QLearningAgent

    # Default hyperparameters
    default_hp = HyperParameters(
        step_size=0.1,
        gamma=0.99,
        epsilon=0.01,
        n_steps=2
    )

    # Define parameter search ranges
    hp_range = {
        "step_size": [0.01, 0.5],
        "epsilon": [0.01, 0.1],
        "n_steps": [1, 7]
    }

    # Environment creation
    env = get_empty_grid(
        render_mode=None,
        max_steps=200,
        wrapping_lst=["ViewSize", "StepReward", "FlattenOnehotObj"],
        wrapping_params=[{"agent_view_size": 3}, {"step_reward": -1}, {}]
    )

    # Instantiate agent with default hyperparameters
    agent = QLearningAgent(env.action_space, default_hp)

    # Run tuning
    best_hp, study, runs_dir = tune_hyperparameters(
        env,
        agent,
        default_hp,
        hp_range,
        n_trials=10,
        num_runs=3,
        num_episodes=20,
        seed_offset=1
    )

    print("Best hyperparameters found:")
    print(best_hp)
    print(f"Study logs saved at: {runs_dir}")