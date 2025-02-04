import os
import numpy as np
import optuna
import datetime
from Agents.Utils.HyperParams import HyperParameters
from Experiments.BaseExperiment import BaseExperiment

def Tune_Hyper_Params(env, agent, hp, hp_range, n_trials=20, num_runs=3, num_episodes=50, seed_offset=1):
    """
    Automatically tune hyper-parameters using Optuna.
    
    Args:
        env: Either a callable that returns a fresh environment instance or an environment instance.
        agent_class: The agent class (e.g., QLearningAgent).
        hp: An instance of HyperParameters containing the default hyper-parameter values.
        n_trials (int): Number of Optuna trials.
        num_episodes (int): Number of episodes to run per trial.
        seed_offset (int): A fixed seed for reproducibility.
        
    Returns:
        best_hp (HyperParameters): An instance with the best hyper-parameters.
        study (optuna.study.Study): The Optuna study object.
    """
    # Extract the default hyper-parameters as a dictionary.
    base_params = hp.to_dict()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = f"Runs/Tuning/{agent.__class__.__name__}_{timestamp}"
    
    def objective(trial):
        new_params = {}
        for key, default_value in base_params.items():
            if key in hp_range:
                if isinstance(default_value, float):
                    new_params[key] = trial.suggest_float(key, hp_range[key][0], hp_range[key][1])
                elif isinstance(default_value, int):                
                    new_params[key] = trial.suggest_int(key, hp_range[key][0], hp_range[key][1])
                else:
                    print(f"Kept default value for {key} as {default_value}")
                    new_params[key] = default_value
            else:
                print(f"Kept default value for {key} as {default_value}")
                new_params[key] = default_value

        # Create a new HyperParameters instance for this trial.
        tuned_hp = HyperParameters(**new_params)
        agent.set_hp(tuned_hp)
        
        # Create a fresh environment instance. If env is callable, call it.
        new_env = env() if callable(env) else env

        # Update agent using the tuned hyper-parameter and reset the agent
        agent.set_hp(tuned_hp)
        agent.reset()

        # Create a unique logging directory for this trial.
        trial_dir = os.path.join(runs_dir, f"trial_{trial.number}_{agent}")

        experiment = BaseExperiment(new_env, agent, exp_dir=trial_dir)

        # Run the experiment for a fixed number of episodes.
        metrics = experiment.multi_run(num_episodes=num_episodes,
                                        num_runs=num_runs, 
                                        seed_offset=seed_offset, 
                                        dump_metrics=True)
        # Compute the  total reward
        rewards = np.array([[episode.get("total_reward") for episode in run]for run in metrics])
        
        # Get the average of last 20 episodes across different runs
        avg_reward = np.mean(np.mean(rewards, axis=0)[-20:]) 
        
        # Return negative reward because we want to maximize reward.
        return -avg_reward

    # Create and run the Optuna study.
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_hp = HyperParameters(**best_params)
    return best_hp, study, runs_dir

# Example usage:
if __name__ == "__main__":
    from Environments.MiniGrid.EmptyGrid import get_empty_grid
    from Agents.TabularAgent.QLearningAgent import QLearningAgent

    # Create a default HyperParameters instance.
    default_hp = HyperParameters(alpha=0.1, gamma=0.99, epsilon=0.01)
    hp_range = {"alpha": [0.01, 0.5],
                "epsilon": [0.01, 0.1]}
                
    env = get_empty_grid(
        render_mode=None,
        max_steps=200,
        wrapping_lst=["ViewSize", "StepReward", "FlattenOnehotObj"],
        wrapping_params=[{"agent_view_size": 3}, {"step_reward": -1}, {}]
    )
    agent = QLearningAgent(env.action_space, default_hp, seed=None)

    # Call the tuning function.
    best_hp, study, runs_dir = Tune_Hyper_Params(env, agent, default_hp, hp_range,
                                       n_trials=100, num_runs=3, 
                                       num_episodes=200, seed_offset=123123)
    
    print("Best hyper-parameters found:")
    print(best_hp)