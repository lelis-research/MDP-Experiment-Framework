import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
import pickle

class BaseExperiment:
    """
    Base class for connecting an agent and an environment.
    
    This class handles running episodes and collecting metrics.
    Subclasses can override methods like run_episode() or run() 
    to customize behavior.
    """
    def __init__(self, env, agent, exp_dir=None):
        """
        Args:
            env: An initialized environment.
            agent: An agent instance.
            max_episodes (int): Number of episodes to run.
        """
        self.env = env
        self.agent = agent
        if exp_dir is not None:
            os.makedirs(exp_dir, exist_ok=True)
            self.exp_dir = exp_dir

    def run_episode(self, seed=None):
        """
        Run a single episode.
        
        Args:
            seed: (Optional) Random seed for the episode.
        
        Returns:
            A dictionary containing metrics such as total_reward and steps.
        """
        observation, info = self.env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = self.agent.act(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.agent.update(observation, reward, terminated, truncated)
            
            total_reward += reward
            steps += 1
        
        return {"total_reward": total_reward, "steps": steps}

    def single_run(self, num_episodes=100, seed_offset=0):
        """
        Run the experiment for a specified number of episodes.
        
        Returns:
            A list of episode metrics for analysis.
        """
        all_metrics = []
        pbar = tqdm(range(1, num_episodes + 1), desc="Running episodes")
        for episode in pbar:
            # Use a seed that may be offset to ensure reproducibility.
            metrics = self.run_episode(seed=episode + seed_offset)
            all_metrics.append(metrics)
                        
            # Update the progress bar.
            pbar.set_postfix({
                "Reward": metrics['total_reward'], 
                "Steps": metrics['steps']
            })
        return all_metrics
    
    def multi_run(self, num_runs=10, num_episodes=100, seed_offset=0, dump_metrics=True):
        """
        Run multiple independent runs of the experiment.
        
        Returns:
            A list of run metrics, where each run's metrics is a list of
            episode metrics.
        """
        all_runs_metrics = []
        for run in range(1, num_runs + 1):
            print(f"Starting Run {run}")
            
            # Set a seed offset for this run.
            seed = run * num_episodes + seed_offset
            run_metrics = self.single_run(num_episodes, seed_offset=seed)
            all_runs_metrics.append(run_metrics)
            
            # Reset the agent's state to ensure independent runs.
            self.agent.reset(seed=seed)
            
            # Save the metrics for this run.
            if dump_metrics:
                file = os.path.join(self.exp_dir, "metrics.pkl")
                with open(file, "wb") as f:
                    pickle.dump(all_runs_metrics, f)
                    
        return all_runs_metrics
    
