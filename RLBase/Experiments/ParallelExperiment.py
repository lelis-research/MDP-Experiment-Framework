from tqdm import tqdm
import os
import pickle
import random
import numpy as np

from . import BaseExperiment
#TODO: THIS CLASS IS NOT COMPLETED 
class ParallelExperiment(BaseExperiment):
    """    
    This class handles running episodes and collecting metrics
    for vectorized Envs.
    """
    def run_episodes_parallel(self, num_episodes, seed):
        """
        Run `num_episodes` *in total* across the vectorized environment.
        
        We'll track when each environment finishes an episode and 
        reset it individually until we've completed `num_episodes`.
        
        Returns:
            A list of episode metrics (dicts) with keys like
            {"ep_return", "ep_length", "seed"}.
        """
        # We assume gymnasium-like API for vector env:
        # obs.shape -> (num_envs, obs_space...)
        # done, truncated are arrays of size num_envs

        num_envs = self.env.num_envs
        # For reproducibility (some vector envs also accept a list of seeds, one per env)
        observations, infos = self.env.reset(seed=seed)
        
        # Tracking for each environment
        episode_rewards = np.zeros(num_envs, dtype=np.float32)
        episode_steps = np.zeros(num_envs, dtype=int)
        episode_counts = 0  # how many episodes have finished so far
        
        all_metrics = []
        pbar = tqdm(range(1, num_episodes + 1), desc="Running episodes")
        # We continue stepping until we finish `num_episodes` episodes in total.
        # We don't rely on all envs finishing simultaneously.
        while episode_counts < num_episodes:
            actions = self.agent.parallel_act(observations)                     
            observations, rewards, dones, truncated, infos = self.env.step(actions)
            self.agent.parallel_update(observations, rewards, dones, truncated)
            
            # Accumulate reward & step counts
            episode_rewards += rewards
            episode_steps += 1

            # For each env that's terminated/truncated, record metrics
            for i in range(num_envs):
                if (dones[i] or truncated[i]) and episode_counts < num_episodes:
                    # We finished an episode in env i
                    metrics = {
                        "ep_return": episode_rewards[i],
                        "ep_length": episode_steps[i],
                        "seed": seed,
                    }

                    # Update the progress bar.
                    pbar.update(1)
                    pbar.set_postfix({
                        "Return": metrics['ep_return'], 
                        "Steps": metrics['ep_length']
                    })
                    

                    all_metrics.append(metrics)
                    
                    episode_counts += 1
                    episode_rewards[i] = 0.0
                    episode_steps[i] = 0

        return all_metrics
    
    def _single_run(self, num_episodes, seed):
        """
        Run the experiment for a specified number of episodes.
        
        Returns:
            A list of episode metrics for analysis.
        """
        self.agent.reset(seed)

        # Use a seed to ensure reproducibility.
        all_metrics = self.run_episodes_parallel(num_episodes, seed)
        
        return all_metrics