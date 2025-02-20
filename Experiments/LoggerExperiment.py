from Experiments.BaseExperiment import BaseExperiment

from torch.utils.tensorboard import SummaryWriter 
import os
from tqdm import tqdm
import pickle
import random

class LoggerExperiment(BaseExperiment):
    """
    This class handles running episodes and collecting metrics.
    It will save the results on TensorBoard
    """
    def __init__(self, env, agent, exp_dir, train=True):
        super().__init__(env, agent, exp_dir, train=train)
        self.writer = SummaryWriter(log_dir=exp_dir)
    
    def _single_run(self, num_episodes, seed, n_run, run_prefix="run"):
        """
        Run the experiment for a specified number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to run.
            seed_offset (int): An offset for the seed.
            writer (SummaryWriter, optional): A TensorBoard writer to log this run's metrics.
                                               If not provided, uses self.writer.
        
        Returns:
            A list of episode metrics for analysis.
        """
        if self._train:
            self.agent.reset(seed)
        all_metrics = []
        pbar = tqdm(range(1, num_episodes + 1), desc="Running episodes")
        for episode in pbar:
            # Use a seed to ensure reproducibility.
            metrics = self.run_episode(episode + seed)
            metrics['agent_seed'] = seed
            all_metrics.append(metrics)
            
            self.writer.add_scalar(f"total_reward/{run_prefix}", 
                                   metrics["total_reward"], episode)
            self.writer.add_scalar(f"steps/{run_prefix}", 
                                   metrics["steps"], episode)
            
            # Update the progress bar.
            pbar.set_postfix({
                "Reward": metrics['total_reward'], 
                "Steps": metrics['steps'],
            })
            if self._checkpoint_freq is not None and episode % self._checkpoint_freq == 0:
                path = os.path.join(self.exp_dir, f"Policy_Run{n_run}_E{episode}")
                self.agent.save(path)
        return all_metrics

    def multi_run(self, num_runs, num_episodes, seed_offset=None, 
                  dump_metrics=True, checkpoint_freq=None, 
                  dump_transitions=False):
        
        all_runs_metrics = super().multi_run(num_runs, num_episodes, seed_offset, 
                                             dump_metrics, checkpoint_freq, 
                                             dump_transitions)
        self.writer.close()
            
        return all_runs_metrics