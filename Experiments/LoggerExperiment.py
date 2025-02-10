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
    def __init__(self, env, agent, exp_dir):
        super().__init__(env, agent, exp_dir)
        self.writer = SummaryWriter(log_dir=exp_dir)
    
    def _single_run(self, num_episodes, seed, run_prefix="run"):
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

        self.agent.reset(seed)
        all_metrics = []
        pbar = tqdm(range(1, num_episodes + 1), desc="Running episodes")
        for episode in pbar:
            # Use a seed to ensure reproducibility.
            metrics = self.run_episode(episode + seed)
            all_metrics.append(metrics)
            
            self.writer.add_scalar(f"total_reward/{run_prefix}", 
                                   metrics["total_reward"], episode)
            self.writer.add_scalar(f"steps/{run_prefix}", 
                                   metrics["steps"], episode)
            
            # Update the progress bar.
            pbar.set_postfix({
                "Reward": metrics['total_reward'], 
                "Steps": metrics['steps']
            })
        return all_metrics

    def multi_run(self, num_runs, num_episodes, seed_offset=None, dump_metrics=True):
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
            seed = random.randint(0, 2**32 - 1) if seed_offset is None else run * num_episodes + seed_offset 
            run_metrics = self._single_run(num_episodes, seed)

            all_runs_metrics.append(run_metrics)
            
            # Reset the agent's state to ensure independent runs.
            self.agent.reset(seed=seed)
            
            # Save the metrics for this run.
            if dump_metrics:
                metrics_file = os.path.join(self.exp_dir, f"metrics.pkl")
                with open(metrics_file, "wb") as f:
                    pickle.dump(all_runs_metrics, f)
       
        self.writer.close()
            
        return all_runs_metrics