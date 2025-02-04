from Experiments.BaseExperiment import BaseExperiment

from torch.utils.tensorboard import SummaryWriter 
import os
from tqdm import tqdm
import pickle
class LoggerExperiment(BaseExperiment):
    """
    This class handles running episodes and collecting metrics.
    It will save the results on TensorBoard
    """
    def __init__(self, env, agent, exp_dir):
        super().__init__(env, agent, exp_dir)
        self.writer = SummaryWriter(log_dir=exp_dir)
    
    def single_run(self, num_episodes=100, seed_offset=0, run_prefix="run"):
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


        all_metrics = []
        pbar = tqdm(range(1, num_episodes + 1), desc="Running episodes")
        for episode in pbar:
            # Use a seed that may be offset to ensure reproducibility.
            metrics = self.run_episode(seed=episode + seed_offset)
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
            run_metrics = self.single_run(num_episodes, seed_offset=seed, run_prefix=f"run_{run}")
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