import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

class MultiExpAnalyzer:
    """
    A class for analyzing and plotting experiment metrics from multiple experiment directories.
    
    The constructor accepts a dictionary where keys are experiment names (used for labels)
    and values are the experiment directories. Each directory should contain a pickle file
    (default name "metrics.pkl") with the experiment results.
    """
    def __init__(self, experiments_dict):
        """
        Args:
            experiments_dict (dict): Keys are experiment names (str) and values are the corresponding
                                     experiment directories (str) where 'metrics.pkl' is saved.
        """
        self.experiments_dict = experiments_dict
        self.metrics_data = {}  # Will map experiment names to the loaded metrics.
    
    def load_metrics(self, filename="metrics.pkl"):
        """
        Load experiment metrics from the directories provided in the constructor.
        
        Args:
            filename (str): Name of the pickle file containing the metrics. Defaults to "metrics.pkl".
        """
        self.metrics_data = {}
        for exp_name, exp_dir in self.experiments_dict.items():
            file_path = os.path.join(exp_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    metrics = pickle.load(f)
                    self.metrics_data[exp_name] = metrics
                
            else:
                print(f"Warning: {file_path} does not exist for experiment '{exp_name}'.")
        
    def plot_combined(self, save_path=None):
        """
        Plot the total rewards for all experiments on the same figure.
        
        For each experiment:
          - Each run is plotted with low transparency.
          - The mean reward (with standard deviation as a shaded area) is overlaid.
          - The plot is labeled with the experiment's key.
        
        Args:
            save_path (str, optional): If provided, the figure will be saved to this path.
        """
        if not self.metrics_data:
            print("No metrics loaded. Please call load_metrics() first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Use a colormap to assign a distinct color for each experiment.
        num_experiments = len(self.metrics_data)
        colors = plt.cm.viridis(np.linspace(0, 1, num_experiments))
        
        for idx, (exp_name, metrics) in enumerate(self.metrics_data.items()):
            # Each 'metrics' is assumed to be a list of runs,
            # where each run is a list of episode metrics (dictionaries with key "total_reward").
            if not metrics:
                continue
            
            # Assume every run has the same number of episodes.
            num_runs = len(metrics)
            num_episodes = len(metrics[0])
            rewards = np.zeros((num_runs, num_episodes))
            
            # Plot each run with low transparency.
            for run_idx, run in enumerate(metrics):
                run_rewards = [ep.get("total_reward", 0) for ep in run]
                rewards[run_idx, :] = run_rewards
                plt.plot(range(1, num_episodes + 1), run_rewards, color=colors[idx], alpha=0.3)
            
            # Compute mean and std deviation across runs for this experiment.
            mean_rewards = np.mean(rewards, axis=0)
            episodes = np.arange(1, num_episodes + 1)
            
            # Overlay the mean reward (solid line) and a shaded region for std deviation.
            plt.plot(episodes, mean_rewards, color=colors[idx], label=exp_name)
      
        
        plt.title("Total Reward per Episode (Combined Experiments)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()


