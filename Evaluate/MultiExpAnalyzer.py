import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

class MultiExpAnalyzer:
    """
    Analyzes and plots experiment metrics from multiple experiment directories.
    
    Expects a dictionary where keys are experiment names (labels)
    and values are directories containing a pickle file (default "metrics.pkl")
    with experiment results.
    """
    def __init__(self, experiments_dict):
        """
        Args:
            experiments_dict (dict): Keys are experiment names (str) and values are the 
                                     corresponding experiment directories (str) where 'metrics.pkl' is saved.
        """
        self.experiments_dict = experiments_dict
        self.metrics_data = {}  # Maps experiment names to loaded metrics.
    
    def load_metrics(self, filename="metrics.pkl"):
        """
        Load experiment metrics from the provided directories.
        
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
        Plot total rewards for all experiments on one figure.
        
        For each experiment:
          - Each run is plotted with low transparency.
          - The mean reward (with standard deviation as a shaded area) is overlaid.
          - The plot is labeled with the experiment's name.
        
        Args:
            save_path (str, optional): If provided, saves the figure to this path.
        """
        if not self.metrics_data:
            print("No metrics loaded. Please call load_metrics() first.")
            return
        
        plt.figure(figsize=(10, 6))
        num_experiments = len(self.metrics_data)
        colors = plt.cm.viridis(np.linspace(0, 1, num_experiments))
        
        for idx, (exp_name, metrics) in enumerate(self.metrics_data.items()):
            if not metrics:
                continue
            
            num_runs = len(metrics)
            num_episodes = len(metrics[0])
            rewards = np.zeros((num_runs, num_episodes))
            
            for run_idx, run in enumerate(metrics):
                run_rewards = [ep.get("total_reward", 0) for ep in run]
                rewards[run_idx, :] = run_rewards
                plt.plot(range(1, num_episodes + 1), run_rewards, color=colors[idx], alpha=0.3)
            
            mean_rewards = np.mean(rewards, axis=0)
            episodes = np.arange(1, num_episodes + 1)
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