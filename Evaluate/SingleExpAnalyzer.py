import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import pickle

class SingleExpAnalyzer:
    """
    Class for analyzing and plotting experiment metrics.
    
    This class accepts metrics from multiple runs and provides methods to
    print summary statistics and generate plots with individual runs in a 
    transparent color, and the mean and standard deviation as a solid line
    with a shaded region.
    """
    def __init__(self, metrics=None, exp_path=None):
        """
        Initialize the analyzer with metrics from multiple runs.

        Args:
            metrics (list): A list of runs, where each run is a list of dictionaries.
                            Each dictionary should contain episode metrics (e.g.,
                            "total_reward" and "steps").
        """
        if metrics is None and exp_path is None:
            raise ValueError("Both Metrics and Exp Path are None")
        if metrics is None:
            metrics_path = os.path.join(exp_path, "metrics.pkl")
            with open(metrics_path, "rb") as f:
                metrics = pickle.load(f)
        self.exp_path = exp_path
        self.metrics = metrics
        self.num_runs = len(metrics)
        self.num_episodes = len(metrics[0]) if self.num_runs > 0 else 0

        # Convert metrics into 2D NumPy arrays:
        # Each row corresponds to a run and each column to an episode.
        self.total_rewards = np.array([
            [episode.get("total_reward") for episode in run]
            for run in metrics
        ])
        self.steps = np.array([
            [episode.get("steps") for episode in run]
            for run in metrics
        ])
        

    def print_summary(self):
        """
        Print summary statistics of the experiment results (overall mean and std).
        """
        # Compute overall mean and std across all runs and episodes.
        avg_reward = np.mean(self.total_rewards)
        std_reward = np.std(self.total_rewards)
        avg_steps = np.mean(self.steps)
        std_steps = np.std(self.steps)

        print("Experiment Summary:")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Average Steps:  {avg_steps:.2f} ± {std_steps:.2f}")


    def plot_combined(self, save_dir=None, show=False):
        """
        Create subplots for both total rewards and steps per episode, with each run
        plotted transparently and the mean with its standard deviation shown.
        """
        episodes = np.arange(1, self.num_episodes + 1)
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot rewards in the first subplot.
        for run in self.total_rewards:
            axs[0].plot(episodes, run, color='blue', alpha=0.3)
        mean_rewards = np.mean(self.total_rewards, axis=0)
        std_rewards = np.std(self.total_rewards, axis=0)
        axs[0].plot(episodes, mean_rewards, color='blue', label="Mean Reward")
        # axs[0].fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
        #                     color='blue', alpha=0.2, label="Std. Dev.")
        axs[0].set_title("Total Reward per Episode")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Total Reward")
        axs[0].legend()
        axs[0].grid(True)

        # Plot steps in the second subplot.
        for run in self.steps:
            axs[1].plot(episodes, run, color='orange', alpha=0.3)
        mean_steps = np.mean(self.steps, axis=0)
        std_steps = np.std(self.steps, axis=0)
        axs[1].plot(episodes, mean_steps, color='orange', label="Mean Steps")
        # axs[1].fill_between(episodes, mean_steps - std_steps, mean_steps + std_steps,
        #                     color='orange', alpha=0.2, label="Std. Dev.")
        axs[1].set_title("Steps per Episode")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Steps")
        axs[1].legend()
        axs[1].grid(True)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "Combined.png"))
        if show:  
            plt.show()

    def save_seeds(self, save_dir):
        """
        Shows the seed for each episode for reproducing the results
        """
        seed_lst = []

        for r, run in enumerate(self.metrics):
            for e, episode in enumerate(run):
                seed = episode.get("seed")
                seed_lst.append(f"run {r}, episode {e} -> seed = {seed}\n")
        
        with open(os.path.join(save_dir, "seed.txt"), "w") as file:
            file.writelines(seed_lst)

    def generate_video(self, run_number, episode_number, video_type="gif"):
        '''
        video_type: gif or mp4
        '''
        frames = self.metrics[run_number - 1][episode_number - 1]['frames']
        if video_type == "gif":
            # Save as a GIF
            filname = os.path.join(self.exp_path, f"run_{run_number}, ep_{episode_number}.gif")
            imageio.mimsave(filname, frames, fps=15)  # Adjust fps if needed

            print(f"GIF saved as {filname}")
        else:
            raise NotImplementedError("Video Type is not Defined")