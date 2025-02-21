import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio

class SingleExpAnalyzer:
    """
    Analyzes and plots metrics from multiple runs of an experiment.
    
    Expects metrics as a list of runs, where each run is a list of episode dictionaries.
    Each episode dictionary should contain keys like "total_reward", "steps", etc.
    """
    def __init__(self, metrics=None, exp_path=None):
        """
        Args:
            metrics (list): List of runs (each run is a list of episode dictionaries).
            exp_path (str): Directory containing a "metrics.pkl" file.
        """
        if metrics is None and exp_path is None:
            raise ValueError("Both metrics and exp_path are None")
        if metrics is None:
            metrics_path = os.path.join(exp_path, "metrics.pkl")
            with open(metrics_path, "rb") as f:
                metrics = pickle.load(f)
        self.exp_path = exp_path
        self.metrics = metrics
        self.num_runs = len(metrics)
        self.num_episodes = len(metrics[0]) if self.num_runs > 0 else 0

        # Convert metrics into 2D NumPy arrays (runs x episodes)
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
        Print overall mean and standard deviation for rewards and steps.
        """
        avg_reward = np.mean(self.total_rewards)
        std_reward = np.std(self.total_rewards)
        avg_steps = np.mean(self.steps)
        std_steps = np.std(self.steps)

        print("Experiment Summary:")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Average Steps:  {avg_steps:.2f} ± {std_steps:.2f}")

    def plot_combined(self, save_dir=None, show=False):
        """
        Plot total rewards and steps per episode.
        
        Each run is plotted transparently; the mean is overlaid.
        
        Args:
            save_dir (str, optional): Directory to save the plot.
            show (bool): Whether to display the plot.
        """
        episodes = np.arange(1, self.num_episodes + 1)
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot total rewards
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

        # Plot steps
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
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, "Combined.png"))
        if show:
            plt.show()

    def save_seeds(self, save_dir):
        """
        Save the seed information for each episode to a text file.
        
        Args:
            save_dir (str): Directory to save the seed file.
        """
        seed_lst = []
        for r, run in enumerate(self.metrics):
            for e, episode in enumerate(run):
                agent_seed = episode.get("agent_seed")
                env_seed = episode.get("env_seed")
                seed_lst.append(f"run {r + 1}, episode {e + 1} -> env_seed = {env_seed}, agent_seed = {agent_seed}\n")
        
        with open(os.path.join(save_dir, "seed.txt"), "w") as file:
            file.writelines(seed_lst)

    def generate_video(self, run_number, episode_number, video_type="gif"):
        """
        Generate a video (currently only GIF supported) from stored frames.
        
        Args:
            run_number (int): Run index (1-indexed).
            episode_number (int): Episode index (1-indexed).
            video_type (str): "gif" or "mp4" (only "gif" is implemented).
        """
        frames = self.metrics[run_number - 1][episode_number - 1]['frames']
        print(f"Number of frames: {len(frames)}")
        if video_type == "gif":
            filename = os.path.join(self.exp_path, f"run_{run_number}_ep_{episode_number}.gif")
            imageio.mimsave(filename, frames, fps=15)  # Adjust fps as needed
            print(f"GIF saved as {filename}")
        else:
            raise NotImplementedError("Only GIF video type is implemented.")