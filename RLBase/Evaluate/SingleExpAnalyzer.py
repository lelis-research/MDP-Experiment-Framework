import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math

class SingleExpAnalyzer:
    """
    Analyzes and plots metrics from multiple runs of an experiment.
    
    Expects metrics as a list of runs, where each run is a list of episode dictionaries.
    Each episode dictionary should contain keys like "ep_return", "steps", etc.
    """
    def __init__(self, metrics=None, exp_path=None):
        """
        Args:
            metrics (list): List of runs (each run is a list of episode dictionaries).
            exp_path (str): Directory containing a "all_metrics.pkl" file.
        """
        if metrics is None and exp_path is None:
            raise ValueError("Both metrics and exp_path are None")
        if metrics is None:
            metrics_path = os.path.join(exp_path, "all_metrics.pkl")
            with open(metrics_path, "rb") as f:
                metrics = pickle.load(f)
        
        self.exp_path = exp_path
        self.metrics = metrics
        self.calculate_rewards_steps()
        
    
    @property
    def num_runs(self):
        return len(self.metrics)
    
    def calculate_rewards_steps(self):
        self.ep_returns = [[episode.get("ep_return") for episode in run] for run in self.metrics]
        self.ep_lengths = [[episode.get("ep_length") for episode in run] for run in self.metrics]
        
    def print_summary(self):
        """
        Print overall mean and standard deviation for rewards and steps.
        """
        avg_return= np.mean(self.ep_returns)
        std_return = np.std(self.ep_returns)
        avg_steps = np.mean(self.ep_lengths)
        std_steps = np.std(self.ep_lengths)

        print("Experiment Summary:")
        print(f"  Average Episode Return: {avg_return:.2f} ± {std_return:.2f}")
        print(f"  Average Episode Length:  {avg_steps:.2f} ± {std_steps:.2f}")
    
    def _plot_reward_per_episode(self, ax, num_episodes, color, label):
            ep_return = np.array([run[:num_episodes] for run in self.ep_returns])
            episodes = np.arange(1, num_episodes + 1)

            for run in ep_return:
                ax.plot(episodes, run, color=color, alpha=min(1/(len(ep_return)), 0.15))
            mean_rewards = np.mean(ep_return, axis=0)
            ax.plot(episodes, mean_rewards, color=color, label=label)
            
            ax.set_title("Sum Reward per Episode")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Sum Reward")
            # ax.legend()
            ax.grid(True)

    def _plot_steps_per_episode(self, ax, num_episodes, color, label):
        steps = np.array([run[:num_episodes] for run in self.ep_lengths])
        episodes = np.arange(1, num_episodes + 1)

        for run in steps:
            ax.plot(episodes, run, color=color, alpha=min(1/(len(steps)), 0.15))
        mean_steps = np.mean(steps, axis=0)
        ax.plot(episodes, mean_steps, color=color, label=label)
    
        ax.set_title("Steps per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        # ax.legend()
        ax.grid(True)
    
    def _plot_reward_per_steps(self, ax, num_steps, color, label):
        ep_returns = self.ep_returns
        steps = self.ep_lengths          
        x_common = np.linspace(0, num_steps, 200)      
        rewards_interpolation = []

        # Build arrays for each run
        for i in range(len(ep_returns)):
            # Convert to numeric arrays
            run_rewards = np.array(ep_returns[i], dtype=float)
            run_steps = np.array(steps[i], dtype=float)
        
            # Get the cumulative steps
            cum_steps = np.cumsum(run_steps)
            
            # Plot each run’s line and points (faint)
            ax.plot(cum_steps, run_rewards, marker='o', alpha=min(1/(len(ep_returns)), 0.15), color=color, markersize=1)

            # Interpolate the reward to fine in between values
            rewards_interpolation.append(np.interp(x_common, cum_steps, run_rewards))

        rewards_interpolation = np.asarray(rewards_interpolation)
        mean_rewards = np.mean(rewards_interpolation, axis=0)
        ax.plot(x_common, mean_rewards, color=color, label=label)
        
        ax.set_title("Sum Rewards per Steps")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Episode Reward")
        # ax.legend()
        ax.grid(True)

    def plot_combined(self, fig=None, axs=None, save_dir=None, show=False, color='blue', label="", show_legend=True):
        """
        Plot total rewards and steps per episode and per steps.
        
        Each run is plotted transparently; the mean is overlaid.
        
        Args:
            fig (fig, optional): Matplotlib figure.
            axs (axs, optional): Matplotlib axs list (must be 3x1).
            save_dir (str, optional): Directory to save the plot.
            show (bool): Whether to display the plot.
        """
        if fig is None or axs is None:
            fig, axs = plt.subplots(3, 1, figsize=(12, 10))

        # find minimum number of episodes across runs
        num_episodes = min([len(run) for run in self.ep_returns])
        num_steps = min([sum(steps) for steps in self.ep_lengths])

        self._plot_reward_per_episode(axs[0], num_episodes, color, label)
        self._plot_reward_per_steps(axs[1], num_steps, color, label)
        self._plot_steps_per_episode(axs[2], num_episodes, color, label)

        if show_legend:
            # Retrieve handles and labels from one of the subplots.
            handles, labels = axs[0].get_legend_handles_labels()
            # Create one legend for the entire figure.
            fig.legend(handles, labels, loc='upper center', ncols=math.ceil(len(labels)/2), shadow=False)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            fig.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, "Combined.png"))
        if show:
            plt.show()

        return fig, axs
    

    def save_seeds(self, save_dir):
        """
        Save the seed information for each episode to a text file.
        
        Args:
            save_dir (str): Directory to save the seed file.
        """
        seed_lst = []
        for r, run in enumerate(self.metrics):
            for e, episode in enumerate(run):
                agent_seed = episode.get("agent_seed", None)
                env_seed = episode.get("env_seed", None)
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