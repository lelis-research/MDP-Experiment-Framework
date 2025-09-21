import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.size": 16,            # base font size
    # "axes.titlesize": 16,       # title
    # "axes.labelsize": 16,       # x and y labels
    # "xtick.labelsize": 14,      # x tick labels
    # "ytick.labelsize": 14,      # y tick labels
    # "legend.fontsize": 14,      # legend
    # "figure.titlesize": 18      # overall figure title
})

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
    
    # def _smooth(self, data, window_size):
    #     """Apply centered moving average with window size self.window_size"""
    #     if window_size <= 1:
    #         return data
    #     window = np.ones(window_size) / window_size
    #     return np.convolve(data, window, mode='same')
    
    def _smooth(self, data, window_size):
        res = np.empty_like(data)
        for j in range(len(data)):
            start_idx = max(0, j - window_size)
            end_idx = min(len(data), j + window_size)
            res[j] = np.mean(data[start_idx:end_idx])
        return res
        
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
    
    def _plot_reward_per_episode(self, ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, ignore_last=False):
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.set_aspect('auto')
        if ignore_last: # sometimes the last episode is not complete
            ep_return = np.array([run[:num_episodes - 1] for run in self.ep_returns])
            episodes = np.arange(1, num_episodes)
        else:
            ep_return = np.array([run[:num_episodes] for run in self.ep_returns])
            episodes = np.arange(1, num_episodes + 1)
        if plot_each:
            for each_return in ep_return:
                smooth_return = self._smooth(each_return, window_size)
                ax.plot(episodes, smooth_return, color=color, alpha=min(4/(len(ep_return)), 0.15))
        
        mean_returns = np.mean(ep_return, axis=0)
        smooth_returns = self._smooth(mean_returns, window_size)
        ax.plot(episodes, smooth_returns, marker, color=color, label=label, markevery=50)
        
        # Optional confidence interval
        if show_ci and ep_return.shape[0] >= 2:
            n = ep_return.shape[0]

            # sample std with ddof=1 -> unbiased; then standard error
            se = np.std(ep_return, axis=0, ddof=1) / np.sqrt(n)
            ci = 1.96 * se   # ~95% CI (normal approx). For very small n, consider t-crit.

            lower = mean_returns - ci
            upper = mean_returns + ci

            # smooth mean and bounds consistently
            lower_s = self._smooth(lower, window_size)
            upper_s = self._smooth(upper, window_size)
            ax.fill_between(episodes, lower_s, upper_s, alpha=0.2, color=color, linewidth=0)
            
        # ax.set_title("Sum Reward per Episode")
        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Return")
        # ax.legend()
        ax.grid(True)

    def _plot_steps_per_episode(self, ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, ignore_last=False):
        ax.set_aspect('auto')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        if ignore_last: # sometimes the last episode is not complete
            steps = np.array([run[:num_episodes - 1] for run in self.ep_lengths])
            episodes = np.arange(1, num_episodes)
        else:
            steps = np.array([run[:num_episodes] for run in self.ep_lengths])
            episodes = np.arange(1, num_episodes + 1)

        if plot_each:
            for each_step in steps:
                smooth_step = self._smooth(each_step, window_size)
                ax.plot(episodes, smooth_step, color=color, alpha=min(4/(len(steps)), 0.15))
        
        
        
        mean_steps = np.mean(steps, axis=0)
        smooth_steps = self._smooth(mean_steps, window_size)
        ax.plot(episodes, smooth_steps, marker, color=color, label=label, markevery=50)
                
        if show_ci and steps.shape[0] >= 2:
            n = steps.shape[0]
            
            # standard error with unbiased std (ddof=1)
            se = np.std(steps, axis=0, ddof=1) / np.sqrt(n)
            ci = 1.96 * se # ~95% normal approx; for small n consider Student-t

            lower = mean_steps - ci
            upper = mean_steps + ci

            lower_s = self._smooth(lower, window_size)
            upper_s = self._smooth(upper, window_size)

            ax.fill_between(episodes, lower_s, upper_s, alpha=0.2, color=color, linewidth=0)
            
        # ax.set_title("Steps per Episode")
        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Environment Steps")
        # ax.legend()
        ax.grid(True)
    
    def _plot_reward_per_steps(self, ax, num_steps, color, marker, label, window_size, plot_each, show_ci, ignore_last=False):
        ax.set_aspect('auto')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ep_returns = self.ep_returns
        steps = self.ep_lengths    
        x_common = np.linspace(0, num_steps, 1000)      
        returns_interpolation = []
        
        # Build arrays for each run
        for i in range(len(ep_returns)):
            # Convert to numeric arrays
            if ignore_last: # sometimes the last episode is not complete
                run_returns = np.array(ep_returns[i], dtype=float)[:-1]
                run_steps = np.array(steps[i], dtype=float)[:-1]
            else:
                run_returns = np.array(ep_returns[i], dtype=float)
                run_steps = np.array(steps[i], dtype=float)
            
            # Get the cumulative steps
            cum_steps = np.cumsum(run_steps)
            interpolated_run_returns = np.interp(x_common, cum_steps, run_returns)
            
            
            if plot_each:
                # Plot each run’s line and points (faint)
                smooth_return = self._smooth(interpolated_run_returns, window_size)
                ax.plot(x_common, smooth_return, marker='o', alpha=min(4/(len(ep_returns)), 0.15), color=color, markersize=1)

            # Interpolate the reward to fine in between values
            returns_interpolation.append(interpolated_run_returns)

        returns_interpolation = np.asarray(returns_interpolation)
        mean_returns = np.mean(returns_interpolation, axis=0)
        smooth_returns = self._smooth(mean_returns, window_size)
        ax.plot(x_common, smooth_returns, marker, color=color, label=label, markevery=50)
        
        # Optional confidence interval (quantile-based)
        if show_ci and returns_interpolation.shape[0] >= 2:
            n = returns_interpolation.shape[0]

            # sample std with ddof=1 -> unbiased; then standard error
            se = np.std(returns_interpolation, axis=0, ddof=1) / np.sqrt(n)
            ci = 1.96 * se   # ~95% CI (normal approx). For very small n, consider t-crit.

            lower = mean_returns - ci
            upper = mean_returns + ci

            # smooth mean and bounds consistently
            lower_s = self._smooth(lower, window_size)
            upper_s = self._smooth(upper, window_size)
            ax.fill_between(x_common, lower_s, upper_s, alpha=0.2, color=color, linewidth=0)
        
        # ax.set_title("Sum Rewards per Steps")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Return")
        # ax.legend()
        ax.grid(True)

    def plot_combined(self, fig=None, axs=None, save_dir=None, show=False, color='blue', marker='-',
                      label="", show_legend=True, window_size=1, plot_each=True, show_ci=True, 
                      title="", ignore_last=False, plt_configs=["r_e", "r_s", "s_e"]):
        """
        Plot total rewards and steps per episode and per steps.
        
        Each run is plotted transparently; the mean is overlaid.
        
        Args:
            fig (fig, optional): Matplotlib figure.
            axs (axs, optional): Matplotlib axs list (must be 3x1).
            save_dir (str, optional): Directory to save the plot.
            show (bool): Whether to display the plot.
        """
        assert all(c in {"r_e", "r_s", "s_e"} for c in plt_configs), \
        f"Invalid entries in plt_configs: {plt_configs}"

        if fig is None or axs is None:
            fig, axs = plt.subplots(len(plt_configs), 1, figsize=(10, 6*len(plt_configs)), constrained_layout=True)

        # find minimum number of episodes across runs
        num_episodes = min([len(run) for run in self.ep_returns])
        num_steps = min([sum(steps) for steps in self.ep_lengths])
        
        ax_counter = 0
        if "r_e" in plt_configs:
            ax = axs[ax_counter] if len(plt_configs) > 1 else axs
            self._plot_reward_per_episode(ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, ignore_last)
            ax_counter += 1
        if "r_s" in plt_configs:
            ax = axs[ax_counter] if len(plt_configs) > 1 else axs
            self._plot_reward_per_steps(ax, num_steps, color, marker, label, window_size, plot_each, show_ci, ignore_last)
            ax_counter += 1
        if "s_e" in plt_configs:
            ax = axs[ax_counter] if len(plt_configs) > 1 else axs
            self._plot_steps_per_episode(ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, ignore_last)
            ax_counter += 1

        if title:
            fig.suptitle(title)
        
        if len(plt_configs) == 1:
            ax.legend(loc="best", frameon=True)
        else:
            if show_legend:
                # Retrieve handles and labels from one of the subplots.
                handles, labels = ax.get_legend_handles_labels()
                
                # Create one legend for the entire figure.
                fig.legend(handles, labels, loc='upper center', ncols=math.ceil(len(labels)/2), shadow=False, bbox_to_anchor=(0.5, 0.965))
                fig.tight_layout(rect=[0, 0, 1.0, 0.95])
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

    def generate_video(self, run_number, episode_number, video_type="gif", name_tag=""):
        """
        Generate a video (currently only GIF supported) from stored frames.
        
        Args:
            run_number (int): Run index (1-indexed).
            episode_number (int): Episode index (1-indexed).
            video_type (str): "gif" or "mp4" (only "gif" is implemented).
        """
        frames = self.metrics[run_number - 1][episode_number - 1]['frames']
        
        if self.exp_path is not None:
            filename = os.path.join(self.exp_path, f"run_{run_number}_ep_{episode_number}_{name_tag}")
        else:
            filename = f"{name_tag}"
       
        print(f"Number of frames: {len(frames)}")
        if video_type == "gif":
            filename = f"{filename}.gif"
            imageio.mimsave(filename, frames, fps=15)  # Adjust fps as needed
            print(f"GIF saved as {filename}")
        else:
            raise NotImplementedError("Only GIF video type is implemented.")