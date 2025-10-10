import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
import matplotlib.ticker as mticker
from .Utils import get_mono_font, normalize_ansi_frames, render_fixed_ansi

plt.rcParams.update({
    "font.size": 24,            # base font size
    # "axes.titlesize": 16,       # title
    # "axes.labelsize": 16,       # x and y labels
    # "xtick.labelsize": 14,      # x tick labels
    # "ytick.labelsize": 14,      # y tick labels
    "legend.fontsize": 16,      # legend
    "figure.titlesize": 24      # overall figure title
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
        
    
    @property
    def num_runs(self):
        return len(self.metrics)
            
    
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
        assert all(c in {"r_e", "r_s", "s_e", "ou_s", "ou_e", "no_s", "no_e"} for c in plt_configs), \
        f"Invalid entries in plt_configs: {plt_configs}"

        if fig is None or axs is None:
            fig, axs = plt.subplots(len(plt_configs), 1, figsize=(10, 6*len(plt_configs)), constrained_layout=True)

        ep_returns = [[episode.get("ep_return") for episode in run] for run in self.metrics]
        ep_lengths = [[episode.get("ep_length") for episode in run] for run in self.metrics]
        
        # find minimum number of episodes across runs
        num_episodes = min([len(run) for run in ep_returns])
        num_steps = min([sum(steps) for steps in ep_lengths])
        
        for i, config in enumerate(plt_configs):
            ax = axs[i] if len(plt_configs) > 1 else axs
            if config == "r_e":
                self._plot_data_per_episode(ep_returns, ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, ignore_last, 
                                        x_label="Episode Number", y_label="Return")
            elif config == "r_s":
                 self._plot_data_per_steps(ep_returns, ep_lengths, ax, num_steps, color, marker, label, window_size, plot_each, show_ci, ignore_last,
                                        x_label="Environment Steps", y_label="Return")
            elif config == "s_e":
                self._plot_data_per_episode(ep_lengths, ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, ignore_last,
                                        x_label="Episode Number", y_label="Episode Length")
            elif config == "ou_s":
                option_usage = [[sum(item["OptionUsageLog"] for item in ep.get("agent_logs", [])) / len(ep.get("agent_logs", [])) 
                            if ep.get("agent_logs") else 0.0
                            for ep in run]
                            for run in self.metrics]
                self._plot_data_per_steps(option_usage, ep_lengths, ax, num_steps, color, marker, label, window_size, plot_each, show_ci, ignore_last,
                                    x_label="Environmnet Steps", y_label="Option Usage")
            elif config == "ou_e":
                option_usage = [[sum(item["OptionUsageLog"] for item in ep.get("agent_logs", [])) / len(ep.get("agent_logs", [])) 
                            if ep.get("agent_logs") else 0.0
                            for ep in run]
                            for run in self.metrics]
                self._plot_data_per_episode(option_usage, ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, ignore_last,
                                            x_label="Episode Number", y_label="Option Usage")

            elif config == "no_s":
                num_options = [[sum(item["NumOptions"] for item in ep.get("agent_logs", [])) / len(ep.get("agent_logs", [])) 
                            if ep.get("agent_logs") else 0.0
                            for ep in run]
                            for run in self.metrics]
                self._plot_data_per_steps(num_options, ep_lengths, ax, num_steps, color, marker, label, window_size, plot_each, show_ci, ignore_last,
                                      x_label="Environment Steps", y_label="Number of Options")
                
            elif config == "no_e":
                num_options = [[sum(item["NumOptions"] for item in ep.get("agent_logs", [])) / len(ep.get("agent_logs", [])) 
                            if ep.get("agent_logs") else 0.0
                            for ep in run]
                            for run in self.metrics]
                self._plot_data_per_episode(num_options, ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, ignore_last,
                                        x_label="Episode Number", y_label="Number of Options")
        
            

        if title:
            fig.suptitle(title)
        
        if len(plt_configs) == 1:
            axs.legend(loc="best", frameon=True)
        else:
            if show_legend:
                # Retrieve handles and labels from one of the subplots.
                handles, labels = axs[0].get_legend_handles_labels()
                
                # Create one legend for the entire figure.
                fig.legend(handles, labels, loc='upper center', ncols=math.ceil(len(labels)/2), shadow=False, bbox_to_anchor=(0.5, 0.96))
                fig.tight_layout(rect=[0, 0, 1.0, 0.96])
            # else:
            #     fig.tight_layout()

        
            
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, "Combined.png"))
        if show:
            plt.show()

        return fig, axs
    

    def _plot_data_per_episode(self, all_runs_data, ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, 
                               ignore_last=False, x_label="", y_label=""):
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.set_aspect('auto')
        if ignore_last: # sometimes the last episode is not complete
            ep_data = np.array([run[:num_episodes - 1] for run in all_runs_data])
            episodes = np.arange(1, num_episodes)
        else:
            ep_data = np.array([run[:num_episodes] for run in all_runs_data])
            episodes = np.arange(1, num_episodes + 1)
        if plot_each:
            for each_data in ep_data:
                smooth_each_data = self._smooth(each_data, window_size)
                ax.plot(episodes, smooth_each_data, color=color, alpha=min(4/(len(ep_data)), 0.15))
        
        mean_data = np.mean(ep_data, axis=0)
        smooth_data = self._smooth(mean_data, window_size)
        ax.plot(episodes, smooth_data, marker, color=color, label=label, markevery=50)
        
        # Optional confidence interval
        if show_ci and ep_data.shape[0] >= 2:
            n = ep_data.shape[0]

            # sample std with ddof=1 -> unbiased; then standard error
            se = np.std(ep_data, axis=0, ddof=1) / np.sqrt(n)
            ci = 1.96 * se   # ~95% CI (normal approx). For very small n, consider t-crit.

            lower = mean_data - ci
            upper = mean_data + ci

            # smooth mean and bounds consistently
            lower_s = self._smooth(lower, window_size)
            upper_s = self._smooth(upper, window_size)
            ax.fill_between(episodes, lower_s, upper_s, alpha=0.2, color=color, linewidth=0)
            
        # ax.set_title("Sum Reward per Episode")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.legend()
        ax.grid(True)
    
    def _plot_data_per_steps(self, all_runs_data, all_runs_ep_lengths, ax, num_steps, color, marker, label, window_size, plot_each, show_ci, 
                             ignore_last=False, x_label="", y_label=""):
        ax.set_aspect('auto')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        steps = all_runs_ep_lengths   
        x_common = np.linspace(0, num_steps, 1000)      
        data_interpolation = []
        
        # Build arrays for each run
        for i in range(len(all_runs_data)):
            # Convert to numeric arrays
            if ignore_last: # sometimes the last episode is not complete
                run_data = np.array(all_runs_data[i], dtype=float)[:-1]
                run_steps = np.array(steps[i], dtype=float)[:-1]
            else:
                run_data = np.array(all_runs_data[i], dtype=float)
                run_steps = np.array(steps[i], dtype=float)
            
            
            # Get the cumulative steps
            cum_steps = np.cumsum(run_steps)
            interpolated_run_data = np.interp(x_common, cum_steps, run_data)
            
            
            if plot_each:
                # Plot each run’s line and points (faint)
                smooth_each_data = self._smooth(interpolated_run_data, window_size)
                ax.plot(x_common, smooth_each_data, marker='o', alpha=min(4/(len(all_runs_data)), 0.15), color=color, markersize=1)

            # Interpolate the reward to fine in between values
            data_interpolation.append(interpolated_run_data)

        data_interpolation = np.asarray(data_interpolation)
        mean_data = np.mean(data_interpolation, axis=0)
        smooth_data = self._smooth(mean_data, window_size)
        ax.plot(x_common, smooth_data, marker, color=color, label=label, markevery=50)
        
        # Optional confidence interval (quantile-based)
        if show_ci and data_interpolation.shape[0] >= 2:
            n = data_interpolation.shape[0]

            # sample std with ddof=1 -> unbiased; then standard error
            se = np.std(data_interpolation, axis=0, ddof=1) / np.sqrt(n)
            ci = 1.96 * se   # ~95% CI (normal approx). For very small n, consider t-crit.

            lower = mean_data - ci
            upper = mean_data + ci

            # smooth mean and bounds consistently
            lower_s = self._smooth(lower, window_size)
            upper_s = self._smooth(upper, window_size)
            ax.fill_between(x_common, lower_s, upper_s, alpha=0.2, color=color, linewidth=0)
        
        # ax.set_title("Sum Rewards per Steps")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.legend()
        ax.grid(True)
        
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

    def generate_video(self, run_number, episode_number, video_type="gif", name_tag="",
                       fps=5, ansi_font_size=14, ansi_scale=2):
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
        
        first = frames[0]
        is_ansi = isinstance(first, (str, bytes))

        # If ANSI: render all to fixed-size images
        if is_ansi:
            font = get_mono_font(size=ansi_font_size)
            frames_lines, max_cols, max_rows = normalize_ansi_frames(frames)
            img_frames = [render_fixed_ansi(lines, max_cols, max_rows, font,scale=ansi_scale)
                          for lines in frames_lines]
        else:
            # Assume RGB numpy arrays (H, W, 3) or lists thereof
            img_frames = frames

            # Optional safety: ensure all frames have same size by padding to max
            # (uncomment if you ever hit shape mismatch)
            # max_h = max(f.shape[0] for f in img_frames)
            # max_w = max(f.shape[1] for f in img_frames)
            # def _pad(img):
            #     h, w = img.shape[:2]
            #     if (h, w) == (max_h, max_w): return img
            #     pad = np.zeros((max_h, max_w, img.shape[2]), dtype=img.dtype)
            #     pad[:h, :w] = img
            #     return pad
            # img_frames = [_pad(f) for f in img_frames]

        
        if video_type == "gif":
            filename = f"{filename}.gif"
            imageio.mimsave(filename, img_frames, fps=fps)  # Adjust fps as needed
            print(f"GIF saved as {filename}")
        else:
            raise NotImplementedError("Only GIF video type is implemented.")