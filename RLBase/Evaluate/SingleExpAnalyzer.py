import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
import matplotlib.ticker as mticker
from PIL import Image, ImageDraw, ImageFont
from collections import Counter, defaultdict


from .Utils import get_mono_font, normalize_ansi_frames, render_fixed_ansi

plt.rcParams.update({
    "font.size": 18,            # base font size
    "legend.fontsize": 16,      # legend
    "figure.titlesize": 24      # overall figure title
})


class SingleExpAnalyzer:
    """
    Analyze and plot metrics from multiple runs of a single experiment.

    Expects metrics shaped as: list of runs -> list of episode dicts.
    Episode dicts typically contain:
      - ep_return (float), ep_length (int)
      - frames (list) and actions (list) if recorded
      - transitions (list) if dump_transitions was enabled
      - agent_seed / env_seed
      - agent_logs (list of per-step dicts) optionally holding keys like OptionUsageLog, NumOptions, OptionClass, OptionIndex.
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

    # ---------------------------
    # Utilities / smoothing
    # ---------------------------
    def _smooth(self, data, window_size):
        """Simple symmetric moving average (edges are truncated)."""
        if window_size is None or window_size <= 1:
            return data
        res = np.empty_like(data, dtype=float)
        for j in range(len(data)):
            start_idx = max(0, j - window_size)
            end_idx = min(len(data), j + window_size + 1)
            res[j] = np.mean(data[start_idx:end_idx])
        return res

    def _smooth_rows(self, arr, window_size):
        """Apply _smooth row-wise to a 2D array [rows x T]."""
        if window_size is None or window_size <= 1:
            return arr
        out = np.empty_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            out[i] = self._smooth(arr[i], window_size)
        return out


    # ---------------------------
    # Basic summaries
    # ---------------------------
    def print_summary(self):
        """
        Print overall mean and standard deviation for rewards and steps.
        """
        ep_returns = []
        ep_lengths = []
        for run in self.metrics:
            ep_returns.extend([episode.get("ep_return") for episode in run])
            ep_lengths.extend([episode.get("ep_length") for episode in run])
        avg_return = np.mean(ep_returns) if ep_returns else float("nan")
        std_return = np.std(ep_returns) if ep_returns else float("nan")
        avg_steps = np.mean(ep_lengths) if ep_lengths else float("nan")
        std_steps = np.std(ep_lengths) if ep_lengths else float("nan")

        print("Experiment Summary:")
        print(f"  Average Episode Return: {avg_return:.2f} ± {std_return:.2f}")
        print(f"  Average Episode Length:  {avg_steps:.2f} ± {std_steps:.2f}")

    # ---------------------------
    # Combined plotting entrypoint
    # ---------------------------
    def plot_combined(self, fig=None, axs=None, save_dir=None, show=False, color='blue', marker='-',
                  label="", show_legend=True, window_size=1, plot_each=True, show_ci=True,
                  title="", ignore_last=False,
                  plt_configs=("r_e", "r_s", "s_e"), index=None, total_index=None):
        """
        Plot totals per episode and per steps, plus optional option-usage panels.
        """

        assert all(c in {"r_e", "r_s", "s_e", "ou_s", "ou_e", "no_s", "no_e"} for c in plt_configs), \
            f"Invalid entries in plt_configs: {plt_configs}"

        if fig is None or axs is None:
            fig, axs = plt.subplots(len(plt_configs), 1, figsize=(10, 6 * len(plt_configs)), constrained_layout=True)

        ep_returns = [[episode.get("ep_return") for episode in run] for run in self.metrics]
        ep_lengths = [[episode.get("ep_length") for episode in run] for run in self.metrics]
        
        num_episodes = min((len(run) for run in ep_returns), default=0)
        num_steps = min((sum(steps) for steps in ep_lengths), default=0)

        for i, config in enumerate(plt_configs):
            ax = axs[i] if len(plt_configs) > 1 else axs

            if config == "r_e":
                self._plot_data_per_episode(ep_returns, ax, num_episodes, color, marker, label,
                                            window_size, plot_each, show_ci, ignore_last,
                                            x_label="Episode Number", y_label="Return")

            elif config == "r_s":
                self._plot_data_per_steps(ep_returns, ep_lengths, ax, num_steps, color, marker, label,
                                        window_size, plot_each, show_ci, ignore_last,
                                        x_label="Environment Steps", y_label="Return")

            elif config == "s_e":
                self._plot_data_per_episode(ep_lengths, ax, num_episodes, color, marker, label,
                                            window_size, plot_each, show_ci, ignore_last,
                                            x_label="Episode Number", y_label="Episode Length")

            elif config == "ou_s":
                option_usage = [[
                    (sum(item.get("OptionUsageLog", 0) for item in ep.get("agent_logs", [])) /
                    len(ep.get("agent_logs", [])))
                    if ep.get("agent_logs") and any("OptionUsageLog" in i for i in ep["agent_logs"]) else 0.0
                    for ep in run] for run in self.metrics]

                if any(any(run) for run in option_usage):
                    self._plot_data_per_steps(option_usage, ep_lengths, ax, num_steps, color, marker, label,
                                            window_size, plot_each, show_ci, ignore_last,
                                            x_label="Environment Steps", y_label="Option Usage")
                else:
                    print("OptionUsageLog doesn't exist — skipping plot.")

            elif config == "ou_e":
                option_usage = [[
                    (sum(item.get("OptionUsageLog", 0) for item in ep.get("agent_logs", [])) /
                    len(ep.get("agent_logs", [])))
                    if ep.get("agent_logs") and any("OptionUsageLog" in i for i in ep["agent_logs"]) else 0.0
                    for ep in run] for run in self.metrics]

                if any(any(run) for run in option_usage):
                    self._plot_data_per_episode(option_usage, ax, num_episodes, color, marker, label,
                                                window_size, plot_each, show_ci, ignore_last,
                                                x_label="Episode Number", y_label="Option Usage")
                else:
                    print("OptionUsageLog doesn't exist — skipping plot.")

            elif config == "no_s":
                num_options = [[
                    (sum(item.get("NumOptions", 0) for item in ep.get("agent_logs", [])) /
                    len(ep.get("agent_logs", [])))
                    if ep.get("agent_logs") and any("NumOptions" in i for i in ep["agent_logs"]) else 0.0
                    for ep in run] for run in self.metrics]

                if any(any(run) for run in num_options):
                    self._plot_data_per_steps(num_options, ep_lengths, ax, num_steps, color, marker, label,
                                            window_size, plot_each, show_ci, ignore_last,
                                            x_label="Environment Steps", y_label="Number of Options")
                else:
                    print("NumOptions doesn't exist — skipping plot.")

            elif config == "no_e":
                num_options = [[
                    (sum(item.get("NumOptions", 0) for item in ep.get("agent_logs", [])) /
                    len(ep.get("agent_logs", [])))
                    if ep.get("agent_logs") and any("NumOptions" in i for i in ep["agent_logs"]) else 0.0
                    for ep in run] for run in self.metrics]

                if any(any(run) for run in num_options):
                    self._plot_data_per_episode(num_options, ax, num_episodes, color, marker, label,
                                                window_size, plot_each, show_ci, ignore_last,
                                                x_label="Episode Number", y_label="Number of Options")
                else:
                    print("NumOptions doesn't exist — skipping plot.")

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

    

    # ---------------------------
    # Per-episode and per-steps primitives
    # ---------------------------
    def _plot_data_per_episode(self, all_runs_data, ax, num_episodes, color, marker, label, window_size, plot_each, show_ci,
                               ignore_last=False, x_label="", y_label=""):
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.set_aspect('auto')
        if num_episodes <= 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return

        if ignore_last:  # sometimes the last episode is not complete
            ep_data = np.array([run[:num_episodes - 1] for run in all_runs_data], dtype=float)
            episodes = np.arange(1, num_episodes)
        else:
            ep_data = np.array([run[:num_episodes] for run in all_runs_data], dtype=float)
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
            se = np.std(ep_data, axis=0, ddof=1) / np.sqrt(n)
            ci = 1.96 * se   # ~95% CI
            lower = mean_data - ci
            upper = mean_data + ci
            lower_s = self._smooth(lower, window_size)
            upper_s = self._smooth(upper, window_size)
            ax.fill_between(episodes, lower_s, upper_s, alpha=0.2, color=color, linewidth=0)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)

    def _plot_data_per_steps(self, all_runs_data, all_runs_ep_lengths, ax, num_steps, color, marker, label, window_size, plot_each, show_ci,
                             ignore_last=False, x_label="", y_label=""):
        ax.set_aspect('auto')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        if num_steps <= 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return

        steps = all_runs_ep_lengths
        x_common = np.linspace(0, num_steps, 1000)
        data_interpolation = []

        # Build arrays for each run
        for i in range(len(all_runs_data)):
            if ignore_last:  # sometimes the last episode is not complete
                run_data = np.array(all_runs_data[i], dtype=float)[:-1]
                run_steps = np.array(steps[i], dtype=float)[:-1]
            else:
                run_data = np.array(all_runs_data[i], dtype=float)
                run_steps = np.array(steps[i], dtype=float)

            cum_steps = np.cumsum(run_steps)
            if len(cum_steps) == 0:
                continue
            interpolated_run_data = np.interp(x_common, cum_steps, run_data)

            if plot_each:
                smooth_each_data = self._smooth(interpolated_run_data, window_size)
                ax.plot(x_common, smooth_each_data, marker='o', alpha=min(4/(len(all_runs_data)), 0.15), color=color, markersize=1)

            data_interpolation.append(interpolated_run_data)

        if not data_interpolation:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return

        data_interpolation = np.asarray(data_interpolation, dtype=float)
        mean_data = np.mean(data_interpolation, axis=0)
        smooth_data = self._smooth(mean_data, window_size)
        ax.plot(x_common, smooth_data, marker, color=color, label=label, markevery=50)

        # Optional confidence interval
        if show_ci and data_interpolation.shape[0] >= 2:
            n = data_interpolation.shape[0]
            se = np.std(data_interpolation, axis=0, ddof=1) / np.sqrt(n)
            ci = 1.96 * se   # ~95% CI
            lower = mean_data - ci
            upper = mean_data + ci
            lower_s = self._smooth(lower, window_size)
            upper_s = self._smooth(upper, window_size)
            ax.fill_between(x_common, lower_s, upper_s, alpha=0.2, color=color, linewidth=0)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)

    # ---------------------------
    # NEW: Option-centric plots
    # ---------------------------

    
    def plot_option_class_usage(self, fig=None, axs=None, save_dir=None, show=False,
                                     color='blue', marker='-', label="", show_legend=True,
                                     window_size=1, plot_each=True, show_ci=True,
                                     title="", ignore_last=False, option_classes=None,
                                     index=None, total_index=None, x_type=None):
        """
        For THIS experiment:
        - one axis per option in `option_classes`
        - draw a line (per-episode) on each axis for this experiment
        """
        assert x_type in ["s", "e"], "x_type must be 's' (steps) or 'e' (episodes)"

        if not option_classes:
            raise ValueError("option_classes must be a non-empty list.")

        if fig is None or axs is None:
            fig, axs = plt.subplots(len(option_classes), 1,
                                    figsize=(10, 6 * len(option_classes)),
                                    constrained_layout=True)

        # Build basic series data and alignment terms
        ep_returns = [[episode.get("ep_return") for episode in run] for run in self.metrics]
        ep_lengths = [[episode.get("ep_length") for episode in run] for run in self.metrics]
        
        num_episodes = min((len(run) for run in ep_returns), default=0)
        num_steps    = min((sum(steps) for steps in ep_lengths), default=0)

        for ax, target_cls in zip(axs, option_classes):
            # Per-episode normalized usage series for THIS option (one list per run)
            ax.set_title(f"{target_cls.__name__}")
            series = [[
                (sum(1 for it in ep.get("agent_logs", [])
                    if it.get("OptionClass") == target_cls) / len(ep.get("agent_logs", [])))
                if ep.get("agent_logs") and any("OptionClass" in it for it in ep["agent_logs"]) else 0.0
                for ep in run
            ] for run in self.metrics]
            
            if not any(any(run) for run in series):
                print(f"OptionClass '{getattr(target_cls, '__name__', target_cls)}' doesn't exist — skipping plot.")
                continue
                
            if x_type == "e":
                self._plot_data_per_episode(
                    series, ax, num_episodes, color, marker, label,
                    window_size, plot_each, show_ci, ignore_last,
                    x_label="Episode Number", y_label="Option Usage"
                )
            else:  # x_type == "s"
                self._plot_data_per_steps(
                    series, ep_lengths, ax, num_steps, color, marker, label,
                    window_size, plot_each, show_ci, ignore_last,
                    x_label="Environment Steps", y_label=f"Option Usage"
                )

        if title:
            fig.suptitle(title)

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
            fig.savefig(os.path.join(save_dir, "OptionClassUsage.png"))

        if show:
            plt.show()

        return fig, axs
        
        
        
    
    # ---------------------------
    # Misc: seed saver and video generator
    # ---------------------------
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

        os.makedirs(save_dir, exist_ok=True)
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
        actions = self.metrics[run_number - 1][episode_number - 1]['actions']
        if "agent_logs" in self.metrics[run_number - 1][episode_number - 1]:
            options = self.metrics[run_number - 1][episode_number - 1]['agent_logs']
            options_index = [option[0].get('OptionIndex', None) for option in options] #[0] is because we only look at the first env in test
        else:
            options_index = [None for _ in range(len(actions))]

        if self.exp_path is not None:
            filename = os.path.join(self.exp_path, f"run_{run_number}_ep_{episode_number}_{name_tag}")
        else:
            filename = f"{name_tag}"
            
        print(f"Number of frames: {len(frames)}")
        if len(frames) == 0:
            print("No frames to generate video.")
            return
        
        first = frames[0]
        
        is_ansi = isinstance(first, (str, bytes))

        # If ANSI: render all to fixed-size images
        if is_ansi:
            font = get_mono_font(size=ansi_font_size)
            frames_lines, max_cols, max_rows = normalize_ansi_frames(frames)
            img_frames = [render_fixed_ansi(lines, max_cols, max_rows, font, scale=ansi_scale)
                          for lines in frames_lines]
        else:
            img_frames = frames

        # --- Overlay actions and options ---
        def _to_pil(img):
            return Image.fromarray(img) if isinstance(img, np.ndarray) else img

        def _overlay_label(img, text, pad=6, alpha=140, font_size=18):
            img = _to_pil(img)
            draw = ImageDraw.Draw(img, "RGBA")
            try:
                font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            box_w, box_h = tw + 2 * pad, th + 2 * pad
            draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0, alpha))
            draw.text((pad, pad), text, fill=(255, 255, 255, 255), font=font)
            return img

        T = len(img_frames)
        A = len(actions)
        O = len(options_index)
        
        annotated = []
        for t, frame in enumerate(img_frames):
            timestep = t
            if t == 0:
                text = "START"
            else:
                ai = min(t - 1, A - 1) if A > 0 else None
                oi = min(t - 1, O - 1) if O > 0 else None
                a_val = actions[ai] if A > 0 else "NA"
                o_val = options_index[oi] if O > 0 else "NA"
                text = f"t:{timestep} | a:{a_val} | opt:{o_val}"
            annotated.append(np.array(_overlay_label(frame, text)))
        img_frames = annotated
        
        # def annotated_frames():
        #     for t, frame in enumerate(img_frames):
        #         timestep = t
        #         if t == 0:
        #             text = "START"
        #         else:
        #             ai = min(t - 1, A - 1) if A > 0 else None
        #             oi = min(t - 1, O - 1) if O > 0 else None
        #             a_val = actions[ai] if A > 0 else "NA"
        #             o_val = options_index[oi] if O > 0 else "NA"
        #             text = f"t:{timestep} | a:{a_val} | opt:{o_val}"
        #         yield np.array(_overlay_label(frame, text))

        if video_type == "gif":
            filename = f"{filename}_{fps}fps.gif"
            imageio.mimsave(filename, img_frames, fps=fps)
            print(f"GIF saved as {filename}")
        else:
            raise NotImplementedError("Only GIF video type is implemented.")
        
        # if video_type == "gif":
        #     filename = f"{filename}.gif"
        #     with imageio.get_writer(filename, mode="I", fps=fps) as writer:
        #         for fr in annotated_frames():
        #             writer.append_data(fr)
        #     print(f"GIF saved as {filename}")
        # else:
        #     raise NotImplementedError("Only GIF video type is implemented.")
