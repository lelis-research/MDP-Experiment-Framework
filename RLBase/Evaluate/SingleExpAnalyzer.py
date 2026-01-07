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
        
    def plot_option_embedding1(self, fig=None, ax=None, save_dir=None, show=False,
                              color='blue', marker='o', label="", show_legend=True,
                              title="", index=None, total_index=None):  
        data = []
        for r, run in enumerate(self.metrics):
            for e, ep in enumerate(run):
                for l, logs in enumerate(ep.get("agent_logs")):
                    if isinstance(logs, list):
                        data.extend(logs)
                    else:
                        data.append(logs)
                    
        # Group points by index
        by_ind = defaultdict(list)
        for d in data:
            by_ind[d['ind']].append(d)

        # Create a color map (5 distinct colors)
        cmap = plt.cm.get_cmap("tab10", len(by_ind))

        plt.figure(figsize=(7, 7))

        for idx, (ind, items) in enumerate(by_ind.items()):
            color = cmap(idx)

            # Plot all proto_e for this ind
            proto_es = np.stack([d['proto_e'] for d in items])
            plt.scatter(
                proto_es[:, 0],
                proto_es[:, 1],
                color=color,
                # marker='x',
                s=8,              # 👈 make markers small (try 4–12)
                linewidths=0.5,   # 👈 thin the 'x' strokes
                alpha=0.1,
                label=f'proto_e (ind={ind})'
            )

            # Plot the corresponding e (usually one per ind)
            e_vals = np.stack([d['e'].cpu().numpy() for d in items])
            plt.scatter(
                e_vals[:, 0],
                e_vals[:, 1],
                color=color,
                marker='o',
                s=32,
                edgecolors='black',
                label=f'e (ind={ind})'
            )
            

        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title("e vs proto_e embeddings")
        # plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()
        
    def plot_option_embedding2(
        self,
        fig=None,
        ax=None,
        save_dir=None,
        show=False,
        show_legend=True,
        title="e vs proto_e embeddings",
        # marker/size controls
        proto_marker="x",
        e_marker="o",
        proto_s=8,
        e_s=32,
        proto_alpha=0.15,
        e_alpha=0.9,
        proto_lw=0.5,
        e_edge_lw=0.8,
        draw_lines=False,      # connect proto_e -> e (useful but can clutter)
    ):
        # -------------------------
        # 1) Collect logs robustly
        # -------------------------
        data = []
        for run in getattr(self, "metrics", []):
            for ep in run:
                for logs in ep.get("agent_logs", []):
                    if isinstance(logs, list):
                        data.extend(logs)
                    elif isinstance(logs, dict):
                        data.append(logs)

        if len(data) == 0:
            print("plot_option_embedding: no data found in agent_logs.")
            return

        # -------------------------
        # 2) Infer dimensionality
        # -------------------------
        def to_np_vec(v):
            if hasattr(v, "detach"):  # torch tensor
                v = v.detach().cpu().numpy()
            else:
                v = np.asarray(v)
            return np.asarray(v, dtype=float).reshape(-1)

        first = data[0]
        d_proto = to_np_vec(first["proto_e"]).shape[0]
        d_e = to_np_vec(first["e"]).shape[0]
        dim = d_proto  # assume they match

        if dim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D embeddings, got dim={dim}. (proto_e={d_proto}, e={d_e})")
        if d_proto != d_e:
            raise ValueError(f"proto_e dim != e dim ({d_proto} vs {d_e}).")

        # -------------------------
        # 3) Group by ind safely
        # -------------------------
        by_ind = defaultdict(list)
        for dct in data:
            ind_val = dct.get("ind", 0)
            # convert torch/numpy scalars to python int
            if hasattr(ind_val, "item"):
                ind_val = ind_val.item()
            ind_val = int(ind_val)
            by_ind[ind_val].append(dct)

        # -------------------------
        # 4) Create fig/ax
        # -------------------------
        created_fig = False
        if fig is None or ax is None:
            created_fig = True
            if dim == 3:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(figsize=(7, 7))

        # -------------------------
        # 5) Colors per ind
        # -------------------------
        cmap = plt.cm.get_cmap("tab10", len(by_ind))

        # -------------------------
        # 6) Plot
        # -------------------------
        for color_i, (ind, items) in enumerate(sorted(by_ind.items())):
            color = cmap(color_i)

            proto_es = np.stack([to_np_vec(dct["proto_e"]) for dct in items])
            e_vals  = np.stack([to_np_vec(dct["e"])       for dct in items])

            if dim == 2:
                ax.scatter(
                    proto_es[:, 0], proto_es[:, 1],
                    color=color,
                    marker=proto_marker,
                    s=proto_s,
                    alpha=proto_alpha,
                    linewidths=proto_lw,
                    label=(f"proto_e (ind={ind})" if show_legend else None),
                )
                ax.scatter(
                    e_vals[:, 0], e_vals[:, 1],
                    color=color,
                    marker=e_marker,
                    s=e_s,
                    alpha=e_alpha,
                    edgecolors="black",
                    linewidths=e_edge_lw,
                    label=(f"e (ind={ind})" if show_legend else None),
                )

                if draw_lines:
                    for pe, ev in zip(proto_es, e_vals):
                        ax.plot([pe[0], ev[0]], [pe[1], ev[1]], color=color, alpha=0.2, linewidth=0.7)

            else:  # dim == 3
                ax.scatter(
                    proto_es[:, 0], proto_es[:, 1], proto_es[:, 2],
                    color=color,
                    marker=proto_marker,
                    s=proto_s,
                    alpha=proto_alpha,
                    linewidths=proto_lw,
                    label=(f"proto_e (ind={ind})" if show_legend else None),
                )
                ax.scatter(
                    e_vals[:, 0], e_vals[:, 1], e_vals[:, 2],
                    color=color,
                    marker=e_marker,
                    s=e_s,
                    alpha=e_alpha,
                    edgecolors="black",
                    linewidths=e_edge_lw,
                    label=(f"e (ind={ind})" if show_legend else None),
                )

                if draw_lines:
                    for pe, ev in zip(proto_es, e_vals):
                        ax.plot([pe[0], ev[0]], [pe[1], ev[1]], [pe[2], ev[2]], color=color, alpha=0.2, linewidth=0.7)

        # -------------------------
        # 7) Cosmetics
        # -------------------------
        if dim == 2:
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.grid(True)
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_zlabel("Dim 3")

        ax.set_title(title)

        if show_legend:
            # Note: this will create 2 entries per ind; fine for 5 inds.
            ax.legend(loc="best", fontsize=9)

        # -------------------------
        # 8) Save / show
        # -------------------------
        if save_dir is not None:
            fig.savefig(save_dir, bbox_inches="tight", dpi=200)

        if show or created_fig:
            plt.show()

        return fig, ax
    
    def plot_option_embedding3(
        self,
        fig=None,
        ax=None,
        save_dir=None,
        show=False,
        show_legend=True,
        title="e vs proto_e embeddings",
        # sizes
        proto_s=8,
        e_s=32,
        proto_alpha=0.15,
        e_alpha=0.9,
        proto_lw=0.5,
        e_edge_lw=0.8,
        draw_lines=False,
        # NEW: episode bucketing
        episode_buckets=5,   # 2=halves, 3=thirds, etc.
    ):
        # ---------- helpers ----------
        def to_np_vec(v):
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            return np.asarray(v, dtype=float).reshape(-1)

        # Marker pool for buckets (repeat if needed)
        bucket_markers = ['x', '^', 's', 'D', 'P', 'v', '*', '<', '>', '1', '2', '3', '4']
        if episode_buckets > len(bucket_markers):
            # just cycle if user picks a huge number
            pass

        # ---------- collect logs + attach ep index ----------
        data = []
        for run in getattr(self, "metrics", []):
            for ep_idx, ep in enumerate(run):
                for logs in ep.get("agent_logs", []):
                    if isinstance(logs, list):
                        for d in logs:
                            if isinstance(d, dict):
                                dd = dict(d)
                                dd["_ep_idx"] = ep_idx
                                data.append(dd)
                    elif isinstance(logs, dict):
                        dd = dict(logs)
                        dd["_ep_idx"] = ep_idx
                        data.append(dd)

        if not data:
            print("plot_option_embedding: no data found in agent_logs.")
            return

        # ---------- infer dimension ----------
        dim = to_np_vec(data[0]["proto_e"]).shape[0]
        if dim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D embeddings, got dim={dim}.")

        # ---------- compute bucket id for each point ----------
        max_ep = max(d["_ep_idx"] for d in data)
        denom = max(1, max_ep + 1)  # number of episodes in run
        for d in data:
            # bucket = floor(ep_idx / num_eps * episode_buckets)
            b = int(np.floor(d["_ep_idx"] * episode_buckets / denom))
            d["_bucket"] = min(b, episode_buckets - 1)

        # ---------- group by ind ----------
        by_ind = defaultdict(list)
        for d in data:
            ind_val = d.get("ind", 0)
            if hasattr(ind_val, "item"):
                ind_val = ind_val.item()
            ind_val = int(ind_val)
            by_ind[ind_val].append(d)

        # ---------- fig/ax ----------
        created_fig = False
        if fig is None or ax is None:
            created_fig = True
            if dim == 3:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(figsize=(7, 7))

        # ---------- colors for ind ----------
        cmap = plt.cm.get_cmap("tab10", len(by_ind))

        # ---------- plot ----------
        # We'll create legend entries more carefully to avoid exploding legends:
        # - one legend for ind colors (optional)
        # - one legend for episode buckets (markers)
        ind_handles = []
        bucket_handles = []

        for color_i, (ind, items) in enumerate(sorted(by_ind.items())):
            color = cmap(color_i)

            # plot e as one marker per ind (constant marker)
            e_vals = np.stack([to_np_vec(d["e"]) for d in items])
            if dim == 2:
                sc_e = ax.scatter(
                    e_vals[:, 0], e_vals[:, 1],
                    color=color, marker='o', s=e_s,
                    alpha=e_alpha, edgecolors="black",
                    linewidths=e_edge_lw,
                    label=None
                )
            else:
                sc_e = ax.scatter(
                    e_vals[:, 0], e_vals[:, 1], e_vals[:, 2],
                    color=color, marker='o', s=e_s,
                    alpha=e_alpha, edgecolors="black",
                    linewidths=e_edge_lw,
                    label=None
                )

            # keep a handle for ind legend (just once per ind)
            ind_handles.append((sc_e, f"ind={ind}"))

            # plot proto_e with marker depending on episode bucket
            for b in range(episode_buckets):
                b_items = [d for d in items if d["_bucket"] == b]
                if not b_items:
                    continue
                proto_es = np.stack([to_np_vec(d["proto_e"]) for d in b_items])
                m = bucket_markers[b % len(bucket_markers)]

                if dim == 2:
                    sc_p = ax.scatter(
                        proto_es[:, 0], proto_es[:, 1],
                        color=color, marker=m,
                        s=proto_s, alpha=proto_alpha,
                        linewidths=proto_lw,
                        label=None
                    )
                else:
                    sc_p = ax.scatter(
                        proto_es[:, 0], proto_es[:, 1], proto_es[:, 2],
                        color=color, marker=m,
                        s=proto_s, alpha=proto_alpha,
                        linewidths=proto_lw,
                        label=None
                    )

                # optional lines (can get busy fast)
                if draw_lines:
                    e_bucket = np.stack([to_np_vec(d["e"]) for d in b_items])
                    for pe, ev in zip(proto_es, e_bucket):
                        if dim == 2:
                            ax.plot([pe[0], ev[0]], [pe[1], ev[1]], color=color, alpha=0.15, linewidth=0.6)
                        else:
                            ax.plot([pe[0], ev[0]], [pe[1], ev[1]], [pe[2], ev[2]], color=color, alpha=0.15, linewidth=0.6)

        # ---------- cosmetics ----------
        if dim == 2:
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
            ax.grid(True)
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")

        ax.set_title(title)

        # ---------- legends (two separate legends: colors for ind, markers for time buckets) ----------
        if show_legend:
            # legend 1: ind colors (use e circles as handles)
            handles1 = [h for h, _ in ind_handles]
            labels1  = [lab for _, lab in ind_handles]
            leg1 = ax.legend(handles1, labels1, loc="upper right", title="Option index (color)")
            ax.add_artist(leg1)

            # legend 2: episode buckets (create dummy handles)
            dummy = []
            dummy_labels = []
            for b in range(episode_buckets):
                m = bucket_markers[b % len(bucket_markers)]
                # create a dummy scatter for legend
                if dim == 2:
                    h = ax.scatter([], [], marker=m, color="black", s=proto_s*2, alpha=0.8)
                else:
                    h = ax.scatter([], [], [], marker=m, color="black", s=proto_s*2, alpha=0.8)
                dummy.append(h)
                # label ranges nicely
                dummy_labels.append(f"bucket {b+1}/{episode_buckets}")
            ax.legend(dummy, dummy_labels, loc="lower right", title="Episode bucket (marker)")

        if save_dir is not None:
            fig.savefig(save_dir, bbox_inches="tight", dpi=200)

        if show or created_fig:
            plt.show()

        return fig, ax
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
