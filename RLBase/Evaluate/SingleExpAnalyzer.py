import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
import matplotlib.ticker as mticker
from PIL import Image, ImageDraw, ImageFont
from .Utils import get_mono_font, normalize_ansi_frames, render_fixed_ansi

plt.rcParams.update({
    "font.size": 24,            # base font size
    "legend.fontsize": 16,      # legend
    "figure.titlesize": 24      # overall figure title
})


class SingleExpAnalyzer:
    """
    Analyzes and plots metrics from multiple runs of an experiment.

    Expects metrics as a list of runs, where each run is a list of episode dictionaries.
    Each episode dictionary should contain keys like "ep_return", "ep_length", etc.
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
    # Option name extraction from existing logs
    # ---------------------------
    def _opt_name_from_item(self, item):
        """Robustly extract a printable option name from a logged dict item."""
        oc = item.get("OptionClass", None)
        if oc is None:
            return "None"
        if isinstance(oc, str):
            return oc
        name = getattr(oc, "__name__", None)
        if name is not None:
            return name
        return str(oc)

    def _episode_option_series(self, episode):
        """
        Returns a list of option names (or 'None') with same length as number of logged steps.
        Assumes per-step logs live under episode['agent_logs'].
        """
        logs = episode.get("agent_logs", []) or []
        return [self._opt_name_from_item(it) for it in logs]

    def _collect_option_name_vocab(self):
        """Collect a sorted list of all option names present across all runs/episodes."""
        vocab = set()
        for run in self.metrics:
            for ep in run:
                for name in self._episode_option_series(ep):
                    vocab.add(name)
        return sorted(vocab)

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
                      plt_configs=["r_e", "r_s", "s_e"]):
        """
        Plot totals per episode and per steps, plus optional option-usage panels.

        Supported plt_configs:
        - Rewards / steps: "r_e", "r_s", "s_e"
        - Logged usage:    "ou_s", "ou_e", "no_s", "no_e"
        - Options (new):   "oc_stack", "oc_heat", "oc_trans"
        """
        assert all(c in {
            "r_e", "r_s", "s_e", "ou_s", "ou_e", "no_s", "no_e",
            "oc_stack", "oc_heat", "oc_trans"
        } for c in plt_configs), f"Invalid entries in plt_configs: {plt_configs}"

        if fig is None or axs is None:
            fig, axs = plt.subplots(len(plt_configs), 1, figsize=(10, 6*len(plt_configs)), constrained_layout=True)

        ep_returns = [[episode.get("ep_return") for episode in run] for run in self.metrics]
        ep_lengths = [[episode.get("ep_length") for episode in run] for run in self.metrics]

        # find minimums for alignment
        num_episodes = min([len(run) for run in ep_returns]) if ep_returns else 0
        num_steps = min([sum(steps) for steps in ep_lengths]) if ep_lengths else 0

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
                option_usage = [[
                    (sum(item.get("OptionUsageLog", 0) for item in ep.get("agent_logs", [])) / len(ep.get("agent_logs", [])))
                    if ep.get("agent_logs") and any("OptionUsageLog" in i for i in ep["agent_logs"]) else 0.0
                    for ep in run] for run in self.metrics]
                if any(any(run) for run in option_usage):
                    self._plot_data_per_steps(option_usage, ep_lengths, ax, num_steps, color, marker, label, window_size, plot_each,
                                              show_ci, ignore_last, x_label="Environment Steps", y_label="Option Usage")
                else:
                    print("OptionUsageLog doesn't exist — skipping plot.")

            elif config == "ou_e":
                option_usage = [[
                    (sum(item.get("OptionUsageLog", 0) for item in ep.get("agent_logs", [])) / len(ep.get("agent_logs", [])))
                    if ep.get("agent_logs") and any("OptionUsageLog" in i for i in ep["agent_logs"]) else 0.0
                    for ep in run] for run in self.metrics]
                if any(any(run) for run in option_usage):
                    self._plot_data_per_episode(option_usage, ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, ignore_last,
                                                x_label="Episode Number", y_label="Option Usage")
                else:
                    print("OptionUsageLog doesn't exist — skipping plot.")

            elif config == "no_s":
                num_options = [[
                    (sum(item.get("NumOptions", 0) for item in ep.get("agent_logs", [])) / len(ep.get("agent_logs", [])))
                    if ep.get("agent_logs") and any("NumOptions" in i for i in ep["agent_logs"]) else 0.0
                    for ep in run] for run in self.metrics]
                if any(any(run) for run in num_options):
                    self._plot_data_per_steps(num_options, ep_lengths, ax, num_steps, color, marker, label, window_size, plot_each, show_ci, ignore_last,
                                              x_label="Environment Steps", y_label="Number of Options")
                else:
                    print("NumOptions doesn't exist — skipping plot.")

            elif config == "no_e":
                num_options = [[
                    (sum(item.get("NumOptions", 0) for item in ep.get("agent_logs", [])) / len(ep.get("agent_logs", [])))
                    if ep.get("agent_logs") and any("NumOptions" in i for i in ep["agent_logs"]) else 0.0
                    for ep in run] for run in self.metrics]
                if any(any(run) for run in num_options):
                    self._plot_data_per_episode(num_options, ax, num_episodes, color, marker, label, window_size, plot_each, show_ci, ignore_last,
                                                x_label="Episode Number", y_label="Number of Options")
                else:
                    print("NumOptions doesn't exist — skipping plot.")

            # ---- NEW option-centric panels wired into plot_combined ----
            elif config == "oc_stack":
                self.plot_option_stack_over_steps(ax=ax, window_size=window_size, label="Option Mix over Steps")

            elif config == "oc_heat":
                # default to run 0 for the heatmap
                if self.num_runs > 0 and len(self.metrics[0]) > 0:
                    self.plot_option_heatmap_over_episodes(run_idx=0, ax=ax, title="Option Usage per Episode (Run 1)")
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')

            elif config == "oc_trans":
                if self.num_runs > 0 and len(self.metrics[0]) > 0:
                    self.plot_option_transition_matrix(run_idx=0, ax=ax, title="Option Transition Matrix (Run 1)")
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')

        if title:
            fig.suptitle(title)

        if len(plt_configs) == 1:
            axs.legend(loc="best", frameon=True)
        else:
            if show_legend:
                # Retrieve handles and labels from one of the subplots.
                handles, labels = axs[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center',
                           ncols=math.ceil(max(1, len(labels))/2),
                           shadow=False, bbox_to_anchor=(0.5, 0.96))
                fig.tight_layout(rect=[0, 0, 1.0, 0.96])
                


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
    def plot_option_stack_over_steps(self, ax=None, window_size=1, label=""):
        """
        Stacked area (fractions) of options over environment steps, averaged over runs.
        """
        import numpy as np
        import matplotlib.ticker as mticker

        vocab = self._collect_option_name_vocab()
        if not vocab:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, "No option usage found", ha='center', va='center')
            return ax

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_aspect('auto')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        series_per_run = []
        x_common = None

        for run in self.metrics:
            names_all, steps_all = [], []
            for ep in run:
                names = self._episode_option_series(ep)
                ep_len = ep.get("ep_length", len(names))
                L = min(len(names), ep_len)
                names_all.extend(names[:L])
                steps_all.append(L)

            if not names_all:
                continue

            T = int(sum(steps_all)) if steps_all else 0
            if T <= 0:
                continue

            t_axis = np.arange(1, len(names_all) + 1, dtype=float)
            if x_common is None:
                x_common = np.linspace(0, max(1, T), 1000)

            onehots = np.zeros((len(vocab), len(names_all)), dtype=float)
            name_to_idx = {n: i for i, n in enumerate(vocab)}
            for t, n in enumerate(names_all):
                onehots[name_to_idx[n], t] = 1.0

            interp = np.vstack([np.interp(x_common, t_axis, row) for row in onehots])
            interp = self._smooth_rows(interp, window_size)

            frac = interp / (interp.sum(axis=0, keepdims=True) + 1e-9)
            series_per_run.append(frac)

        if not series_per_run:
            ax.text(0.5, 0.5, "No option usage found", ha='center', va='center')
        else:
            mean_frac = np.mean(np.stack(series_per_run, axis=0), axis=0)  # [|vocab|, |x|]
            ax.stackplot(x_common, *mean_frac, labels=vocab)
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel("Fraction of Steps (Option)")
            if label:
                ax.set_title(label)
            ax.legend(loc='upper left', ncols=2, frameon=True)
            ax.grid(True)
        return ax

    def plot_option_heatmap_over_episodes(self, run_idx=0, ax=None, title="Option Usage per Episode"):
        """
        Heatmap: rows = option names, cols = episode index; values = fraction of episode steps.
        """
        import numpy as np
        if self.num_runs == 0:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return ax

        run = self.metrics[run_idx]
        vocab = self._collect_option_name_vocab()
        idx = {n: i for i, n in enumerate(vocab)}
        M = np.zeros((len(vocab), len(run)), dtype=float)

        for e, ep in enumerate(run):
            names = self._episode_option_series(ep)
            if not names:
                continue
            total = len(names)
            if total == 0:
                continue
            for n in names:
                M[idx[n], e] += 1.0 / total

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        im = ax.imshow(M, aspect='auto', interpolation='nearest', origin='lower')
        ax.set_yticks(range(len(vocab)))
        ax.set_yticklabels(vocab)
        ax.set_xlabel("Episode")
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Fraction of Episode")
        ax.grid(False)
        return ax

    def plot_option_transition_matrix(self, run_idx=0, ax=None, title="Option Transition Matrix"):
        """
        Row-normalized transition probabilities between options, after collapsing repeats.
        """
        import numpy as np
        if self.num_runs == 0:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return ax

        run = self.metrics[run_idx]
        vocab = self._collect_option_name_vocab()
        idx = {n: i for i, n in enumerate(vocab)}
        C = np.zeros((len(vocab), len(vocab)), dtype=float)

        for ep in run:
            names = self._episode_option_series(ep)
            if not names:
                continue
            seq = [names[0]]
            for n in names[1:]:
                if n != seq[-1]:
                    seq.append(n)
            for a, b in zip(seq[:-1], seq[1:]):
                C[idx[a], idx[b]] += 1.0

        row_sum = C.sum(axis=1, keepdims=True) + 1e-9
        P = C / row_sum

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(P, origin='lower', interpolation='nearest')
        ax.set_xticks(range(len(vocab))); ax.set_xticklabels(vocab, rotation=45, ha='right')
        ax.set_yticks(range(len(vocab))); ax.set_yticklabels(vocab)
        ax.set_xlabel("Next Option"); ax.set_ylabel("Current Option")
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("P(next | current)")
        ax.grid(False)
        return ax

    def plot_episode_option_timeline(self, run_idx=0, ep_idx=0, ax=None, title="Option Timeline"):
        """
        Gantt-style timeline for a single episode: horizontal segments per active option.
        """
        if self.num_runs == 0 or len(self.metrics[run_idx]) == 0:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(12, 2.8))
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return ax

        ep = self.metrics[run_idx][ep_idx]
        names = self._episode_option_series(ep)
        if not names:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(12, 2.8))
            ax.text(0.5, 0.5, "No agent_logs", ha='center', va='center')
            return ax

        segs = []
        cur, start = names[0], 0
        for i, n in enumerate(names[1:], 1):
            if n != cur:
                segs.append((cur, start, i))
                cur, start = n, i
        segs.append((cur, start, len(names)))

        vocab = self._collect_option_name_vocab()
        pos = {n: i for i, n in enumerate(vocab)}

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 2.8))
        for name, s, e in segs:
            y = pos[name]
            ax.broken_barh([(s, e - s)], (y - 0.4, 0.8))
        ax.set_yticks(range(len(vocab))); ax.set_yticklabels(vocab)
        ax.set_xlabel("Timestep")
        ax.set_ylim([-0.5, len(vocab)-0.5])
        ax.set_title(f"{title} (run {run_idx+1}, ep {ep_idx+1})")
        ax.grid(True, axis='x')
        return ax

    def plot_option_total_counts(self, ax=None, title="Total Option Counts (All Runs)"):
        """
        Bar chart of total step counts per option across all runs.
        """
        from collections import Counter
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        cnt = Counter()
        for run in self.metrics:
            for ep in run:
                for n in self._episode_option_series(ep):
                    cnt[n] += 1
        if not cnt:
            ax.text(0.5, 0.5, "No option usage found", ha='center', va='center')
            return ax

        names, vals = zip(*sorted(cnt.items(), key=lambda x: x[1], reverse=True))
        ax.bar(names, vals)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel("Step Count"); ax.set_title(title)
        ax.grid(True, axis='y')
        return ax

    def plot_option_fraction_vs_return(self, run_idx=0, top_k=6, ax=None, title="Option Usage vs Episode Return"):
        """
        For top-k most used options in a run, scatter fraction-of-episode vs episode return.
        """
        import numpy as np
        from collections import Counter
        if self.num_runs == 0:
            return None

        run = self.metrics[run_idx]
        total_cnt = Counter()
        for ep in run:
            for n in self._episode_option_series(ep):
                total_cnt[n] += 1
        top = [n for n, _ in total_cnt.most_common(top_k) if n != "None"]
        if not top:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.text(0.5, 0.5, "No options to correlate", ha='center', va='center')
            return ax

        ncols = min(3, len(top))
        nrows = int(np.ceil(len(top) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
        axes = axes.flatten()

        pane_used = 0
        for i, name in enumerate(top):
            xs, ys = [], []
            for ep in run:
                names = self._episode_option_series(ep)
                if not names:
                    continue
                frac = names.count(name) / max(1, len(names))
                ret = ep.get("ep_return", None)
                if ret is not None:
                    xs.append(frac); ys.append(ret)
            ax_i = axes[i]
            if xs and ys:
                ax_i.scatter(xs, ys, s=12)
            else:
                ax_i.text(0.5, 0.5, "No data", ha='center', va='center')
            ax_i.set_title(name)
            ax_i.set_xlabel("Fraction of Episode using option")
            ax_i.set_ylabel("Episode Return")
            ax_i.grid(True)
            pane_used += 1

        for j in range(pane_used, len(axes)):
            axes[j].axis('off')

        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return axes[0] if pane_used > 0 else None

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
            options_index = [option.get('OptionIndex', None) for option in options]
        else:
            options_index = [None for _ in range(len(actions))]

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

        if video_type == "gif":
            filename = f"{filename}.gif"
            imageio.mimsave(filename, img_frames, fps=fps)
            print(f"GIF saved as {filename}")
        else:
            raise NotImplementedError("Only GIF video type is implemented.")