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
                  plt_configs=("r_s", "r_e", "s_e", "ou_s", "ou_e", "uni_ou_s", "uni_ou_e", "no_s", "no_e"), index=None, total_index=None):
        """
        Plot totals per episode and per steps, plus optional option-usage panels.
        """

        assert all(c in {"r_e", "r_s", "s_e", "ou_s", "ou_e", "no_s", "no_e", "uni_ou_e", "uni_ou_s"} for c in plt_configs), \
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
                # average times option != -1 per episode
                option_usage = [[
                    float(np.mean(ep["agent_logs"]["option_index"] != -1))
                    if ep.get("agent_logs") is not None
                    and "option_index" in ep["agent_logs"]
                    else 0.0
                    for ep in run] 
                    for run in self.metrics]  

                if any(any(v != 0.0 for v in run) for run in option_usage):
                    self._plot_data_per_steps(option_usage, ep_lengths, ax, num_steps, color, marker, label,
                                            window_size, plot_each, show_ci, ignore_last,
                                            x_label="Environment Steps", y_label="Option Usage")
                else:
                    print("OptionUsageLog doesn't exist — skipping plot.")

            elif config == "ou_e":
                # average times option != -1 per episode
                option_usage = [[
                    float(np.mean(ep["agent_logs"]["option_index"] != -1))
                    if ep.get("agent_logs") is not None
                    and "option_index" in ep["agent_logs"]
                    else 0.0
                    for ep in run] 
                    for run in self.metrics]          

                if any(any(v != 0.0 for v in run) for run in option_usage):
                    self._plot_data_per_episode(option_usage, ax, num_episodes, color, marker, label,
                                                window_size, plot_each, show_ci, ignore_last,
                                                x_label="Episode Number", y_label="Option Usage")
                else:
                    print("OptionUsageLog doesn't exist — skipping plot.")
            
            elif config == "uni_ou_e":
                # number of unique options being used per episode
                # option_usage = [[
                #     int(len(np.unique(ep["agent_logs"]["option_index"])))
                #     if ep.get("agent_logs") is not None and "option_index" in ep["agent_logs"]
                #     else 0.0 
                #     for ep in run] 
                #     for run in self.metrics]  
                option_usage = [[
                    len(np.unique(np.array(ep["agent_logs"]["option_index"])[np.array(ep["agent_logs"]["option_index"]) != -1]))
                    if ep.get("agent_logs") is not None and "option_index" in ep["agent_logs"]
                    else 0.0
                    for ep in run]
                    for run in self.metrics]
                               

                if any(any(v != 0.0 for v in run) for run in option_usage):
                    self._plot_data_per_episode(option_usage, ax, num_episodes, color, marker, label,
                                                window_size, plot_each, show_ci, ignore_last,
                                                x_label="Episode Number", y_label="Num Unique Option Usage")
                else:
                    print("OptionUsageLog doesn't exist — skipping plot.")
                    
            elif config == "uni_ou_s":
                # number of unique options being used per step
                # option_usage = [[
                #     int(len(np.unique(ep["agent_logs"]["option_index"])))
                #     if ep.get("agent_logs") is not None and "option_index" in ep["agent_logs"]
                #     else 0.0 
                #     for ep in run] 
                #     for run in self.metrics]     
                option_usage = [[
                    len(np.unique(np.array(ep["agent_logs"]["option_index"])[np.array(ep["agent_logs"]["option_index"]) != -1]))
                    if ep.get("agent_logs") is not None and "option_index" in ep["agent_logs"]
                    else 0.0
                    for ep in run]
                    for run in self.metrics]

                if any(any(v != 0.0 for v in run) for run in option_usage):
                    self._plot_data_per_steps(option_usage, ep_lengths, ax, num_steps, color, marker, label,
                                            window_size, plot_each, show_ci, ignore_last,
                                            x_label="Environment Steps", y_label="Num Unique Option Usage")
                else:
                    print("OptionUsageLog doesn't exist — skipping plot.")
                    
            elif config == "no_s":
                # number of total options per step
                num_options = [[float(np.mean(ep["agent_logs"]["num_options"])) 
                                if ep.get("agent_logs") is not None and "num_options" in ep["agent_logs"] 
                                else 0.0 
                                for ep in run] 
                                for run in self.metrics]

                if any(any(v != 0.0 for v in run) for run in num_options):
                    self._plot_data_per_steps(num_options, ep_lengths, ax, num_steps, color, marker, label,
                                            window_size, plot_each, show_ci, ignore_last,
                                            x_label="Environment Steps", y_label="Number of Options")
                else:
                    print("NumOptions doesn't exist — skipping plot.")

            elif config == "no_e":
                # number of total options per episode
                num_options = [[float(np.mean(ep["agent_logs"]["num_options"])) 
                                if ep.get("agent_logs") is not None and "num_options" in ep["agent_logs"] 
                                else 0.0 
                                for ep in run] 
                                for run in self.metrics]
                if any(any(v != 0.0 for v in run) for run in num_options):
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
                fig.legend(handles, labels, loc='upper center', ncols=math.ceil(len(labels)/3), shadow=False, bbox_to_anchor=(0.5, 0.965))
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
    def plot_option_embedding(
        self,
        fig=None,
        ax=None,
        save_dir=None,
        show=False,
        show_legend=True,
        title="e vs proto_e embeddings",

        # sizes / styling
        proto_s=8,
        e_s=32,
        proto_alpha=0.15,
        e_alpha=0.9,
        proto_lw=0.5,
        e_edge_lw=0.8,
        draw_lines=False,

        # optional episode filtering (episode index within each run)
        max_ep=None,  # inclusive
        min_ep=None,  # inclusive

        # PCA controls
        normalize_before_pca=True,   # L2-normalize rows before PCA (often good for cosine-like embeddings)
        pca_whiten=False,
        pca_random_state=0,
    ):
        """
        NEW expected format:
        episode["agent_logs"] is either None or a dict-of-numpy-arrays:
            agent_logs["e"]        : (T, D)
            agent_logs["proto_e"]  : (T, D)
            agent_logs["ind"]      : (T,) or (T,1)

        Behavior:
        - collects ALL points from ALL runs/episodes (optionally filtered by ep range)
        - ALWAYS reduces to 2D using PCA (even if D==2 already; it will be identity-ish)
        - plots:
            e       as circles
            proto_e as x markers
            both colored by ind
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        from sklearn.decomposition import PCA

        def _to_np(v):
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            return np.asarray(v)

        def _l2_normalize_rows(x, eps=1e-12):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            return x / (n + eps)

        # ---------------------------
        # Collect dict-of-arrays logs across runs/episodes (vectorized)
        # ---------------------------
        E_list, P_list, I_list = [], [], []

        runs = getattr(self, "metrics", [])
        for run in runs:
            for ep_idx, ep in enumerate(run):

                if min_ep is not None and ep_idx < int(min_ep):
                    continue
                if max_ep is not None and ep_idx > int(max_ep):
                    continue

                logs = ep.get("agent_logs", None)
                if logs is None or not isinstance(logs, dict):
                    continue

                if ("e" not in logs) or ("proto_e" not in logs) or ("ind" not in logs):
                    continue

                e = _to_np(logs["e"])
                p = _to_np(logs["proto_e"])
                ind = _to_np(logs["ind"])

                if e.ndim != 2 or p.ndim != 2:
                    continue

                if ind.ndim != 1:
                    ind = ind.reshape(-1)

                # align lengths (just in case)
                T = min(e.shape[0], p.shape[0], ind.shape[0])
                if T <= 0:
                    continue

                E_list.append(e[:T])
                P_list.append(p[:T])
                I_list.append(ind[:T].astype(np.int32, copy=False))

        if not E_list:
            print("plot_option_embedding: no dict-of-arrays data found in agent_logs.")
            return None

        E = np.concatenate(E_list, axis=0)   # (N, D)
        P = np.concatenate(P_list, axis=0)   # (N, D)
        I = np.concatenate(I_list, axis=0)   # (N,)

        if E.shape[1] < 2:
            raise ValueError(f"Expected embedding dim>=2, got D={E.shape[1]}")

        # ---------------------------
        # PCA to 2D (jointly for E and P so they share the same projection)
        # ---------------------------
        X = np.vstack([E, P])  # (2N, D)
        if normalize_before_pca:
            X = _l2_normalize_rows(X)

        n_samples, n_features = X.shape
        if min(n_samples, n_features) < 2:
            # pathological tiny case: just slice/pad
            Y = X[:, :2] if n_features >= 2 else np.pad(X, ((0, 0), (0, 2 - n_features)))
        else:
            pca = PCA(n_components=2, whiten=pca_whiten, random_state=pca_random_state)
            Y = pca.fit_transform(X)

        N = E.shape[0]
        E2 = Y[:N, :]   # (N, 2)
        P2 = Y[N:, :]   # (N, 2)

        # ---------------------------
        # Setup figure / axes (always 2D)
        # ---------------------------
        created_fig = False
        if fig is None or ax is None:
            created_fig = True
            fig, ax = plt.subplots(figsize=(7, 7))
        fig.subplots_adjust(top=0.88, bottom=0.15)

        # ---------------------------
        # Plot grouped by ind
        # ---------------------------
        inds = sorted(set(int(x) for x in I.tolist()))
        cmap = plt.cm.get_cmap("Set1", max(1, len(inds)))
        ind_handles = []

        for color_i, ind in enumerate(inds):
            mask = (I == ind)
            if not np.any(mask):
                continue

            color = cmap(color_i)
            e_vals = E2[mask]
            p_vals = P2[mask]

            sc_e = ax.scatter(
                e_vals[:, 0], e_vals[:, 1],
                color=color, marker="o", s=e_s,
                alpha=e_alpha, edgecolors="black",
                linewidths=e_edge_lw,
            )
            ax.scatter(
                p_vals[:, 0], p_vals[:, 1],
                color=color, marker="x", s=proto_s,
                alpha=proto_alpha, linewidths=proto_lw,
            )

            if draw_lines:
                m = min(len(p_vals), len(e_vals))
                for pe, ev in zip(p_vals[:m], e_vals[:m]):
                    ax.plot([pe[0], ev[0]], [pe[1], ev[1]],
                            color=color, alpha=0.15, linewidth=0.6)

            ind_handles.append((sc_e, f"ind={ind}"))

        # ---------------------------
        # Cosmetics
        # ---------------------------
        ax.set_title(f"{title}  (PCA→2D)")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

        if show_legend and ind_handles:
            handles = [h for h, _ in ind_handles]
            labels = [l for _, l in ind_handles]
            fig.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                ncol=min(5, len(handles)),
                frameon=True,
                fontsize=9,
            )

        # ---------------------------
        # Save / show
        # ---------------------------
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, "OptionEmbedding.png"), bbox_inches="tight", dpi=200)

        if show or created_fig:
            plt.show()

        return fig, ax
    def plot_option_embedding2(
        self,
        fig=None,
        ax=None,
        save_dir=None,
        show=False,
        show_legend=True,
        title="e vs proto_e embeddings",
        # sizes / styling
        proto_s=8,
        e_s=32,
        proto_alpha=0.15,
        e_alpha=0.9,
        proto_lw=0.5,
        e_edge_lw=0.8,
        draw_lines=False,
        # optional episode filtering
        max_ep=None,  # inclusive
        min_ep=None,  # inclusive
    ):
        """
        Scatter plot of:
        - e embeddings: large circles (colored by 'ind')
        - proto_e embeddings: small x markers (colored by 'ind')

        New expected format:
        episode["agent_logs"] is either None or a dict-of-numpy-arrays:
            agent_logs["e"]        : (T, d)
            agent_logs["proto_e"]  : (T, d)
            agent_logs["ind"]      : (T,)
        """

        def _to_np(v):
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            return np.asarray(v)

        # ---------------------------
        # Collect arrays across all runs/episodes (vectorized)
        # ---------------------------
        E_list, P_list, I_list = [], [], []

        runs = getattr(self, "metrics", [])
        for run in runs:
            for ep_idx, ep in enumerate(run):

                # episode filtering (kept same semantics as before)
                if min_ep is not None and ep_idx < int(min_ep):
                    continue
                if max_ep is not None and ep_idx > int(max_ep):
                    continue

                logs = ep.get("agent_logs", None)
                if logs is None:
                    continue
                if not isinstance(logs, dict):
                    # old-format experiment -> ignore (or add backward compat here)
                    continue

                if ("e" not in logs) or ("proto_e" not in logs) or ("ind" not in logs):
                    continue

                e = _to_np(logs["e"])
                p = _to_np(logs["proto_e"])
                ind = _to_np(logs["ind"])

                if e.ndim != 2 or p.ndim != 2:
                    continue

                # ensure ind is (T,)
                if ind.ndim != 1:
                    ind = ind.reshape(-1)

                # align lengths just in case
                T = min(e.shape[0], p.shape[0], ind.shape[0])
                if T <= 0:
                    continue

                E_list.append(e[:T])
                P_list.append(p[:T])
                I_list.append(ind[:T].astype(np.int32, copy=False))

        if not E_list:
            print("plot_option_embedding2: no dict-of-arrays data found in agent_logs.")
            return None

        E = np.concatenate(E_list, axis=0)   # (N, d)
        P = np.concatenate(P_list, axis=0)   # (N, d)
        I = np.concatenate(I_list, axis=0)   # (N,)

        dim = E.shape[1]
        if dim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D embeddings, got dim={dim}.")

        # ---------------------------
        # Setup figure / axes
        # ---------------------------
        created_fig = False
        if fig is None or ax is None:
            created_fig = True
            if dim == 3:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(figsize=(7, 7))
        fig.subplots_adjust(top=0.88, bottom=0.15)

        # ---------------------------
        # Plot grouped by ind
        # ---------------------------
        inds = sorted(set(int(x) for x in I.tolist()))
        cmap = plt.cm.get_cmap("Set1", max(1, len(inds)))
        ind_handles = []

        for color_i, ind in enumerate(inds):
            mask = (I == ind)
            if not np.any(mask):
                continue

            color = cmap(color_i)
            e_vals = E[mask]
            p_vals = P[mask]

            if dim == 2:
                sc_e = ax.scatter(
                    e_vals[:, 0], e_vals[:, 1],
                    color=color, marker="o", s=e_s,
                    alpha=e_alpha, edgecolors="black",
                    linewidths=e_edge_lw,
                )
                ax.scatter(
                    p_vals[:, 0], p_vals[:, 1],
                    color=color, marker="x", s=proto_s,
                    alpha=proto_alpha, linewidths=proto_lw,
                )
                if draw_lines:
                    m = min(len(p_vals), len(e_vals))
                    for pe, ev in zip(p_vals[:m], e_vals[:m]):
                        ax.plot([pe[0], ev[0]], [pe[1], ev[1]], color=color, alpha=0.15, linewidth=0.6)
            else:
                sc_e = ax.scatter(
                    e_vals[:, 0], e_vals[:, 1], e_vals[:, 2],
                    color=color, marker="o", s=e_s,
                    alpha=e_alpha, edgecolors="black",
                    linewidths=e_edge_lw,
                )
                ax.scatter(
                    p_vals[:, 0], p_vals[:, 1], p_vals[:, 2],
                    color=color, marker="x", s=proto_s,
                    alpha=proto_alpha, linewidths=proto_lw,
                )
                if draw_lines:
                    m = min(len(p_vals), len(e_vals))
                    for pe, ev in zip(p_vals[:m], e_vals[:m]):
                        ax.plot([pe[0], ev[0]], [pe[1], ev[1]], [pe[2], ev[2]],
                                color=color, alpha=0.15, linewidth=0.6)

            ind_handles.append((sc_e, f"ind={ind}"))

        # ---------------------------
        # Cosmetics
        # ---------------------------
        ax.set_title(title)
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

        if show_legend and ind_handles:
            handles = [h for h, _ in ind_handles]
            labels = [l for _, l in ind_handles]
            fig.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                ncol=min(5, len(handles)),
                frameon=True,
                fontsize=9,
            )

        # ---------------------------
        # Save / show
        # ---------------------------
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "OptionEmbedding.png")
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
            print(f"Saved {save_path}")

        if show or created_fig:
            plt.show()

        return fig, ax
    def plot_option_embedding_gif(
    self,
    name_tag="",
    show=True,
    title="e vs proto_e embeddings",
    # styling
    proto_s=16,
    e_s=32,
    proto_alpha=0.6,
    e_alpha=0.9,
    proto_lw=0.5,
    e_edge_lw=0.8,
    draw_lines=False,
    # episode controls
    min_ep=0,
    max_ep=None,
    every=1,
    fps=6,
    dpi=80,
    # multi-run behavior
    run_idx=None,
    cumulative_mode=False,
    ):
        """
        Create a GIF showing embedding scatter evolution over episodes.

        Expects per-episode agent_logs to be a dict of numpy arrays:
        agent_logs["e"]        : (T, d)
        agent_logs["proto_e"]  : (T, d)
        agent_logs["ind"]      : (T,)
        """

        print("[OptionEmbeddingGIF] Collecting agent logs...")

        def _to_np(v):
            # allow torch -> numpy, but your logs should already be numpy
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            return np.asarray(v)

        # ---------------------------
        # Collect data in a vectorized way
        # by_ep[ep_idx] = list of chunks (dicts of arrays)
        # each chunk is already "one episode" data, but we keep list for cumulative mode
        # ---------------------------
        by_ep = defaultdict(list)
        runs = getattr(self, "metrics", [])

        if run_idx is None:
            run_iter = enumerate(runs)
        else:
            r = int(run_idx)
            run_iter = [(r, runs[r])]

        for r, run in run_iter:
            for ep_idx, ep in enumerate(run):
                logs = ep.get("agent_logs", None)
                if logs is None:
                    continue
                if not isinstance(logs, dict):
                    # If you still have some old experiments saved in old format, skip or handle separately
                    continue

                if ("e" not in logs) or ("proto_e" not in logs) or ("ind" not in logs):
                    continue

                e = _to_np(logs["e"])
                p = _to_np(logs["proto_e"])
                ind = _to_np(logs["ind"])

                if e.ndim != 2 or p.ndim != 2:
                    continue
                if ind.ndim != 1:
                    ind = ind.reshape(-1)

                T = min(len(ind), e.shape[0], p.shape[0])
                if T <= 0:
                    continue

                chunk = {
                    "e": e[:T],
                    "proto_e": p[:T],
                    "ind": ind[:T].astype(np.int32, copy=False),
                    "_run_idx": r,
                    "_ep_idx": ep_idx,
                }
                by_ep[ep_idx].append(chunk)

        if not by_ep:
            print("[OptionEmbeddingGIF] No dict-of-arrays logs found. Aborting.")
            return None

        # ---------------------------
        # Episode range
        # ---------------------------
        last_ep = max(by_ep.keys())
        lo = int(min_ep) if min_ep is not None else 0
        hi = int(last_ep) if max_ep is None else int(max_ep)

        # keep only eps in range
        by_ep = {k: v for k, v in by_ep.items() if lo <= k <= hi}
        if not by_ep:
            print("[OptionEmbeddingGIF] No data in episode range. Aborting.")
            return None

        print("[OptionEmbeddingGIF] Inferring embedding dimension...")

        # ---------------------------
        # Infer dimension (2D or 3D)
        # ---------------------------
        # find first chunk
        first_chunk = None
        for ep_chunks in by_ep.values():
            if ep_chunks:
                first_chunk = ep_chunks[0]
                break
        if first_chunk is None:
            print("[OptionEmbeddingGIF] No usable chunks. Aborting.")
            return None

        dim = first_chunk["proto_e"].shape[1]
        if dim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D embeddings, got dim={dim}.")

        print("[OptionEmbeddingGIF] Computing axis limits...")

        # ---------------------------
        # Axis limits from ALL points in range
        # ---------------------------
        all_e = []
        all_p = []
        for ep_chunks in by_ep.values():
            for ch in ep_chunks:
                all_e.append(ch["e"])
                all_p.append(ch["proto_e"])

        all_e = np.concatenate(all_e, axis=0)
        all_p = np.concatenate(all_p, axis=0)
        all_pts = np.concatenate([all_e, all_p], axis=0)

        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        pad = 0.08 * (maxs - mins + 1e-9)
        mins -= pad
        maxs += pad

        print("[OptionEmbeddingGIF] Preparing colors...")

        # ---------------------------
        # Colors by ind across all data in range
        # ---------------------------
        all_inds = []
        for ep_chunks in by_ep.values():
            for ch in ep_chunks:
                all_inds.append(ch["ind"])
        all_inds = np.concatenate(all_inds, axis=0)
        inds = sorted(set(int(x) for x in all_inds.tolist()))

        ind_to_color_i = {ind: i for i, ind in enumerate(inds)}
        cmap = plt.cm.get_cmap("Set1", max(1, len(inds)))

        print("[OptionEmbeddingGIF] Rendering frames...")

        # ---------------------------
        # Output path
        # ---------------------------
        if self.exp_path is not None:
            out_path = os.path.join(self.exp_path, f"OptionEmbeddingOverTime_run_{run_idx}_{name_tag}.gif")
        else:
            out_path = f"OptionEmbeddingOverTime_run_{run_idx}_{name_tag}.gif"

        ep_list = list(range(lo, hi + 1, max(1, int(every))))
        frames = []
        cumulative_chunks = []

        for i, t in enumerate(ep_list):
            if i % max(1, len(ep_list)//10) == 0:
                print(f"[OptionEmbeddingGIF]  ├─ rendering frame {i+1}/{len(ep_list)} (ep {t})")

            if cumulative_mode:
                cumulative_chunks.extend(by_ep.get(t, []))
                chunks = cumulative_chunks
            else:
                chunks = by_ep.get(t, [])

            # flatten chunks into one set of arrays for this frame
            if not chunks:
                # still render empty frame to keep timing consistent
                E = np.zeros((0, dim))
                P = np.zeros((0, dim))
                I = np.zeros((0,), dtype=np.int32)
            else:
                E = np.concatenate([ch["e"] for ch in chunks], axis=0)
                P = np.concatenate([ch["proto_e"] for ch in chunks], axis=0)
                I = np.concatenate([ch["ind"] for ch in chunks], axis=0)

            # setup fig
            if dim == 3:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="3d")
                ax.set_xlim(mins[0], maxs[0])
                ax.set_ylim(mins[1], maxs[1])
                ax.set_zlim(mins[2], maxs[2])
            else:
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.set_xlim(mins[0], maxs[0])
                ax.set_ylim(mins[1], maxs[1])
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.axvline(0, color="gray", linewidth=0.5)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True)

            # group by ind and plot
            for ind in inds:
                mask = (I == ind)
                if not np.any(mask):
                    continue
                color = cmap(ind_to_color_i[ind])

                e_vals = E[mask]
                p_vals = P[mask]

                if dim == 2:
                    ax.scatter(
                        e_vals[:, 0], e_vals[:, 1],
                        color=color, marker="o", s=e_s,
                        alpha=e_alpha, edgecolors="black",
                        linewidths=e_edge_lw
                    )
                    ax.scatter(
                        p_vals[:, 0], p_vals[:, 1],
                        color=color, marker="x", s=proto_s,
                        alpha=proto_alpha, linewidths=proto_lw
                    )
                    if draw_lines:
                        m = min(len(p_vals), len(e_vals))
                        for pe, ev in zip(p_vals[:m], e_vals[:m]):
                            ax.plot([pe[0], ev[0]], [pe[1], ev[1]],
                                    color=color, alpha=0.15, linewidth=0.6)
                else:
                    ax.scatter(
                        e_vals[:, 0], e_vals[:, 1], e_vals[:, 2],
                        color=color, marker="o", s=e_s,
                        alpha=e_alpha, edgecolors="black",
                        linewidths=e_edge_lw
                    )
                    ax.scatter(
                        p_vals[:, 0], p_vals[:, 1], p_vals[:, 2],
                        color=color, marker="x", s=proto_s,
                        alpha=proto_alpha, linewidths=proto_lw
                    )
                    if draw_lines:
                        m = min(len(p_vals), len(e_vals))
                        for pe, ev in zip(p_vals[:m], e_vals[:m]):
                            ax.plot([pe[0], ev[0]], [pe[1], ev[1]], [pe[2], ev[2]],
                                    color=color, alpha=0.15, linewidth=0.6)

            ax.set_title(f"{title} | ep {t}")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            if dim == 3:
                ax.set_zlabel("Dim 3")

            fig.set_dpi(dpi)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(frame)
            plt.close(fig)

        print("[OptionEmbeddingGIF] Writing GIF to disk...")
        imageio.mimsave(out_path, frames, fps=fps)
        print("[OptionEmbeddingGIF] Done.")

        if show:
            print(f"GIF saved: {out_path}")

        return out_path
    
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
            options_index = self.metrics[run_number - 1][episode_number - 1]['agent_logs']['option_index']
            # options_index = [option[0].get('option_index', None) for option in options] #[0] is because we only look at the first env in test
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
