# RLBase/Evaluate/MultiExpPlots.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

from .SingleExpAnalyzer import SingleExpAnalyzer

plt.rcParams.update({
    "font.size": 18,            # base font size
    "legend.fontsize": 16,      # legend
    "figure.titlesize": 24      # overall figure title
})


def _build_analyzer(payload):
    if isinstance(payload, str):
        return SingleExpAnalyzer(exp_path=payload)
    elif isinstance(payload, list):
        return SingleExpAnalyzer(metrics=payload)
    else:
        raise ValueError(f"Unknown param type: {type(payload)}")


def _colors():
    return ['red', 'black', 'green', 'purple', 'blue',
            'orange', 'brown', 'pink', 'grey', 'cyan']


def _linestyles():
    # Keep simple, readable sets
    return ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']


def _markers():
    # Standard marker set; combine with linestyles to make a fmt string
    return ['o', '^', 's', 'D', 'v', 'p', 'X', '*', '>', '<']


def _fmt_for_idx(i):
    ls = _linestyles()[i % len(_linestyles())]
    mk = _markers()[i % len(_markers())]
    return f"{ls}{mk}"  # e.g., "-o", "--^", "-.s", ":D"


def plot_experiments(
    agent_dict,
    save_dir,
    name="",
    window_size=10,
    plot_each=False,
    show_ci=False,
    ignore_last=False,
    plt_configs=("r_e", "r_s", "s_e")
):
    """
    Example:
        agent_dict = {
            "labelA": "path/to/exp_A",
            "labelB": [metrics_for_B],
        }
    Produces one figure with stacked rows for each plt_config,
    overlaying curves from all experiments in each row.
    """
    os.makedirs(save_dir, exist_ok=True)

    colors = _colors()
    fig, axs = plt.subplots(len(plt_configs), 1,
                            figsize=(10, 6 * len(plt_configs)),
                            constrained_layout=True)

    generated_name = []
    for i, (exp_label, payload) in enumerate(agent_dict.items()):
        analyzer = _build_analyzer(payload)
        fmt = _fmt_for_idx(i)
        color = colors[i % len(colors)]

        analyzer.plot_combined(
            fig, axs,
            color=color,
            marker=fmt,                  # your analyzer expects a single fmt string here
            label=exp_label,
            show_legend=(i == len(agent_dict) - 1),
            window_size=window_size,
            plot_each=plot_each,
            show_ci=show_ci,
            title=name,
            ignore_last=ignore_last,
            plt_configs=list(plt_configs),
            index=i,
            total_index=len(agent_dict)
        )
        generated_name.append(exp_label)

    out_name = name if name else "_".join(generated_name)
    out_path = os.path.join(save_dir, f"{out_name}.png")

    # Make a bit more room for the suptitle/legend if present
    fig.subplots_adjust(top=0.90)
    fig.savefig(out_path, format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    return out_path


def plot_option_usage(
    agent_dict,
    save_dir,
    name="",
    window_size=10,
    plot_each=False,
    show_ci=False,
    ignore_last=False,
    option_classes=None,
    x_type="s",
):
    """
    One subplot per option in `option_classes`.
    On each axis, overlay a line per experiment (from agent_dict).
    """
    if not option_classes:
        raise ValueError("option_classes must be a non-empty list.")

    os.makedirs(save_dir, exist_ok=True)

    # Grid: up to 4 rows; compute columns from number of options
    n_rows = 6
    n_cols = math.ceil(len(option_classes) / n_rows)

    # Width should scale with columns; height with rows
    fig, axs_grid = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 3.5 * n_rows),
        constrained_layout=True
    )

    # Flatten axes, then keep only as many as we have options
    axs = np.array(axs_grid).ravel().tolist()
    needed = len(option_classes)
    assert len(axs) >= needed, "Internal: not enough axes allocated."

    # Hide any extra axes beyond the number of options
    for ax in axs[needed:]:
        ax.set_visible(False)

    # Keep only the ones we need (order matches option_classes)
    axs = axs[:needed]

    colors = _colors()
    generated_name = []

    # Iterate experiments (memory-lean: build analyzer per experiment)
    for i, (exp_label, payload) in enumerate(agent_dict.items()):
        analyzer = _build_analyzer(payload)
        color = colors[i % len(colors)]
        fmt = _fmt_for_idx(i)

        analyzer.plot_option_class_usage(
            fig, axs,
            color=color,
            marker=fmt,
            label=exp_label,
            show_legend=(i == len(agent_dict) - 1),
            window_size=window_size,
            plot_each=plot_each,
            show_ci=show_ci,
            title=name,
            ignore_last=ignore_last,
            index=i,
            total_index=len(agent_dict),
            option_classes=option_classes,
            x_type=x_type,
        )
        generated_name.append(exp_label)

    out_name = name if name else "_".join(generated_name)
    out_path = os.path.join(save_dir, f"{out_name}.png")

    # Small top margin for figure legend/suptitle
    fig.subplots_adjust(top=0.90)
    fig.savefig(out_path, format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    return out_path

def gather_experiments(exp_dir, name_string_conditions=None, name_string_anti_conditions=None):
    """
    Scan a directory of experiments; return a flat list of runs (metrics).
    - name_string_conditions: list of substrings that MUST be in the folder name
    - name_string_anti_conditions: list of substrings that MUST NOT be in the folder name
    """
    name_string_conditions = name_string_conditions or []
    name_string_anti_conditions = name_string_anti_conditions or []

    run_counter = 0
    file_counter = 0
    metrics_lst = []
    no_metrics_file_lst = []

    for exp in os.listdir(exp_dir):
        satisfy_conditions = all(s in exp for s in name_string_conditions) if name_string_conditions else True
        satisfy_anti = not any(s in exp for s in name_string_anti_conditions) if name_string_anti_conditions else True

        if not (satisfy_conditions and satisfy_anti):
            continue

        exp_path = os.path.join(exp_dir, exp)
        try:
            analyzer = SingleExpAnalyzer(exp_path=exp_path)
            metrics_lst += analyzer.metrics
            run_counter += len(analyzer.metrics)
            file_counter += 1
        except FileNotFoundError:
            no_metrics_file_lst.append(exp_path)

    print("***")
    if file_counter == 0:
        raise FileNotFoundError(f"No experiment folders matched {name_string_conditions} (and not {name_string_anti_conditions}).")
    if no_metrics_file_lst:
        print("These folders matched naming but had no 'all_metrics.pkl':")
        print(*no_metrics_file_lst, sep="\n")

    print(f"Found {run_counter} runs from {file_counter} folders")
    return metrics_lst