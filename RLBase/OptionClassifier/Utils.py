import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrices(train_cm, val_cm, class_names, save_path):
    n = len(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cm, title in zip(axes, [train_cm, val_cm], ["Train", "Val"]):
        cm_np = cm.numpy()
        im = ax.imshow(cm_np, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(class_names, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_option_lens_boxplot(option_lens: dict, save_path: str):
    """
    Parameters
    ----------
    option_lens : dict mapping option_id -> list of action-sequence lengths
                  (from dataset.option_lens())
    save_path   : path to save the figure
    """
    labels = [str(k) for k in option_lens.keys()]
    data   = list(option_lens.values())

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 5))
    ax.boxplot(data, labels=labels)
    ax.set_xlabel("Option ID")
    ax.set_ylabel("Rollout Length")
    ax.set_title("Option Rollout Length Distribution")
    ax.tick_params(axis="x", labelsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_label_histogram(label_counts: dict, save_path: str):
    """
    Parameters
    ----------
    label_counts : dict mapping option_id -> count  (from dataset.label_counts())
    save_path    : path to save the figure
    """
    labels = [str(k) for k in label_counts.keys()]
    counts = list(label_counts.values())

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 5))
    ax.bar(labels, counts)
    ax.set_xlabel("Option ID")
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution")
    ax.tick_params(axis="x", labelsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    