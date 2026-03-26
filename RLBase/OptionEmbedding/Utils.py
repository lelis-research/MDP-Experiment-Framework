import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def extract_representations(
    classifier,
    dataset,
    device: str = "cpu",
    batch_size: int = 512,
):
    """
    Run every sample in *dataset* through classifier.representation.

    Returns
    -------
    x_rep : (N, repr_dim) float tensor  — one vector per rollout
    y     : (N,) long tensor            — class labels
    """
    classifier.representation.eval()
    classifier.representation.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    reps, labels = [], []

    with torch.no_grad():
        for features, label in loader:
            features = {k: v.to(device) for k, v in features.items()}
            rep = classifier.representation(**features)
            reps.append(rep.cpu())
            labels.append(label)

    return torch.cat(reps, dim=0), torch.cat(labels, dim=0)


def compute_class_embeddings(
    x_rep: torch.Tensor,
    y: torch.Tensor,
    normalize_rollouts: bool = True,
    normalize_class_means: bool = True,
    eps: float = 1e-8,
):
    """
    Average per-sample representations into one vector per option class.

    Parameters
    ----------
    x_rep               : (N, D) float tensor
    y                   : (N,) long tensor of class labels
    normalize_rollouts  : L2-normalise each row of x_rep before averaging
    normalize_class_means : L2-normalise each class mean after averaging

    Returns
    -------
    class_emb : (num_options, D) float tensor
    counts    : (num_options,) long tensor — samples per class
    """
    x = x_rep.float()
    y = y.long()

    if normalize_rollouts:
        x = F.normalize(x, dim=1, eps=eps)

    num_options = int(y.max().item()) + 1
    D = x.shape[1]

    class_emb = torch.zeros(num_options, D, dtype=x.dtype)
    counts    = torch.zeros(num_options, 1, dtype=x.dtype)

    class_emb.index_add_(0, y, x)
    counts.index_add_(0, y, torch.ones(y.shape[0], 1, dtype=x.dtype))
    class_emb = class_emb / counts.clamp_min(1.0)

    if normalize_class_means:
        class_emb = F.normalize(class_emb, dim=1, eps=eps)

    return class_emb, counts.squeeze(1).long()


def pca_project(
    x: torch.Tensor,
    out_dim: int,
    center: bool = True,
    eps: float = 1e-8,
):
    """
    Project *x* onto its top-`out_dim` principal components.

    Returns
    -------
    x_proj                  : (N, out_dim) float tensor
    explained_variance_ratio: (out_dim,) float tensor
    """
    x = x.float()
    N, D = x.shape

    mean = x.mean(dim=0, keepdim=True) if center else torch.zeros(1, D, dtype=x.dtype)
    x_c  = x - mean

    k = min(out_dim, min(N, D))
    _, S, Vh = torch.linalg.svd(x_c, full_matrices=False)
    components = Vh[:k].T          # (D, k)
    x_proj     = x_c @ components  # (N, k)

    total_var  = (S ** 2).sum() / max(N - 1, 1)
    expl_var_r = (S[:k] ** 2) / (max(N - 1, 1) * total_var + eps)

    return x_proj, expl_var_r


def plot_distance_matrix(
    class_emb: torch.Tensor,
    save_path: str = None,
    metric: str = "cosine",
    decimals: int = 2,
    title: str = None,
):
    """
    Plot the pairwise distance matrix between option embeddings.

    Parameters
    ----------
    class_emb : (K, D) float tensor — one embedding per option
    save_path : file path to save the figure (None = show interactively)
    metric    : "cosine" or "l2"
    decimals  : decimal places for cell annotations
    title     : plot title (auto-generated if None)
    """
    x = class_emb.float()
    if metric == "cosine":
        x_n = F.normalize(x, dim=1)
        dist = (1.0 - x_n @ x_n.T).clamp(0.0)
    elif metric == "l2":
        dist = torch.cdist(x, x, p=2)
    else:
        raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")

    dist = dist.cpu()
    K = dist.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist.numpy(), vmin=0.0)

    ax.set_title(title or f"Option Distance Matrix ({metric})")
    ax.set_xlabel("Option ID")
    ax.set_ylabel("Option ID")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))

    threshold = float(dist.max().item()) / 2.0
    for i in range(K):
        for j in range(K):
            val = float(dist[i, j].item())
            color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.{decimals}f}", ha="center", va="center",
                    color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_embeddings_2d(
    class_emb: torch.Tensor,
    save_path: str = None,
    title: str = "Option Embeddings (2D PCA)",
    center: bool = True,
):
    """
    Project option embeddings to 2D via PCA and plot each option as a labelled point.

    Parameters
    ----------
    class_emb : (K, D) float tensor — one embedding per option
    save_path : file path to save the figure (None = show interactively)
    title     : plot title
    center    : whether to mean-centre before SVD
    """
    x = class_emb.float()
    K, D = x.shape

    if D > 2:
        mean = x.mean(dim=0, keepdim=True) if center else torch.zeros(1, D, dtype=x.dtype)
        x_c = x - mean
        _, _, Vh = torch.linalg.svd(x_c, full_matrices=False)
        x_2d = (x_c @ Vh[:2].T).cpu().numpy()
        xlabel, ylabel = "PC 1", "PC 2"
    elif D == 2:
        x_2d = x.cpu().numpy()
        xlabel, ylabel = "dim 0", "dim 1"
    else:
        raise ValueError(f"class_emb must have at least 2 dims, got {D}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x_2d[:, 0], x_2d[:, 1], s=80)
    for i in range(K):
        ax.text(x_2d[i, 0], x_2d[i, 1], str(i),
                fontsize=10, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()