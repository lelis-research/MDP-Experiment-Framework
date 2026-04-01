import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def extract_representations(
    model,
    dataset,
    device: str = "cpu",
    batch_size: int = 512,
):
    """
    Run every sample in *dataset* through model.

    model can be:
      - an OptionClassifier (has .representation attribute)
      - a raw network (e.g. agent.encoder.encoder) called with **features

    Returns
    -------
    x_rep : (N, repr_dim) float tensor  — one vector per rollout
    y     : (N,) long tensor            — class labels
    """
    net = None
    if model is not None:
        net = model.representation if hasattr(model, "representation") else model
        net.eval()
        net.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    reps, labels = [], []

    with torch.no_grad():
        for features, label in loader:
            features = {k: v.to(device) for k, v in features.items()}
            if net is not None:
                rep = net(**features)
            else:
                rep = torch.cat([v.flatten(start_dim=1) for v in features.values()], dim=1)
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


# ---------------------------------------------------------------------------
# EmbToRep — learned embedding table + MLP decoder
# ---------------------------------------------------------------------------

class EmbToRep(nn.Module):
    """
    Maps option IDs to target feature vectors via a learned embedding + MLP decoder.

    Architecture: Embedding(num_options, emb_dim) -> tanh -> Linear -> ReLU -> Linear
    """

    def __init__(self, num_options: int, emb_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.emb = nn.Embedding(num_options, emb_dim)
        nn.init.uniform_(self.emb.weight, a=-0.05, b=0.05)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def get_embeddings(self, option_id: torch.Tensor | None = None) -> torch.Tensor:
        """Return tanh-squashed embeddings for given IDs (or all IDs if None)."""
        e = self.emb.weight if option_id is None else self.emb(option_id)
        return torch.tanh(e)

    def forward(self, option_id: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.get_embeddings(option_id))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _TargetDataset(Dataset):
    def __init__(self, option_id: torch.Tensor, option_feature: torch.Tensor):
        self.option_id      = option_id.long()
        self.option_feature = option_feature.float()

    def __len__(self):
        return self.option_id.shape[0]

    def __getitem__(self, i):
        return self.option_id[i], self.option_feature[i]


def _pairwise_dist(x: torch.Tensor, metric: str, eps: float = 1e-8) -> torch.Tensor:
    if metric == "cosine":
        x_n = F.normalize(x.float(), dim=1, eps=eps)
        return (1.0 - x_n @ x_n.T).clamp(0.0)
    if metric == "l2":
        return torch.cdist(x.float(), x.float(), p=2)
    raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")


def _dist_to_probs(dist: torch.Tensor, temperature: float = 1.0, eps: float = 1e-8):
    t = max(float(temperature), eps)
    logits = -dist / t
    diag = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diag, -1e9)
    return torch.softmax(logits, dim=1), torch.log_softmax(logits, dim=1)


def _kl_emb_vs_target(
    model: EmbToRep,
    option_id: torch.Tensor,
    option_feature: torch.Tensor,
    metric: str = "cosine",
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL( dist-probs of target feature prototypes || dist-probs of embeddings )."""
    unique_ids, inverse = torch.unique(option_id, sorted=True, return_inverse=True)
    k = int(unique_ids.numel())
    if k < 2:
        return model.emb.weight.sum() * 0.0  # differentiable zero

    feat = option_feature.float()
    fdim = feat.shape[1]
    feat_sum = torch.zeros(k, fdim, device=feat.device, dtype=feat.dtype)
    feat_sum.index_add_(0, inverse, feat)
    counts = torch.bincount(inverse, minlength=k).to(feat.dtype).unsqueeze(1).clamp_min(1.0)
    feat_proto = feat_sum / counts

    emb_vec   = model.emb(unique_ids)
    dist_feat = _pairwise_dist(feat_proto, metric=metric, eps=eps)
    dist_emb  = _pairwise_dist(emb_vec,    metric=metric, eps=eps)

    p_feat, _    = _dist_to_probs(dist_feat, temperature=temperature, eps=eps)
    _, log_p_emb = _dist_to_probs(dist_emb,  temperature=temperature, eps=eps)

    return F.kl_div(log_p_emb, p_feat, reduction="batchmean")


def _make_mse_kl_loss(kl_weight: float, metric: str, temperature: float):
    def loss_fn(model, option_id, option_feature):
        mse = F.mse_loss(model(option_id), option_feature)
        kl  = _kl_emb_vs_target(model, option_id, option_feature, metric=metric, temperature=temperature)
        return mse + kl_weight * kl
    return loss_fn


def _make_metrics_fn(metric: str, temperature: float):
    def metric_fn(model, option_id, option_feature):
        with torch.no_grad():
            pred = model(option_id)
            kl   = _kl_emb_vs_target(model, option_id, option_feature, metric=metric, temperature=temperature)
            return {
                "mse": F.mse_loss(pred, option_feature).item(),
                "l1":  F.l1_loss(pred, option_feature).item(),
                "kl":  float(kl.item()),
            }
    return metric_fn


def _run_epoch(model, loader, device, loss_fn, optimizer=None, metric_fn=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_n = 0
    sums = {"loss": 0.0}
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for option_id, option_feature in loader:
            option_id      = option_id.to(device)
            option_feature = option_feature.to(device)
            loss = loss_fn(model, option_id, option_feature)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = option_id.shape[0]
            total_n += bs
            sums["loss"] += float(loss.item()) * bs
            if metric_fn is not None:
                for k, v in metric_fn(model, option_id, option_feature).items():
                    sums.setdefault(k, 0.0)
                    sums[k] += float(v) * bs
    return {k: v / max(total_n, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# train_learned_embedding — public API
# ---------------------------------------------------------------------------

def train_learned_embedding(
    x_rep: torch.Tensor,
    y: torch.Tensor,
    num_options: int,
    device: str = "cpu",
    emb_dim: int = 4,
    kl_weight: float = 0.05,
    kl_metric: str = "cosine",
    kl_temperature: float = 1.0,
    num_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> torch.Tensor:
    """
    Train a small EmbToRep network to replace compute_class_embeddings.

    Instead of averaging x_rep per class, learns a compact embedding table
    (num_options, emb_dim) by minimising MSE + kl_weight * KL on the dataset
    (y, x_rep).  The KL term encourages pairwise distances in embedding space
    to match pairwise distances in feature space.

    Parameters
    ----------
    x_rep         : (N, D) per-sample representations from extract_representations
    y             : (N,)   option class labels
    num_options   : number of distinct options
    emb_dim       : dimension of the learned embedding table (default 4)
    kl_weight     : weight on the KL structure-matching term
    kl_metric     : "cosine" or "l2" — metric for KL distance matrices
    kl_temperature: softmax temperature for distance-to-probability conversion
    num_epochs    : training epochs
    lr            : Adam learning rate
    batch_size    : DataLoader batch size

    Returns
    -------
    learned_emb : (num_options, emb_dim) — learned embedding table (tanh-squashed)
    """
    ds = _TargetDataset(option_id=y, option_feature=x_rep)
    n_train = int(len(ds) * 0.8)
    n_val   = len(ds) - n_train
    train_set, val_set = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)

    out_dim   = x_rep.shape[1]
    model     = EmbToRep(num_options=num_options, emb_dim=emb_dim, out_dim=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn   = _make_mse_kl_loss(kl_weight=kl_weight, metric=kl_metric, temperature=kl_temperature)
    metric_fn = _make_metrics_fn(metric=kl_metric, temperature=kl_temperature)

    for e in range(num_epochs):
        tr = _run_epoch(model, train_loader, device=device, loss_fn=loss_fn, optimizer=optimizer, metric_fn=metric_fn)
        va = _run_epoch(model, val_loader,   device=device, loss_fn=loss_fn, optimizer=None,      metric_fn=metric_fn)
        print(
            f"[EmbToRep] Epoch {e+1}/{num_epochs} | "
            f"Train loss={tr['loss']:.6f} mse={tr.get('mse', 0):.6f} "
            f"l1={tr.get('l1', 0):.6f} kl={tr.get('kl', 0):.6f} | "
            f"Val   loss={va['loss']:.6f} mse={va.get('mse', 0):.6f} "
            f"l1={va.get('l1', 0):.6f} kl={va.get('kl', 0):.6f}"
        )

    with torch.no_grad():
        learned_emb = model.get_embeddings().cpu()

    return learned_emb