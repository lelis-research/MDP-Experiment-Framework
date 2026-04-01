import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .Utils import extract_representations, compute_class_embeddings, pca_project, train_learned_embedding



# ---------------------------------------------------------------------------
# OptionEmbedding
# ---------------------------------------------------------------------------

class OptionEmbedding:
    """
    Parameters
    ----------
    model                 : trained OptionClassifier, agent encoder, or None (raw features)
    dataset               : OptionRolloutDataset used to extract representations
    mode                  : "repr" | "repr_pca" | "repr_learned" | "repr_pca_learned"
    pca_dim               : target dimension for PCA (used when mode contains "pca")
    normalize_rollouts    : L2-normalise per-sample reps before averaging (repr/repr_pca only)
    normalize_class_means : L2-normalise each class mean embedding (repr/repr_pca only)
    device                : device for the forward pass / training
    batch_size            : batch size for representation extraction

    Learned-mode parameters (repr_learned / repr_pca_learned)
    ---------------------------------------------------------
    emb_dim       : dimension of the learned embedding table (default 4)
    kl_weight     : weight on the KL structure-matching term (default 0.05)
    kl_metric     : "cosine" or "l2" for the KL distance matrices (default "cosine")
    kl_temperature: softmax temperature for distance-to-probability conversion (default 1.0)
    num_epochs    : training epochs (default 20)
    lr            : Adam learning rate (default 1e-3)
    """

    def __init__(
        self,
        model,
        dataset,
        mode: str = "repr",
        pca_dim: int = 4,
        normalize_rollouts: bool = True,
        normalize_class_means: bool = True,
        device: str = "cpu",
        batch_size: int = 512,
        # learned-mode params
        emb_dim: int = 4,
        kl_weight: float = 0.05,
        kl_metric: str = "cosine",
        kl_temperature: float = 1.0,
        num_epochs: int = 20,
        lr: float = 1e-3,
    ):
        assert mode in ("repr", "repr_pca", "repr_learned", "repr_pca_learned"), \
            f"Unknown mode: {mode!r}"

        self.mode        = mode
        self.pca_dim     = pca_dim
        self.num_options = dataset.num_classes

        # 1. extract per-sample representations
        x_rep, y = extract_representations(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
        )

        # 2. optionally PCA-project
        self.explained_variance_ratio = None
        if mode in ("repr_pca", "repr_pca_learned"):
            x_rep, evr = pca_project(x_rep, out_dim=pca_dim)
            self.explained_variance_ratio = evr
            print(f"PCA {pca_dim}d — explained variance ratio sum: {evr.sum().item():.4f}")

        if mode in ("repr", "repr_pca"):
            # 3a. average per class
            self._embeddings, self.counts = compute_class_embeddings(
                x_rep, y,
                normalize_rollouts=normalize_rollouts,
                normalize_class_means=normalize_class_means,
            )
        elif mode in ("repr_learned", "repr_pca_learned"):
            # 3b. learn embedding table via MSE + KL
            self._embeddings = train_learned_embedding(
                x_rep=x_rep,
                y=y,
                num_options=self.num_options,
                device=device,
                emb_dim=emb_dim,
                kl_weight=kl_weight,
                kl_metric=kl_metric,
                kl_temperature=kl_temperature,
                num_epochs=num_epochs,
                lr=lr,
            )
            self.counts = torch.bincount(y, minlength=self.num_options)
        else:
            raise ValueError(f"Unknown mode: {mode!r}") 

        print(f"OptionEmbedding [{mode}]  shape={tuple(self._embeddings.shape)}  "
              f"counts={self.counts.tolist()}")

    # ------------------------------------------------------------------

    def get_embeddings(self) -> torch.Tensor:
        """Return the (num_options, emb_dim) embedding table."""
        return self._embeddings.clone()

    def save(self, file_path: str):
        torch.save({
            "mode":                     self.mode,
            "pca_dim":                  self.pca_dim,
            "num_options":              self.num_options,
            "embeddings":               self._embeddings,
            "counts":                   self.counts,
            "explained_variance_ratio": self.explained_variance_ratio,
        }, file_path)

    @classmethod
    def load(cls, file_path: str) -> "OptionEmbedding":
        ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
        obj  = cls.__new__(cls)
        obj.mode                     = ckpt["mode"]
        obj.pca_dim                  = ckpt["pca_dim"]
        obj.num_options              = ckpt["num_options"]
        obj._embeddings              = ckpt["embeddings"]
        obj.counts                   = ckpt["counts"]
        obj.explained_variance_ratio = ckpt["explained_variance_ratio"]
        return obj