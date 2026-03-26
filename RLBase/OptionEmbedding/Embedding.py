import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .Utils import extract_representations, compute_class_embeddings, pca_project



# ---------------------------------------------------------------------------
# OptionEmbedding
# ---------------------------------------------------------------------------

class OptionEmbedding:
    """
    Parameters
    ----------
    classifier          : trained OptionClassifier
    dataset             : OptionRolloutDataset used to extract representations
    mode                : "repr" | "repr_pca"
    pca_dim             : target dimension for PCA (only used when mode="repr_pca")
    normalize_rollouts  : L2-normalise per-sample reps before averaging
    normalize_class_means : L2-normalise each class mean embedding
    device              : device for the classifier forward pass
    batch_size          : batch size for representation extraction
    """

    def __init__(
        self,
        classifier,
        dataset,
        mode: str = "repr",
        pca_dim: int = 4,
        normalize_rollouts: bool = True,
        normalize_class_means: bool = True,
        device: str = "cpu",
        batch_size: int = 512,
    ):
        assert mode in ("repr", "repr_pca"), f"Unknown mode: {mode!r}"

        self.mode     = mode
        self.pca_dim  = pca_dim
        self.num_options = dataset.num_classes

        # 1. extract per-sample representations
        x_rep, y = extract_representations(
            classifier=classifier,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
        )

        # 2. optionally PCA-project
        self.explained_variance_ratio = None
        if mode == "repr_pca":
            x_rep, evr = pca_project(x_rep, out_dim=pca_dim)
            self.explained_variance_ratio = evr
            print(f"PCA {pca_dim}d — explained variance ratio sum: {evr.sum().item():.4f}")

        # 3. average per class
        self._embeddings, self.counts = compute_class_embeddings(
            x_rep, y,
            normalize_rollouts=normalize_rollouts,
            normalize_class_means=normalize_class_means,
        )

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
