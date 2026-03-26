"""
OptionClassifier
================
A lightweight supervised classifier that maps option-rollout features
(produced by Features.py) to option class labels.

The network is built with the project's NetworkGen factory so any
architecture registered in NETWORK_PRESETS can be used.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional

from ..Networks.NetworkFactory import NetworkGen, prepare_network_config
from ..Networks.Presets import NETWORK_PRESETS


class OptionClassifier:
    """
    Parameters
    ----------
    hyper_params  : HyperParameters with at least:
        - network      : preset key (str) or raw list of layer dicts
        - num_classes  : int
        - feature_dim  : int
        - step_size    : float  (default 3e-4)
        - eps          : float  (default 1e-8)
        - max_grad_norm: float | None  (default 1.0)
    device        : torch device string
    """

    def __init__(self, feature_dict: dict, num_classes: int, network_type: str, repr_dim: int, device: str = "cpu",
                 class_weights: Optional[torch.Tensor] = None, kl_weight: float = 0.0):
        self.feature_dict = feature_dict
        self.num_classes  = num_classes
        self.network_type = network_type
        self.repr_dim = repr_dim
        self.device = device
        self.kl_weight = kl_weight

        network_cfg = NETWORK_PRESETS[network_type]

        # Representation network: feature_dim -> repr_dim
        repr_descs = prepare_network_config(
            network_cfg,
            input_dims=feature_dict,
            output_dim=repr_dim,
        )
        self.representation = NetworkGen(layer_descriptions=repr_descs).to(device)

        # Linear head: repr_dim -> num_classes
        self.head = nn.Linear(repr_dim, num_classes).to(device)

        self.optimizer = optim.Adam(
            list(self.representation.parameters()) + list(self.head.parameters()),
            lr=3e-4,
            eps=1e-8,
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )

    # ------------------------------------------------------------------
    # training / evaluation
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """
        Run one full epoch over *dataloader*.

        Returns
        -------
        dict with keys "loss" and "accuracy" (both averaged over the epoch)
        """
        self.representation.train()
        self.head.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        max_grad_norm = 1.0

        for features, labels in dataloader:
            features = {key: val.to(self.device) for key, val in features.items()}
            labels   = labels.to(self.device)
            
            logits = self.head(self.representation(**features))
            loss   = self.criterion(logits, labels)

            if self.kl_weight > 0.0:
                feat_cat = torch.cat([v.flatten(1) for v in features.values()], dim=1)
                loss = loss + self.kl_weight * _kl_distmat_loss(labels, feat_cat, logits)

            self.optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self.representation.parameters()) + list(self.head.parameters()),
                    max_grad_norm,
                )
            self.optimizer.step()

            total_loss    += loss.item() * labels.shape[0]
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.shape[0]

        return {
            "loss":     total_loss     / total_samples,
            "accuracy": total_correct  / total_samples,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """
        Evaluate on a dataloader without gradient updates.

        Returns
        -------
        dict with keys "loss" and "accuracy"
        """
        self.representation.eval()
        self.head.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for features, labels in dataloader:
            features = {key: val.to(self.device) for key, val in features.items()}
            labels   = labels.to(self.device)

            logits = self.head(self.representation(**features))
            loss   = self.criterion(logits, labels)

            total_loss    += loss.item() * labels.shape[0]
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.shape[0]

        return {
            "loss":     total_loss     / total_samples,
            "accuracy": total_correct  / total_samples,
        }

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (N, feature_dim) float tensor

        Returns
        -------
        class_indices : (N,) long tensor
        """
        self.representation.eval()
        self.head.eval()
        features = {key: val.to(self.device) for key, val in features.items()}
        logits = self.head(self.representation(**features))
        return logits.argmax(1)

    # ------------------------------------------------------------------
    # confusion matrix
    # ------------------------------------------------------------------

    @torch.no_grad()
    def confusion_matrix(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Compute the confusion matrix over a dataloader.

        Returns
        -------
        matrix : (num_classes, num_classes) long tensor
            matrix[true, pred] = count
        """
        self.representation.eval()
        self.head.eval()
        matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

        for features, labels in dataloader:
            features = {key: val.to(self.device) for key, val in features.items()}
            labels   = labels.to(self.device)

            preds = self.head(self.representation(**features)).argmax(1)
            for true, pred in zip(labels, preds):
                matrix[true.item(), pred.item()] += 1

        return matrix

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, file_path=None):
        checkpoint = {
                "feature_dict": self.feature_dict,
                "num_classes": self.num_classes,
                "network_type": self.network_type,
                "repr_dim": self.repr_dim,
                "device": self.device,
                "class_weights": self.criterion.weight.cpu() if self.criterion.weight is not None else None,
                "kl_weight": self.kl_weight,
                
                "representation_state_dict": self.representation.state_dict(),
                "head_state_dict":           self.head.state_dict(),
                "optimizer_state_dict":      self.optimizer.state_dict(),
            }
        if file_path is not None:
            torch.save(checkpoint, file_path)
        return checkpoint

    @classmethod
    def load(cls, file_path: str, checkpoint=None) -> "OptionClassifier":
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        instance = cls(checkpoint["feature_dict"], checkpoint["num_classes"], 
                       checkpoint["network_type"], checkpoint["repr_dim"], 
                       device=checkpoint["device"], class_weights=checkpoint["class_weights"], 
                       kl_weight=checkpoint["kl_weight"])
        
        instance.representation.load_state_dict(checkpoint["representation_state_dict"])
        instance.head.load_state_dict(checkpoint["head_state_dict"])
        instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return instance



# ------------------------------------------------------------------
# KL distance-matrix regularizer helpers
# ------------------------------------------------------------------

def _pairwise_distance_matrix(x: torch.Tensor, metric: str = "cosine", eps: float = 1e-8) -> torch.Tensor:
    if metric == "cosine":
        x_norm = F.normalize(x, p=2, dim=1)
        return (1.0 - x_norm @ x_norm.t()).clamp(min=0.0)
    else:  # euclidean
        return torch.cdist(x, x, p=2)


def _distance_matrix_to_probs(dist: torch.Tensor, temperature: float = 1.0, eps: float = 1e-8):
    log_p = F.log_softmax(-dist / temperature, dim=-1)
    return log_p.exp(), log_p


def _kl_distmat_loss(
    option_id: torch.Tensor,
    option_feature: torch.Tensor,
    logits: torch.Tensor,
    metric: str = "cosine",
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    unique_ids, inverse = torch.unique(option_id, sorted=True, return_inverse=True)
    k = int(unique_ids.numel())
    if k < 2:
        return logits.sum() * 0.0

    feat = option_feature.float()
    if feat.ndim > 2:
        feat = feat.flatten(start_dim=1)

    counts = torch.bincount(inverse, minlength=k).to(feat.dtype).unsqueeze(1).clamp_min(1.0)

    feat_sum = torch.zeros(k, feat.shape[1], device=feat.device, dtype=feat.dtype)
    feat_sum.index_add_(0, inverse, feat)
    feat_proto = feat_sum / counts  # (k, fdim)

    probs = torch.softmax(logits / temperature, dim=-1)  # (N, num_classes)
    prob_sum = torch.zeros(k, logits.shape[1], device=logits.device, dtype=logits.dtype)
    prob_sum.index_add_(0, inverse, probs)
    class_proto = prob_sum / counts  # (k, num_classes)

    dist_feat  = _pairwise_distance_matrix(feat_proto,  metric=metric, eps=eps)
    dist_class = _pairwise_distance_matrix(class_proto, metric=metric, eps=eps)

    p_feat,   _           = _distance_matrix_to_probs(dist_feat,  temperature=temperature, eps=eps)
    _,        log_p_class = _distance_matrix_to_probs(dist_class, temperature=temperature, eps=eps)

    return F.kl_div(log_p_class, p_feat, reduction="batchmean")
