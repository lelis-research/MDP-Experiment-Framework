"""
Dataset for option-classifier training.

Streams pickled records written by VQOptionCriticAgent.dump_option_rollout()
and pre-computes features for all records at construction time.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from typing import Callable, Optional

from .Features import FEATURE_FN_DICT

class OptionRolloutDataset(Dataset):
    """
    Parameters
    ----------
    dump_path        : path to the pickle file written by the VQ agent
    feature_type     : one of "last_start", "sf", "reverse_sf", "mean_enc", "action_hist"
    encoder          : Encoder instance (required for SF-based features), else None
    gamma            : discount factor used in SF features
    num_actions      : number of primitive actions (for action_hist)

    Attributes
    ----------
    num_classes      : number of distinct options found in the dump
    feature_dim      : dimension of each pre-computed feature vector
    id_to_class      : dict mapping raw option_id -> contiguous class index
    class_to_id      : inverse mapping
    """

    def __init__(
        self,
        data_path: str,
        feature_type: str = "delta_last",
        encoder=None,
        gamma: float = 0.99,
        num_actions: int = 7,
    ):
        self.feature_type = feature_type
        self.encoder      = encoder
        self.gamma        = gamma
        self.num_actions  = num_actions

        # ---- load all records ------------------------------------------------
        records = []
        with open(data_path, "rb") as f:
            while True:
                try:
                    records.append(pickle.load(f))
                except EOFError:
                    break

        if len(records) == 0:
            raise ValueError(f"No records found in {data_path}")

        # ---- preprocessing ---------------------------------------------------
        records = self._filter_malformed(records)

        # ---- build label remapping -------------------------------------------
        raw_ids = [r["option_id"] for r in records]
        unique_ids = sorted(set(raw_ids))
        self.id_to_class = {oid: i for i, oid in enumerate(unique_ids)}
        self.class_to_id = {i: oid for oid, i in self.id_to_class.items()}
        self.num_classes  = len(unique_ids)
        
        # ---- pre-compute features --------------------------------------------
        fn = FEATURE_FN_DICT.get(feature_type)

        features = []
        for rec in records:
            feat = fn(rec)
            features.append(feat)

        self.features = {
            key: torch.stack([f[key] for f in features], dim=0).float()
            for key in features[0].keys()
        }
        
        self.labels   = torch.tensor(
            [self.id_to_class[oid] for oid in raw_ids], dtype=torch.long
        )
        self.feature_dict = {key: val.shape[1:] for key, val in self.features.items()}
        self._option_lens = [len(r["actions"]) for r in records]

    # ------------------------------------------------------------------
    @staticmethod
    def _filter_malformed(records: list) -> list:
        """
        For each option that has at least one rollout with action length > 1,
        drop all its length-1 rollouts (likely the option didn't run properly).
        Options whose every rollout is length 1 are left untouched.
        """
        from collections import defaultdict
        max_len = defaultdict(int)
        for r in records:
            max_len[r["option_id"]] = max(max_len[r["option_id"]], len(r["actions"]))

        return [
            r for r in records
            if not (max_len[r["option_id"]] > 1 and len(r["actions"]) == 1)
        ]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.features.items()}, self.labels[idx]       
    
    # ------------------------------------------------------------------
    def option_lens(self) -> dict:
        """
        Returns
        -------
        dict mapping option_id -> list of action-sequence lengths across all
        rollouts for that option
        """
        result = {self.class_to_id[i]: [] for i in range(self.num_classes)}
        for class_idx, length in zip(self.labels.tolist(), self._option_lens):
            result[self.class_to_id[class_idx]].append(length)
        return result

    # ------------------------------------------------------------------
    def label_counts(self) -> dict:
        """
        Returns
        -------
        dict mapping option_id -> count of samples for that option
        """
        counts = {}
        for class_idx, count in enumerate(torch.bincount(self.labels, minlength=self.num_classes).tolist()):
            counts[self.class_to_id[class_idx]] = count
        return counts 

def train_val_split(dataset: OptionRolloutDataset,
                    train_fraction: float = 0.8,
                    seed: int = 42):
    """
    Split an OptionRolloutDataset into train and validation subsets.

    Returns
    -------
    train_ds, val_ds  : torch.utils.data.Subset objects
    """
    n_train = int(len(dataset) * train_fraction)
    n_val   = len(dataset) - n_train
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)
