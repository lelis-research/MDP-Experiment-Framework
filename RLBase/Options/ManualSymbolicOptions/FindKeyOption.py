from ..Utils import BaseOption
from ..Utils import discrete_levin_loss_on_trajectory
from ...registry import register_option
from ...loaders import load_policy, load_feature_extractor
from ..Utils import save_options_list, load_options_list

import random
import torch
import numpy as np
from tqdm import tqdm
import copy
from multiprocessing import Pool
import os
import numpy as np

# MiniGrid action ids
ACTION_LEFT   = 0
ACTION_RIGHT  = 1
ACTION_FWD    = 2
ACTION_PICKUP = 3

@register_option
class FindKey(BaseOption):
    def __init__(self, key_code=5, use_channel="auto", pickup_in_front=True, version=1):
        self.key_code = int(key_code)
        self.use_channel = str(use_channel)        # "auto" | "2" | "0"
        self.pickup_in_front = bool(pickup_in_front)
        self.version = int(version)

    # ----------------------- Option API -----------------------

    def select_action(self, observation):
        """
        Greedy controller:
          - find nearest visible key
          - if key is directly in front (or on our tile), PICKUP
          - else turn toward it horizontally, otherwise go FORWARD
        """
        img = observation #["image"]
        obj = self._object_plane(img)              # (H, W)

        H, W = obj.shape
        cy, cx = H // 2, W // 2

        ys, xs = np.where(obj == self.key_code)
        if ys.size == 0:
            return ACTION_FWD                      # harmless default

        # nearest key by manhattan distance
        d = np.abs(ys - cy) + np.abs(xs - cx)
        k = int(np.argmin(d))
        ty, tx = int(ys[k]), int(xs[k])

        dy, dx = ty - cy, tx - cx

        # pickup if right in front, or on our tile
        if self.pickup_in_front and dy == -1 and dx == 0:
            return ACTION_PICKUP
        if dy == 0 and dx == 0:
            return ACTION_PICKUP

        # simple greedy steering
        if dx > 0:   return ACTION_RIGHT
        if dx < 0:   return ACTION_LEFT
        if dy < 0:   return ACTION_FWD
        # if it's behind us, start turning (pick a side consistently)
        return ACTION_RIGHT

    def is_terminated(self, observation):
        """Terminate when no key is visible (assumes we picked it up)."""
        img = observation#["image"]
        obj = self._object_plane(img)
        return not np.any(obj == self.key_code)

    def save(self, file_path=None):
        checkpoint = {
            "option_class": self.__class__.__name__,
            "version": self.version,
            "config": {
                "key_code": self.key_code,
                "use_channel": self.use_channel,
                "pickup_in_front": self.pickup_in_front,
                "version": self.version,
            },
        }
        if file_path is not None:
            torch.save(checkpoint, f"{file_path}_options.t")
        return checkpoint

    @classmethod
    def load_from_file(cls, file_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        cfg = checkpoint.get("config", {})
        
        instance = cls(
            key_code=cfg.get("key_code", 5),
            use_channel=cfg.get("use_channel", "auto"),
            pickup_in_front=cfg.get("pickup_in_front", True),
            version=cfg.get("version", 1),
        )
        return instance
    
    # ----------------------- Helpers -----------------------
    def _object_plane(self, image: np.ndarray) -> np.ndarray:
        """
        Return (H,W) plane of object codes/types.
        - "2": use image[...,2]
        - "0": use image[...,0]
        - "auto": prefer channel 2 if it seems to hold the key code, else channel 0
        """
        if image.ndim == 2:
            return image
        assert image.ndim == 3 and image.shape[-1] >= 1, "Expected (H,W,3) symbolic image"
        if self.use_channel == "2":
            return image[..., 2]
        if self.use_channel == "0":
            return image[..., 0]
        # auto
        if image.shape[-1] >= 3 and np.any(image[..., 2] == self.key_code):
            return image[..., 2]
        return image[..., 0]