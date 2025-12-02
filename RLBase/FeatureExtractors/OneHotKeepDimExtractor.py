from gymnasium.spaces import Box, Discrete, Dict, MultiDiscrete
import numpy as np
import torch

from ..registry import register_feature_extractor
from .Base import BaseFeature

ALLOWED_SPACES = (Box, Discrete, Dict, MultiDiscrete)


@register_feature_extractor
class OneHotKeepDimFeature(BaseFeature):
    """
    One-hot encode observations but keep their structural dimensions.

    Shapes (for a batch size B):

    - Discrete(n):
        obs: (B,)           -> one-hot: (B, n)

    - MultiDiscrete(nvec = [n1, n2, ..., nk]):
        obs: (B, k)         -> one-hot: (B, n1, n2, ..., nk)
        (we treat the whole vector as a point in an n1 x n2 x ... x nk grid)

    - Box(int, shape=S), with uniform integer bounds low, high:
        obs: (B, *S)        -> one-hot: (B, *S, n_cat)
        where n_cat = high - low + 1

    - Dict: one-hot each subspace separately and return a dict of tensors
            with the same keys, preserving structure.
    """

    def __init__(self, observation_space, device="cpu"):
        super().__init__(observation_space, device=device, allowed_spaces=ALLOWED_SPACES)
        # Here we store the output "feature shapes" per top-level key
        # This is mainly informational; your NetworkGen can decide how to use it.
        self._features_dict = self._compute_feature_shapes(observation_space)

    # ---------- Shapes / meta ----------

    def _compute_feature_shapes(self, space):
        """
        Return a description of the one-hot feature shapes.

        - For non-Dict: return {"x": shape_tuple}
        - For Dict: return {key: shape_tuple or nested dict}
        """
        if isinstance(space, Discrete):
            return {"x": (space.n,)}

        if isinstance(space, MultiDiscrete):
            # We one-hot into a grid of shape nvec
            return {"x": tuple(int(n) for n in space.nvec)}

        if isinstance(space, Box):
            if not np.issubdtype(space.dtype, np.integer):
                raise TypeError(
                    f"OneHotKeepDimFeature: Box space must have integer dtype, got {space.dtype}"
                )
            low = np.array(space.low).astype(np.int64)
            high = np.array(space.high).astype(np.int64)
            if not (np.all(low == low.flat[0]) and np.all(high == high.flat[0])):
                raise NotImplementedError(
                    "OneHotKeepDimFeature: non-uniform Box low/high not supported yet."
                )
            n_cat = int(high.flat[0] - low.flat[0] + 1)
            return {"x": tuple(space.shape) + (n_cat,)}

        if isinstance(space, Dict):
            # Per-key shapes
            shapes = {}
            for key, subspace in space.spaces.items():
                sub_shapes = self._compute_feature_shapes(subspace)
                # sub_shapes is {"x": ...}, but here we just store shape
                shapes[key] = sub_shapes["x"]
            return shapes

        raise TypeError(f"Unsupported space {type(space)} in OneHotKeepDimFeature.")

    @property
    def features_dict(self):
        # For Dict obs: this will be {key: shape}
        # For non-Dict: {"x": shape}
        return self._features_dict

    # ---------- Main call ----------

    def __call__(self, observation):
        space = self.observation_space

        if isinstance(space, Dict):
            # Return dict of tensors, same keys as observation_space
            encoded = {}
            for key, subspace in space.spaces.items():
                sub_obs = observation[key]
                encoded[key] = self._encode_space(sub_obs, subspace)
            return encoded

        else:
            # Single tensor under key "x"
            x = self._encode_space(observation, space)
            return {"x": x}

    # ---------- Encoders per space ----------

    def _encode_space(self, observation, space):
        if isinstance(space, Discrete):
            return self._encode_discrete(observation, space)

        if isinstance(space, MultiDiscrete):
            return self._encode_multidiscrete(observation, space)

        if isinstance(space, Box):
            return self._encode_box(observation, space)

        if isinstance(space, Dict):
            # Nested dict case (if you ever have it)
            encoded = {}
            for key, subspace in space.spaces.items():
                sub_obs = observation[key]
                encoded[key] = self._encode_space(sub_obs, subspace)
            return encoded

        raise TypeError(f"Unsupported space {type(space)} in _encode_space: {type(space)}")

    def _encode_discrete(self, observation, space: Discrete):
        """
        obs: np.ndarray of shape (B,)
        -> one-hot: (B, n)
        """
        assert isinstance(observation, np.ndarray), \
            f"Expected observation to be np.ndarray, got {type(observation)}"
        assert observation.ndim == 1, \
            f"Expected observation shape (B,), got {observation.shape}"

        B = observation.shape[0]
        n = int(space.n)

        onehot = np.zeros((B, n), dtype=np.float32)
        rows = np.arange(B)
        cols = observation.astype(np.int64)
        onehot[rows, cols] = 1.0

        return torch.from_numpy(onehot).to(self.device, dtype=torch.float32)

    def _encode_multidiscrete(self, observation, space: MultiDiscrete):
        """
        obs: np.ndarray of shape (B, K)
        nvec: array of shape (K,)
        -> one-hot grid: (B, n1, n2, ..., nK)
           where nvec = [n1, n2, ..., nK]

        Each batch element b sets a single 1 at [b, a1, a2, ..., aK].
        """
        assert isinstance(observation, np.ndarray), \
            f"Expected observation to be np.ndarray, got {type(observation)}"
        assert observation.ndim == 2, \
            f"Expected observation shape (B, K), got {observation.shape}"

        B, K = observation.shape
        nvec = space.nvec.astype(np.int64)
        assert K == len(nvec), \
            f"MultiDiscrete: observation dimension {K} != len(nvec) {len(nvec)}"

        onehot_shape = (B, *nvec.tolist())
        onehot = np.zeros(onehot_shape, dtype=np.float32)

        obs_int = observation.astype(np.int64)

        # This is easier with a loop over batch (usually small)
        for b in range(B):
            idx = tuple(obs_int[b].tolist())
            onehot[(b,) + idx] = 1.0

        return torch.from_numpy(onehot).to(self.device, dtype=torch.float32)

    def _encode_box(self, observation, space: Box):
        """
        Integer Box with uniform low/high.

        obs: np.ndarray of shape (B, *S)
        -> one-hot: (B, *S, n_cat)
        """
        assert isinstance(observation, np.ndarray), \
            f"Expected observation to be np.ndarray, got {type(observation)}"
        assert observation.shape[1:] == space.shape, \
            f"Expected observation shape (B, {space.shape}), got {observation.shape}"
        if not np.issubdtype(space.dtype, np.integer):
            raise TypeError(
                f"OneHotKeepDimFeature: Box space must have integer dtype, got {space.dtype}"
            )

        B = observation.shape[0]
        S = space.shape

        low = np.array(space.low).astype(np.int64)
        high = np.array(space.high).astype(np.int64)
        if not (np.all(low == low.flat[0]) and np.all(high == high.flat[0])):
            raise NotImplementedError(
                "OneHotKeepDimFeature: non-uniform Box low/high not supported yet."
            )
        low0 = int(low.flat[0])
        high0 = int(high.flat[0])
        n_cat = high0 - low0 + 1
        assert n_cat > 0, "Box high/low must define at least 1 category."

        flat_obs = observation.astype(np.int64)  # (B, *S)
        shifted = flat_obs - low0  # should be in [0, n_cat-1]
        if (shifted < 0).any() or (shifted >= n_cat).any():
            raise ValueError("Box observation out of bounds after shifting by low.")

        # Create output
        onehot = np.zeros((B, *S, n_cat), dtype=np.float32)

        # We'll flatten the spatial dims, one-hot per element, then reshape back
        N = int(np.prod(S))

        for b in range(B):
            vals = shifted[b].reshape(-1)  # (N,)
            # one-hot for all N elements
            tmp = np.zeros((N, n_cat), dtype=np.float32)
            rows = np.arange(N)
            tmp[rows, vals] = 1.0
            onehot[b] = tmp.reshape(*S, n_cat)

        return torch.from_numpy(onehot).to(self.device, dtype=torch.float32)
    
    
def _sample_batch(space, batch_size):
    # ---- Dict space ----
    if isinstance(space, Dict):
        return {
            key: _sample_batch(subspace, batch_size)
            for key, subspace in space.spaces.items()
        }


    # ---- Discrete ----
    if isinstance(space, Discrete):
        return np.array([space.sample() for _ in range(batch_size)])
        
    
    # ---- Box ----
    if isinstance(space, Box):
        return np.array([space.sample() for _ in range(batch_size)])

    # ---- MultiDiscrete ----
    if isinstance(space, MultiDiscrete):
        return np.array([space.sample() for _ in range(batch_size)])

def print_sample(space, sample):
    if isinstance(space, Dict):
        for key, subspace in space.spaces.items():
            print(f"Key: {key}")
            print_sample(subspace, sample[key])
    else:
        if isinstance(sample, np.ndarray):
            print(f"Sample shape: {sample.shape}")
        else:
            print(f"Sample type: {type(sample)}, Sample value: {sample}")


if __name__ == "__main__":
    spaces = {
        "discrete": Discrete(5),
        "multi_discrete": MultiDiscrete([3, 2, 4, 5]),
        "int_box": Box(low=0, high=10, shape=(3, 2), dtype=np.int32),
        "dict": Dict(
            {
                "grid": Box(low=0, high=20, shape=(6,), dtype=np.int32),
                "img": Box(low=0, high=255, shape=(3,7), dtype=np.int32),
                "state": Discrete(4),
                "multi": MultiDiscrete([2,3,4]),
            }
        ),
    }
    for space_name, space in spaces.items():
        print(f"\nSpace: {space_name}")
        batch_sample = _sample_batch(space, batch_size=2)
        print_sample(space, batch_sample)
        
        extractor = OneHotKeepDimFeature(space)
        batch = extractor(batch_sample)
        
        print("Features Dict: ", extractor.features_dict)
        
        if isinstance(space, Dict):
            for key in space.spaces.keys():
                print(f"Features for key '{key}': ", batch[key].shape)
        else:
            print("Features: ", batch["x"].shape, )
        
        

          
