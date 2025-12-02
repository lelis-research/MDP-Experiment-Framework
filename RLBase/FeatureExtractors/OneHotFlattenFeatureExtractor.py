from gymnasium.spaces import Box, Discrete, Dict, MultiDiscrete
import numpy as np
import torch

from ..registry import register_feature_extractor
from .Base import BaseFeature

ALLOWED_SPACES = (Box, Discrete, Dict, MultiDiscrete)


@register_feature_extractor
class OneHotFlattenFeature(BaseFeature):
    """
    One-hot encode observations for:
      - Discrete
      - MultiDiscrete
      - Box with integer dtype
      - Dict of the above (concatenated)

    Output: {"x": torch.FloatTensor} with shape (batch_size, one_hot_dim)
    """

    def __init__(self, observation_space, device="cpu"):
        super().__init__(observation_space, device=device, allowed_spaces=ALLOWED_SPACES)
        self._features_dict = {"x": self._onehot_dim(observation_space)}

    # ---------- Dimension computation ----------

    def _onehot_dim(self, space):
        """Compute flattened one-hot dimension for a space."""
        if isinstance(space, Discrete):
            # One-hot over [0, n-1]
            return int(space.n)

        elif isinstance(space, MultiDiscrete):
            # Concatenate one-hot for each component
            # component i has nvec[i] categories
            return int(np.sum(space.nvec))

        elif isinstance(space, Box):
            # Require integer Box
            if not np.issubdtype(space.dtype, np.integer):
                raise TypeError(
                    f"OneHotFeature: Box space must have integer dtype, got {space.dtype}"
                )
            high = np.array(space.high, dtype=np.int64).ravel()
            low = np.array(space.low, dtype=np.int64).ravel()
            n_cat = (high - low + 1)
            if np.any(n_cat <= 0):
                raise ValueError("OneHotFeature: Box high/low must define at least 1 category.")
            # Each element position has its own number of categories; we concatenate all
            return int(n_cat.sum())

        elif isinstance(space, Dict):
            # Sum one-hot dim of all subspaces
            return sum(self._onehot_dim(subspace) for subspace in space.spaces.values())

        else:
            raise TypeError(f"Unsupported space {type(space)} in OneHotFeature.")

    # ---------- Main call ----------

    def __call__(self, observation):
        if isinstance(self.observation_space, Discrete):
            onehot = self._encode_discrete(observation, self.observation_space)

        elif isinstance(self.observation_space, MultiDiscrete):
            onehot = self._encode_multidiscrete(observation, self.observation_space)

        elif isinstance(self.observation_space, Box):
            onehot = self._encode_box(observation, self.observation_space)

        elif isinstance(self.observation_space, Dict):
            onehot = self._encode_dict(observation, self.observation_space)

        else:
            raise ValueError(
                f"Observation Space {self.observation_space} "
                f"is not supported by {self.__class__.__name__}"
            )

        return {"x": torch.from_numpy(onehot).to(self.device, dtype=torch.float32)}

    @property
    def features_dict(self):
        return self._features_dict

    # ---------- Encoders per space ----------

    def _encode_discrete(self, observation, space: Discrete):
        """
        observation: np.ndarray of shape (B,)
        space: Discrete(n)
        Returns: np.ndarray of shape (B, n)
        """
        assert isinstance(observation, np.ndarray), \
            f"Expected observation to be np.ndarray, got {type(observation)}"
        assert observation.ndim == 1, \
            f"Expected observation to have shape (batch_size,), got {observation.shape}"

        B = observation.shape[0]
        n = space.n

        onehot = np.zeros((B, n), dtype=np.float32)
        rows = np.arange(B)
        cols = observation.astype(np.int64)
        onehot[rows, cols] = 1.0
        return onehot

    def _encode_multidiscrete(self, observation, space: MultiDiscrete):
        """
        observation: np.ndarray of shape (B, K)
        space: MultiDiscrete(nvec) where nvec shape is (K,)
        Returns: np.ndarray of shape (B, sum(nvec))
        """
        assert isinstance(observation, np.ndarray), \
            f"Expected observation to be np.ndarray, got {type(observation)}"
        assert observation.shape[1:] == space.shape, \
            f"Expected observation to have shape (B, {space.shape}), got {observation.shape}"

        B, K = observation.shape
        nvec = space.nvec.astype(np.int64)

        total_dim = int(nvec.sum())
        onehot = np.zeros((B, total_dim), dtype=np.float32)

        # offsets for each component
        offsets = np.concatenate([[0], np.cumsum(nvec[:-1])])  # shape (K,)

        obs_int = observation.astype(np.int64)
        for k in range(K):
            # component k has nvec[k] categories and offset offsets[k]
            rows = np.arange(B)
            cols = offsets[k] + obs_int[:, k]
            onehot[rows, cols] = 1.0

        return onehot

    def _encode_box(self, observation, space: Box):
        """
        Integer Box:
          - observation: np.ndarray of shape (B, *shape)
          - space.low / space.high define integer ranges per element

        Returns: np.ndarray of shape (B, sum_i n_cat_i)
        where n_cat_i = high_i - low_i + 1 for each element i.
        """
        assert isinstance(observation, np.ndarray), \
            f"Expected observation to be np.ndarray, got {type(observation)}"
        assert observation.shape[1:] == space.shape, \
            f"Expected observation to have shape (B, {space.shape}), got {observation.shape}"
        if not np.issubdtype(space.dtype, np.integer):
            raise TypeError(
                f"OneHotFeature: Box space must have integer dtype, got {space.dtype}"
            )

        B = observation.shape[0]
        flat_obs = observation.reshape(B, -1).astype(np.int64)  # (B, N)
        N = flat_obs.shape[1]

        low = np.array(space.low, dtype=np.int64).ravel()
        high = np.array(space.high, dtype=np.int64).ravel()
        assert low.shape[0] == N and high.shape[0] == N, \
            "Box low/high shapes must match flattened observation."

        n_cat = (high - low + 1)
        if np.any(n_cat <= 0):
            raise ValueError("OneHotFeature: Box high/low must define at least 1 category.")

        total_dim = int(n_cat.sum())
        onehot = np.zeros((B, total_dim), dtype=np.float32)

        # per-element offsets in the final vector
        offsets = np.concatenate([[0], np.cumsum(n_cat[:-1])])  # (N,)

        # shift obs into [0, n_cat_i - 1]
        shifted = flat_obs - low  # (B, N)

        # compute indices in the flattened one-hot dimension
        idx = offsets[None, :] + shifted  # (B, N)

        rows = np.repeat(np.arange(B), N)
        cols = idx.ravel()
        onehot[rows, cols] = 1.0

        return onehot

    def _encode_dict(self, observation: dict, space: Dict):
        """
        Dict of subspaces, each of which is Discrete / MultiDiscrete / Box(int) / nested Dict.
        Concatenate per-subspace one-hot encodings along feature dimension.
        """
        assert isinstance(observation, dict), \
            f"Expected observation to be dict, got {type(observation)}"

        encoded_items = []

        for key, subspace in space.spaces.items():
            value = observation[key]

            if isinstance(subspace, Discrete):
                onehot = self._encode_discrete(value, subspace)
            elif isinstance(subspace, MultiDiscrete):
                onehot = self._encode_multidiscrete(value, subspace)
            elif isinstance(subspace, Box):
                onehot = self._encode_box(value, subspace)
            elif isinstance(subspace, Dict):
                onehot = self._encode_dict(value, subspace)
            else:
                raise TypeError(
                    f"Unsupported subspace {type(subspace).__name__} for key {key}"
                )

            encoded_items.append(onehot)

        concated = np.concatenate(encoded_items, axis=1)
        return concated
    
    

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
        "int_box": Box(low=0, high=3, shape=(3, 2), dtype=np.int32),
        "dict": Dict(
            {
                "grid": Box(low=0, high=2, shape=(6,), dtype=np.int32),
                "img": Box(low=0, high=2, shape=(3,7), dtype=np.int32),
                "state": Discrete(4),
                "multi": MultiDiscrete([2,3,4]),
            }
        ),
    }
    for space_name, space in spaces.items():
        print(f"\nSpace: {space_name}")
        batch_sample = _sample_batch(space, batch_size=2)
        print_sample(space, batch_sample)
        
        extractor = OneHotFlattenFeature(space)
        batch = extractor(batch_sample)
        print("Features: ", batch["x"].shape, extractor.features_dict)
        
        

          
