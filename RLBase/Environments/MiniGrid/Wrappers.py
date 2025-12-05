import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from minigrid.wrappers import (
    ViewSizeWrapper,
    FullyObsWrapper,
)
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX


class DropMissionWrapper(ObservationWrapper):
    """Remove the 'mission' field from MiniGrid observations."""
    def __init__(self, env):
        super().__init__(env)

        # Copy the original observation space
        old_space = env.observation_space

        if isinstance(old_space, spaces.Dict) and "mission" in old_space.spaces:
            # Create a new dict space without the mission key
            new_spaces = {k: v for k, v in old_space.spaces.items() if k != "mission"}
            self.observation_space = spaces.Dict(new_spaces)
        else:
            # Nothing to change
            self.observation_space = old_space
            
    def observation(self, observation):
        if isinstance(observation, dict) and "mission" in observation:
            obs = dict(observation)
            obs.pop("mission", None)
            return obs
        return observation

class OneHotImageDirWrapper(ObservationWrapper):
    """
    One-hot encode:
      - partially observable agent view (obs["image"])
      - agent direction (obs["direction"]) as a 4-dim one-hot vector.
    """

    def __init__(self, env, tile_size=8, num_dirs=4):
        super().__init__(env)

        self.tile_size = tile_size
        self.num_dirs = num_dirs

        # ---- Image one-hot shape (same as original wrapper) ----
        obs_shape = env.observation_space["image"].shape
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX) + 1

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], num_bits),
            dtype="uint8",
        )

        # ---- Direction one-hot space ----
        # Original is usually Discrete(4) with values {0,1,2,3}
        dir_space = env.observation_space["direction"]
        assert isinstance(dir_space, spaces.Discrete)
        assert dir_space.n <= num_dirs, "Expected <= num_dirs directions"

        new_dir_space = spaces.Box(
            low=0,
            high=1,
            shape=(num_dirs,),
            dtype="uint8",
        )

        # ---- Build new Dict observation space ----
        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "image": new_image_space,
                "direction": new_dir_space,
            }
        )

    def observation(self, obs):
        # ----- One-hot the image (same as original wrapper) -----
        img = obs["image"]
        out_img = np.zeros(
            self.observation_space.spaces["image"].shape,
            dtype="uint8",
        )

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                obj_type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out_img[i, j, obj_type] = 1
                out_img[i, j, len(OBJECT_TO_IDX) + color] = 1
                out_img[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        # ----- One-hot the direction -----
        d = obs["direction"]  # integer in {0,1,2,3}
        out_dir = np.zeros(self.num_dirs, dtype="uint8")
        out_dir[d] = 1

        # Keep other keys (e.g., "mission") unchanged
        return {
            **obs,
            "image": out_img,
            "direction": out_dir,
        }

class FixedSeedWrapper(gym.Wrapper):
    """
    Force MiniGrid env to always reset with the same seed.

    Any seed passed from outside (e.g. VectorEnv, env.reset(seed=...))
    is ignored; we always use self.fixed_seed.
    """

    def __init__(self, env, seed: int = 0):
        super().__init__(env)
        self.fixed_seed = int(seed)

    def reset(self, *, seed=None, options=None):
        # Ignore external seed completely, always use fixed_seed
        obs, info = self.env.reset(seed=self.fixed_seed, options=options)
        return obs, info


WRAPPING_TO_WRAPPER = {
    "ViewSize": ViewSizeWrapper,
    "FullyObs": FullyObsWrapper,
    "DropMission": DropMissionWrapper,
    "OneHotImageDir": OneHotImageDirWrapper,
    "FixedSeed": FixedSeedWrapper,
}
