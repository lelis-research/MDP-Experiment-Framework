import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from minigrid.wrappers import (
    ViewSizeWrapper,
    FullyObsWrapper,
)
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR, DIR_TO_VEC


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

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        # ---- Image one-hot shape (same as original wrapper) ----
        obs_shape = env.observation_space["image"].shape
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX) + 1 # +1 is for the state that is none

        onehot_image_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_shape[0], obs_shape[1], num_bits),
            dtype="uint8",
        )

        # ---- Direction one-hot space ----
        # Original is usually Discrete(4) with values {0,1,2,3}
        dir_space = env.observation_space["direction"]
        assert isinstance(dir_space, spaces.Discrete)
        assert dir_space.n <= len(DIR_TO_VEC), "Expected <= num_dirs directions"

        onehot_dir_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(DIR_TO_VEC),),
            dtype="uint8",
        )

        # ---- Build new Dict observation space ----
        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "onehot_image": onehot_image_space,
                "onehot_direction": onehot_dir_space,
            }
        )

    def observation(self, obs):
        # ----- One-hot the image (same as original wrapper) -----
        img = obs["image"]
        onehot_img = np.zeros(
            self.observation_space.spaces["onehot_image"].shape,
            dtype="uint8",
        )

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                obj_type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                onehot_img[i, j, obj_type] = 1
                onehot_img[i, j, len(OBJECT_TO_IDX) + color] = 1
                onehot_img[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        # ----- One-hot the direction -----
        d = obs["direction"]  # integer in {0,1,2,3}
        if d < 0 or d >= len(DIR_TO_VEC):
            raise ValueError(f"direction out of range: {d}")

        onehot_dir = np.zeros(len(DIR_TO_VEC), dtype="uint8")
        onehot_dir[d] = 1

        # Keep other keys (e.g., "mission") unchanged
        return {
            **obs,
            "onehot_image": onehot_img,
            "onehot_direction": onehot_dir,
        }
        
class OneHotImageDirCarryWrapper(ObservationWrapper):
    """
    One-hot encode:
      - partially observable agent view (obs["image"])
      - agent direction (obs["direction"]) as a 4-dim one-hot vector.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        # ---- Image one-hot shape (same as original wrapper) ----
        obs_shape = env.observation_space["image"].shape
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX) + 1

        onehot_image_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_shape[0], obs_shape[1], num_bits),
            dtype="uint8",
        )

        # ---- Direction one-hot space ----
        # Original is usually Discrete(4) with values {0,1,2,3}
        dir_space = env.observation_space["direction"]
        assert isinstance(dir_space, spaces.Discrete)
        assert dir_space.n <= len(DIR_TO_VEC), "Expected <= num_dirs directions"

        onehot_dir_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(DIR_TO_VEC),),
            dtype="uint8",
        )
        
        # ---- Carrying one-hot space ----
        carry_space = env.observation_space["carrying"]
        assert isinstance(carry_space, spaces.Box)
        assert carry_space.shape == (2,), f"Expected carrying shape (2,), got {carry_space.shape}"
        self.num_carry_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + 1 # +1 for "none"/empty
        onehot_carry_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_carry_bits,),
            dtype="uint8",
        )

        # ---- Build new Dict observation space ----
        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "onehot_image": onehot_image_space,
                "onehot_direction": onehot_dir_space,
                "onehot_carrying": onehot_carry_space,
            }
        )

    def observation(self, obs):
        # ----- One-hot the image (same as original wrapper) -----
        img = obs["image"]
        onehot_img = np.zeros(
            self.observation_space.spaces["onehot_image"].shape,
            dtype="uint8",
        )

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                obj_type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                onehot_img[i, j, obj_type] = 1
                onehot_img[i, j, len(OBJECT_TO_IDX) + color] = 1
                onehot_img[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        # ----- One-hot the direction -----
        d = obs["direction"]  # integer in {0,1,2,3}
        if not (0 <= d < len(DIR_TO_VEC)):
            raise ValueError(f"direction out of range: {d}")
        onehot_dir = np.zeros(len(DIR_TO_VEC), dtype="uint8")
        onehot_dir[d] = 1
        
        
        # ----- One-hot the carrying (object, color) -----
        carry = obs["carrying"]  # shape (2,), int values
        obj_type = int(carry[0])
        color = int(carry[1])

        onehot_carry = np.zeros(self.num_carry_bits, dtype="uint8")

        if obj_type == -1 and color == -1:
            onehot_carry[-1] = 1  # empty
        else:
            if not (0 <= obj_type < len(OBJECT_TO_IDX)):
                raise ValueError(f"carrying obj_type out of range: {obj_type}")
            if not (0 <= color < len(COLOR_TO_IDX)):
                raise ValueError(f"carrying color out of range: {color}")
            onehot_carry[obj_type] = 1
            onehot_carry[len(OBJECT_TO_IDX) + color] = 1

        # Keep other keys (e.g., "mission") unchanged
        return {
            **obs,
            "onehot_image": onehot_img,
            "onehot_direction": onehot_dir,
            "onehot_carrying": onehot_carry,
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


class TextObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        base_space = env.observation_space

        # Rough upper bound on text length:
        #   - each cell -> 2 chars
        #   - each row  -> newline
        # even if you only print the current room, this is a safe upper bound
        if hasattr(env, "width") and hasattr(env, "height"):
            max_text_len = env.width * env.height * 2 + env.height
        else:
            max_text_len = 1024  # generic fallback

        if isinstance(base_space, spaces.Dict):
            # Copy existing dict space and add a "text" field
            new_spaces = dict(base_space.spaces)
            new_spaces["text"] = spaces.Text(max_length=max_text_len)
            self.observation_space = spaces.Dict(new_spaces)
        else:
            # Fallback: wrap original obs into {"obs": ..., "text": ...}
            self.observation_space = spaces.Dict(
                {
                    "obs": base_space,
                    "text": spaces.Text(max_length=max_text_len),
                }
            )

        self._max_text_len = max_text_len
        

    def observation(self, observation):
        env = self.unwrapped

        if hasattr(env, "text_obs"):
            txt = env.text_obs()
        else:
            txt = env.pprint_grid()

        # Make sure it's a string and not longer than declared
        if not isinstance(txt, str):
            txt = str(txt)
        if len(txt) > self._max_text_len:
            txt = txt[: self._max_text_len]
        # If env already returns a dict (like MiniGrid), just add "text"
        if isinstance(observation, dict):
            observation["text"] = txt
            return observation

        # Fallback: wrap non-dict obs
        return {"obs": observation, "text": txt}

WRAPPING_TO_WRAPPER = {
    "ViewSize": ViewSizeWrapper,
    "FullyObs": FullyObsWrapper,
    "DropMission": DropMissionWrapper,
    "OneHotImageDir": OneHotImageDirWrapper,
    "OneHotImageDirCarry": OneHotImageDirCarryWrapper,
    "FixedSeed": FixedSeedWrapper,
    "TextObs" :TextObsWrapper,
}
