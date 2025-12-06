import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from minigrid.wrappers import (
    ViewSizeWrapper,
    FullyObsWrapper,
)
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR


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

class AddTextWrapper(ObservationWrapper):
    """
    Wraps a MiniGrid env so that observations include a 'text' field
    describing the visible grid in human-readable form.

    Input obs: Dict with key 'image' of shape (H, W, 3), where channel 0/1/2
               are (object_idx, color_idx, state_idx).
    Output obs: Dict with keys:
        - 'image': same as input
        - 'text': str description
    """
    def __init__(self, env, max_text_len: int = 512):
        super().__init__(env)

        # Original obs space should be a Dict with "image"
        assert isinstance(env.observation_space, spaces.Dict), \
            "Expected Dict observation space for MiniGrid"
        assert "image" in env.observation_space.spaces, \
            "Expected key 'image' in observation space"

        # Extend obs space with a Text space for the description
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "text": spaces.Text(max_length=max_text_len),
        })

        # Optional: small lookup for direction names if you want to use agent_dir
        self._dir_names = ["right", "down", "left", "up"]

    # ---- core API ----
    def observation(self, observation):
        """
        observation: the obs dict produced by the wrapped env.
        We will:
          - read observation["image"]
          - generate a textual description
          - return a new dict with both 'image' and 'text'
        """
        img = observation["image"]    # shape (H, W, 3), uint8
        text = self._image_to_text(img)
        
        # Return extended observation
        obs = dict(observation)
        obs["text"] = text
        
        return obs

    # ---- helpers ----
    def _image_to_text(self, img: np.ndarray) -> str:
        """
        Turn the MiniGrid (H, W, 3) tensor into a compact text description.
        """
        H, W, _ = img.shape

        # Try to get agent position / direction if underlying env exposes it
        # (standard MiniGridEnv does)
        agent_info = ""
        try:
            ax, ay = tuple(self.unwrapped.agent_pos)
            d = int(self.unwrapped.agent_dir)
            dir_name = self._dir_names[d] if 0 <= d < len(self._dir_names) else "unknown"
            agent_info = f"You are at ({ax}, {ay}) facing {dir_name}. "
        except Exception:
            agent_info = ""

        object_descriptions = []

        for y in range(H):
            for x in range(W):
                obj_idx, color_idx, state_idx = img[y, x]

                obj_idx = int(obj_idx)
                color_idx = int(color_idx)
                state_idx = int(state_idx)

                obj_name = IDX_TO_OBJECT.get(obj_idx, "unknown")
                color_name = IDX_TO_COLOR.get(color_idx, "unknown")

                # Skip trivial cells
                if obj_name in ["unseen", "empty", "floor"]:
                    continue

                # State is optional (mostly used for doors/boxes)
                state_name = None
                for k, v in STATE_TO_IDX.items():
                    if v == state_idx:
                        state_name = k
                        break

                if obj_name == "agent":
                    # We already describe the agent via agent_pos/dir; skip here
                    continue

                # Example text: "yellow locked door at (x, y)"
                parts = []
                if color_name != "unknown":
                    parts.append(color_name)
                if state_name is not None:
                    parts.append(state_name)
                parts.append(obj_name)

                desc = " ".join(parts) + f" at ({x}, {y})"
                object_descriptions.append(desc)

        if len(object_descriptions) == 0:
            objects_str = "You see nothing of interest."
        else:
            objects_str = "Visible: " + "; ".join(object_descriptions) + "."

        text = (agent_info + objects_str).strip()

        # Optional truncation to match Text space (just in case)
        max_len = self.observation_space.spaces["text"].max_length
        if max_len is not None and len(text) > max_len:
            text = text[: max_len]

        return text
       

WRAPPING_TO_WRAPPER = {
    "ViewSize": ViewSizeWrapper,
    "FullyObs": FullyObsWrapper,
    "DropMission": DropMissionWrapper,
    "OneHotImageDir": OneHotImageDirWrapper,
    "FixedSeed": FixedSeedWrapper,
    "AddText": AddTextWrapper,
}
