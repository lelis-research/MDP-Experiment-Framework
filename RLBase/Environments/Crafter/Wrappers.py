import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Crafter semantic ID → readable name  (standard Crafter material IDs)
# ---------------------------------------------------------------------------
_SEMANTIC_NAMES = {
    0:  None,           # void / out-of-bounds
    1:  "water",
    2:  "grass",
    3:  "stone",
    4:  "path",
    5:  "sand",
    6:  "tree",
    7:  "lava",
    8:  "coal",
    9:  "iron",
    10: "diamond",
    11: "table",
    12: "furnace",
    13: "plant",
    14: "ripe plant",
    15: "cow",
    16: "zombie",
    17: "skeleton",
    18: "arrow",
    19: "player",
}

_BG_ITEMS   = {"grass", "path", "sand"}   # boring background tiles
_VITALS     = ["health", "food", "drink", "energy"]
_RESOURCES  = ["wood", "stone", "coal", "iron", "diamond", "sapling"]
_TOOLS      = ["wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
               "wood_sword",   "stone_sword",   "iron_sword"]


def _dir_str(dr: int, dc: int) -> str:
    """Convert (row_offset, col_offset) → human-readable direction string."""
    parts = []
    if dr < 0:
        n = -dr
        parts.append(f"{n} step{'s' if n > 1 else ''} north")
    elif dr > 0:
        parts.append(f"{dr} step{'s' if dr > 1 else ''} south")
    if dc > 0:
        parts.append(f"{dc} step{'s' if dc > 1 else ''} east")
    elif dc < 0:
        n = -dc
        parts.append(f"{n} step{'s' if n > 1 else ''} west")
    return " and ".join(parts) if parts else "here"


class IdentityWrapper(gym.Wrapper):
    """No-op wrapper to keep the wrapper chain composable."""

    def __init__(self, env):
        super().__init__(env)


class TransposeImageWrapper(gym.ObservationWrapper):
    """Convert (H, W, C) image observations to (C, H, W) for PyTorch-style channel-first layout."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 3, (
            "TransposeImageWrapper requires a 3-D Box observation space (H, W, C)."
        )
        h, w, c = obs_space.shape
        self.observation_space = spaces.Box(
            low=obs_space.low.transpose(2, 0, 1),
            high=obs_space.high.transpose(2, 0, 1),
            dtype=obs_space.dtype,
        )

    def observation(self, obs):
        return obs.transpose(2, 0, 1)


class NormalizeImageWrapper(gym.ObservationWrapper):
    """Normalize uint8 pixel observations from [0, 255] to float32 [0.0, 1.0]."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box), (
            "NormalizeImageWrapper requires a Box observation space."
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert RGB (H, W, 3) observations to grayscale (H, W, 1) using standard luminance weights."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box) and obs_space.shape[-1] == 3, (
            "GrayscaleWrapper requires an (H, W, 3) Box observation space."
        )
        h, w, _ = obs_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )
        # ITU-R BT.601 luminance coefficients
        self._weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def observation(self, obs):
        gray = (obs.astype(np.float32) @ self._weights).clip(0, 255).astype(np.uint8)
        return gray[:, :, np.newaxis]


class AchievementRewardWrapper(gym.Wrapper):
    """Add bonus reward for each new achievement unlocked during the episode.

    Crafter's info dict contains an ``achievements`` key mapping achievement names
    to boolean (unlocked / not unlocked) values.  This wrapper grants a one-time
    ``bonus`` reward the first time each achievement is unlocked per episode and
    adds it on top of the original environment reward.
    """

    def __init__(self, env, bonus: float = 1.0):
        super().__init__(env)
        self.bonus = bonus
        self._unlocked: set = set()

    def reset(self, **kwargs):
        self._unlocked = set()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        achievements = info.get("achievements", {})
        for name, unlocked in achievements.items():
            if unlocked and name not in self._unlocked:
                self._unlocked.add(name)
                reward = reward + self.bonus
        return obs, reward, terminated, truncated, info


class RecordAchievementsWrapper(gym.Wrapper):
    """Accumulate which achievements were unlocked over the episode and expose
    them in ``info["episode_achievements"]`` upon termination/truncation."""

    def __init__(self, env):
        super().__init__(env)
        self._episode_achievements: set = set()

    def reset(self, **kwargs):
        self._episode_achievements = set()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        achievements = info.get("achievements", {})
        for name, unlocked in achievements.items():
            if unlocked:
                self._episode_achievements.add(name)
        if terminated or truncated:
            info["episode_achievements"] = set(self._episode_achievements)
        return obs, reward, terminated, truncated, info


class SymbolicObsWrapper(gym.Wrapper):
    """Convert Crafter's pixel observation into a symbolic text description.

    The returned observation is a dict::

        {
            "obs":              str  – full multi-line description (backward compat)
            "status":           str  – compact vitals, e.g. "health=9, food=6, drink=4, energy=8"
            "inventory_str":    str  – compact inventory, e.g. "wood=3" or "empty"
            "surroundings":     str  – compact surroundings, e.g. "tree (north), water (east)"
            "available_actions":str  – comma-separated action names
        }

    The description is built from ``info["inventory"]`` and ``info["semantic"]``
    which Crafter populates on every step/reset.

    Args:
        env: The Crafter environment to wrap.
        skip_background (bool): Drop grass/path/sand from the surroundings list.
        max_view_dist (int | None): Manhattan-distance cap for reported objects.
            ``None`` keeps everything in the 9×9 view.
    """

    def __init__(self, env, skip_background: bool = True, max_view_dist: int = None):
        super().__init__(env)
        self.skip_background = skip_background
        self.max_view_dist = max_view_dist

        # Cache action names once from the underlying env
        try:
            actions_enum = env.unwrapped.actions
            self._action_names = [a.name for a in actions_enum]
        except AttributeError:
            self._action_names = []

        self.observation_space = spaces.Dict({
            "obs":               spaces.Text(min_length=0, max_length=4096),
            "status":            spaces.Text(min_length=0, max_length=256),
            "inventory_str":     spaces.Text(min_length=0, max_length=256),
            "surroundings":      spaces.Text(min_length=0, max_length=1024),
            "available_actions": spaces.Text(min_length=0, max_length=512),
        })

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._describe(info), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._describe(info), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Description helpers
    # ------------------------------------------------------------------
    def _describe(self, info: dict) -> dict:
        status       = self._compact_vitals(info)
        inventory_str = self._compact_inventory(info)
        surroundings = self._compact_surroundings(info)
        available_actions = ", ".join(self._action_names)

        parts = [
            self._describe_vitals(info),
            self._describe_inventory(info),
            self._describe_surroundings(info),
        ]
        full_obs = "\n".join(p for p in parts if p)

        return {
            "obs":               full_obs,
            "status":            status,
            "inventory_str":     inventory_str,
            "surroundings":      surroundings,
            "available_actions": available_actions,
        }

    # -- compact single-line helpers (used by LLMAgent) -----------------
    def _compact_vitals(self, info: dict) -> str:
        inv = info.get("inventory", {})
        return ", ".join(f"{v}={inv.get(v, '?')}" for v in _VITALS)

    def _compact_inventory(self, info: dict) -> str:
        inv = info.get("inventory", {})
        items = [f"{name}={inv[name]}" for name in _RESOURCES + _TOOLS if inv.get(name, 0) > 0]
        return ", ".join(items) if items else "empty"

    def _compact_surroundings(self, info: dict) -> str:
        sem_raw = info.get("semantic")
        if sem_raw is None:
            return ""
        sem = np.asarray(sem_raw)
        h, w = sem.shape
        pr, pc = h // 2, w // 2
        closest: dict[str, tuple[int, int, int]] = {}
        for r in range(h):
            for c in range(w):
                name = _SEMANTIC_NAMES.get(int(sem[r, c]))
                if name is None or name == "player":
                    continue
                if self.skip_background and name in _BG_ITEMS:
                    continue
                dr, dc = r - pr, c - pc
                dist = abs(dr) + abs(dc)
                if self.max_view_dist is not None and dist > self.max_view_dist:
                    continue
                if name not in closest or dist < closest[name][0]:
                    closest[name] = (dist, dr, dc)
        if not closest:
            return ""
        return ", ".join(
            f"{name} ({_dir_str(dr, dc)})"
            for name, (_, dr, dc) in sorted(closest.items(), key=lambda x: x[1][0])
        )

    # -- verbose multi-line helpers (used for full "obs" field) ---------
    def _describe_vitals(self, info: dict) -> str:
        inv = info.get("inventory", {})
        lines = ["[Vitals]"]
        for v in _VITALS:
            val = inv.get(v, "?")
            lines.append(f"  {v}: {val}/9")
        return "\n".join(lines)

    def _describe_inventory(self, info: dict) -> str:
        inv = info.get("inventory", {})
        items = [
            f"  {name}: {inv[name]}"
            for name in _RESOURCES + _TOOLS
            if inv.get(name, 0) > 0
        ]
        if not items:
            return "[Inventory] empty"
        return "[Inventory]\n" + "\n".join(items)

    def _describe_surroundings(self, info: dict) -> str:
        semantic = info.get("semantic")
        if semantic is None:
            return ""

        sem = np.asarray(semantic)
        h, w = sem.shape
        pr, pc = h // 2, w // 2   # player is at the center of the view

        # Collect closest instance of each distinct object type
        closest: dict[str, tuple[int, int, int]] = {}  # name → (dist, dr, dc)
        for r in range(h):
            for c in range(w):
                name = _SEMANTIC_NAMES.get(int(sem[r, c]))
                if name is None or name == "player":
                    continue
                if self.skip_background and name in _BG_ITEMS:
                    continue
                dr, dc = r - pr, c - pc
                dist = abs(dr) + abs(dc)
                if self.max_view_dist is not None and dist > self.max_view_dist:
                    continue
                if name not in closest or dist < closest[name][0]:
                    closest[name] = (dist, dr, dc)

        if not closest:
            return ""

        lines = ["[Surroundings]"]
        for name, (_, dr, dc) in sorted(closest.items(), key=lambda x: x[1][0]):
            lines.append(f"  {name}: {_dir_str(dr, dc)}")
        return "\n".join(lines)


WRAPPING_TO_WRAPPER = {
    "Identity": IdentityWrapper,
    "TransposeImage": TransposeImageWrapper,
    "NormalizeImage": NormalizeImageWrapper,
    "Grayscale": GrayscaleWrapper,
    "AchievementReward": AchievementRewardWrapper,
    "RecordAchievements": RecordAchievementsWrapper,
    "SymbolicObs": SymbolicObsWrapper,
}
