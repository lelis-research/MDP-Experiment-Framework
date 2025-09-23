
# ADD WRAPPERS HERE





WRAPPING_TO_WRAPPER = {

}


import copy
import importlib

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ensure MiniHack envs are registered with Gymnasium when this module is imported
try:  # pragma: no cover
    import minihack  # noqa: F401
except Exception:  # pragma: no cover
    minihack = None

import traceback
from nle.env.tasks import NetHackStaircase


class MiniHackWrap(gym.Env):
    """
    Minimal MiniHack wrapper that:
    - Uses built-in agent-centric crop observations (e.g., 'chars_crop' or 'glyphs_crop').
    - Appends goal distance (dx, dy) to the observation (relative to crop center).
    - Shapes reward: step_reward for each step, goal_reward when terminated.

    Assumptions:
    - '*_crop' keys are present in observations (e.g., 'chars_crop', 'glyphs_crop').
    - Goal location is denoted by the '>' character in the 'chars' crop.

    Disclaimer (MiniHack-Corridor-specific tuning):
    - The default compact one-hot vocabulary targets symbols commonly visible in
      MiniHack-Corridor-R2-v0: [' ', '-', '|', '#', '.', '<', '>', '@'] plus an
      optional "other" class. Other MiniHack tasks can expose additional tiles,
      colors, specials, messages, or different goal markers.
    - If you switch environments, consider:
        * Extending `char_vocab` to include the symbols present in that task;
        * Disabling compact mapping by setting `include_other_class=False` and/or
          using a broader vocabulary;
        * Turning off character one-hot (`one_hot=False`) or using `glyphs_crop`
          by setting `use_chars=False`;
        * Adjusting goal detection if the goal character is not '>' via
          `goal_chars=(...)`.
    - This wrapper assumes the env provides full `chars` alongside `*_crop` so
      global goal deltas can be computed; if not, dx/dy fall back to zeros.
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int | None = None,
        view_size: int = 9,
        step_reward: float = -1.0,
        goal_reward: float = 1.0,
        goal_chars: tuple[str, ...] = (">",),
        use_chars: bool = True,
        one_hot: bool = True,
        n_char_classes: int = 256,
        include_dxdy: bool = False,
        char_vocab: tuple[str, ...] | None = None,
        include_other_class: bool = True,
    ):
        super().__init__()
        self.env = env
        self._env_ctor = self._infer_env_ctor(env)
        self.seed_ = seed
        self.view_size = int(view_size)
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        self.goal_chars = tuple(goal_chars)
        self.use_chars = bool(use_chars)
        self.one_hot = bool(one_hot)
        self.n_char_classes = int(n_char_classes)
        self.include_dxdy = bool(include_dxdy)
        self.include_other_class = bool(include_other_class)

        # Precompute eye for one-hot to avoid reallocs
        self._eye_chars = None
        self._lut_chars = None  # maps ASCII code -> compact index
        self._char_vocab = None
        self._other_char = "?"
        if self.one_hot and self.use_chars:
            # Default compact vocabulary tailored for MiniHack-Corridor
            # Visible chars: space, '-', '|', '#', '.', '<', '>', '@'
            if char_vocab is None:
                char_vocab = (" ", "-", "|", "#", ".", "<", ">", "@", "+", "-")
            self._char_vocab = tuple(char_vocab)
            # Build ASCII code list and LUT to compact indices
            vocab_codes = np.array([ord(c) for c in char_vocab], dtype=np.int64)
            vocab_size = len(vocab_codes) + (1 if self.include_other_class else 0)
            self._eye_chars = np.eye(vocab_size, dtype=np.float32)
            # LUT spans 256 codes; default to 'other' index if enabled else 0
            other_idx = vocab_size - 1 if self.include_other_class else 0
            lut = np.full(256, other_idx, dtype=np.int64)
            for i, code in enumerate(vocab_codes):
                lut[code] = i
            self._lut_chars = lut

        # Delegate action space to underlying env
        self.action_space = env.action_space

        # Infer observation space from env.observation_space without resetting.
        # This avoids triggering MiniHack/NLE reset during VectorEnv construction,
        # which can cause low-level crashes (SIGFPE) in some environments.
        self.last_obs = None
        self._last_info: dict[str, object] | None = None
        self._last_done_flag = False

        # Determine crop subspace shape
        crop_subspace = None
        try:
            obs_space = getattr(self.env, "observation_space", None)
            if isinstance(obs_space, spaces.Dict):
                if self.use_chars and "chars_crop" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["chars_crop"]
                elif (not self.use_chars) and "glyphs_crop" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["glyphs_crop"]
                elif self.use_chars and "chars" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["chars"]
                elif "glyphs" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["glyphs"]
        except Exception:
            crop_subspace = None

        if crop_subspace is not None and hasattr(crop_subspace, "shape"):
            crop_shape = tuple(int(x) for x in crop_subspace.shape)
        else:
            # Fallback: assume square crop of view_size if unknown
            crop_shape = (int(self.view_size), int(self.view_size))
        self._crop_shape = crop_shape
        self._crop_size = int(np.prod(crop_shape)) if len(crop_shape) > 0 else 0

        # Compute flattened feature size
        if self.one_hot and self.use_chars:
            vocab_size = self._eye_chars.shape[0] if self._eye_chars is not None else int(self.n_char_classes)
            flat_size = int(np.prod(crop_shape)) * int(vocab_size)
        else:
            flat_size = int(np.prod(crop_shape))
        if self.include_dxdy:
            flat_size += 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
        )

        self.spec = getattr(self.env, "spec", None)

        self.steps = 0

        # Initialize environment state immediately
        # so that last_obs and internal counters are set.
        # This mirrors other env wrappers in the repo.
        self.reset(seed=self.seed_)

    # ----- Pickling helpers -----
    @staticmethod
    def _safe_copy_kwargs(kwargs):
        if not kwargs:
            return {}
        try:
            return copy.deepcopy(kwargs)
        except Exception:
            return dict(kwargs)

    def _infer_env_ctor(self, env: gym.Env | None) -> dict[str, object] | None:
        if env is None:
            return None
        spec = getattr(env, "spec", None)
        ctor: dict[str, object] = {
            "id": None,
            "kwargs": {},
            "entry_point": None,
        }
        if spec is not None:
            ctor["id"] = getattr(spec, "id", None)
            entry_point = getattr(spec, "entry_point", None)
            if isinstance(entry_point, str):
                ctor["entry_point"] = entry_point
            elif entry_point is not None:
                qualname = getattr(entry_point, "__qualname__", getattr(entry_point, "__name__", None))
                if qualname is not None:
                    ctor["entry_point"] = f"{entry_point.__module__}.{qualname}"
                else:
                    ctor["entry_point"] = str(entry_point)
            ctor["kwargs"] = self._safe_copy_kwargs(getattr(spec, "kwargs", None))
        else:
            ctor["entry_point"] = f"{env.__module__}.{env.__class__.__qualname__}"
        return ctor

    @staticmethod
    def _call_env_method(env: gym.Env | None, names: tuple[str, ...], *args, **kwargs):
        if env is None:
            return False, None
        for name in names:
            method = getattr(env, name, None)
            if callable(method):
                try:
                    return True, method(*args, **kwargs)
                except TypeError:
                    continue
        return False, None

    def _capture_env_state(self, env: gym.Env | None):
        candidates = (
            ("clone_state", ("restore_state", "set_state")),
            ("clone_full_state", ("restore_full_state", "restore_state", "set_state")),
            ("get_state", ("set_state", "restore_state")),
            ("state_dict", ("load_state_dict", "set_state_dict")),
        )
        for getter, setters in candidates:
            call_variants = (
                ((), {}),
                ((), {"include_information": True}),
            )
            for args, kwargs in call_variants:
                success, snapshot = self._call_env_method(env, (getter,), *args, **kwargs)
                if success and snapshot is not None:
                    return {
                        "getter": getter,
                        "setters": setters,
                        "snapshot": snapshot,
                        "call_kwargs": kwargs,
                    }
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is env:
            unwrapped = None
        if unwrapped is not None:
            data = self._capture_env_state(unwrapped)
            if data is not None:
                data.setdefault("unwrap_count", 0)
                data["unwrap_count"] += 1
                return data
        return None

    def _restore_env_state(self, env: gym.Env, payload: dict[str, object]) -> bool:
        if not payload:
            return False
        unwrap_count = payload.get("unwrap_count", 0)
        target_env = env
        for _ in range(int(unwrap_count)):
            target_env = getattr(target_env, "unwrapped", target_env)
        snapshot = payload.get("snapshot")
        setters = payload.get("setters", ())
        for setter in setters:
            success, _ = self._call_env_method(target_env, (setter,), snapshot)
            if success:
                return True
        extra_setters = (
            "restore_state",
            "set_state",
            "restore_full_state",
            "load_state_dict",
            "from_state",
        )
        for setter in extra_setters:
            success, _ = self._call_env_method(target_env, (setter,), snapshot)
            if success:
                return True
        return False

    def __getstate__(self):
        state = self.__dict__.copy()
        env = state.pop("env", None)
        pickled_env = None
        if env is not None:
            pickled_env = {
                "ctor": copy.deepcopy(self._env_ctor) if self._env_ctor else None,
                "state": self._capture_env_state(env),
                "done": bool(self._last_done_flag),
                "info": copy.deepcopy(self._last_info) if self._last_info is not None else None,
            }
        state["_pickled_env"] = pickled_env
        return state

    def __setstate__(self, state):
        pickled_env = state.pop("_pickled_env", None)
        self.__dict__.update(state)

        env = None
        ctor = None
        if pickled_env:
            ctor = pickled_env.get("ctor")
        if ctor and ctor.get("id"):
            env_kwargs = ctor.get("kwargs") or {}
            env = gym.make(ctor["id"], **env_kwargs)
        elif ctor and ctor.get("entry_point"):
            entry_point = ctor["entry_point"]
            if isinstance(entry_point, str) and entry_point:
                if ":" in entry_point:
                    module_name, qualname = entry_point.split(":", 1)
                else:
                    module_name, _, qualname = entry_point.rpartition(".")
                try:
                    module = importlib.import_module(module_name)
                    env_cls = getattr(module, qualname)
                    env = env_cls(**(ctor.get("kwargs") or {}))
                except (ModuleNotFoundError, AttributeError):
                    env = None
        if env is None:
            raise RuntimeError("Failed to reconstruct MiniHack environment during unpickling.")

        if self.seed_ is not None:
            try:
                env.unwrapped.seed(self.seed_)
            except Exception:
                pass

        restored = False
        if pickled_env and pickled_env.get("state"):
            restored = self._restore_env_state(env, pickled_env["state"])
        if not restored:
            try:
                reset_result = env.reset(seed=self.seed_)
            except TypeError:
                reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, _ = reset_result
            else:
                obs = reset_result
            self.last_obs = obs
            self.steps = 0
            self._last_info = {}
            self._last_done_flag = False
        else:
            if "done" in (pickled_env or {}):
                self._last_done_flag = bool(pickled_env.get("done", False))
            if "info" in (pickled_env or {}):
                self._last_info = pickled_env.get("info")

        self.env = env
        self._env_ctor = self._infer_env_ctor(env)
        self.action_space = env.action_space
        self.spec = getattr(env, "spec", None)

    # ----- Observation helpers -----
    def _grid(self, obs) -> np.ndarray:
        # Prefer built-in cropped keys
        if self.use_chars and "chars_crop" in obs:
            return obs["chars_crop"]
        if (not self.use_chars) and "glyphs_crop" in obs:
            return obs["glyphs_crop"]
        # Fallbacks if ever needed
        if self.use_chars and "chars" in obs:
            return obs["chars"]
        return obs.get("glyphs", obs.get("chars"))

    def _agent_xy(self, obs) -> tuple[int, int]:
        bl = obs.get("blstats")
        if bl is None:
            return 0, 0
        return int(bl[0]), int(bl[1])

    def _global_goal_delta(self, obs) -> tuple[float, float]:
        """ Compute dx,dy using full observation (not the crop).
        dx = agent_x - goal_x, dy = agent_y - goal_y
        """
        chars = obs.get("chars")
        print(chars.shape)
        print("chars")
        for row in np.char.mod('%c', chars):
            print("".join(row))
        if chars is None:
            return 0.0, 0.0
        ax, ay = self._agent_xy(obs)
        targets = [ord(c) for c in self.goal_chars]
        ys, xs = np.where(np.isin(chars, targets))
        if xs.size == 0:
            return 0.0, 0.0
        dists = np.abs(xs - ax) + np.abs(ys - ay)
        i = int(np.argmin(dists))
        gx, gy = int(xs[i]), int(ys[i])
        return float(ax - gx), float(ay - gy)

    def _encode_crop(self, crop: np.ndarray) -> np.ndarray:
        # One-hot encode characters if requested
        if self.one_hot and self.use_chars:
            flat_codes = crop.astype(np.int64).reshape(-1)
            if self._lut_chars is not None:
                mapped = self._lut_chars[np.clip(flat_codes, 0, 255)]
                oh = self._eye_chars[mapped]
            else:
                # Fallback to 256-way one-hot
                flat_codes = np.clip(flat_codes, 0, self.n_char_classes - 1)
                oh = self._eye_chars[flat_codes]
            return oh.reshape(-1)
        # Fallback to raw values
        return crop.astype(np.float32).flatten()

    def _build_observation(self, obs) -> np.ndarray:
        grid = self._grid(obs)
        crop = grid  # always crop view
        # For dx,dy compute using full observation (global goal location)
        dx = dy = 0.0
        if self.include_dxdy:
            dx, dy = self._global_goal_delta(obs)
        enc = self._encode_crop(crop)
        if self.include_dxdy:
            enc = np.concatenate([enc, np.array([dx, dy], dtype=np.float32)])
        return enc.astype(np.float32)

    def print_one_hot_observation(self, encoded_obs: np.ndarray) -> None:
        """Decode a flattened one-hot observation back to characters and print."""
        if not (self.one_hot and self.use_chars):
            raise RuntimeError("print_one_hot_observation requires character one-hot observations.")
        if self._eye_chars is None or self._char_vocab is None:
            raise RuntimeError("Character vocabulary has not been initialized; cannot decode observation.")

        obs = np.asarray(encoded_obs, dtype=np.float32)
        if self.include_dxdy:
            if obs.size < 2:
                raise ValueError("Encoded observation is too short to contain dx/dy components.")
            obs = obs[:-2]

        vocab_size = self._eye_chars.shape[0]
        if obs.size % vocab_size != 0:
            raise ValueError(
                "Encoded observation length is not divisible by the vocabulary size; cannot reshape to crop grid."
            )

        if self._crop_size == 0:
            raise ValueError("Crop size is zero; cannot decode observation")

        flattened = obs.reshape(-1, vocab_size)
        indices = np.argmax(flattened, axis=1)

        chars = list(self._char_vocab)
        if self.include_other_class:
            chars.append(self._other_char)
        if len(chars) != vocab_size:
            raise ValueError("Character mapping length does not match vocabulary size.")

        char_grid = np.array(chars, dtype="<U1")[indices]
        try:
            char_grid = char_grid.reshape(self._crop_shape)
        except ValueError as exc:
            raise ValueError("Cannot reshape decoded characters to crop shape.") from exc

        for row in char_grid:
            print("".join(row.tolist()))

    # ----- Gym API -----
    def reset(self, *, seed: int | None = None, options=None):
        # traceback.print_stack()
        self.steps = 0
        if seed is not None:
            self.seed_ = seed
        else:
            seed = self.seed_
        self.env.unwrapped.seed(seed)
        obs, info = self.env.reset(options=options)
        self.last_obs = obs
        self._last_info = info
        self._last_done_flag = False
        return self._build_observation(obs), info

    def get_observation(self):
        return self._build_observation(self.last_obs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.last_obs = obs
        self._last_info = info
        
        reward = self.goal_reward if terminated and info["end_status"] == NetHackStaircase.StepStatus.TASK_SUCCESSFUL else self.step_reward

        # CAVIAT: No option implementation
        info.update({"action_size": 1, "steps": self.steps})
        done = bool(terminated) or bool(truncated)
        self._last_done_flag = done
        return self._build_observation(obs), reward, bool(terminated), bool(truncated), info

    def get_observation_space(self):
        return self.observation_space.shape[0]

    def get_action_space(self):
        return self.action_space.n

    def is_over(self) -> bool:
        # Prefer underlying env-provided signal if available
        underlying = getattr(self.env, "is_over", None)
        if callable(underlying):
            try:
                result = underlying()
                if result is not None:
                    return bool(result)
            except Exception:
                pass

        if self._last_done_flag:
            return True

        # Fallback: check whether agent currently stands on the goal tile
        if self.last_obs is not None:
            try:
                obs = self.last_obs
                chars = obs.get("chars") if isinstance(obs, dict) else None
                if chars is not None:
                    ax, ay = self._agent_xy(obs)
                    targets = [ord(c) for c in self.goal_chars]
                    if 0 <= ay < chars.shape[0] and 0 <= ax < chars.shape[1]:
                        if chars[int(ay), int(ax)] in targets:
                            return True
            except Exception:
                return False

        return False

    def render(self):
        return self.env.render()

    def seed(self, seed: int):
        self.seed_ = seed
        self.env.unwrapped.seed(seed)
        self.env.reset()

    def close(self):
        return self.env.close()


def make_env_minihack(*, env_id: str = "MiniHack-Corridor-R2-v0", seed: int = 0, view_size: int = 9):
    """Vector-friendly builder returning a thunk that constructs the wrapped env."""
    def thunk():
        if minihack is None:
            raise ImportError("minihack is not installed or failed to import; please `pip install minihack`.")
        base = gym.make(
            env_id,
            observation_keys=(
                "chars",
                "glyphs",
                "blstats",
                "chars_crop",
                "glyphs_crop",
            ),
        )
        env = MiniHackWrap(base, seed=seed, view_size=view_size, step_reward=-1.0, goal_reward=1000.0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def get_minihack_env(*, env_id: str = "MiniHack-Corridor-R2-v0", seed: int = 0, view_size: int = 9):
    if minihack is None:
        raise ImportError("minihack is not installed or failed to import; please `pip install minihack`.")
    base = gym.make(
        env_id,
        observation_keys=(
            "chars",
            "glyphs",
            "blstats",
            "chars_crop",
            "glyphs_crop",
        ),
    )
    env = MiniHackWrap(base, seed=seed, view_size=view_size, step_reward=-1.0, goal_reward=1000.0)
    return env
