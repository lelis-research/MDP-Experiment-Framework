import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from gymnasium import spaces


class CombineObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Calculate total size
        total_dim = (
            env.observation_space["achieved_goal"].shape[0] +
            env.observation_space["desired_goal"].shape[0] +
            env.observation_space["observation"].shape[0]
        )

        # Define the new observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )
        
    def observation(self, observation):
        # Concatenate all pieces into one flat array
        return np.concatenate([
            observation["achieved_goal"],
            observation["desired_goal"],
            observation["observation"]
        ]).astype(np.float32)

class AddHealthyRewardWrapper(gym.Wrapper):
    def __init__(self, env, healthy_const=0.001, control_const=0.001):
        super().__init__(env)
        self.healthy_const = healthy_const
        self.control_const = control_const

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        healthy_reward = info["reward_survive"]
        control_reward = info["reward_ctrl"]
        reward += self.healthy_const * healthy_reward + self.control_const * control_reward
        return obs, reward, terminated, truncated, info
        
class ClipObsWrapper(gym.wrappers.TransformObservation):
    """
    Clips observations elementwise to [clip_low, clip_high].
    Works best when the (possibly wrapped) env has a Box observation space.
    """
    def __init__(self, env, clip_low=-10.0, clip_high=10.0):
        # Build the transform function
        transform_fn = lambda obs: np.clip(obs, clip_low, clip_high)

        # Derive the new observation space if it's a Box; otherwise keep as-is
        obs_space = getattr(env, "observation_space", None)
        if isinstance(obs_space, spaces.Box) and obs_space.shape is not None:
            dtype = np.result_type(np.float32, obs_space.dtype)  # stay float
            new_space = spaces.Box(
                low=np.full(obs_space.shape, clip_low, dtype=dtype),
                high=np.full(obs_space.shape, clip_high, dtype=dtype),
                shape=obs_space.shape,
                dtype=dtype,
            )
        else:
            new_space = obs_space  # fallback: leave space unchanged

        super().__init__(env, transform_fn, observation_space=new_space)

class ClipRewardWrapper(gym.wrappers.TransformReward):
    """
    Clips rewards elementwise to [clip_low, clip_high].
    """
    def __init__(self, env, clip_low=-10.0, clip_high=10.0):
        transform_fn = lambda r: np.clip(r, clip_low, clip_high)
        super().__init__(env, transform_fn)
        
class RecordRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store the original reward in the info dict.
        info['actual_reward'] = reward
        return obs, reward, terminated, truncated, info
        
# Dictionary mapping string keys to corresponding wrapper classes.

WRAPPING_TO_WRAPPER = {
    "CombineObs": CombineObsWrapper,
    "NormalizeObs": gym.wrappers.NormalizeObservation,
    "NormalizeReward":gym.wrappers.NormalizeReward,
    "ClipObs": ClipObsWrapper,
    "ClipReward": ClipRewardWrapper,
    "ClipAction": gym.wrappers.ClipAction,
    "RecordReward": RecordRewardWrapper,
    "AddHealthyReward":AddHealthyRewardWrapper,
}