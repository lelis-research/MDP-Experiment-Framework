import gymnasium as gym
import numpy as np
from functools import partial

class IdentityWrapper(gym.Wrapper):
    """No-op wrapper to keep the wrapper chain composable."""
    def __init__(self, env):
        super().__init__(env)

class RecordRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store the original reward in the info dict.
        info['actual_reward'] = reward
        return obs, reward, terminated, truncated, info

class CombineObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Calculate total size
        total_dim = (
            env.observation_space["achieved_goal"].shape[0] +
            env.observation_space["desired_goal"].shape[0] +
            env.observation_space["observation"].shape[0]
        )

        # Define the new observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )
        
    def observation(self, observation):
        # Concatenate all pieces into one flat array
        return np.concatenate([
            observation["achieved_goal"],
            observation["desired_goal"],
            observation["observation"]
        ]).astype(np.float32)
        


WRAPPING_TO_WRAPPER = {
    # no-op
    "Identity": IdentityWrapper,

    # logging
    "RecordReward": RecordRewardWrapper,

    # observation structure
    "CombineObs": CombineObsWrapper,

    # normalization (Gym-style, not VecNormalize-style)
    "NormalizeObservation": gym.wrappers.NormalizeObservation,
    "NormalizeReward": gym.wrappers.NormalizeReward,

    # clipping
    "ClipAction": gym.wrappers.ClipAction,
    "ClipObservation": partial(
        gym.wrappers.TransformObservation,
        func=lambda obs: np.clip(obs, -10.0, 10.0),
    ),
    "ClipReward": partial(
        gym.wrappers.TransformReward,
        func=lambda reward: np.clip(reward, -10.0, 10.0),
    ),
}