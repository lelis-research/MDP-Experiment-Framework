import gymnasium as gym
import numpy as np
from typing import Mapping
from minigrid.wrappers import ViewSizeWrapper, ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper, SymbolicObsWrapper
from gymnasium.wrappers import FrameStackObservation


from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from minigrid.core.constants import COLOR_NAMES, IDX_TO_OBJECT
from minigrid.core.world_object import Ball

# RewardWrapper that adds a constant step reward to the environment's reward.
class StepRewardWrapper(RewardWrapper):
    def __init__(self, env, step_reward=0):
        super().__init__(env)
        self.step_reward = step_reward
    
    def reward(self, reward):
        return reward + self.step_reward
    

# ActionWrapper that remaps a discrete action index to a predefined list of actions.
class CompactActionWrapper(ActionWrapper):
    def __init__(self, env, actions_lst=[0, 1, 2, 3, 4, 5, 6]):
        super().__init__(env)
        self.actions_lst = actions_lst
        self.action_space = gym.spaces.Discrete(len(actions_lst))
    
    def action(self, action):
        return self.actions_lst[action]

class FixedSeedWrapper(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self._seed = seed

    def reset(self, **kwargs):
        # force the same seed every reset
        kwargs.pop('seed', None)
        return super().reset(seed=self._seed, **kwargs)  

# ObservationWrapper that flattens the image observation by one-hot encoding object indices
# and concatenating the agent's direction.
class FlatOnehotObjectObsWrapper(ObservationWrapper):
    def __init__(self, env, object_to_onehot=None):
        super().__init__(env)
        # Create default one-hot mapping for each object index if not provided.
        if object_to_onehot is None:
            self.object_to_onehot = {}
            for idx in IDX_TO_OBJECT:  
                one_hot_array = np.zeros(len(IDX_TO_OBJECT))
                one_hot_array[idx] = 1
                self.object_to_onehot[idx] = one_hot_array
        else:
            self.object_to_onehot = object_to_onehot
        
        one_hot_dim = len(list(self.object_to_onehot.values())[0])
        # Compute the flattened observation shape: one-hot for each grid cell + one for direction.
        flatten_shape = (
            self.observation_space['image'].shape[0] *
            self.observation_space['image'].shape[1] *
            one_hot_dim + 4 # 4 is for the directions
        )
        self.observation_space = gym.spaces.Box(low=0,
                                                high=100,
                                                shape=[flatten_shape],
                                                dtype=np.float64)
    
    def observation(self, observation):
        # Extract the object indices from the image (assumed to be in the first channel).
        flatten_object_obs = observation['image'][:,:,0].flatten()
        # Convert each object index into its one-hot representation.
        one_hot = np.array([self.object_to_onehot[int(x)] for x in flatten_object_obs]).flatten()
        # Concatenate the flattened one-hot array with the agent's direction.
        one_hot_dir = np.zeros([4])
        one_hot_dir[observation['direction']] = 1
        new_obs = np.concatenate((one_hot, one_hot_dir))
        return new_obs


# A Wrapper that randomly adds distractions (in this case balls) to the environment
# The seed will fix the position of these distractions across different resets

class DistractorBall(Ball):
    def can_pickup(self):
        return False

    def can_overlap(self):
        return True
  
class FixedRandomDistractorWrapper(gym.Wrapper):
    def __init__(self, env, num_distractors=50, seed=42):
        super().__init__(env)
        self.num_distractors = num_distractors
        # pre-sample the distractor positions + colors once
        rng = np.random.RandomState(seed)
        base = self.env.unwrapped
        W, H = base.width, base.height
        color = "grey"
        self._distractors = []
        placed = 0
        attempts = 0
        while placed < num_distractors and attempts < 1000:
            x, y = rng.randint(1, W-1), rng.randint(1, H-1)
            if base.grid.get(x, y) is None and (x, y, color) not in self._distractors:
                # color = rng.choice(COLOR_NAMES)
                self._distractors.append((x, y, color))
                placed += 1
            attempts += 1

        if placed < num_distractors:
            print(f"[Warning] Only sampled {placed}/{num_distractors} distractor spots")

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        # apply the *same* distractors back onto the new grid
        base = self.env.unwrapped
        for x, y, color in self._distractors:
            # only place if still empty (should be)
            if base.grid.get(x, y) is None:
                base.put_obj(DistractorBall(color), x, y)

        return obs


class RecordRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store the original reward in the info dict.
        info['actual_reward'] = reward
        return obs, reward, terminated, truncated, info
    
class DropMissionWrapper(gym.ObservationWrapper):
    """Remove string 'mission' from obs and observation_space."""
    def __init__(self, env):
        super().__init__(env)
        if isinstance(self.observation_space, gym.spaces.Dict) and "mission" in self.observation_space.spaces:
            new_spaces = dict(self.observation_space.spaces)
            new_spaces.pop("mission")
            self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        if isinstance(obs, dict) and "mission" in obs:
            obs = dict(obs)
            obs.pop("mission")
        return obs

class AgentPosWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["agent_pos"] = self.unwrapped.agent_pos
        return obs, reward, terminated, truncated, info
    

 
 
 
   

# Dictionary mapping string keys to corresponding wrapper classes.
WRAPPING_TO_WRAPPER = {
    "ViewSize": ViewSizeWrapper,
    "ImgObs": ImgObsWrapper,
    "StepReward": StepRewardWrapper,
    "CompactAction": CompactActionWrapper,
    "FlattenOnehotObj": FlatOnehotObjectObsWrapper,
    "FixedSeed": FixedSeedWrapper,
    "FixedRandomDistractor": FixedRandomDistractorWrapper,
    "RGBImgObs": RGBImgObsWrapper,
    "RGBImgPartialObs": RGBImgPartialObsWrapper,
    "RecordReward": RecordRewardWrapper,
    "FrameStack": FrameStackObservation, # stack_size: int
    "DropMission": DropMissionWrapper,
    "AgentPos": AgentPosWrapper,
    "SymbolicObs": SymbolicObsWrapper,
}