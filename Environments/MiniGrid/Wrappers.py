import gymnasium as gym
import numpy as np
from minigrid.wrappers import ViewSizeWrapper, ImgObsWrapper
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from minigrid.core.constants import IDX_TO_OBJECT





class StepRewardWrapper(RewardWrapper):
    def __init__(self, env, step_reward=0):
        super().__init__(env)
        self.step_reward = step_reward
    
    def reward(self, reward):
        return reward + self.step_reward

class CompactActionWrapper(ActionWrapper):
    def __init__(self, env, actions_lst=[0, 1, 2, 3, 4, 5, 6]):
        super().__init__(env)
        self.actions_lst = actions_lst
        self.action_space = gym.spaces.Discrete(len(actions_lst))
    
    def action(self, action):
        return self.actions_lst[action]

class FlatOnehotObjectObsWrapper(ObservationWrapper):
    def __init__(self, env, object_to_onehot=None):
        super().__init__(env)
        if object_to_onehot is None:
            self.object_to_onehot = {}
            for idx in IDX_TO_OBJECT:  
                one_hot_array = np.zeros(len(IDX_TO_OBJECT))
                one_hot_array[idx] = 1
                self.object_to_onehot[idx] = one_hot_array
        else:
            self.object_to_onehot = object_to_onehot
        
        one_hot_dim = len(list(self.object_to_onehot.values())[0]) # One dimensional
        flatten_shape = self.observation_space['image'].shape[0] * \
                        self.observation_space['image'].shape[1] * \
                        one_hot_dim + 1 # Plus one for concatenating direction
        self.observation_space = gym.spaces.Box(low=0,
                                                high=100,
                                                shape=[flatten_shape], dtype=np.float64)
    def observation(self, observation):
        # (OBJECT_IDX, COLOR_IDX, STATE)
        flatten_object_obs = observation['image'][:,:,0].flatten()
        one_hot = np.array([self.object_to_onehot[int(x)] for x in flatten_object_obs]).flatten()
        new_obs = np.concatenate((one_hot,
                                  [observation['direction']]))
        return new_obs
        


WRAPPING_TO_WRAPPER = {
    "ViewSize": ViewSizeWrapper,
    "ImgObs": ImgObsWrapper,
    "StepReward": StepRewardWrapper,
    "CompactAction": CompactActionWrapper,
    "FlattenOnehotObj": FlatOnehotObjectObsWrapper,
}
