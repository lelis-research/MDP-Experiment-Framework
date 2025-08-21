import gymnasium as gym
import numpy as np
from minigrid.wrappers import ViewSizeWrapper, ImgObsWrapper
from gymnasium.core import ActionWrapper, ObservationWrapper, RewardWrapper
from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX, OBJECT_TO_IDX, IDX_TO_OBJECT
from minigrid.core.world_object import Ball, Box, Wall
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
class FixedRandomDistractorWrapper(ObservationWrapper):
    def __init__(self, env, num_distractors=50, seed=42):
        super().__init__(env)
        self.num_distractors = num_distractors

        # Sample fixed distractor (x,y,color) in WORLD coordinates
        rng = np.random.RandomState(seed)
        base = self.env.unwrapped
        W, H = base.width, base.height

        self._distractors = []
        placed, attempts = 0, 0
        while placed < num_distractors and attempts < 5000:
            x, y = rng.randint(0, W), rng.randint(0, H)
            # Only allow on empty cells in the *initial* layout (for consistency)
            if base.grid.get(x, y) is None:
                color = rng.choice(COLOR_NAMES)
                self._distractors.append((int(x), int(y), str(color)))
                placed += 1
            attempts += 1

        if placed < num_distractors:
            print(f"[VisualDistractors] Only sampled {placed}/{num_distractors} distractors")

        # Cache encoding for balls per color
        self._ball_enc_by_color = {
            c: np.array([OBJECT_TO_IDX["ball"], COLOR_TO_IDX[c], 0], dtype=np.uint8)
            for c in COLOR_NAMES
        }

    def observation(self, obs):
        """
        obs: Dict with 'image' key (MiniGrid symbolic encoding).
        We overlay 'ball' encodings at distractor spots that fall within the current obs.
        """
        if not isinstance(obs, dict) or "image" not in obs:
            # If you're using a custom obs wrapper, adapt here
            return obs

        img = obs["image"].copy()  # (H, W, 3) in symbolic form: [type, color, state]

        # Distinguish between fully observed vs agent-centric view
        base = self.env.unwrapped
        H, W, C = img.shape
        is_full = (W == base.width) and (H == base.height)

        if is_full:
            # Fully observed: world coords map 1:1 to obs coords
            for x, y, color in self._distractors:
                if 0 <= x < W and 0 <= y < H:
                    img[y, x] = self._ball_enc_by_color[color]
        else:
            # Partially observed (agent-centric). Map world -> view coords.
            # MiniGrid agent_dir: 0:right, 1:down, 2:left, 3:up
            ax, ay = base.agent_pos
            ad = base.agent_dir

            # We assume view is a square of size H x W (e.g., 7x7), centered in front of the agent.
            # Compute top-left (tlx,tly) of the view window in WORLD coords for each direction.
            vs = H  # assume square
            half = vs // 2

            def in_view(wx, wy):
                """Return (vx, vy) if in view; else None"""
                # Compute top-left of the view window and mapping formula by agent_dir
                if ad == 0:  # facing right (+x)
                    tlx, tly = ax, ay - half
                    vx, vy = wx - tlx, wy - tly
                elif ad == 1:  # facing down (+y)
                    tlx, tly = ax - half, ay
                    # rotate 90 deg: world (wx,wy) maps to view (vx,vy)
                    # When facing down, +y is "forward". View x increases to the left in world.
                    vx, vy = (ax + half) - wx, wy - tly
                elif ad == 2:  # facing left (-x)
                    tlx, tly = ax - (vs - 1), ay - half
                    vx, vy = tlx + (vs - 1) - wx, (tly + (vs - 1)) - (wy + (vs - 1) - (tly))
                    # Simplify: vx = (ax - (vs-1)) + (vs-1) - wx = ax - wx
                    vx, vy = ax - wx, (ay + half) - wy
                else:  # ad == 3, facing up (-y)
                    tlx, tly = ax - half, ay - (vs - 1)
                    # rotate -90 deg
                    vx, vy = wx - tlx, (ay + half) - wy

                if 0 <= vx < W and 0 <= vy < H:
                    return int(vx), int(vy)
                return None

            for x, y, color in self._distractors:
                mapped = in_view(x, y)
                if mapped is not None:
                    vx, vy = mapped
                    img[vy, vx] = self._ball_enc_by_color[color]

        new_obs = dict(obs)
        new_obs["image"] = img
        return new_obs

  

    
# Dictionary mapping string keys to corresponding wrapper classes.
WRAPPING_TO_WRAPPER = {
    "ViewSize": ViewSizeWrapper,
    "ImgObs": ImgObsWrapper,
    "StepReward": StepRewardWrapper,
    "CompactAction": CompactActionWrapper,
    "FlattenOnehotObj": FlatOnehotObjectObsWrapper,
    "FixedSeed": FixedSeedWrapper,
    "FixedRandomDistractor": FixedRandomDistractorWrapper,
}