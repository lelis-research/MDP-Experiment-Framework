import gymnasium as gym
import minihack

env = gym.make(
   "MiniHack-River-v0",
   observation_keys=("pixel", "glyphs", "colors", "chars"),
   max_episode_steps=100,
)
env.reset() # each reset generates a new environment instance
env.step(1)  # move agent '@' north
env.render()
