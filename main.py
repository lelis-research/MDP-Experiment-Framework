from Environments.MiniGrid.EmptyGrid import get_empty_grid

from Agents.RandomAgent.RandomAgent import RandomAgent
from Agents.TabularAgent.QLearningAgent import QLearningAgent
from Agents.TabularAgent.NStepQLearningAgent import NStepQLearningAgent
from Agents.TabularAgent.SarsaAgent import SarsaAgent
from Agents.TabularAgent.DoubleQLearningAgent import DoubleQLearningAgent
from Agents.DeepAgent.DQNAgent import DQNAgent
from Agents.DeepAgent.DoubleDQNAgent import DoubleDQNAgent
from Agents.Utils.HyperParams import HyperParameters

from Experiments.BaseExperiment import BaseExperiment
from Experiments.LoggerExperiment import LoggerExperiment
from Evaluate.SingleExpAnalyzer import SingleExpAnalyzer

import pickle
import os
import datetime

import numpy as np
import random
import torch
def main():
    

    # Create the environment with any desired wrappers
    runs_dir = "Runs/"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    

    env = get_empty_grid(
        render_mode=None,
        max_steps=200,
        wrapping_lst=["ViewSize", "StepReward", "FlattenOnehotObj"],
        wrapping_params=[{"agent_view_size": 3}, {"step_reward": -1}, {}]
    )

    seed = 200

    # agent = RandomAgent(env.action_space)

    # hp = HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1)
    # agent = QLearningAgent(env.action_space, hp)

    # hp = HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1)
    # agent = SarsaAgent(env.action_space, hp)

    # hp = HyperParameters(step_size=0.1, gamma=0.99, epsilon=0.1)
    # agent = DoubleQLearningAgent(env.action_space, hp)

    # hp = HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1, n_steps=3)
    # agent = NStepQLearningAgent(env.action_space, hp)

    # hp = HyperParameters(step_size=0.01, gamma=0.99, epsilon=0.1, 
    #                      replay_buffer_cap=512, batch_size=32,
    #                      target_update_freq=20)
    # agent = DQNAgent(env.action_space, env.observation_space, hp)

    hp = HyperParameters(step_size=0.01, gamma=0.99, epsilon=0.1, 
                         replay_buffer_cap=512, batch_size=32,
                         target_update_freq=20)
    agent = DoubleDQNAgent(env.action_space, env.observation_space, hp)

    # Create and run the experiment
    exp_name = f"{agent}_seed [{seed}]_{timestamp}"
    exp_dir = os.path.join(runs_dir, exp_name)

    experiment = LoggerExperiment(env, agent, exp_dir)
    metrics = experiment.multi_run(num_runs=3, num_episodes=400, seed_offset=seed)    
    

    # Analyze and plot the results.
    analyzer = SingleExpAnalyzer(metrics)
    analyzer.plot_combined(save_dir=exp_dir)
    analyzer.save_seeds(save_dir=exp_dir)


if __name__ == "__main__":
    main()