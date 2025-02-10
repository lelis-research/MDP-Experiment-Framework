from Agents.RandomAgent.RandomAgent import RandomAgent
from Agents.TabularAgent.QLearningAgent import QLearningAgent
from Agents.TabularAgent.NStepQLearningAgent import NStepQLearningAgent
from Agents.TabularAgent.SarsaAgent import SarsaAgent
from Agents.TabularAgent.DoubleQLearningAgent import DoubleQLearningAgent

from Agents.DeepAgent.DQNAgent import DQNAgent
from Agents.DeepAgent.DoubleDQNAgent import DoubleDQNAgent
from Agents.DeepAgent.ReinforceAgent import ReinforceAgent
from Agents.DeepAgent.ReinforceWithBaseline import ReinforceAgentWithBaseline

from Agents.Utils.HyperParams import HyperParameters

AGENT_DICT = {
    "RandomAgent": lambda env: RandomAgent(env.action_space),
    "QLearningAgent": lambda env: QLearningAgent(
        env.action_space, HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1)
    ),
    "SarsaAgent": lambda env: SarsaAgent(
        env.action_space, HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1)
    ),
    "DoubleQLearningAgent": lambda env: DoubleQLearningAgent(
        env.action_space, HyperParameters(step_size=0.1, gamma=0.99, epsilon=0.1)
    ),
    "NStepQLearningAgent": lambda env: NStepQLearningAgent(
        env.action_space, HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1, n_steps=3)
    ),
    "DQNAgent": lambda env: DQNAgent(
        env.action_space, env.observation_space,
        HyperParameters(step_size=0.01, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20)
    ),
    "DoubleDQNAgent": lambda env: DoubleDQNAgent(
        env.action_space, env.observation_space,
        HyperParameters(step_size=0.01, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20)
    ),
    "ReinforceAgent": lambda env: ReinforceAgent(
        env.action_space, env.observation_space,
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1)
    ),
    "ReinforceAgentWithBaseline": lambda env: ReinforceAgentWithBaseline(
        env.action_space, env.observation_space,
        HyperParameters(step_size=0.01, gamma=0.99, epsilon=0.1,
                        actor_step_size=0.001, critic_step_size=0.001)
    ),
}
