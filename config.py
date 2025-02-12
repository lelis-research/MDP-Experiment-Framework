from Agents.RandomAgent.RandomAgent import RandomAgent
from Agents.TabularAgent.QLearningAgent import QLearningAgent
from Agents.TabularAgent.NStepQLearningAgent import NStepQLearningAgent
from Agents.TabularAgent.SarsaAgent import SarsaAgent
from Agents.TabularAgent.DoubleQLearningAgent import DoubleQLearningAgent

from Agents.DeepAgent.DQNAgent import DQNAgent
from Agents.DeepAgent.DoubleDQNAgent import DoubleDQNAgent
from Agents.DeepAgent.ReinforceAgent import ReinforceAgent
from Agents.DeepAgent.ReinforceWithBaseline import ReinforceAgentWithBaseline
from Agents.DeepAgent.ActorCriticAgent import ActorCriticAgent
from Agents.DeepAgent.PPOAgent import PPOAgent

from Agents.Utils.HyperParams import HyperParameters

def get_env_action_space(env):
    return env.single_action_space if hasattr(env, 'single_action_space') else env.action_space

def get_env_observation_space(env):
    return env.single_observation_space if hasattr(env, 'single_observation_space') else env.observation_space

def get_num_envs(env):
    return env.num_envs if hasattr(env, 'num_envs') else 1

AGENT_DICT = {
    "Random": lambda env: RandomAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        None,
        get_num_envs(env)
    ),
    "QLearning": lambda env: QLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1),
        get_num_envs(env)
    ),
    "Sarsa": lambda env: SarsaAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1),
        get_num_envs(env)
    ),
    "DoubleQLearning": lambda env: DoubleQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.1, gamma=0.99, epsilon=0.1),
        get_num_envs(env)
    ),
    "NStepQLearning": lambda env: NStepQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.01, n_steps=1),
        get_num_envs(env)
    ),
    "DQN": lambda env: DQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.01, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20),
        get_num_envs(env)
    ),
    "DoubleDQN": lambda env: DoubleDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.01, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20),
        get_num_envs(env)
    ),
    "Reinforce": lambda env: ReinforceAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1),
        get_num_envs(env)
    ),
    "ReinforceWithBaseline": lambda env: ReinforceAgentWithBaseline(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, epsilon=0.1,
                        actor_step_size=0.001, critic_step_size=0.001),
        get_num_envs(env)
    ),
    "ActorCritic": lambda env: ActorCriticAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, epsilon=0.1, rollout_steps=5,
                        actor_step_size=0.001, critic_step_size=0.001),
        get_num_envs(env)
    ),
    "PPOAgent": lambda env: PPOAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, clip_range=0.2,
                        batch_size=256, rollout_steps=1024, num_epochs=1,
                        actor_step_size=3e-4, critic_step_size=3e-4, 
                        value_loss_coef=0.5, entropy_coef=0.0)
    )
}
