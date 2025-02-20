from Agents.RandomAgent.RandomAgent import RandomAgent
from Agents.TabularAgent.QLearningAgent import QLearningAgent
from Agents.TabularAgent.NStepQLearningAgent import NStepQLearningAgent
from Agents.TabularAgent.SarsaAgent import SarsaAgent
from Agents.TabularAgent.DoubleQLearningAgent import DoubleQLearningAgent

from Agents.DeepAgent.ValueBased.DQNAgent import DQNAgent
from Agents.DeepAgent.ValueBased.DoubleDQNAgent import DoubleDQNAgent
from Agents.DeepAgent.ValueBased.NStepDQNAgent import NStepDQNAgent
from Agents.DeepAgent.PolicyGradient.ReinforceAgent import ReinforceAgent
from Agents.DeepAgent.PolicyGradient.ReinforceWithBaseline import ReinforceAgentWithBaseline
from Agents.DeepAgent.PolicyGradient.A2C_v1 import A2CAgentV1
from Agents.DeepAgent.PolicyGradient.A2C_v2 import A2CAgentV2
from Agents.DeepAgent.PolicyGradient.PPOAgent import PPOAgent

from Agents.Utils.HyperParams import HyperParameters

def get_env_action_space(env):
    return env.single_action_space if hasattr(env, 'single_action_space') else env.action_space

def get_env_observation_space(env):
    return env.single_observation_space if hasattr(env, 'single_observation_space') else env.observation_space

def get_num_envs(env):
    return env.num_envs if hasattr(env, 'num_envs') else 1

two_hidden_layers_network_1 = [
    {"type": "linear", "out_features": 128},
    {"type": "relu"},
    {"type": "linear", "in_features": 128, "out_features": 128},
    {"type": "relu"},
    {"type": "linear", "in_features": 128}
]
linear_network_1 = [
    {"type": "linear"}
]

AGENT_DICT = {
    "Random": lambda env: RandomAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        None,
        get_num_envs(env)
    ),

    #Tabular Agents
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
    
    # Deep Agents
    "DQN": lambda env: DQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20,
                        value_network=linear_network_1,
                        ),
        get_num_envs(env)
    ),
    "DoubleDQN": lambda env: DoubleDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.01, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20,
                        value_network=two_hidden_layers_network_1,
                        ),
        get_num_envs(env)
    ),
    "NStepDQN": lambda env: NStepDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20, n_steps=10,
                        value_network=two_hidden_layers_network_1,
                        ),
        get_num_envs(env)
    ),
    "Reinforce": lambda env: ReinforceAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1,
                        actor_network=two_hidden_layers_network_1,
                        ),
        get_num_envs(env)
    ),
    "ReinforceWithBaseline": lambda env: ReinforceAgentWithBaseline(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, epsilon=0.1,
                        actor_network=two_hidden_layers_network_1,
                        actor_step_size=0.001,
                        critic_network=two_hidden_layers_network_1,
                        critic_step_size=0.001,
                        ),
        get_num_envs(env)
    ),
    "A2C_v1": lambda env: A2CAgentV1(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, epsilon=0.1, rollout_steps=5,
                        actor_network=two_hidden_layers_network_1,
                        actor_step_size=0.001,
                        critic_network=two_hidden_layers_network_1,
                        critic_step_size=0.001,
                        ),
        get_num_envs(env)
    ),
    "A2C_v2": lambda env: A2CAgentV2(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, epsilon=0.1, rollout_steps=5,
                        actor_network=two_hidden_layers_network_1,
                        actor_step_size=0.001,
                        critic_network=two_hidden_layers_network_1,
                        critic_step_size=0.001,
                        ),
        get_num_envs(env)
    ),
    "PPO": lambda env: PPOAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, clip_range=0.2,
                        mini_batch_size=64, rollout_steps=2048, 
                        entropy_coef=0.01, num_epochs=10,
                        actor_network=two_hidden_layers_network_1,
                        actor_step_size=3e-4,
                        critic_network=two_hidden_layers_network_1,
                        critic_step_size=1e-4, 
                        ),
        get_num_envs(env)
    )
}
