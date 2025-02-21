from Agents.Utils import (
    TabularFeature,
    FLattenFeature,
    ImageFeature,
    HyperParameters,
)
from Agents.RandomAgent import RandomAgent
from Agents.TabularAgent import (
    QLearningAgent,
    NStepQLearningAgent,
    SarsaAgent,
    DoubleQLearningAgent,
)
from Agents.DeepAgent.ValueBased import (
    DQNAgent,
    DoubleDQNAgent,
    NStepDQNAgent,
)
from Agents.DeepAgent.PolicyGradient import (
    ReinforceAgent,
    ReinforceWithBaselineAgent,
    A2CAgentV1,
    A2CAgentV2,
    PPOAgent,
)

def get_env_action_space(env):
    return env.single_action_space if hasattr(env, 'single_action_space') else env.action_space

def get_env_observation_space(env):
    return env.single_observation_space if hasattr(env, 'single_observation_space') else env.observation_space

def get_num_envs(env):
    return env.num_envs if hasattr(env, 'num_envs') else 1

conv_network_1 = [
    {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "stride": 1},
    {"type": "relu"},
    {"type": "conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1},
    {"type": "relu"},
    {"type": "flatten"},
    {"type": "linear", "out_features": 512},
    {"type": "relu"},
    {"type": "linear", "in_features": 512}
]
fc_network_1 = [
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
        HyperParameters(),
        get_num_envs(env)
    ),

    #Tabular Agents
    "QLearning": lambda env: QLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.2, gamma=0.99, epsilon=0.1),
        get_num_envs(env),
        TabularFeature,
    ),
    "Sarsa": lambda env: SarsaAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1),
        get_num_envs(env),
        TabularFeature,
    ),
    "DoubleQLearning": lambda env: DoubleQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.1, gamma=0.99, epsilon=0.1),
        get_num_envs(env),
        TabularFeature,
    ),
    "NStepQLearning": lambda env: NStepQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.01, n_steps=1),
        get_num_envs(env),
        TabularFeature,
    ),
    
    # Deep Agents
    "DQN": lambda env: DQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20,
                        value_network=conv_network_1,
                        ),
        get_num_envs(env),
        ImageFeature,
    ),
    "DoubleDQN": lambda env: DoubleDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.01, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20,
                        value_network=fc_network_1,
                        ),
        get_num_envs(env),
        FLattenFeature,
    ),
    "NStepDQN": lambda env: NStepDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20, n_steps=10,
                        value_network=conv_network_1,
                        ),
        get_num_envs(env),
        ImageFeature,
    ),
    "Reinforce": lambda env: ReinforceAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1,
                        actor_network=fc_network_1,
                        ),
        get_num_envs(env),
        FLattenFeature,
    ),
    "ReinforceWithBaseline": lambda env: ReinforceWithBaselineAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, epsilon=0.1,
                        actor_network=fc_network_1,
                        actor_step_size=0.001,
                        critic_network=fc_network_1,
                        critic_step_size=0.001,
                        ),
        get_num_envs(env),
        FLattenFeature,
    ),
    "A2C_v1": lambda env: A2CAgentV1(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, epsilon=0.1, rollout_steps=5,
                        actor_network=fc_network_1,
                        actor_step_size=0.001,
                        critic_network=fc_network_1,
                        critic_step_size=0.001,
                        ),
        get_num_envs(env),
        FLattenFeature,
    ),
    "A2C_v2": lambda env: A2CAgentV2(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, epsilon=0.1, rollout_steps=5,
                        actor_network=fc_network_1,
                        actor_step_size=0.001,
                        critic_network=fc_network_1,
                        critic_step_size=0.001,
                        ),
        get_num_envs(env),
        FLattenFeature,
    ),
    "PPO": lambda env: PPOAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, clip_range=0.2,
                        mini_batch_size=64, rollout_steps=2048, 
                        entropy_coef=0.01, num_epochs=10,
                        actor_network=fc_network_1,
                        actor_step_size=3e-4,
                        critic_network=fc_network_1,
                        critic_step_size=1e-4, 
                        ),
        get_num_envs(env),
        FLattenFeature,
    )
}
