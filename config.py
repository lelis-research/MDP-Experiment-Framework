from RLBase.Agents.Utils import (
    TabularFeature,
    FLattenFeature,
    ImageFeature,
    HyperParameters,
)
from RLBase.Agents.RandomAgent import RandomAgent
from RLBase.Agents.TabularAgent import (
    QLearningAgent,
    NStepQLearningAgent,
    SarsaAgent,
    DoubleQLearningAgent,
    MaskedQLearningAgent
)
from RLBase.Agents.DeepAgent.ValueBased import (
    DQNAgent,
    DoubleDQNAgent,
    NStepDQNAgent,
    MaskedDQNAgent,
)
from RLBase.Agents.DeepAgent.PolicyGradient import (
    ReinforceAgent,
    ReinforceWithBaselineAgent,
    A2CAgentV1,
    A2CAgentV2,
    PPOAgent,
)
from RLBase import load_option


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

env_wrapping= ["ViewSize", "ImgObs", "StepReward"] #FlattenOnehotObj
wrapping_params = [{"agent_view_size": 5}, {}, {"step_reward": -1}]
env_params = {}#{"chain_length": 20}

device="cpu" # cpu, mps, cuda

AGENT_DICT = {
    RandomAgent.name: lambda env: RandomAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(),
        get_num_envs(env),
    ),

    #Tabular Agents
    QLearningAgent.name: lambda env: QLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.2, gamma=0.99, epsilon=0.1),
        get_num_envs(env),
        TabularFeature,
    ),
    SarsaAgent.name: lambda env: SarsaAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1),
        get_num_envs(env),
        TabularFeature,
    ),
    DoubleQLearningAgent.name: lambda env: DoubleQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.1, gamma=0.99, epsilon=0.1),
        get_num_envs(env),
        TabularFeature,
    ),
    NStepQLearningAgent.name: lambda env: NStepQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.01, n_steps=1),
        get_num_envs(env),
        TabularFeature,
    ),
    
    MaskedQLearningAgent.name: lambda env: MaskedQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.2, gamma=0.99, epsilon=0.1),
        get_num_envs(env),
        TabularFeature,
        initial_options=load_option("Runs/Train/MiniGrid-Empty-5x5-v0_{}_DQN_seed[123123]_20250306_113108/Run1_Last_options.t"),        
    ),
    
    # Deep Agents
    DQNAgent.name: lambda env: DQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20,
                        value_network=fc_network_1,
                        ),
        get_num_envs(env),
        FLattenFeature,
        device=device,
    ),
    MaskedDQNAgent.name: lambda env: MaskedDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20,
                        value_network=fc_network_1,
                        ),
        get_num_envs(env),
        FLattenFeature,
        initial_options=load_option("Runs/Train/MiniGrid-DoorKey-5x5-v0_{}_DQN_seed[123123]_20250306_162847/Run1_Last_options.t"),
    ),

    DoubleDQNAgent.name: lambda env: DoubleDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.01, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20,
                        value_network=fc_network_1,
                        ),
        get_num_envs(env),
        FLattenFeature,
        device=device
    ),
    NStepDQNAgent.name: lambda env: NStepDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1, 
                        replay_buffer_cap=512, batch_size=32,
                        target_update_freq=20, n_steps=10,
                        value_network=fc_network_1,
                        ),
        get_num_envs(env),
        FLattenFeature,
        device=device
    ),
    
    ReinforceAgent.name: lambda env: ReinforceAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1,
                        actor_network=fc_network_1,
                        ),
        get_num_envs(env),
        FLattenFeature,
        device=device
    ),
    ReinforceWithBaselineAgent.name: lambda env: ReinforceWithBaselineAgent(
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
        device=device
    ),
    A2CAgentV1.name: lambda env: A2CAgentV1(
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
        device=device
    ),
    A2CAgentV2.name: lambda env: A2CAgentV2(
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
        device=device
    ),
    PPOAgent.name: lambda env: PPOAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, clip_range=0.2,
                        mini_batch_size=64, rollout_steps=2048, 
                        entropy_coef=0.01, num_epochs=10,
                        actor_network=conv_network_1,
                        actor_step_size=3e-4,
                        critic_network=conv_network_1,
                        critic_step_size=1e-4, 
                        ),
        get_num_envs(env),
        ImageFeature,
        device=device
    )
}
