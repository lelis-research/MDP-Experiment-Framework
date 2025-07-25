from RLBase.Agents.Utils import (
    TabularFeature,
    FLattenFeature,
    ImageFeature,
    HyperParameters,
)
from RLBase.Agents.HumanAgent import HumanAgent
from RLBase.Agents.RandomAgent import RandomAgent, OptionRandomAgent
from RLBase.Agents.TabularAgent import (
    QLearningAgent,
    NStepQLearningAgent,
    SarsaAgent,
    DoubleQLearningAgent,
    # MaskedQLearningAgent
)
from RLBase.Agents.DeepAgent.ValueBased import (
    DQNAgent,
    DoubleDQNAgent,
    NStepDQNAgent,
    OptionDQNAgent,
)
from RLBase.Agents.DeepAgent.PolicyGradient import (
    ReinforceAgent,
    ReinforceWithBaselineAgent,
    A2CAgent,
    PPOAgent,
)
from RLBase.Options.Utils import load_options_list


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
    {"type": "linear", "out_features": 64},
    {"type": "relu"},
    {"type": "linear", "in_features": 64},

]
linear_network_1 = [
    {"type": "linear"}
]

device="cpu" # cpu, mps, cuda#

AGENT_DICT = {
    HumanAgent.name: lambda env, info: HumanAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(actions_enum=env.unwrapped.actions), #enum of the actions and their name
        get_num_envs(env),
        FLattenFeature,
        options_lst=load_options_list("Runs/Options/MaskedOptionLearner/MaxLen-20_Mask-input-l1_0/selected_options_10.t"),
        device=device
    ),
    RandomAgent.name: lambda env, info: RandomAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(),
        get_num_envs(env),
    ),
    OptionRandomAgent.name: lambda env, info: OptionRandomAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(),
        get_num_envs(env),
        options_lst=load_options_list(info["option_path"]),
    ),

    #Tabular Agents
    QLearningAgent.name: lambda env, info: QLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.2, gamma=0.99, epsilon=0.1),
        get_num_envs(env),
        TabularFeature,
    ),
    SarsaAgent.name: lambda env, info: SarsaAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.1),
        get_num_envs(env),
        TabularFeature,
    ),
    DoubleQLearningAgent.name: lambda env, info: DoubleQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.1, gamma=0.99, epsilon=0.1),
        get_num_envs(env),
        TabularFeature,
    ),
    NStepQLearningAgent.name: lambda env, info: NStepQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.5, gamma=0.99, epsilon=0.01, n_steps=1),
        get_num_envs(env),
        TabularFeature,
    ),
    
    # MaskedQLearningAgent.name: lambda env: MaskedQLearningAgent(
    #     get_env_action_space(env), 
    #     get_env_observation_space(env),
    #     HyperParameters(step_size=0.2, gamma=0.99, epsilon=0.1),
    #     get_num_envs(env),
    #     TabularFeature,
    #     initial_options=load_option("Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 20}/DQN/_seed[123123]_20250312_092541/R1_T5_N1_L['1']_S100_options.t"),        
    # ),
    
    # Deep Agents
    DQNAgent.name: lambda env, info: DQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.01, 
                        replay_buffer_cap=10000, batch_size=128,
                        target_update_freq=20,
                        value_network=fc_network_1,
                        ),
        get_num_envs(env),
        FLattenFeature,
        device=device,
    ),
    
    OptionDQNAgent.name: lambda env, info: OptionDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=info['step_size'], gamma=0.99, epsilon=info['epsilon'], 
                        replay_buffer_cap=100000, batch_size=info['batch_size'],
                        target_update_freq=info['target_update_freq'],
                        value_network=fc_network_1,
                        ),
        get_num_envs(env),
        FLattenFeature,
        options_lst=load_options_list(info["option_path"]),
        device=device
    ),
    
    DoubleDQNAgent.name: lambda env, info: DoubleDQNAgent(
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
    NStepDQNAgent.name: lambda env, info: NStepDQNAgent(
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
    
    ReinforceAgent.name: lambda env, info: ReinforceAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(step_size=0.001, gamma=0.99, epsilon=0.1,
                        actor_network=fc_network_1,
                        ),
        get_num_envs(env),
        FLattenFeature,
        device=device
    ),
    ReinforceWithBaselineAgent.name: lambda env, info: ReinforceWithBaselineAgent(
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
    A2CAgent.name: lambda env, info: A2CAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(gamma=0.99, lamda=0.95, rollout_steps=9,
                        actor_network=fc_network_1,
                        actor_step_size=0.0001,
                        critic_network=fc_network_1,
                        critic_step_size=0.0001,
                        ),
        get_num_envs(env),
        FLattenFeature,
        device=device
    ),
    PPOAgent.name: lambda env, info: PPOAgent(
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
