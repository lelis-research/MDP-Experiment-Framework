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
    OptionA2CAgent,
    OptionPPOAgent,
)
from RLBase.Options.Utils import load_options_list
from Configs.networks import NETWORKS
import torch

def get_env_action_space(env):
    return env.single_action_space if hasattr(env, 'single_action_space') else env.action_space

def get_env_observation_space(env):
    return env.single_observation_space if hasattr(env, 'single_observation_space') else env.observation_space

def get_num_envs(env):
    return env.num_envs if hasattr(env, 'num_envs') else 1

def get_device(preferred_device):
    if preferred_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif preferred_device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    elif preferred_device == "cpu":
        device = "cpu"
    else:
        print(f"⚠️ Warning: {preferred_device} not available, falling back to CPU.")
        device = "cpu"
    return device

preferred_device = "cpu"  # cpu, mps, cuda
device = get_device(preferred_device)
print(f"Using device: {device}")

AGENT_DICT = {
    HumanAgent.name: lambda env, info: HumanAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(actions_enum=env.unwrapped.actions), #enum of the actions and their name
        get_num_envs(env),
        FLattenFeature,
        options_lst=[], #load_options_list("Runs/Options/MaskedOptionLearner/MaxLen-20_Mask-input-l1_Regularized-0.01_0/selected_options_10.t"),
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
        HyperParameters(
            step_size=info.get("step_size", 0.2),
            gamma=info.get("gamma", 0.99),
            epsilon=info.get("epsilon", 0.1),
        ),
        get_num_envs(env),
        TabularFeature,
    ),
    SarsaAgent.name: lambda env, info: SarsaAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.5),
            gamma=info.get("gamma", 0.99),
            epsilon=info.get("epsilon", 0.1),
        ),
        get_num_envs(env),
        TabularFeature,
    ),
    DoubleQLearningAgent.name: lambda env, info: DoubleQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.5),
            gamma=info.get("gamma", 0.99),
            epsilon=info.get("epsilon", 0.1),
        ),
        get_num_envs(env),
        TabularFeature,
    ),
    NStepQLearningAgent.name: lambda env, info: NStepQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.5),
            gamma=info.get("gamma", 0.99),
            epsilon=info.get("epsilon", 0.01),
            n_steps=info.get("n_steps", 1),
        ),
        get_num_envs(env),
        TabularFeature,
    ),
    
    
    # Deep Agents
    DQNAgent.name: lambda env, info: DQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.001),
            gamma=info.get("gamma", 0.99),
            epsilon=info.get("epsilon", 0.01),
            replay_buffer_cap=info.get("replay_buffer_cap", 10000),
            batch_size=info.get("batch_size", 128),
            target_update_freq=info.get("target_update_freq", 20),
            value_network=NETWORKS[info.get("value_network", "fc_network_1")],
        ),
        get_num_envs(env),
        FLattenFeature,
        device=device,
    ),
    
    OptionDQNAgent.name: lambda env, info: OptionDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.001),
            gamma=info.get("gamma", 0.99),
            epsilon=info.get("epsilon", 0.01),
            replay_buffer_cap=info.get("replay_buffer_cap", 100000),
            batch_size=info.get("batch_size", 128),
            target_update_freq=info.get("target_update_freq", 20),
            value_network=NETWORKS[info.get("value_network", "fc_network_1")],
        ),
        get_num_envs(env),
        FLattenFeature,
        options_lst=load_options_list(info["option_path"]),
        device=device
    ),
    
    DoubleDQNAgent.name: lambda env, info: DoubleDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.001),
            gamma=info.get("gamma", 0.99),
            epsilon=info.get("epsilon", 0.01),
            replay_buffer_cap=info.get("replay_buffer_cap", 100000),
            batch_size=info.get("batch_size", 128),
            target_update_freq=info.get("target_update_freq", 20),
            value_network=NETWORKS[info.get("value_network", "fc_network_1")],
        ),
        get_num_envs(env),
        FLattenFeature,
        device=device
    ),
    NStepDQNAgent.name: lambda env, info: NStepDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.001),
            gamma=info.get("gamma", 0.99),
            epsilon=info.get("epsilon", 0.01),
            replay_buffer_cap=info.get("replay_buffer_cap", 100000),
            batch_size=info.get("batch_size", 128),
            target_update_freq=info.get("target_update_freq", 20),
            n_steps=info.get("n_steps", 10),
            value_network=NETWORKS[info.get("value_network", "fc_network_1")],
        ),
        get_num_envs(env),
        FLattenFeature,
        device=device
    ),
    
    ReinforceAgent.name: lambda env, info: ReinforceAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.001),
            gamma=info.get("gamma", 0.99),
            epsilon=info.get("epsilon", 0.01),
            actor_network=NETWORKS[info.get("actor_network", "fc_network_1")],
        ),
        get_num_envs(env),
        FLattenFeature,
        device=device
    ),
    ReinforceWithBaselineAgent.name: lambda env, info: ReinforceWithBaselineAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            epsilon=info.get("epsilon", 0.01),
            actor_network=NETWORKS[info.get("actor_network", "fc_network_1")],
            actor_step_size=info.get("actor_step_size", 0.001),
            critic_network=NETWORKS[info.get("critic_network", "fc_network_1")],
            critic_step_size=info.get("critic_step_size", 0.001),
        ),
        get_num_envs(env),
        FLattenFeature,
        device=device
    ),
    A2CAgent.name: lambda env, info: A2CAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            lamda=info.get("lamda", 0.95),
            rollout_steps=info.get("rollout_steps", 128),
            actor_network=NETWORKS[info.get("actor_network", "conv_network_1")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            critic_network=NETWORKS[info.get("critic_network", "conv_network_1")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            norm_adv_flag=info.get("norm_adv_flag", True),
            entropy_coef=info.get("entropy_coef", 0.0),
            anneal_step_size_flag=info.get("anneal_step_size_flag", False),
            total_updates=info.get("total_updates", 2e6)
        ),
        get_num_envs(env),
        ImageFeature,
        device=device
    ),
    OptionA2CAgent.name: lambda env, info: OptionA2CAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            lamda=info.get("lamda", 0.95),
            rollout_steps=info.get("rollout_steps", 20),
            actor_network=NETWORKS[info.get("actor_network", "conv_network_1")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            critic_network=NETWORKS[info.get("critic_network", "conv_network_1")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            norm_adv_flag=info.get("norm_adv_flag", True),
            entropy_coef=info.get("entropy_coef", 0.0),
            anneal_step_size_flag=info.get("anneal_step_size_flag", False),
            total_updates=info.get("total_updates", 2e6)
        ),
        get_num_envs(env),
        ImageFeature,
        options_lst=load_options_list(info["option_path"]),
        device=device
    ),
    
    PPOAgent.name: lambda env, info: PPOAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            lamda=info.get("lamda", 0.95),
            mini_batch_size=info.get("mini_batch_size", 64),
            rollout_steps=info.get("rollout_steps", 256),
            num_epochs=info.get("num_epochs", 5),
            
            clip_range_actor_init=info.get("clip_range_actor_init", 0.2),
            anneal_clip_range_actor=info.get("anneal_clip_range_actor", True),
            clip_range_critic_init=info.get("clip_range_critic_init", None), # None means no clipping
            anneal_clip_range_critic=info.get("anneal_clip_range_critic", False),
            target_kl=info.get("target_kl", 0.02),
            
            actor_network=NETWORKS[info.get("actor_network", "conv_network_2")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            actor_eps = info.get("actor_eps", 1e-5),
            critic_network=NETWORKS[info.get("critic_network", "conv_network_2")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            critic_eps = info.get("critic_eps", 1e-5),
            
            anneal_step_size_flag=info.get("anneal_step_size_flag", True),
            total_steps=info.get("total_steps", 2_000_000 // 256),
            
            norm_adv_flag=info.get("norm_adv_flag", True),
            critic_coef=info.get("critic_coef", 0.5),
            entropy_coef=info.get("entropy_coef", 0.02),
            max_grad_norm=info.get("max_grad_norm", 0.5),
            
        ),
        get_num_envs(env),
        FLattenFeature,
        device=device
    ),
    
    OptionPPOAgent.name: lambda env, info: OptionPPOAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            lamda=info.get("lamda", 0.95),
            mini_batch_size=info.get("mini_batch_size", 64),
            rollout_steps=info.get("rollout_steps", 256),
            num_epochs=info.get("num_epochs", 5),
            
            clip_range_actor_init=info.get("clip_range_actor_init", 0.2),
            anneal_clip_range_actor=info.get("anneal_clip_range_actor", True),
            clip_range_critic_init=info.get("clip_range_critic_init", None), # None means no clipping
            anneal_clip_range_critic=info.get("anneal_clip_range_critic", False),
            target_kl=info.get("target_kl", 0.02),
            
            actor_network=NETWORKS[info.get("actor_network", "conv_network_2")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            actor_eps = info.get("actor_eps", 1e-5),
            critic_network=NETWORKS[info.get("critic_network", "conv_network_2")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            critic_eps = info.get("critic_eps", 1e-5),
            
            anneal_step_size_flag=info.get("anneal_step_size_flag", True),
            total_steps=info.get("total_steps", 2_000_000 // 256),
            
            norm_adv_flag=info.get("norm_adv_flag", True),
            critic_coef=info.get("critic_coef", 0.5),
            entropy_coef=info.get("entropy_coef", 0.02),
            max_grad_norm=info.get("max_grad_norm", 0.5),
            
        ),
        get_num_envs(env),
        ImageFeature,
        options_lst=load_options_list(info["option_path"]),
        device=device
    )
}