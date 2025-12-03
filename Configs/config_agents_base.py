from RLBase.Agents.Utils import HyperParameters
from RLBase.FeatureExtractors import *
from RLBase.Agents import *
from RLBase.Networks import NETWORK_PRESETS
# from RLBase.Options import load_options_list
# from RLBase.Options.SymbolicOptions.LLMOptions import FindKeyOption, OpenDoorOption, create_all_options
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

AGENT_DICT = {
    # HumanAgent.name: lambda env, info: HumanAgent(
    #     get_env_action_space(env), 
    #     get_env_observation_space(env),
    #     HyperParameters(actions_enum=env.envs[0].unwrapped.actions), #enum of the actions and their name
    #     get_num_envs(env),
    #     MirrorFeature,
    #     options_lst=[],#create_all_options(), 
    #     device=device
    # ),
    # ContinualHumanAgent.name: lambda env, info: ContinualHumanAgent(
    #     get_env_action_space(env), 
    #     get_env_observation_space(env),
    #     HyperParameters(actions_enum=env.unwrapped.actions), #enum of the actions and their name
    #     get_num_envs(env),
    #     MirrorFeature,
    #     option_learner_class=ManualContinualOptionLearner,
    #     device=device
    # ),
    RandomAgent.name: lambda env, info: RandomAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(),
        get_num_envs(env),
        TabularFeature,
    ),
    # OptionRandomAgent.name: lambda env, info: OptionRandomAgent(
    #     get_env_action_space(env), 
    #     get_env_observation_space(env),
    #     HyperParameters(),
    #     get_num_envs(env),
    #     options_lst=create_all_options(),
    # ),

    #Tabular Agents
    QLearningAgent.name: lambda env, info: QLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.1),
            gamma=info.get("gamma", 0.99),
            epsilon_start=info.get("epsilon_start", 1.0),
            epsilon_end=info.get("epsilon_end", 0.05),
            epsilon_decay_steps=info.get("epsilon_decay_steps", 10000),
            n_steps=info.get("n_steps", 3),
        ),
        get_num_envs(env),
        TabularFeature,
    ),
    # OptionQLearningAgent.name: lambda env, info: OptionQLearningAgent(
    #     get_env_action_space(env), 
    #     get_env_observation_space(env),
    #     HyperParameters(
    #         step_size=info.get("step_size", 0.2),
    #         gamma=info.get("gamma", 0.99),
    #         epsilon_start=info.get("epsilon_start", 1.0),
    #         epsilon_end=info.get("epsilon_end", 0.001),
    #         epilon_decay_steps=info.get("epilon_decay_steps", 400_000),
    #         discount_option_flag=info.get("discount_option_flag", True),
    #         update_action_within_option_flag=info.get("update_action_within_option_flag", False),
    #         n_steps=info.get("n_steps", 1),
    #     ),
    #     get_num_envs(env),
    #     TabularFeature,
    #     options_lst=create_all_options(), 
    # ),
    # ContinualOptionQLearningAgent.name: lambda env, info: ContinualOptionQLearningAgent(
    #     get_env_action_space(env), 
    #     get_env_observation_space(env),
    #     HyperParameters(
    #         step_size=info.get("step_size", 0.1),
    #         gamma=info.get("gamma", 0.99),
    #         epsilon_start=info.get("epsilon_start", 1.0),
    #         epsilon_end=info.get("epsilon_end", 0.01),
    #         epilon_decay_steps=info.get("epilon_decay_steps", 100_000),
    #         discount_option_flag=info.get("discount_option_flag", True),
    #         update_action_within_option_flag=info.get("update_action_within_option_flag", False),
    #         n_steps=info.get("n_steps", 1),
            
    #         option_init_mode=info.get("option_init_mode", "reset"),
    #         uncertainty_mode=info.get("uncertainty_mode", "entropy"),
    #         uncertainty_beta= float(info.get("uncertainty_beta", 1.0)),
    #         uncertainty_tau=float(info.get("uncertainty_tau", 1.0)),
    #         uncertainty_kappa=float(info.get("uncertainty_kappa", 1.0)),
            
    #         option_explore_mode=info.get("option_explore_mode", "none"), # none, e_greedy, schedule
    #         sch_budget=info.get("sch_budget", 2),
    #         sch_rho=info.get("sch_rho", 0.5),            
    #     ),
    #     get_num_envs(env),
    #     TabularFeature,
    #     ManualContinualOptionLearner,
    #     initial_options_lst=[],
    # ),
    
    
    # Deep Agents
    DQNAgent.name: lambda env, info: DQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 1e-4),
            gamma=info.get("gamma", 0.99),
            epsilon_start=info.get("epsilon_start", 1.0),
            epsilon_end=info.get("epsilon_end", 0.05),
            epsilon_decay_steps=info.get("epsilon_decay_steps", 10000),
            n_steps=info.get("n_steps", 1),
            warmup_buffer_size=info.get("warmup_buffer_size", 100),
            replay_buffer_cap=info.get("replay_buffer_cap", 200_000),
            batch_size=info.get("batch_size", 32),
            target_update_freq=info.get("target_update_freq", 20),
            flag_double_dqn_target=info.get("flag_double_dqn_target", False),
            value_network=NETWORK_PRESETS[info.get("value_network", "mlp1")],
        ),
        get_num_envs(env),
        OneHotFlattenFeature,
        device=device,
    ),
    
    # OptionDQNAgent.name: lambda env, info: OptionDQNAgent(
    #     get_env_action_space(env), 
    #     get_env_observation_space(env),
    #     HyperParameters(
    #         step_size=info.get("step_size", 1e-4),
    #         gamma=info.get("gamma", 0.99),
    #         epsilon_start=info.get("epsilon_start", 1.0),
    #         epsilon_end=info.get("epsilon_end", 0.05),
    #         epsilon_decay_steps=info.get("epsilon_decay_steps", 160_000),
    #         n_steps=info.get("n_steps", 1),
    #         warmup_buffer_size=info.get("warmup_buffer_size", 10_000), 
    #         replay_buffer_cap=info.get("replay_buffer_cap", 200_000),
    #         batch_size=info.get("batch_size", 32),
    #         target_update_freq=info.get("target_update_freq", 2000),
    #         flag_double_dqn_target=info.get("flag_double_dqn_target", True),
    #         value_network=NETWORKS[info.get("value_network", "conv2_network")],
    #     ),
    #     get_num_envs(env),
    #     OneHotImageDirCarryFeature,
    #     options_lst=create_all_options(),
    #     device=device
    # ),
    

    A2CAgent.name: lambda env, info: A2CAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            lamda=info.get("lamda", 0.95),
            rollout_steps=info.get("rollout_steps", 256),
            
            actor_network=NETWORK_PRESETS[info.get("actor_network", "mlp1")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            actor_eps = info.get("actor_eps", 1e-5),
            
            critic_network=NETWORK_PRESETS[info.get("critic_network", "mlp1")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            critic_eps = info.get("critic_eps", 1e-5),
            
            critic_coef=info.get("critic_coef", 0.5),
            entropy_coef=info.get("entropy_coef", 0.02),
            
            anneal_step_size_flag=info.get("anneal_step_size_flag", False),
            total_steps=info.get("total_steps", 100_000),
            update_type=info.get("update_type", "sync"), # sync, per_env
            norm_adv_flag=info.get("norm_adv_flag", True),
        ),
        get_num_envs(env),
        OneHotFlattenFeature,
        device=device
    ),
    # OptionA2CAgent.name: lambda env, info: OptionA2CAgent(
    #     get_env_action_space(env), 
    #     get_env_observation_space(env),
    #     HyperParameters(
    #         gamma=info.get("gamma", 0.99),
    #         lamda=info.get("lamda", 0.95),
    #         rollout_steps=info.get("rollout_steps", 20),
    #         actor_network=NETWORKS[info.get("actor_network", "conv_network_1")],
    #         actor_step_size=info.get("actor_step_size", 3e-4),
    #         critic_network=NETWORKS[info.get("critic_network", "conv_network_1")],
    #         critic_step_size=info.get("critic_step_size", 3e-4),
    #         norm_adv_flag=info.get("norm_adv_flag", False),
    #         entropy_coef=info.get("entropy_coef", 0.0),
    #         anneal_step_size_flag=info.get("anneal_step_size_flag", False),
    #         total_updates=info.get("total_updates", 2e6)
    #     ),
    #     get_num_envs(env),
    #     FLattenFeature,
    #     options_lst=load_options_list(info["option_path"]),
    #     device=device
    # ),
    
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
            
            actor_network=NETWORK_PRESETS[info.get("actor_network", "mlp1")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            actor_eps = info.get("actor_eps", 1e-5),
            
            critic_network=NETWORK_PRESETS[info.get("critic_network", "mlp1")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            critic_eps = info.get("critic_eps", 1e-5),
            
            anneal_step_size_flag=info.get("anneal_step_size_flag", False),
            total_steps=info.get("total_steps", 100_000),
            update_type=info.get("update_type", "sync"), # sync, per_env
            norm_adv_flag=info.get("norm_adv_flag", True),
            
            critic_coef=info.get("critic_coef", 0.5),
            entropy_coef=info.get("entropy_coef", 0.02),
            max_grad_norm=info.get("max_grad_norm", 0.5),
            
        ),
        get_num_envs(env),
        OneHotFlattenFeature,
        device=device
    ),
    
    # OptionPPOAgent.name: lambda env, info: OptionPPOAgent(
    #     get_env_action_space(env), 
    #     get_env_observation_space(env),
    #     HyperParameters(
    #         gamma=info.get("gamma", 0.99),
    #         lamda=info.get("lamda", 0.95),
    #         mini_batch_size=info.get("mini_batch_size", 64),
    #         rollout_steps=info.get("rollout_steps", 256),
    #         num_epochs=info.get("num_epochs", 5),
            
    #         clip_range_actor_init=info.get("clip_range_actor_init", 0.2),
    #         anneal_clip_range_actor=info.get("anneal_clip_range_actor", True),
    #         clip_range_critic_init=info.get("clip_range_critic_init", None), # None means no clipping
    #         anneal_clip_range_critic=info.get("anneal_clip_range_critic", False),
    #         target_kl=info.get("target_kl", 0.02),
            
    #         actor_network=NETWORKS[info.get("actor_network", "conv_network_2")],
    #         actor_step_size=info.get("actor_step_size", 3e-4),
    #         actor_eps = info.get("actor_eps", 1e-5),
    #         critic_network=NETWORKS[info.get("critic_network", "conv_network_2")],
    #         critic_step_size=info.get("critic_step_size", 3e-4),
    #         critic_eps = info.get("critic_eps", 1e-5),
            
    #         anneal_step_size_flag=info.get("anneal_step_size_flag", True),
    #         total_steps=info.get("total_steps", 2_000_000 // 256),
            
    #         norm_adv_flag=info.get("norm_adv_flag", True),
    #         critic_coef=info.get("critic_coef", 0.5),
    #         entropy_coef=info.get("entropy_coef", 0.02),
    #         max_grad_norm=info.get("max_grad_norm", 0.5),
            
    #     ),
    #     get_num_envs(env),
    #     ImageFeature,
    #     options_lst=load_options_list(info["option_path"]),
    #     device=device
    # )
}
