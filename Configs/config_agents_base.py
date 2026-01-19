from RLBase.Agents.Utils import HyperParameters
from RLBase.FeatureExtractors import *
from RLBase.Agents import *
from RLBase.Networks import NETWORK_PRESETS
# from RLBase.Options import load_options_list
from RLBase.Options.SymbolicOptions.PreDesigned import *
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
    HumanAgent.name: lambda env, info: HumanAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            actions_enum=env.envs[0].unwrapped.actions
        ), #enum of the actions and their name
        get_num_envs(env),
        TabularFeature,
        device=device
    ),
    OptionHumanAgent.name: lambda env, info: OptionHumanAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            actions_enum=env.envs[0].unwrapped.actions,
            gamma=0.9,
        ), 
        get_num_envs(env),
        TabularFeature,
        init_option_lst=manual_options,
        device=device
    ),
    
    RandomAgent.name: lambda env, info: RandomAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            ),
        get_num_envs(env),
        FlattenFeature,
    ),
    OptionRandomAgent.name: lambda env, info: OptionRandomAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            ),
        get_num_envs(env),
        FlattenFeature,
        init_option_lst=[GoToRedGoalOption(), GoToGreenGoalOption()], 
    ),

    #Tabular Agents
    QLearningAgent.name: lambda env, info: QLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.1),
            gamma=info.get("gamma", 0.99),
            epsilon_start=info.get("epsilon_start", 1.0),
            epsilon_end=info.get("epsilon_end", 0.05),
            epsilon_decay_steps=info.get("epsilon_decay_steps", 40000),
            n_steps=info.get("n_steps", 3),
            
            replay_buffer_size=info.get("replay_buffer_size", 256), # give a buffer size for Dyna; None is not Dyna
            batch_size=info.get("batch_size", 1), # for the Dyna
            warmup_buffer_size=info.get("warmup_buffer_size", 128), #for the Dyna
        ),
        get_num_envs(env),
        TabularFeature,
    ),
    OptionQLearningAgent.name: lambda env, info: OptionQLearningAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            step_size=info.get("step_size", 0.1),
            gamma=info.get("gamma", 0.99),
            epsilon_start=info.get("epsilon_start", 1.0),
            epsilon_end=info.get("epsilon_end", 0.05),
            epsilon_decay_steps=info.get("epsilon_decay_steps", 10000),
            n_steps=info.get("n_steps", 5),
            
            replay_buffer_size=info.get("replay_buffer_size", 256), # give a buffer size for Dyna; None is not Dyna
            batch_size=info.get("batch_size", 1), # for the Dyna
            warmup_buffer_size=info.get("warmup_buffer_size", 10), #for the Dyna
        ),
        get_num_envs(env),
        TabularFeature,
        init_option_lst=[GoToRedGoalOption(), GoToGreenGoalOption()], 
    ),
    
    
    # Deep Agents
    DQNAgent.name: lambda env, info: DQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            n_steps=info.get("n_steps", 3), # n-step td return
            
            epsilon_start=info.get("epsilon_start", 1.0),
            epsilon_end=info.get("epsilon_end", 0.05),
            epsilon_decay_steps=info.get("epsilon_decay_steps", 15_000),
            enable_noisy_nets=info.get("enable_noisy_nets", True), # Noisy Nets instead of e-greedy
            
            replay_buffer_size=info.get("replay_buffer_size", 200_000),            
            warmup_buffer_size=info.get("warmup_buffer_size", 500),
            
            batch_size=info.get("batch_size", 64),
            update_freq=info.get("update_freq", 4), # update the network every this step
            target_update_freq=info.get("target_update_freq", 20), 
            
            
            value_network=NETWORK_PRESETS[info.get("value_network", "MiniGrid/DQN/mlp_noisy")],
            step_size=info.get("step_size", 1e-3),
            enable_double_dqn_target=info.get("enable_double_dqn_target", True), # Double DQN
            enable_dueling_networks=info.get("enable_dueling_networks", False), # Dueling Net
            enable_huber_loss=info.get("enable_huber_loss", True), # Hubber Loss
            max_grad_norm=info.get("max_grad_norm", None), # Clip Gradients
        ),
        get_num_envs(env),
        OneHotFlattenFeature,
        device=device,
    ),
    
    OptionDQNAgent.name: lambda env, info: OptionDQNAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            n_steps=info.get("n_steps", 3), # n-step td return
            
            epsilon_start=info.get("epsilon_start", 1.0),
            epsilon_end=info.get("epsilon_end", 0.05),
            epsilon_decay_steps=info.get("epsilon_decay_steps", 15_000),
            enable_noisy_nets=info.get("enable_noisy_nets", True), # Noisy Nets instead of e-greedy
            
            replay_buffer_size=info.get("replay_buffer_size", 200_000),            
            warmup_buffer_size=info.get("warmup_buffer_size", 500),
            
            batch_size=info.get("batch_size", 64),
            update_freq=info.get("update_freq", 4), # update the network every this step
            target_update_freq=info.get("target_update_freq", 20), 
            
            
            value_network=NETWORK_PRESETS[info.get("value_network", "MiniGrid/DQN/mlp_noisy")],
            step_size=info.get("step_size", 1e-3),
            enable_double_dqn_target=info.get("enable_double_dqn_target", True), # Double DQN
            enable_dueling_networks=info.get("enable_dueling_networks", False), # Dueling Net
            enable_huber_loss=info.get("enable_huber_loss", True), # Hubber Loss
            max_grad_norm=info.get("max_grad_norm", None), # Clip Gradients
        ),
        get_num_envs(env),
        OneHotFlattenFeature,
        init_option_lst=[GoToRedGoalOption(), GoToGreenGoalOption()],
        device=device
    ),
    

    A2CAgent.name: lambda env, info: A2CAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            lamda=info.get("lamda", 0.95),
            rollout_steps=info.get("rollout_steps", 32),
            
            actor_network=NETWORK_PRESETS[info.get("actor_network", "MiniGrid/PPO/mlp_actor")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            actor_eps = info.get("actor_eps", 1e-8),
            
            critic_network=NETWORK_PRESETS[info.get("critic_network", "MiniGrid/PPO/mlp_actor")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            critic_eps = info.get("critic_eps", 1e-8),
            
            critic_coef=info.get("critic_coef", 0.5),
            entropy_coef=info.get("entropy_coef", 0.0),
            max_grad_norm=info.get("max_grad_norm", 0.5),
            
            min_logstd=info.get("min_logstd", None), #None means no clipping for logstd (-20.0)
            max_logstd=info.get("max_logstd", None), #None means no clipping for logstd (+2.0)
            
            enable_stepsize_anneal=info.get("enable_stepsize_anneal", False),
            total_steps=info.get("total_steps", 200_000), # used for anealing step size
            update_type=info.get("update_type", "per_env"), # sync, per_env
            enable_advantage_normalization=info.get("enable_advantage_normalization", True),
        ),
        get_num_envs(env),
        OneHotFlattenFeature,
        device=device
    ),
    OptionA2CAgent.name: lambda env, info: OptionA2CAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            lamda=info.get("lamda", 0.95),
            rollout_steps=info.get("rollout_steps", 32),
            
            actor_network=NETWORK_PRESETS[info.get("actor_network", "MiniGrid/PPO/mlp_actor")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            actor_eps = info.get("actor_eps", 1e-8),
            
            critic_network=NETWORK_PRESETS[info.get("critic_network", "MiniGrid/PPO/mlp_actor")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            critic_eps = info.get("critic_eps", 1e-8),
            
            critic_coef=info.get("critic_coef", 0.5),
            entropy_coef=info.get("entropy_coef", 0.0),
            max_grad_norm=info.get("max_grad_norm", 0.5),
            
            min_logstd=info.get("min_logstd", None), #None means no clipping for logstd (-20.0)
            max_logstd=info.get("max_logstd", None), #None means no clipping for logstd (+2.0)
            
            enable_stepsize_anneal=info.get("enable_stepsize_anneal", False),
            total_steps=info.get("total_steps", 200_000), # used for anealing step size
            update_type=info.get("update_type", "per_env"), # sync, per_env
            enable_advantage_normalization=info.get("enable_advantage_normalization", True),
        ),
        get_num_envs(env),
        OneHotFlattenFeature,
        init_option_lst=[GoToRedGoalOption(), GoToGreenGoalOption()],
        device=device
    ),
    
    PPOAgent.name: lambda env, info: PPOAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            lamda=info.get("lamda", 0.95),
            rollout_steps=info.get("rollout_steps", 2048),
            mini_batch_size=info.get("mini_batch_size", 64),
            num_epochs=info.get("num_epochs", 10),
            target_kl=info.get("target_kl", None), #None means no early stop
            
            actor_network=NETWORK_PRESETS[info.get("actor_network", "MiniGrid/PPO/conv_imgdir_actor")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            actor_eps = info.get("actor_eps", 1e-8),
            clip_range_actor_init=info.get("clip_range_actor_init", 0.2),
            anneal_clip_range_actor=info.get("anneal_clip_range_actor", False),
            
            critic_network=NETWORK_PRESETS[info.get("critic_network", "MiniGrid/PPO/conv_imgdir_critic")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            critic_eps = info.get("critic_eps", 1e-8),
            clip_range_critic_init=info.get("clip_range_critic_init", 0.2), # None means no clipping
            anneal_clip_range_critic=info.get("anneal_clip_range_critic", False),
            
            critic_coef=info.get("critic_coef", 0.5),
            entropy_coef=info.get("entropy_coef", 0.0),
            max_grad_norm=info.get("max_grad_norm", 0.5),
            
            min_logstd=info.get("min_logstd", None), #None means no clipping for logstd
            max_logstd=info.get("max_logstd", None), #None means no clipping for logstd
            
            enable_stepsize_anneal=info.get("enable_stepsize_anneal", False),
            total_steps=info.get("total_steps", 500_000), # used for anealing step size
            update_type=info.get("update_type", "per_env"), # sync, per_env
            enable_advantage_normalization=info.get("enable_advantage_normalization", True),
            enable_transform_action=info.get("enable_transform_action", True),
            
        ),
        get_num_envs(env),
        MirrorFeature,
        device=device
    ),
    
    OptionPPOAgent.name: lambda env, info: OptionPPOAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            lamda=info.get("lamda", 0.95),
            rollout_steps=info.get("rollout_steps", 2048),
            mini_batch_size=info.get("mini_batch_size", 64),
            num_epochs=info.get("num_epochs", 10),
            target_kl=info.get("target_kl", None), #None means no early stop
            
            actor_network=NETWORK_PRESETS[info.get("actor_network", "MiniGrid/PPO/mlp_actor")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            actor_eps = info.get("actor_eps", 1e-8),
            clip_range_actor_init=info.get("clip_range_actor_init", 0.2),
            anneal_clip_range_actor=info.get("anneal_clip_range_actor", False),
            
            critic_network=NETWORK_PRESETS[info.get("critic_network", "MiniGrid/PPO/mlp_actor")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            critic_eps = info.get("critic_eps", 1e-8),
            clip_range_critic_init=info.get("clip_range_critic_init", 0.2), # None means no clipping
            anneal_clip_range_critic=info.get("anneal_clip_range_critic", False),
            
            critic_coef=info.get("critic_coef", 0.5),
            entropy_coef=info.get("entropy_coef", 0.0),
            max_grad_norm=info.get("max_grad_norm", 0.5),
            
            min_logstd=info.get("min_logstd", None), #None means no clipping for logstd
            max_logstd=info.get("max_logstd", None), #None means no clipping for logstd
            
            enable_stepsize_anneal=info.get("enable_stepsize_anneal", False),
            total_steps=info.get("total_steps", 100_000), # used for anealing step size
            update_type=info.get("update_type", "per_env"), # sync, per_env
            enable_advantage_normalization=info.get("enable_advantage_normalization", True),
            enable_transform_action=info.get("enable_transform_action", True),
        ),
        get_num_envs(env),
        OneHotFlattenFeature,
        init_option_lst=[GoToRedGoalOption(), GoToGreenGoalOption()],
        device=device
    ),
    
    TD3Agent.name: lambda env, info: TD3Agent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(
            gamma=info.get("gamma", 0.99),
            
            exploration_noise=info.get("exploration_noise", 0.1),
            initial_random_steps=info.get("initial_random_steps", 1000),
            
            replay_buffer_size=info.get("replay_buffer_size", 200_000),
            warmup_buffer_size=info.get("warmup_buffer_size", 2048),
            
            batch_size=info.get("batch_size", 64),
            num_updates=info.get("num_updates", 1),
            policy_delay=info.get("policy_delay", 2),
            
            actor_network=NETWORK_PRESETS[info.get("actor_network", "MuJoCo/TD3/mlp_actor")],
            actor_step_size=info.get("actor_step_size", 3e-4),
            actor_eps = info.get("actor_eps", 1e-8),
            need_squash=info.get("need_squash", True), # False if the model already has tanh at the end layer
            
            critic_network=NETWORK_PRESETS[info.get("critic_network", "MuJoCo/TD3/mlp_critic")],
            critic_step_size=info.get("critic_step_size", 3e-4),
            critic_eps = info.get("critic_eps", 1e-8),
            
            target_policy_noise=info.get("target_policy_noise", 0.1),
            target_policy_noise_clip=info.get("target_policy_noise_clip", 1.0),
            target_network_update_tau=info.get("target_network_update_tau", 0.1),
            max_grad_norm=info.get("max_grad_norm", 0.5),
            
        ),
        get_num_envs(env),
        FlattenFeature,
        device=device
    ),

    VQOptionCriticAgent.name: lambda env, info: VQOptionCriticAgent(
        get_env_action_space(env), 
        get_env_observation_space(env),
        HyperParameters(            
            # Encoder Params
            encoder_network=NETWORK_PRESETS[info.get("encoder_network", "MiniGrid/VQOptionCritic/mlp_encoder")],
            encoder_step_size=info.get("encoder_step_size", 3e-4),
            encoder_eps=info.get("encoder_eps", 1e-5),
            encoder_dim=info.get("encoder_dim", 256),
            
            # High Level Params
            hl = HyperParameters(
                gamma=info.get("gamma", 0.99),
                lamda=info.get("hl_lamda", 0.95),
                rollout_steps=info.get("hl_rollout_steps", 2048),
                mini_batch_size=info.get("hl_mini_batch_size", 64),
                num_epochs=info.get("hl_num_epochs", 10),
                target_kl=info.get("hl_target_kl", None), #None means no early stop
                
                actor_network=NETWORK_PRESETS[info.get("hl_actor_network", "MiniGrid/PPO/conv_imgdir_actor")],
                actor_step_size=info.get("hl_actor_step_size", 3e-4),
                actor_eps = info.get("hl_actor_eps", 1e-8),
                clip_range_actor_init=info.get("hl_clip_range_actor_init", 0.2),
                anneal_clip_range_actor=info.get("hl_anneal_clip_range_actor", False),
                
                critic_network=NETWORK_PRESETS[info.get("hl_critic_network", "MiniGrid/PPO/conv_imgdir_critic")],
                critic_step_size=info.get("hl_critic_step_size", 3e-4),
                critic_eps = info.get("hl_critic_eps", 1e-8),
                clip_range_critic_init=info.get("hl_clip_range_critic_init", 0.2), # None means no clipping
                anneal_clip_range_critic=info.get("hl_anneal_clip_range_critic", False),
                
                critic_coef=info.get("hl_critic_coef", 0.5),
                entropy_coef=info.get("hl_entropy_coef", 0.0),
                max_grad_norm=info.get("hl_max_grad_norm", 0.5),
                
                min_logstd=info.get("hl_min_logstd", None), #None means no clipping for logstd
                max_logstd=info.get("hl_max_logstd", None), #None means no clipping for logstd
                
                enable_stepsize_anneal=info.get("hl_enable_stepsize_anneal", False),
                total_steps=info.get("hl_total_steps", 200_000), # used for anealing step size
                # update_type=info.get("hl_update_type", "per_env"), # sync, per_env
                enable_advantage_normalization=info.get("hl_enable_advantage_normalization", True),
                enable_transform_action=info.get("hl_enable_transform_action", True),
                
                commit_coef = info.get("commit_coef", 0.2),
            ),
            
            
            
            # CodeBook Params
            codebook = HyperParameters(
                embedding_dim = info.get("codebook_embedding_dim", 2),
                embedding_low = info.get("codebook_embedding_low", -1),
                embedding_high = info.get("codebook_embedding_high", +1),
                
                step_size=info.get("codebook_step_size", 3e-4),
                eps=info.get("codebook_eps", 1e-5),
                max_grad_norm=info.get("codebook_max_grad_norm", 1.0)
            )
            
            
        ),
        get_num_envs(env),
        MirrorFeature,
        init_option_lst=manual_options,
        device=device
    ),
}
