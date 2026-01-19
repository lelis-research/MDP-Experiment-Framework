
##################################### VQOptionCritic for MiniGrid #####################################
INFO_tier1='{
  "gamma": 0.99,
  "hl_lamda": 0.95,

  "hl_actor_network": "MiniGrid/PPO/conv_imgdircarry_actor",
  "hl_actor_eps": 1e-8,
  "hl_clip_range_actor_init": 0.2,
  "hl_anneal_clip_range_actor": false,

  "hl_critic_network": "MiniGrid/PPO/conv_imgdircarry_critic",
  "hl_critic_eps": 1e-8,
  "hl_clip_range_critic_init": null,
  "hl_anneal_clip_range_critic": false,

  "hl_critic_coef": 0.5,
  "hl_max_grad_norm": 0.5,

  "hl_target_kl": null,
  "hl_min_logstd": null,
  "hl_max_logstd": null,

  "hl_enable_stepsize_anneal": false,
  "hl_total_steps": 200000,

  "hl_enable_advantage_normalization": true,
  "hl_enable_transform_action": true,

  "codebook_embedding_dim": 2,
  "codebook_embedding_low": -1.0,
  "codebook_embedding_high": 1.0,
  "codebook_eps": 1e-5,
  "codebook_max_grad_norm": 1.0,

  "hl_rollout_steps": 1024,
  "hl_mini_batch_size": 128,
  "hl_num_epochs": 10
}'
# 540 Configs
HP_SEARCH_SPACE_tier1='{
  "hl_actor_step_size": [1e-4, 3e-4, 1e-3],
  "hl_critic_step_size": [1e-4, 3e-4, 1e-3],

  "hl_entropy_coef": [0.0, 0.0003, 0.001, 0.003, 0.01],

  "commit_coef": [0.05, 0.1, 0.2, 0.4],
  "codebook_step_size": [1e-4, 3e-4, 1e-3]
}'

INFO_tier2='{
  "gamma": 0.99,
  "hl_lamda": 0.95,

  "hl_actor_network": "MiniGrid/PPO/conv_imgdircarry_actor",
  "hl_actor_eps": 1e-8,
  "hl_clip_range_actor_init": 0.2,
  "hl_anneal_clip_range_actor": false,

  "hl_critic_network": "MiniGrid/PPO/conv_imgdircarry_critic",
  "hl_critic_eps": 1e-8,
  "hl_clip_range_critic_init": null,
  "hl_anneal_clip_range_critic": false,

  "hl_critic_coef": 0.5,
  "hl_max_grad_norm": 0.5,

  "hl_target_kl": null,
  "hl_min_logstd": null,
  "hl_max_logstd": null,

  "hl_enable_stepsize_anneal": false,
  "hl_total_steps": 200000,

  "hl_enable_advantage_normalization": true,
  "hl_enable_transform_action": true,

  "codebook_embedding_dim": 2,
  "codebook_embedding_low": -1.0,
  "codebook_embedding_high": 1.0,
  "codebook_eps": 1e-5,
  "codebook_max_grad_norm": 1.0,

  "hl_actor_step_size": FROM_TIER1,
  "hl_critic_step_size": FROM_TIER1,
  "hl_entropy_coef": FROM_TIER1,
  "commit_coef": FROM_TIER1,
  "codebook_step_size": FROM_TIER1
}'

HP_SEARCH_SPACE_tier2='{
  "hl_rollout_steps": [512, 1024, 2048],
  "hl_mini_batch_size": [64, 128, 256],
  "hl_num_epochs": [5, 10, 20]
}'

##################################### PPO / OptionPPO for MiniGrid #####################################
INFO_tier1='{
  "gamma": 0.99,
  "lamda": 0.95,

  "actor_network": "MiniGrid/PPO/conv_imgdircarry_actor",
  "critic_network": "MiniGrid/PPO/conv_imgdircarry_critic",

  "actor_eps": 1e-8,
  "critic_eps": 1e-8,

  "clip_range_actor_init": 0.2,
  "anneal_clip_range_actor": false,

  "clip_range_critic_init": null,
  "anneal_clip_range_critic": false,

  "critic_coef": 0.5,
  "max_grad_norm": 0.5,

  "min_logstd": null,
  "max_logstd": null,

  "enable_stepsize_anneal": false,
  "total_steps": 500000,

  "update_type": "per_env",
  "enable_advantage_normalization": true,
  "enable_transform_action": true,

  "target_kl": null,

  "rollout_steps": 1024,
  "mini_batch_size": 128,
  "num_epochs": 10
}'
# 45 configs
HP_SEARCH_SPACE_tier1='{
  "actor_step_size": [1e-4, 3e-4, 1e-3],
  "critic_step_size": [1e-4, 3e-4, 1e-3],
  "entropy_coef": [0.0, 0.0003, 0.001, 0.003, 0.01]
}'

INFO_tier2='{
  "gamma": 0.99,
  "lamda": 0.95,

  "actor_network": "MiniGrid/PPO/conv_imgdir_actor",
  "critic_network": "MiniGrid/PPO/conv_imgdir_critic",

  "actor_eps": 1e-8,
  "critic_eps": 1e-8,

  "clip_range_actor_init": 0.2,
  "anneal_clip_range_actor": false,

  "clip_range_critic_init": null,
  "anneal_clip_range_critic": false,

  "critic_coef": 0.5,
  "max_grad_norm": 0.5,

  "min_logstd": null,
  "max_logstd": null,

  "enable_stepsize_anneal": false,
  "total_steps": 500000,

  "update_type": "per_env",
  "enable_advantage_normalization": true,
  "enable_transform_action": true,

  "target_kl": null,

  "actor_step_size": FROM_TIER1,
  "critic_step_size": FROM_TIER1,
  "entropy_coef": FROM_TIER1
}'

HP_SEARCH_SPACE_tier2='{
  "rollout_steps": [512, 1024, 2048],
  "mini_batch_size": [64, 128, 256],
  "num_epochs": [5, 10, 20]
}'