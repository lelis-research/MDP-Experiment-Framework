"""
Reusable network presets for the graph-based builder.

These configs are dictionaries understood by NetworkGen/prepare_network_config.
Keep them simple and composable; users can copy/modify as needed.
"""

# =========================
# MiniGrid — DQN
# =========================

# --- DQN Conv trunk (one-hot image input: img) ---
# Output head "out" will be set by prepare_network_config(..., output_dim=action_dim)
MINIGRID_DQN_CONV = [
    {"type":"input",  "id":"img", "input_key":"img"},

    {"type":"conv2d", "id":"c1", "from":"img", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r1", "from":"c1"},

    {"type":"conv2d", "id":"c2", "from":"r1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r2", "from":"c2"},

    {"type":"flatten","id":"flat","from":"r2"},

    {"type":"linear", "id":"fc",  "from":"flat", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_fc","from":"fc"},

    {"type":"linear", "id":"out", "from":"r_fc",
     "init_params":{"name":"orthogonal","gain":1.0}},
]

# --- DQN Conv trunk + Noisy head ---
MINIGRID_DQN_CONV_NOISY = [
    {"type":"input",  "id":"img", "input_key":"img"},

    {"type":"conv2d", "id":"c1", "from":"img", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r1", "from":"c1"},

    {"type":"conv2d", "id":"c2", "from":"r1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r2", "from":"c2"},

    {"type":"flatten","id":"flat","from":"r2"},

    {"type":"linear", "id":"fc",  "from":"flat", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_fc","from":"fc"},

    {"type":"noisy_linear", "id":"out", "from":"r_fc", "sigma_init":0.5,
     "init_params":{"name":"orthogonal","gain":1.0}},
]

# --- DQN MLP (flattened one-hot image vector input: x) ---
MINIGRID_DQN_MLP = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"r1", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r2",  "from":"fc2"},

    {"type":"linear", "id":"out", "from":"r2",
     "init_params":{"name":"orthogonal","gain":1.0}},
]

# --- DQN MLP + Noisy head ---
MINIGRID_DQN_MLP_NOISY = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"r1", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r2",  "from":"fc2"},

    {"type":"noisy_linear", "id":"out", "from":"r2", "sigma_init":0.5,
     "init_params":{"name":"orthogonal","gain":1.0}},
]

# --- Dueling MLP (flattened one-hot image vector input: x) ---
MINIGRID_DQN_DUELING_MLP = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"r1", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r2",  "from":"fc2"},

    # Value head V(s)
    {"type":"linear", "id":"V", "from":"r2",
     "init_params":{"name":"orthogonal","gain":1.0}},

    # Advantage head A(s,a) (out_features filled by output_dims)
    {"type":"linear", "id":"A", "from":"r2",
     "init_params":{"name":"orthogonal","gain":1.0}},
]

# --- Dueling MLP + Noisy heads ---
MINIGRID_DQN_DUELING_MLP_NOISY = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"r1", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r2",  "from":"fc2"},

    {"type":"noisy_linear", "id":"V", "from":"r2", "sigma_init":0.5,
     "init_params":{"name":"orthogonal","gain":1.0}},

    {"type":"noisy_linear", "id":"A", "from":"r2", "sigma_init":0.5,
     "init_params":{"name":"orthogonal","gain":1.0}},
]

# --- Dueling Conv (one-hot image input: img) ---
MINIGRID_DQN_DUELING_CONV = [
    {"type":"input",  "id":"img", "input_key":"img"},

    {"type":"conv2d", "id":"c1", "from":"img", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r1", "from":"c1"},

    {"type":"conv2d", "id":"c2", "from":"r1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r2", "from":"c2"},

    {"type":"flatten","id":"flat","from":"r2"},

    {"type":"linear", "id":"fc",  "from":"flat", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_fc","from":"fc"},

    # Value head V(s)
    {"type":"linear", "id":"V", "from":"r_fc",
     "init_params":{"name":"orthogonal","gain":1.0}},

    # Advantage head A(s,a) (out_features filled by output_dims)
    {"type":"linear", "id":"A", "from":"r_fc",
     "init_params":{"name":"orthogonal","gain":1.0}},
]

# --- Dueling Conv + Noisy heads ---
MINIGRID_DQN_DUELING_CONV_NOISY = [
    {"type":"input",  "id":"img", "input_key":"img"},

    {"type":"conv2d", "id":"c1", "from":"img", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r1", "from":"c1"},

    {"type":"conv2d", "id":"c2", "from":"r1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r2", "from":"c2"},

    {"type":"flatten","id":"flat","from":"r2"},

    {"type":"linear", "id":"fc",  "from":"flat", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_fc","from":"fc"},

    {"type":"noisy_linear", "id":"V", "from":"r_fc", "sigma_init":0.5,
     "init_params":{"name":"orthogonal","gain":1.0}},

    {"type":"noisy_linear", "id":"A", "from":"r_fc", "sigma_init":0.5,
     "init_params":{"name":"orthogonal","gain":1.0}},
]
# =========================
# MiniGrid — PPO (actor/critic)
# =========================

# --- PPO Conv actor (discrete actions) ---
# set output_dim=action_dim
MINIGRID_PPO_CONV_ACTOR = [
    {"type":"input",  "id":"img", "input_key":"img"},

    {"type":"conv2d", "id":"c1", "from":"img", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r1", "from":"c1"},

    {"type":"conv2d", "id":"c2", "from":"r1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r2", "from":"c2"},

    {"type":"flatten","id":"flat","from":"r2"},

    {"type":"linear", "id":"fc",  "from":"flat", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_fc","from":"fc"},

    {"type":"linear", "id":"out", "from":"r_fc",
     "init_params":{"name":"orthogonal","gain":0.01}},
]

# --- PPO Conv critic ---
# set output_dim=1
MINIGRID_PPO_CONV_CRITIC = [
    {"type":"input",  "id":"img", "input_key":"img"},

    {"type":"conv2d", "id":"c1", "from":"img", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r1", "from":"c1"},

    {"type":"conv2d", "id":"c2", "from":"r1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r2", "from":"c2"},

    {"type":"flatten","id":"flat","from":"r2"},

    {"type":"linear", "id":"fc",  "from":"flat", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_fc","from":"fc"},

    {"type":"linear", "id":"out", "from":"r_fc",
     "init_params":{"name":"orthogonal","gain":1.0}},
]

# --- PPO MLP actor/critic (flattened features: x) ---
MINIGRID_PPO_MLP_ACTOR = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"r1", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r2",  "from":"fc2"},

    {"type":"linear", "id":"out", "from":"r2",
     "init_params":{"name":"orthogonal","gain":0.01}},
    
]

MINIGRID_PPO_MLP_CRITIC = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"r1", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r2",  "from":"fc2"},

    {"type":"linear", "id":"out", "from":"r2",
     "init_params":{"name":"orthogonal","gain":1.0}},
    
]


MINIGRID_PPO_CONV_IMGDIRCARRY_ACTOR = [
    # ---- inputs ----
    {"type": "input", "id": "img",   "input_key": "onehot_image"},
    {"type": "input", "id": "dir",   "input_key": "onehot_direction"},
    {"type": "input", "id": "carry", "input_key": "onehot_carrying"},

    # ---- image tower ----
    {"type":"conv2d", "id":"c1", "from":"img", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r1", "from":"c1"},

    {"type":"conv2d", "id":"c2", "from":"r1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r2", "from":"c2"},

    {"type":"flatten","id":"flat_img","from":"r2"},

    {"type":"linear", "id":"fc_img",  "from":"flat_img", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_img","from":"fc_img"},

    # ---- vector tower (dir+carry) ----
    {"type":"concat", "id":"vec_in", "from":["dir", "carry"], "dim":1, "flatten":True},

    {"type":"linear", "id":"fc_vec", "from":"vec_in", "out_features":64,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_vec","from":"fc_vec"},

    # ---- fuse ----
    {"type":"concat", "id":"fused", "from":["r_img", "r_vec"], "dim":1, "flatten":True},

    {"type":"linear", "id":"fc_fuse", "from":"fused", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_fuse","from":"fc_fuse"},

    {"type":"linear", "id":"out", "from":"r_fuse",
     "init_params":{"name":"orthogonal","gain":0.01}},
]

MINIGRID_PPO_CONV_IMGDIRCARRY_CRITIC = [
    # ---- inputs ----
    {"type": "input", "id": "img",   "input_key": "onehot_image"},
    {"type": "input", "id": "dir",   "input_key": "onehot_direction"},
    {"type": "input", "id": "carry", "input_key": "onehot_carrying"},

    # ---- image tower ----
    {"type":"conv2d", "id":"c1", "from":"img", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r1", "from":"c1"},

    {"type":"conv2d", "id":"c2", "from":"r1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r2", "from":"c2"},

    {"type":"flatten","id":"flat_img","from":"r2"},

    {"type":"linear", "id":"fc_img",  "from":"flat_img", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_img","from":"fc_img"},

    # ---- vector tower (dir+carry) ----
    {"type":"concat", "id":"vec_in", "from":["dir", "carry"], "dim":1, "flatten":True},

    {"type":"linear", "id":"fc_vec", "from":"vec_in", "out_features":64,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_vec","from":"fc_vec"},

    # ---- fuse ----
    {"type":"concat", "id":"fused", "from":["r_img", "r_vec"], "dim":1, "flatten":True},

    {"type":"linear", "id":"fc_fuse", "from":"fused", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_fuse","from":"fc_fuse"},

    {"type":"linear", "id":"out", "from":"r_fuse",
     "init_params":{"name":"orthogonal","gain":1.0}},
]

MINIGRID_PPO_CONV_IMGDIR_ACTOR = [
    # ---- inputs ----
    {"type": "input", "id": "img", "input_key": "onehot_image"},
    {"type": "input", "id": "dir", "input_key": "onehot_direction"},

    # ---- image tower ----
    {"type":"permute", "id":"img_permuted", "from":"img", "dims":[0,3,1,2]},  # BHWC -> BCHW
    {"type":"conv2d", "id":"c1", "from":"img_permuted", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r1", "from":"c1"},

    {"type":"conv2d", "id":"c2", "from":"r1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r2", "from":"c2"},

    {"type":"flatten","id":"flat_img","from":"r2"},

    {"type":"linear", "id":"fc_img", "from":"flat_img", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_img","from":"fc_img"},

    # ---- vector tower (dir only) ----
    {"type":"flatten","id":"dir_flat","from":"dir"},  # safe even if already (B,4)

    {"type":"linear", "id":"fc_dir", "from":"dir_flat", "out_features":64,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_dir","from":"fc_dir"},

    # ---- fuse ----
    {"type":"concat", "id":"fused", "from":["r_img", "r_dir"], "dim":1, "flatten":True},

    {"type":"linear", "id":"fc_fuse", "from":"fused", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_fuse","from":"fc_fuse"},

    {"type":"linear", "id":"out", "from":"r_fuse",
     "init_params":{"name":"orthogonal","gain":0.01}},
]

MINIGRID_PPO_CONV_IMGDIR_CRITIC = [
    # ---- inputs ----
    {"type": "input", "id": "img", "input_key": "onehot_image"},
    {"type": "input", "id": "dir", "input_key": "onehot_direction"},

    # ---- image tower ----
    {"type":"permute", "id":"img_permuted", "from":"img", "dims":[0,3,1,2]},  # BHWC -> BCHW
    {"type":"conv2d", "id":"c1", "from":"img_permuted", "out_channels":32, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r1", "from":"c1"},

    {"type":"conv2d", "id":"c2", "from":"r1", "out_channels":64, "kernel_size":3, "stride":1, "padding":1,
     "init_params":{"name":"kaiming_normal","nonlinearity":"relu","mode":"fan_out"}},
    {"type":"relu",   "id":"r2", "from":"c2"},

    {"type":"flatten","id":"flat_img","from":"r2"},

    {"type":"linear", "id":"fc_img", "from":"flat_img", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_img","from":"fc_img"},

    # ---- vector tower (dir only) ----
    {"type":"flatten","id":"dir_flat","from":"dir"},

    {"type":"linear", "id":"fc_dir", "from":"dir_flat", "out_features":64,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_dir","from":"fc_dir"},

    # ---- fuse ----
    {"type":"concat", "id":"fused", "from":["r_img", "r_dir"], "dim":1, "flatten":True},

    {"type":"linear", "id":"fc_fuse", "from":"fused", "out_features":256,
     "init_params":{"name":"kaiming_uniform","nonlinearity":"relu","mode":"fan_in"}},
    {"type":"relu",   "id":"r_fuse","from":"fc_fuse"},

    {"type":"linear", "id":"out", "from":"r_fuse",
     "init_params":{"name":"orthogonal","gain":1.0}},
]
# =========================
# MuJoCo — PPO (continuous)
# =========================
# Typical: tanh MLP with orthogonal init (2.0 hidden, 0.01 actor output)
MUJOCO_PPO_MLP_ACTOR_TANH = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":64,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"tanh",   "id":"t1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"t1", "out_features":64,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"tanh",   "id":"t2",  "from":"fc2"},

    {"type":"linear", "id":"out", "from":"t2",
     "init_params":{"name":"orthogonal","gain":0.01}},
]

MUJOCO_PPO_MLP_CRITIC_TANH = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":64,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"tanh",   "id":"t1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"t1", "out_features":64,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"tanh",   "id":"t2",  "from":"fc2"},

    {"type":"linear", "id":"out", "from":"t2",
     "init_params":{"name":"orthogonal","gain":1.0}},
]


# =========================
# MuJoCo — TD3 (continuous)
# =========================
# TD3 actor: state -> action (later you tanh + scale in policy code)
MUJOCO_TD3_MLP_ACTOR = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":256,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"r1", "out_features":256,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"relu",   "id":"r2",  "from":"fc2"},

    {"type":"linear", "id":"out", "from":"r2",
     "init_params":{"name":"orthogonal","gain":0.01}},
]

# TD3 critic: (x, a) -> Q
# expects input_keys "x" and "a" (you provide both in forward)
MUJOCO_TD3_MLP_CRITIC = [
    {"type":"input",  "id":"x", "input_key":"x"},
    {"type":"input",  "id":"a", "input_key":"a"},
    {"type":"concat", "id":"xa", "from":["x","a"], "dim":1, "flatten":True},

    {"type":"linear", "id":"fc1", "from":"xa", "out_features":256,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"r1", "out_features":256,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"relu",   "id":"r2",  "from":"fc2"},

    {"type":"linear", "id":"out", "from":"r2",
     "init_params":{"name":"orthogonal","gain":1.0}},
]

# =========================
# MiniGrid — VQOptionCritic (discrete)
# =========================
MINIGRID_ENCODER_MLP = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":64,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"out", "from":"r1",
     "init_params":{"name":"orthogonal","gain":0.01}},
]

MINIGRID_HL_MLP_ACTOR = [
    {"type":"input",  "id":"x", "input_key":"x"},

    {"type":"linear", "id":"fc1", "from":"x", "out_features":64,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"r1", "out_features":64,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"relu",   "id":"r2",  "from":"fc2"},

    {"type":"linear", "id":"out", "from":"r2",
     "init_params":{"name":"orthogonal","gain":0.01}},
]

# expects input_keys "x" and "a" (you provide both in forward)
MINIGRID_HL_MLP_CRITIC = [
    {"type":"input",  "id":"x", "input_key":"x"},
    {"type":"input",  "id":"o", "input_key":"o"},
    {"type":"concat", "id":"xo", "from":["x","o"], "dim":1, "flatten":True},

    {"type":"linear", "id":"fc1", "from":"xo", "out_features":64,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"relu",   "id":"r1",  "from":"fc1"},

    {"type":"linear", "id":"fc2", "from":"r1", "out_features":64,
     "init_params":{"name":"orthogonal","gain":2.0}},
    {"type":"relu",   "id":"r2",  "from":"fc2"},

    {"type":"linear", "id":"out", "from":"r2",
     "init_params":{"name":"orthogonal","gain":1.0}},
]


# =========================
# Public registry
# =========================
NETWORK_PRESETS = {
    None: None,

    # MiniGrid DQN
    "MiniGrid/DQN/conv": MINIGRID_DQN_CONV,
    "MiniGrid/DQN/conv_noisy": MINIGRID_DQN_CONV_NOISY,
    "MiniGrid/DQN/mlp": MINIGRID_DQN_MLP,
    "MiniGrid/DQN/mlp_noisy": MINIGRID_DQN_MLP_NOISY,
    "MiniGrid/DQN/dueling_mlp": MINIGRID_DQN_DUELING_MLP,
    "MiniGrid/DQN/dueling_mlp_noisy": MINIGRID_DQN_DUELING_MLP_NOISY,
    "MiniGrid/DQN/dueling_conv": MINIGRID_DQN_DUELING_CONV,
    "MiniGrid/DQN/dueling_conv_noisy": MINIGRID_DQN_DUELING_CONV_NOISY,

    # MiniGrid PPO
    "MiniGrid/PPO/conv_actor": MINIGRID_PPO_CONV_ACTOR,
    "MiniGrid/PPO/conv_critic": MINIGRID_PPO_CONV_CRITIC,
    "MiniGrid/PPO/mlp_actor": MINIGRID_PPO_MLP_ACTOR,
    "MiniGrid/PPO/mlp_critic": MINIGRID_PPO_MLP_CRITIC,
    "MiniGrid/PPO/conv_imgdircarry_actor": MINIGRID_PPO_CONV_IMGDIRCARRY_ACTOR,
    "MiniGrid/PPO/conv_imgdircarry_critic": MINIGRID_PPO_CONV_IMGDIRCARRY_CRITIC,
    "MiniGrid/PPO/conv_imgdir_actor": MINIGRID_PPO_CONV_IMGDIR_ACTOR,
    "MiniGrid/PPO/conv_imgdir_critic": MINIGRID_PPO_CONV_IMGDIR_CRITIC,

    # MuJoCo PPO
    "MuJoCo/PPO/mlp_actor_tanh": MUJOCO_PPO_MLP_ACTOR_TANH,
    "MuJoCo/PPO/mlp_critic_tanh": MUJOCO_PPO_MLP_CRITIC_TANH,

    # MuJoCo TD3
    "MuJoCo/TD3/mlp_actor": MUJOCO_TD3_MLP_ACTOR,
    "MuJoCo/TD3/mlp_critic": MUJOCO_TD3_MLP_CRITIC,
    
    # MiniGrid VQOptionCritic
    "MiniGrid/VQOptionCritic/mlp_encoder": MINIGRID_ENCODER_MLP,
    "MiniGrid/VQOptionCritic/mlp_hl_actor": MINIGRID_HL_MLP_ACTOR,
    "MiniGrid/VQOptionCritic/mlp_hl_critic": MINIGRID_HL_MLP_CRITIC,
}

if __name__ == "__main__":
    """
    Tiny smoke tests:
      1) Train the MLP preset on synthetic data.
      2) Train the small Conv+MLP preset on synthetic images + dummy vector input.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    try:
        from .NetworkFactory import NetworkGen, prepare_network_config
    except ImportError:
        # Allow running as a standalone script
        from NetworkFactory import NetworkGen, prepare_network_config

    torch.manual_seed(0)
    feature_dim = 64
    num_classes = 5

    # Synthetic supervised dataset
    x = torch.randn(512, feature_dim)
    y = torch.randint(0, num_classes, (512,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    # Build model from preset
    cfg = prepare_network_config(MLP_MEDIUM, input_dims={"x": feature_dim}, output_dim=num_classes)
    model = NetworkGen(cfg)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(x=batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch+1}: loss={loss.item():.4f}")

    # CNN smoke test
    img_shape = (3, 16, 16)
    x_img = torch.randn(512, *img_shape)
    y_img = torch.randint(0, num_classes, (512,))
    img_loader = DataLoader(TensorDataset(x_img, y_img), batch_size=32, shuffle=True)

    conv_cfg = prepare_network_config(CONV_MERGE_SMALL, input_dims={"img": img_shape, "dir_carry": num_classes}, output_dim=num_classes)
    conv_model = NetworkGen(conv_cfg)
    conv_criterion = torch.nn.CrossEntropyLoss()
    conv_optimizer = torch.optim.Adam(conv_model.parameters(), lr=1e-3)

    # Use a one-hot of the label as the vector branch input to exercise concat
    for epoch in range(200):
        for batch_x, batch_y in img_loader:
            dir_vec = torch.nn.functional.one_hot(batch_y, num_classes).float()
            conv_optimizer.zero_grad()
            logits = conv_model(img=batch_x, dir_carry=dir_vec)
            loss = conv_criterion(logits, batch_y)
            loss.backward()
            conv_optimizer.step()
        print(f"[CNN] epoch {epoch+1}: loss={loss.item():.4f}")
