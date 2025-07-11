from Configs.base_config import *



fc_network_1 = [
    {"type": "linear", "out_features": 64},
    {"type": "relu"},
    {"type": "linear", "in_features": 64},

]


env_wrapping= ["ViewSize", "FlattenOnehotObj", "FixedSeed"]
wrapping_params = [{"agent_view_size": 9}, {}, {"seed": 8000}]
env_params = {}

device="cpu" # cpu, mps, cuda

AGENT_DICT = {
       A2CAgent.name: lambda env: A2CAgent(
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
}