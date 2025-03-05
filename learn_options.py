 
import os

from Experiments import BaseExperiment
from Environments import get_env
from OfflineOptionLearner.MaskOptions import MaskedOptionLearner_v1


if __name__ == "__main__":

    exp_path = "Runs/Train/MiniGrid-Empty-5x5-v0_{}_DQN_seed[123123]_20250227_140907"
    # exp_path = "Runs/Train/MiniGrid-Empty-5x5-v0_{}_DQN_seed[123123]_20250227_160949"

    args = BaseExperiment.load_args(exp_path)
    config = BaseExperiment.load_config(exp_path)

    # Load Environment
    env = get_env(
        env_name=args.env,
        num_envs=args.num_envs,
        max_steps=args.episode_max_steps,
        render_mode=args.render_mode,
        env_params=config.env_params,
        wrapping_lst=config.env_wrapping,
        wrapping_params=config.wrapping_params,
    )

    # Load Agent
    agent = config.AGENT_DICT[args.agent](env)
    agent.reset(args.seed)    
    agent.load(os.path.join(exp_path, "Policy_Run1_Last.t"))

    # Load Transitions
    all_transitions = BaseExperiment.load_transitions(exp_path)

    # Learning Options
    option_learner = MaskedOptionLearner_v1(agent, all_transitions)
    options_lst = option_learner.random_search(masked_layers=["1"], num_options=5, iteration=10)

    options_dir = os.path.join(exp_path, "options")
    if not os.path.exists(options_dir):
        os.makedirs(options_dir) 

    for e, option in enumerate(options_lst):
        option.save(f"{options_dir}/{e}")
    print(options_lst)