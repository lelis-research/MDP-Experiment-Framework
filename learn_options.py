 
import os

from RLBase.Experiments import BaseExperiment
from RLBase.Environments import get_env
from RLBase.Options.MaskedOptions import LevinLossMaskedOptionLearner
from RLBase import load_option, load_agent, load_policy


def extract_trajectories(transitions):
    trajectories = []
    # Iterate over each run in the transitions data.
    for run in transitions:
        # Each run can contain multiple episodes.
        for episode in run:
            # For each episode, create a trajectory by extracting the state and action.
            trajectory = [(obs, action) for obs, action, *_ in episode]
            if len(trajectory) > 0:
                trajectories.append(trajectory)
    return trajectories

if __name__ == "__main__":
    
    exp_path = "Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 20}_DQN_seed[123123]_20250310_185250"
    
    args = BaseExperiment.load_args(exp_path)
    config = BaseExperiment.load_config(exp_path)

    # # Load Environment
    # env = get_env(
    #     env_name=args.env,
    #     num_envs=args.num_envs,
    #     max_steps=args.episode_max_steps,
    #     render_mode=args.render_mode,
    #     env_params=config.env_params,
    #     wrapping_lst=config.env_wrapping,
    #     wrapping_params=config.wrapping_params,
    # )
    
    # Load Agent
    agent = load_agent(os.path.join(exp_path, "Run1_Last_agent.t"))

    # Load Transitions
    all_transitions = BaseExperiment.load_transitions(exp_path)
    trajectories = extract_trajectories(all_transitions)
    print(f"{len(trajectories)} many trajectories extracted")
    
    # Learning Options
    option_learner = LevinLossMaskedOptionLearner(agent.action_space, agent.observation_space, agent.policy, trajectories[-10:], agent.feature_extractor)
    options = option_learner.learn(num_options=5, search_budget=10, verbose=True, masked_layers=None)

    #Store Options
    option_path = os.path.join(exp_path, "Run1_Last_all") 
    options.save(option_path)

    #Load Options
    options = load_option(f"{option_path}_options.t")
    print("Number of Loaded Options:",  options.n)
