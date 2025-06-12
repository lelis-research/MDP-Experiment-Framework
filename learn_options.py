 
import os
import argparse

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
            # For each episode, create a trajectory by extracting the observation and action.
            trajectory = [(obs, action) for obs, action, *_ in episode]
            if len(trajectory) > 0:
                trajectories.append(trajectory)
    return trajectories

def parse():
    parser = argparse.ArgumentParser(description="Parser for mask search variables")
    
    # Number of options (e.g., different strategies or configurations)
    parser.add_argument("--num_options", type=int, default=1, 
                        help="Number of options (default: 1)")
    
    # Masked layers as a list (use space-separated values on the command line)
    parser.add_argument("--masked_layers", nargs='+', default=["input"],
                        help="List of layers to mask (default: ['input'])")
    
    # Run index for identification purposes
    parser.add_argument("--run_ind", type=int, default=1,
                        help="Run index (default: 1)")
    
    # Last percentage of trajectories to consider
    parser.add_argument("--last_percentage_of_trajectories", type=int, default=100,
                        help="Percentage of trajectories to consider from the end (default: 100)")
    
    # Search budget, e.g., number of function evaluations
    parser.add_argument("--search_budget", type=int, default=100,
                        help="Search budget (default: 100 evaluations)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    num_options = args.num_options
    masked_layers=args.masked_layers
    run_ind = args.run_ind
    last_percentage_of_trajectories = args.last_percentage_of_trajectories
    search_budget = args.search_budget

    exp_path = "Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 20}/DQN/_seed[123123]_20250312_092541"

    args = BaseExperiment.load_args(exp_path)
    
    # Load Agent
    agent = load_agent(os.path.join(exp_path, f"Run{run_ind}_Last_agent.t"))

    # Load Transitions
    all_transitions = BaseExperiment.load_transitions(exp_path)
    trajectories = extract_trajectories([all_transitions[run_ind - 1]])
    print(f"{len(trajectories)}: all trajectories")
    
    # Learning Options
    start_ind = int(-(last_percentage_of_trajectories * len(trajectories) / 100))
    print(f"{len(trajectories[start_ind:])}: used trajectories")
    option_learner = LevinLossMaskedOptionLearner(agent.action_space, agent.observation_space, agent.policy, trajectories[start_ind:], agent.feature_extractor)
    options = option_learner.learn(num_options=num_options, search_budget=search_budget, verbose=True, masked_layers=masked_layers)

    #Store Options
    option_path = os.path.join(exp_path, f"R{run_ind}_T{last_percentage_of_trajectories}_N{num_options}_L{masked_layers}_S{search_budget}") 
    options.save(option_path)

    #Load Options
    options = load_option(f"{option_path}_options.t")
    print("Number of Loaded Options:",  options.n)
