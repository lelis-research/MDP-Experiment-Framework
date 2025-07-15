 
import os
import argparse

from RLBase import load_option, load_agent, load_policy
from Configs.loader import load_config

def parse():
    parser = argparse.ArgumentParser(description="Parser for mask search variables")
    
    # Type of options
    parser.add_argument("--option_type", type=str, default="MaskedOptionLearner", help="type of options")
    
    # Config file name
    parser.add_argument("--config", type=str, default="config_options", help="path to the experiment config file")
    
    # Add a name tag
    parser.add_argument("--name_tag", type=str, default="", help="name tag for experiment folder")
    
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=123123, help="Random seed for reproducibility")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    config_path = os.path.join("Configs", f"{args.config}.py")
    config = load_config(config_path)
    runs_dir = "Runs/Options/"
    os.makedirs(runs_dir, exist_ok=True)  
    
    exp_dir = os.path.join(runs_dir, args.option_type, args.name_tag)
    os.makedirs(exp_dir, exist_ok=True)

    
    exp_path_lst = ["Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-1000)/A2C/0_seed[0]",
                     "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-2000)/A2C/0_seed[0]",
                     ]
    run_ind_lst = [1, 1]
    option_learner = config.OPTION_DICT[args.option_type](exp_path_lst, run_ind_lst)
    options = option_learner.learn(verbose=True, seed=args.seed) 
    exit(0)


    #Store Options
    option_path = os.path.join(exp_dir, "options.t") 
    options.save(option_path)

    #Load Options
    options = load_option(f"{option_path}_options.t")
    print("Number of Loaded Options:",  options.n)
