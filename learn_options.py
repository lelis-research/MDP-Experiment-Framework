 
import os
import argparse
import torch
import json
torch.multiprocessing.set_sharing_strategy('file_system')

from RLBase import load_option, load_agent, load_policy
from RLBase.Options.Utils import save_options_list, load_options_list
from Configs.loader import load_config



def parse():
    parser = argparse.ArgumentParser(description="Parser for mask search variables")
    
    # Type of options
    parser.add_argument("--option_type", type=str, default="MaskedOptionLearner", help="type of options")
    
    # Config file name
    parser.add_argument("--config", type=str, default="config_options_base", help="path to the experiment config file")
    
    # Add a name tag
    parser.add_argument("--name_tag", type=str, default="", help="name tag for experiment folder")
    
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=123123, help="Random seed for reproducibility")
    
    # Experiment paths (one or more)
    parser.add_argument("--exp_path_lst", type=str, nargs='+', required=True, help="List of experiment directory paths")
    
    # Corresponding run indices (one or more)
    parser.add_argument("--run_ind_lst", type=int, nargs='+', required=True, help="List of run indices for each experiment path")
    
    #Number of workers
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for parallelization")
    
    # Info for agent specification
    parser.add_argument("--info", type=json.loads, help='JSON dict, e.g. \'{"masked_layers":["input","1"]}\'')
    
    args = parser.parse_args()
    if len(args.exp_path_lst) != len(args.run_ind_lst):
        parser.error("--exp_path_lst and --run_ind_lst must have the same number of elements")
    return args

if __name__ == "__main__":
    args = parse()
    config_path = os.path.join("Configs", f"{args.config}.py")
    config = load_config(config_path)
    runs_dir = "Runs/Options/"
    os.makedirs(runs_dir, exist_ok=True)  
    
    exp_dir = os.path.join(runs_dir, args.option_type, args.name_tag)
    os.makedirs(exp_dir, exist_ok=True)
    
    option_learner = config.OPTION_DICT[args.option_type](args.exp_path_lst, args.run_ind_lst, args.info)
    for file in os.listdir(exp_dir):
        if file == f"selected_options_{option_learner.hyper_params.max_num_options}.t":
            print("selected options already existed")
            exit(0)
            
    options_lst = option_learner.learn(verbose=True, seed=args.seed, exp_dir=exp_dir, num_workers=args.num_workers) 

