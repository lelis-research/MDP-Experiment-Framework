import os
import argcomplete
import argparse
import json
from pathlib import Path

from RLBase.Trainers import OnlineTrainer
from RLBase.Evaluate import SingleExpAnalyzer
from RLBase.Environments import get_env, ENV_LST
from RLBase import load_policy, load_agent
# TODO: the test doesn't work for the OptionAgents because 
# the OnlineTrainer doesn't call the update method of agent
# which is needed for terminating an option

def discover_agent_files(train_path: Path):
    files = sorted(train_path.glob("*_agent.t"))
    if not files:
        fb = train_path / "Run1_Best_agent.t"
        if fb.exists():
            files = [fb]
    
    # files = [train_path / "Run1_Best_agent.t"]
    return files

def parse():
    parser = argparse.ArgumentParser()
    # Environment name
    parser.add_argument("--exp_dir", type=str, default=None, help="which exp")
    # Environment name
    parser.add_argument("--env", type=str, default=None, choices=ENV_LST, help="which environment")
    # List of wrappers for the environment
    parser.add_argument("--env_wrapping",   type=json.loads, default=None, help="list of wrappers")
    # A list of dictionary of the parameters for each wrapper
    parser.add_argument("--wrapping_params", type=json.loads, default=None, help="list of dictionary represeting the parameters for each wrapper")
    # A dictionary of the environment parameters
    parser.add_argument("--env_params",     type=json.loads, default=None, help="dictionary of the env parameters")
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=123123, help="Random seed for reproducibility")
    # Number of runs
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs")
    # Maximum steps per episode
    parser.add_argument("--episode_max_steps", type=int, default=None, help="maximum number of steps in each episode")
    # Render mode for the environment
    parser.add_argument("--render_mode", type=str, default="rgb_array_list", choices=[None, "human", "rgb_array_list", "ansi"], help="render mode for the environment")
    # Flag to store transitions during the experiment
    parser.add_argument("--store_transitions", action='store_true', help="store the transitions during the experiment")
    # Add a name tag
    parser.add_argument("--name_tag", type=str, default="", help="name tag for experiment folder")
    # Info for agent specification
    parser.add_argument("--info", type=json.loads, help='JSON dict, e.g. \'{"lr":0.001,"epochs":10}\'')
    
    
    argcomplete.autocomplete(parser)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    
    if args.exp_dir is None:
        exp_name = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/QLearning/_seed[123123]"
    else:
        exp_name = args.exp_dir
    
    train_path = Path(exp_name)
    test_root = Path(str(train_path).replace("Runs/Train/", "Runs/Test/", 1))


    # If saved the frames during training you can visualize them like this
    # analyzer = SingleExpAnalyzer(exp_path=train_path)
    # analyzer.generate_video(1, 1)

    exp_args = OnlineTrainer.load_args(train_path)
    config = OnlineTrainer.load_config(train_path)
    
    exp_args.env = args.env if args.env is not None else exp_args.env
    exp_args.env_params = args.env_params if args.env_params is not None else exp_args.env_params
    exp_args.env_wrapping = args.env_wrapping if args.env_wrapping is not None else exp_args.env_wrapping
    exp_args.wrapping_params = args.wrapping_params if args.wrapping_params is not None else exp_args.wrapping_params
    exp_args.episode_max_steps = args.episode_max_steps if args.episode_max_steps is not None else exp_args.episode_max_steps
    
    # exp_args.wrapping_params = [{"agent_view_size":9},{},{"seed":2000},{"num_distractors": 5, "seed": 100}]
    
    env = get_env(
        env_name=exp_args.env,
        num_envs=1,
        max_steps=exp_args.episode_max_steps,
        render_mode=args.render_mode, #args.render_mode, rgb_array_list, ansi
        env_params=exp_args.env_params,
        wrapping_lst=exp_args.env_wrapping,
        wrapping_params=exp_args.wrapping_params,
    )
    
    agent_files = discover_agent_files(train_path)
    if not agent_files:
        raise FileNotFoundError(f"No '*_agent.t' under {train_path}")
    
    for ckpt in agent_files:
        agent_name = Path(ckpt).stem                      # e.g. "Run3_Final_agent"
        test_path = test_root / agent_name                # …/Runs/Test/.../Run3_Final_agent/
        os.makedirs(test_path, exist_ok=True)

        agent = load_agent(ckpt)
        experiment = OnlineTrainer(env, agent, str(test_path), train=False, args=exp_args)
        metrics = experiment.multi_run(num_runs=args.num_runs, num_episodes=1, seed_offset=args.seed)

        analyzer = SingleExpAnalyzer(exp_path=str(test_path))
        for r in range(1, args.num_runs + 1):
            analyzer.generate_video(r, 1, fps=50, name_tag=agent_name)  # << keep the name in the filename
        analyzer.save_seeds(str(test_path))
        analyzer.plot_combined(save_dir=str(test_path))
