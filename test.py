import os
import argcomplete
import argparse
import json

from RLBase.Experiments import BaseExperiment
from RLBase.Evaluate import SingleExpAnalyzer
from RLBase.Environments import get_env, ENV_LST
from RLBase import load_policy, load_agent

def parse():
    parser = argparse.ArgumentParser()
    # Environment name
    parser.add_argument("--env", type=str, default=None, choices=ENV_LST, help="which environment")
    # List of wrappers for the environment
    parser.add_argument("--env_wrapping",   type=json.loads, default="[]", help="list of wrappers")
    # A list of dictionary of the parameters for each wrapper
    parser.add_argument("--wrapping_params", type=json.loads, default="[]", help="list of dictionary represeting the parameters for each wrapper")
    # A dictionary of the environment parameters
    parser.add_argument("--env_params",     type=json.loads, default="{}", help="dictionary of the env parameters")
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=123123, help="Random seed for reproducibility")
    # Number of runs
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs")
    # Number of episodes per run
    parser.add_argument("--num_episodes", type=int, default=3, help="number of episodes in each run")
    # Maximum steps per episode
    parser.add_argument("--episode_max_steps", type=int, default=None, help="maximum number of steps in each episode")
    # Render mode for the environment
    parser.add_argument("--render_mode", type=str, default=None, choices=[None, "human", "rgb_array_list", "ansi"], help="render mode for the environment")
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
    
    exp_name = "MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-20)/PPO/7_seed[7]"
    train_path = f"Runs/Train/{exp_name}"
    test_path = f"Runs/Test/{exp_name}"

    # If saved the frames during training you can visualize them like this
    # analyzer = SingleExpAnalyzer(exp_path=train_path)
    # analyzer.generate_video(1, 1)

    exp_args = BaseExperiment.load_args(train_path)
    config = BaseExperiment.load_config(train_path)
    
    if args.env is not None:
        exp_args.env = args.env
        exp_args.env_params = args.env_params
        exp_args.episode_max_steps = args.episode_max_steps
        exp_args.env_wrapping = args.env_wrapping
        exp_args.wrapping_params = args.wrapping_params
    
    # exp_args.wrapping_params = [{"agent_view_size":9},{},{"seed":2000},{"num_distractors": 5, "seed": 100}]
    
    env = get_env(
        env_name=exp_args.env,
        num_envs=1,
        max_steps=exp_args.episode_max_steps,
        render_mode="ansi", #args.render_mode,
        env_params=exp_args.env_params,
        wrapping_lst=exp_args.env_wrapping,
        wrapping_params=exp_args.wrapping_params,
    )
    agent = load_agent(os.path.join(train_path, "Run1_Best_agent.t"))

    experiment = BaseExperiment(env, agent, test_path, train=False, args=exp_args)
    print(args.num_episodes, args.num_runs)
    metrics = experiment.multi_run(num_runs=args.num_runs, num_episodes=args.num_episodes, seed_offset=args.seed)
    
    analyzer = SingleExpAnalyzer(exp_path=test_path)
    for r in range(1, args.num_runs+1):
        for e in range(1, args.num_episodes+1):
            analyzer.generate_video(r, e)
    analyzer.save_seeds(test_path)
