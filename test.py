import os
import argcomplete
import argparse
import json

from RLBase.Experiments import BaseExperiment
from RLBase.Evaluate import SingleExpAnalyzer
from RLBase.Environments import get_env, ENV_LST
from RLBase import load_policy, load_agent

# def parse():
#     parser = argparse.ArgumentParser()
#     # Environment name
#     parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0", choices=ENV_LST, help="which environment")
#     # List of wrappers for the environment
#     parser.add_argument("--env_wrapping",   type=json.loads, default="[]", help="list of wrappers")
#     # A list of dictionary of the parameters for each wrapper
#     parser.add_argument("--wrapping_params", type=json.loads, default="[]", help="list of dictionary represeting the parameters for each wrapper")
#     # A dictionary of the environment parameters
#     parser.add_argument("--env_params",     type=json.loads, default="{}", help="dictionary of the env parameters")
#     # Random seed for reproducibility
#     parser.add_argument("--seed", type=int, default=123123, help="Random seed for reproducibility")
#     # Number of runs
#     parser.add_argument("--num_runs", type=int, default=1, help="number of runs")
#     # Number of episodes per run
#     parser.add_argument("--num_episodes", type=int, default=5, help="number of episodes in each run")
#     # Maximum steps per episode
#     parser.add_argument("--episode_max_steps", type=int, default=None, help="maximum number of steps in each episode")
#     # Render mode for the environment
#     parser.add_argument("--render_mode", type=str, default=None, choices=[None, "human", "rgb_array_list"], help="render mode for the environment")
#     # Flag to store transitions during the experiment
#     parser.add_argument("--store_transitions", action='store_true', help="store the transitions during the experiment")
#     # Add a name tag
#     parser.add_argument("--name_tag", type=str, default="", help="name tag for experiment folder")
#     # Info for agent specification
#     parser.add_argument("--info", type=json.loads, help='JSON dict, e.g. \'{"lr":0.001,"epochs":10}\'')
    
    
#     argcomplete.autocomplete(parser)
#     return parser.parse_args()

if __name__ == "__main__":
    exp_name = "MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-1000)/A2C/1_seed[1]"
    train_path = f"Runs/Train/{exp_name}"
    test_path = f"Runs/Test/{exp_name}"

    # If saved the frames during training you can visualize them like this
    # analyzer = SingleExpAnalyzer(exp_path=train_path)
    # analyzer.generate_video(1, 1)

    args = BaseExperiment.load_args(train_path)
    config = BaseExperiment.load_config(train_path)
    
    args.env = "MiniGrid-FourRooms-v0"
    args.env_wrapping = ["ViewSize","FlattenOnehotObj","FixedSeed"]#,"FixedRandomDistractor"]
    args.wrapping_params = [{"agent_view_size":9},{},{"seed":5000}]#,{"num_distractors": 30, "seed": 100}]
    env = get_env(
        env_name=args.env,
        num_envs=args.num_envs,
        max_steps=args.episode_max_steps,
        render_mode="rgb_array_list", #args.render_mode,
        env_params=args.env_params,
        wrapping_lst=args.env_wrapping,
        wrapping_params=args.wrapping_params,
    )
    agent = load_agent(os.path.join(train_path, "Run1_Last_agent.t"))
    
    # policy = load_policy(os.path.join(train_path, "Run1_Last_policy.t"))
    # print(policy)
    # agent.hp.update(epsilon=0.0)

    experiment = BaseExperiment(env, agent, test_path, train=False, args=args)
    metrics = experiment.multi_run(num_runs=1, num_episodes=5, seed_offset=1)
    analyzer = SingleExpAnalyzer(exp_path=test_path)
    analyzer.generate_video(1, 1)
    analyzer.generate_video(1, 2)
    analyzer.generate_video(1, 3)
    analyzer.generate_video(1, 4)
    analyzer.generate_video(1, 5)


    analyzer.save_seeds(test_path)
