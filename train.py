import argparse
import os
import datetime
import argcomplete
from functools import partial

from RLBase.Evaluate import SingleExpAnalyzer
from RLBase.Experiments import LoggerExperiment, BaseExperiment, ParallelExperiment
from RLBase.Environments import get_env, ENV_LST
from config import AGENT_DICT, env_wrapping, wrapping_params, env_params

def parse():
    parser = argparse.ArgumentParser()
    # Agent type to run
    parser.add_argument("--agent", type=str, default="Random", choices=list(AGENT_DICT.keys()), help="Which agent to run")
    # Environment name
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0", choices=ENV_LST, help="which environment")
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=123123, help="Random seed for reproducibility")
    # Number of runs
    parser.add_argument("--num_runs", type=int, default=3, help="number of runs")
    # Number of episodes per run
    parser.add_argument("--num_episodes", type=int, default=0, help="number of episodes in each run")
    # Number of total environment steps per run
    parser.add_argument("--total_steps", type=int, default=0, help="number of episodes in each run")
    # Maximum steps per episode
    parser.add_argument("--episode_max_steps", type=int, default=None, help="maximum number of steps in each episode")
    # Number of parallel environments
    parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments")
    # Render mode for the environment
    parser.add_argument("--render_mode", type=str, default=None, choices=[None, "human", "rgb_array_list"], help="render mode for the environment")
    # Flag to store transitions during the experiment
    parser.add_argument("--store_transitions", action='store_true', help="store the transitions during the experiment")
    # Frequency of checking saving checkpoints
    parser.add_argument("--checkpoint_freq", type=int, default=None, help="frequency of saving checkpoints")
    # Add a name tag
    parser.add_argument("--name_tag", type=str, default="", help="name tag for experiment folder")
    # Number of parallel workers
    parser.add_argument("--num_workers", type=int, default=1, help="number of parallel workers")
    
    argcomplete.autocomplete(parser)
    return parser.parse_args()

def main():
    args = parse()
    runs_dir = "Runs/Train/"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)  # Create directory if it doesn't exist
    
    env_fn = partial(
        get_env,
        env_name     = args.env,
        num_envs     = args.num_envs,
        max_steps    = args.episode_max_steps,
        render_mode  = args.render_mode,
        env_params   = env_params,
        wrapping_lst = env_wrapping,
        wrapping_params = wrapping_params,
        )
    # Instantiate agent using factory
    agent_fn = lambda env: AGENT_DICT[args.agent](env)


    # Define experiment name and directory with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.name_tag}_seed[{args.seed}]_{timestamp}"
    exp_dir = os.path.join(runs_dir, f"{args.env}_{env_params}", args.agent, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Choose experiment type based on number of environments
    if args.num_envs == 1:
        experiment = LoggerExperiment(env_fn, agent_fn, exp_dir, config="config.py", args=args)
    else:
        experiment = ParallelExperiment(env_fn, agent_fn, exp_dir)
    
    # Run the experiment and collect metrics
    metrics = experiment.multi_run(num_runs=args.num_runs, num_episodes=args.num_episodes, total_steps=args.total_steps,
                                   seed_offset=args.seed, dump_transitions=args.store_transitions, 
                                   checkpoint_freq=args.checkpoint_freq, num_workers=args.num_workers)

    # Analyze and plot results
    analyzer = SingleExpAnalyzer(metrics=metrics)
    analyzer.plot_combined(save_dir=exp_dir, show_legend=False)
    analyzer.save_seeds(save_dir=exp_dir)

if __name__ == "__main__":
    main()