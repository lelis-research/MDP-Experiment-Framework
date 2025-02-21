import argparse
import os
import datetime

from Evaluate import SingleExpAnalyzer
from Experiments import LoggerExperiment, BaseExperiment, ParallelExperiment
from Environments import get_env, ENV_LST

from config import AGENT_DICT, env_wrapping, wrapping_params

def parse():
    import argparse
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
    parser.add_argument("--num_episodes", type=int, default=200, help="number of episodes in each run")
    # Maximum steps per episode
    parser.add_argument("--episode_max_steps", type=int, default=500, help="maximum number of steps in each episode")
    # Number of parallel environments
    parser.add_argument("--num_envs", type=int, default=1, help="number of parallel environments")
    # Render mode for the environment
    parser.add_argument("--render_mode", type=str, default=None, choices=[None, "human", "rgb_array_list"], help="render mode for the environment")
    # Flag to store transitions during the experiment
    parser.add_argument("--store_transitions", action='store_true', help="store the transitions during the experiment")
    # Frequency of checking saving checkpoints
    parser.add_argument("--checkpoint_freq", type=int, default=None, help="frequency of saving checkpoints")
    return parser.parse_args()

def main():
    args = parse()
    runs_dir = "Runs/Train/"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)  # Create directory if it doesn't exist
    
    # Create environment with wrappers
    env = get_env(
            env_name=args.env,
            num_envs=args.num_envs,
            render_mode=args.render_mode,
            max_steps=args.episode_max_steps,
            wrapping_lst=env_wrapping,
            wrapping_params=wrapping_params,
        )
    
    # Instantiate agent using factory
    agent = AGENT_DICT[args.agent](env)

    # Define experiment name and directory with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.env}_{args.agent}_seed[{args.seed}]_{timestamp}"
    exp_dir = os.path.join(runs_dir, exp_name)

    # Choose experiment type based on number of environments
    if args.num_envs == 1:
        experiment = LoggerExperiment(env, agent, exp_dir)
    else:
        experiment = ParallelExperiment(env, agent, exp_dir)
    
    # Run the experiment and collect metrics
    metrics = experiment.multi_run(num_runs=args.num_runs, num_episodes=args.num_episodes, 
                                   seed_offset=args.seed, dump_transitions=args.store_transitions,
                                   checkpoint_freq=args.checkpoint_freq)

    # Analyze and plot results
    analyzer = SingleExpAnalyzer(metrics=metrics)
    analyzer.plot_combined(save_dir=exp_dir)
    analyzer.save_seeds(save_dir=exp_dir)

if __name__ == "__main__":
    main()