# main.py
import argparse
import os
import datetime
from Evaluate.SingleExpAnalyzer import SingleExpAnalyzer
from Experiments.LoggerExperiment import LoggerExperiment
from Experiments.BaseExperiment import BaseExperiment
from Experiments.ParallelExperiment import ParallelExperiment
from Environments.MiniGrid.GetEnvironment import *
from config import AGENT_DICT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        default="Random",
        choices=list(AGENT_DICT.keys()),
        help="Which agent to run"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="MiniGrid-Empty-5x5-v0",
        choices=ENV_LST,
        help="which environment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123123,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="number of runs"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=200,
        help="number of episode in each run"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="number of parallel environments"
    )
    
    args = parser.parse_args()
    runs_dir = "Runs/"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.num_envs == 1:
        env = get_single_env(
            env_name=args.env_name,
            render_mode="rgb_array_list", # human, rgb_array_list
            max_steps=200,
            wrapping_lst=["ViewSize", "FlattenOnehotObj"],
            wrapping_params=[{"agent_view_size": 3}, {}],
        )
        experiment_class = BaseExperiment
    elif args.num_envs > 1:
        env = get_parallel_env(
            env_name=args.env_name,
            num_envs=args.num_envs,
            render_mode=None,
            max_steps=200,
            wrapping_lst=["ViewSize", "StepReward", "FlattenOnehotObj"],
            wrapping_params=[{"agent_view_size": 3}, {"step_reward": -1}, {},],
        )
        experiment_class = ParallelExperiment

    # Instantiate the agent using our factory
    agent = AGENT_DICT[args.agent](env)

    # Create and run the experiment
    exp_name = f"{args.agent}_{args.env_name}_seed[{args.seed}]_{timestamp}"
    exp_dir = os.path.join(runs_dir, exp_name)

    experiment = experiment_class(env, agent, exp_dir)
    metrics = experiment.multi_run(num_runs=args.num_runs, num_episodes=args.num_episodes, seed_offset=args.seed)

    # Analyze and plot results
    analyzer = SingleExpAnalyzer(metrics=metrics)
    analyzer.plot_combined(save_dir=exp_dir)
    analyzer.save_seeds(save_dir=exp_dir)


if __name__ == "__main__":
    main()