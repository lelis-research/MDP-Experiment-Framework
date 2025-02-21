import argparse
import os
import datetime

from Evaluate import SingleExpAnalyzer
from Experiments import LoggerExperiment, BaseExperiment, ParallelExperiment
from Environments import get_env, ENV_LST

from config import AGENT_DICT
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        default="Random",
        choices=list(AGENT_DICT.keys()),
        help="Which agent to run"
    )
    parser.add_argument(
        "--env",
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
        "--episode_max_steps",
        type=int,
        default=500,
        help="maximum number of steps in each episode"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="number of parallel environments"
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        choices=[None, "human", "rgb_array_list"],
        help="number of parallel environments"
    )
    parser.add_argument(
        "--store_transitions",
        action='store_true',
        help="store the transitions during the experiment"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse()    
    runs_dir = "Runs/Train/"
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    
    # Env Creation
    wrapping_lst = ["ViewSize", "FlattenOnehotObj", "StepReward"] #"ViewSize", "StepReward", "FlattenOnehotObj"
    wrapping_params = [{"agent_view_size": 5}, {}, {"step_reward": -1}] #{"agent_view_size": 3}, {"step_reward": -1}, {} 
    env = get_env(
            env_name=args.env,
            num_envs=args.num_envs,
            render_mode=args.render_mode,
            max_steps=args.episode_max_steps,
            wrapping_lst=wrapping_lst,
            wrapping_params=wrapping_params,
        )    
    
    # Instantiate the agent using our factory
    agent = AGENT_DICT[args.agent](env)

    # Create and run the experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.env}_{args.agent}_seed[{args.seed}]_{timestamp}"
    exp_dir = os.path.join(runs_dir, exp_name)

    if args.num_envs == 1:
        experiment = LoggerExperiment(env, agent, exp_dir)
    else:
        experiment = ParallelExperiment(env, agent, exp_dir)

    metrics = experiment.multi_run(num_runs=args.num_runs, num_episodes=args.num_episodes, 
                                   seed_offset=args.seed, dump_transitions=args.store_transitions)

    # Analyze and plot results
    analyzer = SingleExpAnalyzer(metrics=metrics)
    analyzer.plot_combined(save_dir=exp_dir)
    analyzer.save_seeds(save_dir=exp_dir)


if __name__ == "__main__":
    main()