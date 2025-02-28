import os
import pickle
from matplotlib import pyplot as plt

from Experiments import BaseExperiment
from Evaluate import AnalyzeMultiExp, SingleExpAnalyzer
from Environments import get_env


def visualize(experiment_path, run_number, episode_number):
    analyzer = SingleExpAnalyzer(exp_path=experiment_path)
    analyzer.generate_video(run_number, episode_number)


if __name__ == "__main__":
    # agent_dict = {
    #         "qlearning": "Runs/Train/MiniGrid-Empty-5x5-v0_QLearning_seed[123123]_20250224_164320",
    #         "sarsa": "Runs/Train/MiniGrid-Empty-5x5-v0_Sarsa_seed[123123]_20250224_164336",
    #         "nstep": "Runs/Train/MiniGrid-Empty-5x5-v0_NStepQLearning_seed[123123]_20250224_171603"
    #         # Add more experiments as needed.
    #     }
    # AnalyzeMultiExp(agent_dict, "Runs/Test")
    # exit(0)


    exp_name = "MiniGrid-Empty-5x5-v0_DQN_seed[123123]_20250225_093143"
    train_path = f"Runs/Train/{exp_name}"
    test_path = f"Runs/Test/{exp_name}"
    # visualize(train_path, 1, 200)

    args = BaseExperiment.load_args(train_path)
    config = BaseExperiment.load_config(train_path)

    env = get_env(
        env_name=args.env,
        num_envs=args.num_envs,
        max_steps=args.episode_max_steps,
        render_mode=args.render_mode,
        env_params=config.env_params,
        wrapping_lst=config.env_wrapping,
        wrapping_params=config.wrapping_params,
    )

    agent = config.AGENT_DICT[args.agent](env)
    agent.reset(123123)    
    agent.load(os.path.join(train_path, "Policy_Run1_Last.t"))
    
    experiment = BaseExperiment(env, agent, test_path, train=False, args=args)
    metrics = experiment.multi_run(num_runs=1, num_episodes=1, seed_offset=123123)
    analyzer = SingleExpAnalyzer(exp_path=test_path)
    analyzer.generate_video(1, 1)
    analyzer.save_seeds(test_path)
