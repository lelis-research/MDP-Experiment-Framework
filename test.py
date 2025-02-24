import os
import pickle
from matplotlib import pyplot as plt

from Experiments import BaseExperiment
from Evaluate import AnalyzeMultiExp, SingleExpAnalyzer
from Environments import get_env
from config import AGENT_DICT


def visualize(experiment_path, run_number, episode_number):
    analyzer = SingleExpAnalyzer(exp_path=experiment_path)
    analyzer.generate_video(run_number, episode_number)


if __name__ == "__main__":
    agent_dict = {
            "exp1": "Runs/Train/MiniGrid-Empty-5x5-v0_QLearning_seed[123123]_20250224_155037",
            "exp2": "Runs/Train/MiniGrid-Empty-5x5-v0_QLearning_seed[123123]_20250224_155131",
            # Add more experiments as needed.
        }
    AnalyzeMultiExp(agent_dict, "Runs/Test")
    exit(0)


    agent_type = "PPO"
    exp_name = "MiniGrid-Empty-5x5-v0_PPO_seed[123123]_20250220_145005"
    train_path = f"Runs/Train/{exp_name}"
    test_path = f"Runs/Test/{exp_name}"
    visualize(train_path, 1, 200)

    with open(os.path.join(train_path, "env.pkl"), "rb") as file:
        env_config = pickle.load(file)
    env = get_env(**env_config, render_mode="rgb_array_list")
    agent = AGENT_DICT[agent_type](env)
    agent.reset(123123)    
    agent.load(os.path.join(train_path, "Policy_Run1_Last.t"))
    
    experiment = BaseExperiment(env, agent, test_path, train=False)
    metrics = experiment.multi_run(num_runs=1, num_episodes=1, seed_offset=123123)
    analyzer = SingleExpAnalyzer(exp_path=test_path)
    analyzer.generate_video(1, 1)
    analyzer.save_seeds(test_path)
