import os
import pickle

from Experiments.BaseExperiment import BaseExperiment
from Evaluate.MultiExpAnalyzer import MultiExpAnalyzer
from Evaluate.SingleExpAnalyzer import SingleExpAnalyzer
from Environments.GetEnvironment import *
from agent_config import AGENT_DICT
from Agents.Utils.HyperParams import HyperParameters

def compare_agents(agents_dict):
    '''
    Example:
        agents_dict = {
            "Best": "Runs/QLearningAgent(HyperParameters({'alpha': 0.284, 'gamma': 0.99, 'epsilon': 0.078}))_seed [2500]_20250204_163353",
            "Not Best": "Runs/QLearningAgent(HyperParameters({'alpha': 0.5, 'gamma': 0.99, 'epsilon': 0.1}))_seed [2500]_20250204_163410",
            # Add more experiments as needed.
        }
    '''
    analyzer = MultiExpAnalyzer(agents_dict)
    analyzer.load_metrics()  # Loads metrics.pkl from each experiment directory.
    analyzer.plot_combined(save_path="Runs/combined_rewards.png")

def visualize(experiment_path, run_number, episode_number):
    analyzer = SingleExpAnalyzer(exp_path=experiment_path)
    analyzer.generate_video(run_number, episode_number)

if __name__ == "__main__":
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
