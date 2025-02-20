import os
import pickle

from Evaluate.MultiExpAnalyzer import MultiExpAnalyzer
from Evaluate.SingleExpAnalyzer import SingleExpAnalyzer
from Environments.GetEnvironment import *
from agent_config import AGENT_DICT

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
    exp_path = "Runs/Train/MiniGrid-Empty-5x5-v0_DQN_seed[123123]_20250220_111635"
    with open(os.path.join(exp_path, "env.pkl"), "rb") as file:
        env_config = pickle.load(file)
    env = get_env(**env_config)
    agent = AGENT_DICT["DQN"](env)
    agent.reset(0)
    agent.load(os.path.join(exp_path, "Policy_Run1_Last.t"))
    print(agent.policy.network)