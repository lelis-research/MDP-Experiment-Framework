from Evaluate.MultiExpAnalyzer import MultiExpAnalyzer
from Evaluate.SingleExpAnalyzer import SingleExpAnalyzer
import os
import pickle

def compare_agents(agents_dict):
    analyzer = MultiExpAnalyzer(agents_dict)
    analyzer.load_metrics()  # Loads metrics.pkl from each experiment directory.
    analyzer.plot_combined(save_path="Runs/combined_rewards.png")

def visualize(experiment_path, run_number, episode_number):
    analyzer = SingleExpAnalyzer(exp_path=experiment_path)
    analyzer.generate_video(run_number, episode_number)

if __name__ == "__main__":
    # Define a dictionary where keys are experiment names and values are experiment directories.
    agents_dict = {
        "Best": "Runs/QLearningAgent(HyperParameters({'alpha': 0.284, 'gamma': 0.99, 'epsilon': 0.078}))_seed [2500]_20250204_163353",
        "Not Best": "Runs/QLearningAgent(HyperParameters({'alpha': 0.5, 'gamma': 0.99, 'epsilon': 0.1}))_seed [2500]_20250204_163410",
        # Add more experiments as needed.
    }
    experiment_path = "Runs/Random_MiniGrid-Empty-5x5-v0_seed[123123]_20250213_144316"
    visualize(experiment_path, run_number=1, episode_number=1)