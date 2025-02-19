import os
import pickle

from Evaluate.MultiExpAnalyzer import MultiExpAnalyzer
from Evaluate.SingleExpAnalyzer import SingleExpAnalyzer
from Environments.MiniGrid.GetEnvironment import *
from config import AGENT_DICT

def compare_agents(agents_dict):
    analyzer = MultiExpAnalyzer(agents_dict)
    analyzer.load_metrics()  # Loads metrics.pkl from each experiment directory.
    analyzer.plot_combined(save_path="Runs/combined_rewards.png")

def visualize(experiment_path, run_number, episode_number):
    analyzer = SingleExpAnalyzer(exp_path=experiment_path)
    analyzer.generate_video(run_number, episode_number)

if __name__ == "__main__":
    
    '''
    # *****************   COMPARISON ANALYSIS   ********************
    # Define a dictionary where keys are experiment names and values are experiment directories.
    agents_dict = {
        "Best": "Runs/QLearningAgent(HyperParameters({'alpha': 0.284, 'gamma': 0.99, 'epsilon': 0.078}))_seed [2500]_20250204_163353",
        "Not Best": "Runs/QLearningAgent(HyperParameters({'alpha': 0.5, 'gamma': 0.99, 'epsilon': 0.1}))_seed [2500]_20250204_163410",
        # Add more experiments as needed.
    }
    compare_agents(agents_dict)
    '''

    '''
    # *****************   VISUALIZE ANALYSIS   ********************
    experiment_path = "Runs/MiniGrid-Empty-5x5-v0_NStepDQN_seed[123123]_20250219_131902"
    visualize(experiment_path, run_number=1, episode_number=20)
    '''
    
    # *****************   LOAD ANALYSIS   ********************
    wrapping_lst = ["ViewSize", "StepReward", "FlattenOnehotObj"] #"ViewSize", "StepReward", "FlattenOnehotObj"
    wrapping_params = [{"agent_view_size": 3}, {"step_reward": -1}, {}] #{"agent_view_size": 3}, {"step_reward": -1}, {} 
    env_max_step = 200
    env_name = "MiniGrid-Empty-5x5-v0"
    agent_name = "DQN"
    exp_path = "Runs/MiniGrid-Empty-5x5-v0_['ViewSize', 'StepReward', 'FlattenOnehotObj']_DQN_seed[123123]_20250219_154112"
    analyzer = SingleExpAnalyzer(exp_path=exp_path)
    print(len(analyzer.transitions), len(analyzer.transitions[0]), analyzer.transitions[0][0])
    
    env = get_single_env(
        env_name=env_name,
        render_mode="rgb_array_list", # human, rgb_array_list
        max_steps=env_max_step,
        wrapping_lst=wrapping_lst,
        wrapping_params=wrapping_params,
    )
    print(env.observation_space)
    agent = AGENT_DICT[agent_name](env)
    agent.reset(0)
    agent.load(os.path.join(exp_path, "Policy_R1_Last.t"))
    print(agent.policy.network)