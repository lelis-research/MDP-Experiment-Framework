import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .SingleExpAnalyzer import SingleExpAnalyzer

def AnalyzeMultiExp(agent_dict, save_dir, name_tag=""):
    '''
    Example:
        agents_dict = {
            "Best": "Runs/QLearningAgent(HyperParameters({'alpha': 0.284, 'gamma': 0.99, 'epsilon': 0.078}))_seed [2500]_20250204_163353",
            "Not Best": "Runs/QLearningAgent(HyperParameters({'alpha': 0.5, 'gamma': 0.99, 'epsilon': 0.1}))_seed [2500]_20250204_163410",
            # Add more experiments as needed.
        }
    '''
    os.makedirs(save_dir, exist_ok=True)
    
    # num_experiments = len(agent_dict)
    # colors = plt.cm.viridis(np.linspace(0, 1, num_experiments))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    name = ""
    for i, exp in enumerate(agent_dict):
        analyzer = SingleExpAnalyzer(exp_path=agent_dict[exp])
        analyzer.plot_combined(fig, axs, color=colors[i], label=exp, show_legend=(i==len(agent_dict)-1)) # show legend only for the last which combines all of them
        name += f"{exp}_"
    
    path = os.path.join(save_dir, f"{name}_{name_tag}.png")
    fig.savefig(path)

