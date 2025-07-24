import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .SingleExpAnalyzer import SingleExpAnalyzer

def plot_experiments(agent_dict, save_dir, name="", window_size=10, plot_each=False):
    '''
    Example:
        agents_dict = {
            "label": "exp_path",
        }
        OR
        agents_dict = {
            "label": [metrics],
        }
    '''
    os.makedirs(save_dir, exist_ok=True)
    
    # num_experiments = len(agent_dict)
    # colors = plt.cm.viridis(np.linspace(0, 1, num_experiments))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    generated_name = ""
    for i, exp in enumerate(agent_dict):
        if type(agent_dict[exp]) == str:
            analyzer = SingleExpAnalyzer(exp_path=agent_dict[exp])
        elif type(agent_dict[exp]) == list:
            analyzer = SingleExpAnalyzer(metrics=agent_dict[exp])
        else:
            raise ValueError(f"Unknown param type: {type(agent_dict[exp])}")
        analyzer.plot_combined(fig, axs, color=colors[i], label=exp, show_legend=(i==len(agent_dict)-1), window_size=window_size, plot_each=plot_each) # show legend only for the last which combines all of them
        generated_name += f"{exp}_"

    name = generated_name if name == "" else name
    path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(path)


def gather_experiments(exp_dir, name_string_conditions=[], name_string_anti_conditions=[]):
    '''
    Search for the experiments in this directory for the names that includes name_string_conditions
    and return the metrics
    '''
    run_counter = 0
    file_counter = 0
    metrics_lst = []
    for exp in os.listdir(exp_dir):
        satisfy_conditions = True
        for string in name_string_conditions:
            # exp name must contain all the necessary string conditions
            if string not in exp:
                satisfy_conditions = False
                break
            satisfy_conditions = True
        
        satisfy_anti_conditions = True
        for string in name_string_anti_conditions:
            # exp name must NOT contain any of the anti conditions
            if string in exp:
                satisfy_anti_conditions = False
                break
            satisfy_anti_conditions = True


        if satisfy_conditions and satisfy_anti_conditions:
            exp_path = os.path.join(exp_dir, exp)
            analyzer = SingleExpAnalyzer(exp_path=exp_path)
            metrics_lst += analyzer.metrics

            run_counter += len(analyzer.metrics)
            file_counter += 1
    if file_counter == 0:
        raise FileNotFoundError(f"No files find with {name_string_conditions} conditions")
    
    print(f"Found {run_counter} runs from total {file_counter} files")
    return metrics_lst
