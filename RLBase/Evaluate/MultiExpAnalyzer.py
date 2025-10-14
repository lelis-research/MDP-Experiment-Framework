import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gc

from .SingleExpAnalyzer import SingleExpAnalyzer

plt.rcParams.update({
    "font.size": 24,            # base font size
    # "axes.titlesize": 16,       # title
    # "axes.labelsize": 16,       # x and y labels
    # "xtick.labelsize": 14,      # x tick labels
    # "ytick.labelsize": 14,      # y tick labels
    "legend.fontsize": 16,      # legend
    "figure.titlesize": 24      # overall figure title
})


def plot_experiments(agent_dict, save_dir, name="", window_size=10, plot_each=False, show_ci=False, ignore_last=False, plt_configs=["r_e", "r_s", "s_e"]):
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
    colors = ['red', 'black', 'green', 'purple', 'blue', 'orange', 'brown', 'pink', 'grey', 'cyan']
    markers = ['--', '-', '-.', ':*', '-o', ':s', '--^', '-.D', '-.v', ':p', '-X']
    fig, axs = plt.subplots(len(plt_configs), 1, figsize=(10, 6*len(plt_configs)), constrained_layout=True)
    generated_name = ""
    for i, exp in enumerate(agent_dict):
        if type(agent_dict[exp]) == str:
            analyzer = SingleExpAnalyzer(exp_path=agent_dict[exp])
        elif type(agent_dict[exp]) == list:
            analyzer = SingleExpAnalyzer(metrics=agent_dict[exp])
        else:
            raise ValueError(f"Unknown param type: {type(agent_dict[exp])}")
        analyzer.plot_combined(fig, axs, color=colors[i], marker=markers[i], label=exp, 
                               show_legend=(i==len(agent_dict)-1), window_size=window_size, 
                               plot_each=plot_each, show_ci=show_ci, 
                               title=name, ignore_last=ignore_last, plt_configs=plt_configs) # show legend only for the last which combines all of them
        generated_name += f"{exp}_"
        

    name = generated_name if name == "" else name
    path = os.path.join(save_dir, f"{name}.png")

    fig.savefig(path, format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)


def gather_experiments(exp_dir, name_string_conditions=[], name_string_anti_conditions=[]):
    '''
    Search for the experiments in this directory for the names that includes name_string_conditions j
    and return the metrics
    '''
    run_counter = 0
    file_counter = 0
    metrics_lst = []
    no_metrics_file_lst= []
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
            try:
                analyzer = SingleExpAnalyzer(exp_path=exp_path)
                metrics_lst += analyzer.metrics
                run_counter += len(analyzer.metrics)
                file_counter += 1
                
            except FileNotFoundError:
                # The metrics file doesn't exists
                no_metrics_file_lst.append(exp_path)
                
    print("***")
    if file_counter == 0:
        raise FileNotFoundError(f"No files find with {name_string_conditions} conditions")
    elif len(no_metrics_file_lst) > 0:
        print("The following files satisfies the naming but doesn't have a metrics file in them:")
        print(*no_metrics_file_lst, sep="\n")
    
    print(f"Found {run_counter} runs from total {file_counter} files")
    return metrics_lst
