from RLBase.Evaluate import plot_experiments, gather_experiments, plot_option_usage
from RLBase.Evaluate import SingleExpAnalyzer

def multi_exp_results():
    VQOptionCritic = "Runs/Train/MiniGrid-EmptyTwoGoals-v0_/FullyObs/VQOptionCritic"

    
    agent_dict = {       
        "NoOption": gather_experiments(VQOptionCritic, name_string_conditions=["No_Options"], name_string_anti_conditions=[]),
        "Two-Option": gather_experiments(VQOptionCritic, name_string_conditions=["2_Options"], name_string_anti_conditions=[]),
        "Three-Option": gather_experiments(VQOptionCritic, name_string_conditions=["3_Options"], name_string_anti_conditions=["later", "very_later"]),
        "Three-Option-Later": gather_experiments(VQOptionCritic, name_string_conditions=["3_Options_later"], name_string_anti_conditions=["very_later"]),
        "Three-Option-Very-Later": gather_experiments(VQOptionCritic, name_string_conditions=["3_Options_very_later"], name_string_anti_conditions=[]),
    }
    
    plot_experiments(agent_dict, "Runs/Figures", name=f"Test_Agents", window_size=1, show_ci=True, ignore_last=True, 
                     plt_configs=["r_s"], plot_each=False)

    # plot_experiments(agent_dict, "Runs/Figures", name=f"test_steps", window_size=1, show_ci=True, ignore_last=True, 
    #                  plt_configs=["r_s", "ou_s", "oc_s"], plot_each=False)
    
    # plot_experiments(agent_dict, "Runs/Figures", name=f"test_dqn", window_size=1, show_ci=True, ignore_last=True, 
    #                  plt_configs=["r_s", "ou_s", "no_s"], plot_each=False)
    
    # plot_option_usage(agent_dict, "Runs/Figures", name=f"test2", window_size=1, show_ci=True, ignore_last=True,
    #                   x_type="s", plot_each=False, option_classes=ALL_OPTIONS)

def single_exp_results():
    exp_dir = "Runs/Train/MiniGrid-EmptyTwoGoals-v0_/FullyObs/VQOptionCritic/test_moving_emb1_seed[123123]"
    exp_dir = "Runs/Train/MazeRoomsEnv-v0_/VQOptionCritic/_seed[123123]"
    analyzer = SingleExpAnalyzer(exp_path=exp_dir)

    analyzer.plot_option_embedding(min_ep=998, max_ep=999)

if __name__ == "__main__":
    # multi_exp_results()
    single_exp_results()    
    
    
    
    
    
    
 