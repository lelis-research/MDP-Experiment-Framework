from RLBase.Evaluate import plot_experiments, gather_experiments, plot_option_usage

if __name__ == "__main__":
    
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
    
    
    
    
    
 