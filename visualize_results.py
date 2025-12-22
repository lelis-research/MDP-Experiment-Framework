from RLBase.Evaluate import plot_experiments, gather_experiments, plot_option_usage

if __name__ == "__main__":
    
    Option_Random = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/OptionRandom"
    Option_QLearning = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/OptionQLearning"
    Option_DQN = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/OptionDQN"
    Option_A2C = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/OptionA2C"
    Option_PPO = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/OptionPPO"
    
    Random = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/Random"
    QLearning = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/QLearning"
    DQN = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/DQN"
    A2C = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/A2C"
    PPO = "Runs/Train/MiniGrid-EmptyTwoGoals-6x6-v0_/FullyObs/PPO"
    
    agent_dict = {       
        # "Option_Random": gather_experiments(Option_Random, name_string_conditions=[], name_string_anti_conditions=[]),
        # "Option_QLearning": gather_experiments(Option_QLearning, name_string_conditions=[], name_string_anti_conditions=[]),
        # "Option_DQN": gather_experiments(Option_DQN, name_string_conditions=[], name_string_anti_conditions=[]),
        # "Option_A2C": gather_experiments(Option_A2C, name_string_conditions=[], name_string_anti_conditions=[]),
        # "Option_PPO": gather_experiments(Option_PPO, name_string_conditions=[], name_string_anti_conditions=[]),
        
        "Random": gather_experiments(Random, name_string_conditions=[], name_string_anti_conditions=[]),
        "QLearning": gather_experiments(QLearning, name_string_conditions=[], name_string_anti_conditions=[]),
        "DQN": gather_experiments(DQN, name_string_conditions=[], name_string_anti_conditions=[]),
        "A2C": gather_experiments(A2C, name_string_conditions=[], name_string_anti_conditions=[]),
        "PPO": gather_experiments(PPO, name_string_conditions=[], name_string_anti_conditions=[]),
    }
    
    plot_experiments(agent_dict, "Runs/Figures", name=f"Test_Agents", window_size=1, show_ci=True, ignore_last=True, 
                     plt_configs=["r_s"], plot_each=False)

    # plot_experiments(agent_dict, "Runs/Figures", name=f"test_steps", window_size=1, show_ci=True, ignore_last=True, 
    #                  plt_configs=["r_s", "ou_s", "oc_s"], plot_each=False)
    
    # plot_experiments(agent_dict, "Runs/Figures", name=f"test_dqn", window_size=1, show_ci=True, ignore_last=True, 
    #                  plt_configs=["r_s", "ou_s", "no_s"], plot_each=False)
    
    # plot_option_usage(agent_dict, "Runs/Figures", name=f"test2", window_size=1, show_ci=True, ignore_last=True,
    #                   x_type="s", plot_each=False, option_classes=ALL_OPTIONS)
    
    
    
    
    
 