from RLBase.Evaluate import plot_experiments, gather_experiments

if __name__ == "__main__":
    
    ContinualOptionQLearning_8x8 = "Runs/Train/MiniGrid-DoorKey-8x8-v0_/FullyObs_FixedSeed(seed-10)/ContinualOptionQLearning"
    OptionQLearning_8x8 = "Runs/Train/MiniGrid-DoorKey-8x8-v0_/FullyObs_FixedSeed(seed-10)/OptionQLearning"
    
    QLearning_Seq = "Runs/Train/SequentialDiagonalGoalsEnv-v0_/FullyObs_FixedSeed(seed-10)/QLearning"

    
    agent_dict = {
        # "Continual_QLearning_F": gather_experiments(ContinualOptionQLearning_8x8, name_string_conditions=["Discount-true_Option-len-20_Update-action-false_"], name_string_anti_conditions=[]),
        # "Continual_QLearning_T": gather_experiments(ContinualOptionQLearning_8x8, name_string_conditions=["Discount-true_Option-len-20_Update-action-true_"], name_string_anti_conditions=[]),
        # "Option_QLearning_F": gather_experiments(OptionQLearning_8x8, name_string_conditions=["Discount-true_Option-len-20_Update-action-false_again_"], name_string_anti_conditions=[]),     
        # "Option_QLearning_T": gather_experiments(OptionQLearning_8x8, name_string_conditions=["Discount-true_Option-len-20_Update-action-true_"], name_string_anti_conditions=[]),     
        
        "QLearning_1": gather_experiments(QLearning_Seq, name_string_conditions=["n_steps-1_"], name_string_anti_conditions=[]),
        "QLearning_5": gather_experiments(QLearning_Seq, name_string_conditions=["n_steps-5_"], name_string_anti_conditions=[]),
        "QLearning_10": gather_experiments(QLearning_Seq, name_string_conditions=["n_steps-10_"], name_string_anti_conditions=[]),
        "QLearning_20": gather_experiments(QLearning_Seq, name_string_conditions=["n_steps-20_"], name_string_anti_conditions=[]),    
    }

    plot_experiments(agent_dict, "Runs/Figures", name=f"Diag_Seq_again", window_size=1, show_ci=True, ignore_last=True, 
                     plt_configs=["r_s", "r_e"], plot_each=False)
