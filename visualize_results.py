from RLBase.Evaluate import plot_experiments, gather_experiments

if __name__ == "__main__":
    OptionQLearning_5x5 = "Runs/Train/MiniGrid-DoorKey-5x5-v0_/FullyObs_FixedSeed(seed-10)/OptionQLearning"
    OptionQLearning_8x8 = "Runs/Train/MiniGrid-DoorKey-8x8-v0_/FullyObs_FixedSeed(seed-10)/OptionQLearning"
    QLearning_5x5 = "Runs/Train/MiniGrid-DoorKey-5x5-v0_/FullyObs_FixedSeed(seed-10)/QLearning"
    QLearning_8x8 = "Runs/Train/MiniGrid-DoorKey-8x8-v0_/FullyObs_FixedSeed(seed-10)/QLearning"
    agent_dict = {
        # "Gamma_1": gather_experiments(OptionQLearning_5x5, name_string_conditions=["Effective_Discount-false_option_len-1_"], name_string_anti_conditions=[]),
        # "Gamma_10": gather_experiments(OptionQLearning_5x5, name_string_conditions=["Effective_Discount-false_option_len-10_"], name_string_anti_conditions=[]),
        # "Gamma_100": gather_experiments(OptionQLearning_5x5, name_string_conditions=["Effective_Discount-false_option_len-100_"], name_string_anti_conditions=[]),
        # "Discounted_1": gather_experiments(OptionQLearning_5x5, name_string_conditions=["Effective_Discount-true_option_len-1_"], name_string_anti_conditions=[]),
        # "Discounted_10": gather_experiments(OptionQLearning_5x5, name_string_conditions=["Effective_Discount-true_option_len-10_"], name_string_anti_conditions=[]),
        # "Discounted_100": gather_experiments(OptionQLearning_5x5, name_string_conditions=["Effective_Discount-true_option_len-100_"], name_string_anti_conditions=[]),
        # "QLearning": gather_experiments(QLearning_5x5, name_string_conditions=[], name_string_anti_conditions=[]),
        
        "Gamma_1": gather_experiments(OptionQLearning_8x8, name_string_conditions=["Effective_Discount-false_option_len-1_"], name_string_anti_conditions=[]),
        "Gamma_10": gather_experiments(OptionQLearning_8x8, name_string_conditions=["Effective_Discount-false_option_len-10_"], name_string_anti_conditions=[]),
        "Gamma_100": gather_experiments(OptionQLearning_8x8, name_string_conditions=["Effective_Discount-false_option_len-100_"], name_string_anti_conditions=[]),
        "Discounted_1": gather_experiments(OptionQLearning_8x8, name_string_conditions=["Effective_Discount-true_option_len-1_"], name_string_anti_conditions=[]),
        "Discounted_10": gather_experiments(OptionQLearning_8x8, name_string_conditions=["Effective_Discount-true_option_len-10_"], name_string_anti_conditions=[]),
        "Discounted_100": gather_experiments(OptionQLearning_8x8, name_string_conditions=["Effective_Discount-true_option_len-100_"], name_string_anti_conditions=[]),
        "QLearning": gather_experiments(QLearning_8x8, name_string_conditions=[], name_string_anti_conditions=[]),
            
    }

    plot_experiments(agent_dict, "Runs/Figures", name=f"DoorKey8x8_FullyObs", window_size=1, show_ci=True, ignore_last=True, plt_configs=["r_s", "r_e", "ou_s", "ou_e"], plot_each=False)
