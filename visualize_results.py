from RLBase.Evaluate import plot_experiments, gather_experiments

if __name__ == "__main__":
    OptionQLearning_Effective_Discount = "Runs/Train/MiniGrid-DoorKey-5x5-v0_/SymbolicObs_FixedSeed(seed-10)/OptionQLearning"
    QLearning = "Runs/Train/MiniGrid-DoorKey-5x5-v0_/SymbolicObs_FixedSeed(seed-10)/QLearning"
    agent_dict = {
        "OptionQLearning_Effective_Discount": gather_experiments(OptionQLearning_Effective_Discount, name_string_conditions=["EffectiveDiscount"], name_string_anti_conditions=[]),
        "OptionQLearning": gather_experiments(OptionQLearning_Effective_Discount, name_string_conditions=[], name_string_anti_conditions=["EffectiveDiscount"]),
        # "QLearning": gather_experiments(QLearning, name_string_conditions=[], name_string_anti_conditions=[]),
            
    }

    plot_experiments(agent_dict, "Runs/Figures", name=f"OptionUsage_DoorKey", window_size=1, show_ci=True, ignore_last=True, plt_configs=["ou_e", "ou_s"], plot_each=False)
