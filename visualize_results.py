from RLBase.Evaluate import plot_experiments, gather_experiments

if __name__ == "__main__":
    
    ContinualOptionQLearning_8x8 = "Runs/Train/MiniGrid-DoorKey-8x8-v0_/FullyObs_FixedSeed(seed-10)/ContinualOptionQLearning"
    OptionQLearning_8x8 = "Runs/Train/MiniGrid-DoorKey-8x8-v0_/FullyObs_FixedSeed(seed-8)/OptionQLearning"
    QLearning_8x8 = "Runs/Train/MiniGrid-DoorKey-8x8-v0_/FullyObs_FixedSeed(seed-8)/QLearning"
    
    QLearning_Seq = "Runs/Train/SequentialDiagonalGoalsEnv-v0_/FullyObs_FixedSeed(seed-10)/QLearning"

    ContinualOptionQLearning_TwoGoals = "Runs/Train/TwoRoomKeyDoorTwoGoalEnv-v0_/FullyObs_FixedSeed(seed-1)/ContinualOptionQLearning"
    OptionQLearning_TwoGoals = "Runs/Train/TwoRoomKeyDoorTwoGoalEnv-v0_/FullyObs_FixedSeed(seed-1)/OptionQLearning"
    QLearning_TwoGoals = "Runs/Train/TwoRoomKeyDoorTwoGoalEnv-v0_/FullyObs_FixedSeed(seed-1)/QLearning"
    
    agent_dict = {
       "QLearning": gather_experiments(QLearning_TwoGoals, name_string_conditions=[], name_string_anti_conditions=[]),
       "OptionQLearning": gather_experiments(OptionQLearning_TwoGoals, name_string_conditions=[], name_string_anti_conditions=[]),
       "ContinualOptionQLearning_avg": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_avg_"], name_string_anti_conditions=[]),
       "ContinualOptionQLearning_max": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_max_"], name_string_anti_conditions=[]),
       "ContinualOptionQLearning_zero": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_zero_"], name_string_anti_conditions=[]),
       "ContinualOptionQLearning_reset": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["reset_"], name_string_anti_conditions=[]),
       
        # "QLearning_1": gather_experiments(QLearning_Seq, name_string_conditions=["n_steps-1_"], name_string_anti_conditions=[]),
        # "QLearning_5": gather_experiments(QLearning_Seq, name_string_conditions=["n_steps-5_"], name_string_anti_conditions=[]),
        # "QLearning_10": gather_experiments(QLearning_Seq, name_string_conditions=["n_steps-10_"], name_string_anti_conditions=[]),
        # "QLearning_20": gather_experiments(QLearning_Seq, name_string_conditions=["n_steps-20_"], name_string_anti_conditions=[]),    
    }

    plot_experiments(agent_dict, "Runs/Figures", name=f"TwoGoals_result", window_size=1, show_ci=True, ignore_last=True, 
                     plt_configs=["r_s", "ou_s", "no_s"], plot_each=False)
 