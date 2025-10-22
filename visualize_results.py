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
        
        # "Continual_max": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_max-"], name_string_anti_conditions=["egreedy", "schedule"]),
        # "Continual_max-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_max-egreedy-"], name_string_anti_conditions=[]),
        "Continual_max-schedule": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_max-schedule-"], name_string_anti_conditions=[]),
        
        # "Continual_zero": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_zero-"], name_string_anti_conditions=["egreedy", "schedule"]),
        # "Continual_zero-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_zero-egreedy-"], name_string_anti_conditions=[]),
        "Continual_zero-schedule": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_zero-schedule-"], name_string_anti_conditions=[]),
        
        # "Continual_reset": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["reset-"], name_string_anti_conditions=["egreedy", "schedule"]),
        # "Continual_reset-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["reset-egreedy-"], name_string_anti_conditions=[]),
        "Continual_reset-schedule": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["reset-schedule-"], name_string_anti_conditions=[]),
        
        # "Continual-entropy-beta0": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-entropy-beta0"], name_string_anti_conditions=["egreedy", "schedule"]),
        # "Continual-entropy-beta0-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-entropy-beta0-egreedy-"], name_string_anti_conditions=[]),
        "Continual-entropy-beta0-schedule": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-entropy-beta0-schedule"], name_string_anti_conditions=[]),
        
        # "Continual-entropy-beta1": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-entropy-beta1"], name_string_anti_conditions=["egreedy", "schedule"]),
        # "Continual-entropy-beta1-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-entropy-beta1-egreedy-"], name_string_anti_conditions=[]),
        "Continual-entropy-beta1-schedule": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-entropy-beta1-schedule"], name_string_anti_conditions=[]),
        

        # "Continual-margin-beta0": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-margin-beta0"], name_string_anti_conditions=["egreedy", "schedule"]),
        # "Continual-margin-beta0-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-margin-beta0-egreedy-"], name_string_anti_conditions=[]),
        "Continual-margin-beta0-schedule": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-margin-beta0-schedule"], name_string_anti_conditions=[]),
        
        # "Continual-margin-beta1": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-margin-beta1"], name_string_anti_conditions=["egreedy", "schedule"]),
        # "Continual-margin-beta1-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-margin-beta1-egreedy-"], name_string_anti_conditions=[]),
        "Continual-margin-beta1-schedule": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-margin-beta1-schedule"], name_string_anti_conditions=[]),
        
        
        
    }

    plot_experiments(agent_dict, "Runs/Figures", name=f"TwoGoals_schedule", window_size=1, show_ci=True, ignore_last=True, 
                     plt_configs=["r_s", "ou_s", "no_s"], plot_each=False)
 