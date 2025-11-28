from RLBase.Evaluate import plot_experiments, gather_experiments, plot_option_usage

if __name__ == "__main__":
    
    ContinualOptionQLearning_8x8 = "Runs/Train/MiniGrid-DoorKey-8x8-v0_/FullyObs_FixedSeed(seed-10)/ContinualOptionQLearning"
    OptionQLearning_8x8 = "Runs/Train/MiniGrid-DoorKey-8x8-v0_/FullyObs_FixedSeed(seed-8)/OptionQLearning"
    QLearning_8x8 = "Runs/Train/MiniGrid-DoorKey-8x8-v0_/FullyObs_FixedSeed(seed-8)/QLearning"
    
    QLearning_Seq = "Runs/Train/SequentialDiagonalGoalsEnv-v0_/FullyObs_FixedSeed(seed-10)/QLearning"

    ContinualOptionQLearning_TwoGoals = "Runs/Train/TwoRoomKeyDoorTwoGoalEnv-v0_/FullyObs_FixedSeed(seed-2)/ContinualOptionQLearning"
    OptionQLearning_TwoGoals = "Runs/Train/TwoRoomKeyDoorTwoGoalEnv-v0_/FullyObs_FixedSeed(seed-2)/OptionQLearning"
    QLearning_TwoGoals = "Runs/Train/TwoRoomKeyDoorTwoGoalEnv-v0_/FullyObs_FixedSeed(seed-2)/QLearning"
    
    ContinualOptionQLearning_Big = "Runs/Train/BigCurriculumEnv-v0_/FullyObs_FixedSeed(seed-2)/ContinualOptionQLearning"
    OptionQLearning_Big = "Runs/Train/BigCurriculumEnv-v0_/FullyObs_FixedSeed(seed-2)/OptionQLearning"
    QLearning_Big = "Runs/Train/BigCurriculumEnv-v0_/FullyObs_FixedSeed(seed-2)/QLearning"
    
    Continuals_anti_cond = ["egreedy", "schedule"]
    Continuals_anti_cond2 = []
    
    DQN = "Runs/Train/BigCurriculumEnv-v0_/FlattenOnehotObj/DQN"
    DQN2 = "Runs/Train/BigCurriculumEnv-v0_/DQN"
    
    agent_dict = {
        # "QLearning": gather_experiments(QLearning_Big, name_string_conditions=["400K-"], name_string_anti_conditions=Continuals_anti_cond2),
        # "OptionQLearning": gather_experiments(OptionQLearning_Big, name_string_conditions=["400K-"], name_string_anti_conditions=Continuals_anti_cond2),
        
        # "Continual_max": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-init_max-"], name_string_anti_conditions=Continuals_anti_cond),
        # "Continual_max-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_max-egreedy-"], name_string_anti_conditions=[]),
        # "Continual_max-schedule": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-schedule-init_max-"], name_string_anti_conditions=Continuals_anti_cond2),
        
        # "Continual_zero": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-init_zero-"], name_string_anti_conditions=Continuals_anti_cond),
        # "Continual_zero-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_zero-egreedy-"], name_string_anti_conditions=[]),
        # "Continual_zero-schedule": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-schedule-init_zero-"], name_string_anti_conditions=Continuals_anti_cond2),
        
        # "Continual_reset": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-reset-"], name_string_anti_conditions=Continuals_anti_cond),
        # "Continual_reset-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["reset-egreedy-"], name_string_anti_conditions=[]),
        # "Continual_reset-schedule": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-schedule-reset-"], name_string_anti_conditions=Continuals_anti_cond2),
        
        # "Continual-entropy-beta0": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-init_uncertainty-entropy-beta0"], name_string_anti_conditions=Continuals_anti_cond),
        # "Continual-entropy-beta0-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-entropy-beta0-egreedy-"], name_string_anti_conditions=[]),
        # "Continual-entropy-beta0-schedule": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-schedule-init_uncertainty-entropy-beta0-"], name_string_anti_conditions=Continuals_anti_cond2),
        
        # "Continual-entropy-beta1": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-init_uncertainty-entropy-beta1"], name_string_anti_conditions=Continuals_anti_cond),
        # "Continual-entropy-beta1-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-entropy-beta1-egreedy-"], name_string_anti_conditions=[]),
        # "Continual-entropy-beta1-schedule": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-schedule-init_uncertainty-entropy-beta1-"], name_string_anti_conditions=Continuals_anti_cond2),
        

        # "Continual-margin-beta0": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-init_uncertainty-margin-beta0"], name_string_anti_conditions=Continuals_anti_cond),
        # "Continual-margin-beta0-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-margin-beta0-egreedy-"], name_string_anti_conditions=[]),
        # "Continual-margin-beta0-schedule": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-schedule-init_uncertainty-margin-beta0-"], name_string_anti_conditions=Continuals_anti_cond2),
        
        # "Continual-margin-beta1": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-init_uncertainty-margin-beta1"], name_string_anti_conditions=Continuals_anti_cond),
        # "Continual-margin-beta1-egreedy": gather_experiments(ContinualOptionQLearning_TwoGoals, name_string_conditions=["init_uncertainty-margin-beta1-egreedy-"], name_string_anti_conditions=[]),
        # "Continual-margin-beta1-schedule": gather_experiments(ContinualOptionQLearning_Big, name_string_conditions=["400K-schedule-init_uncertainty-margin-beta1-"], name_string_anti_conditions=Continuals_anti_cond2),
        
        
        # "DQN": gather_experiments(DQN, name_string_conditions=[], name_string_anti_conditions=[]),
        "DQN2": gather_experiments(DQN2, name_string_conditions=[], name_string_anti_conditions=[]),
        
        
        
    }

    # plot_experiments(agent_dict, "Runs/Figures", name=f"test_steps", window_size=1, show_ci=True, ignore_last=True, 
    #                  plt_configs=["r_s", "ou_s", "oc_s"], plot_each=False)
    
    plot_experiments(agent_dict, "Runs/Figures", name=f"test_dqn", window_size=1, show_ci=True, ignore_last=True, 
                     plt_configs=["r_s", "ou_s", "no_s"], plot_each=False)
    
    # plot_option_usage(agent_dict, "Runs/Figures", name=f"test2", window_size=1, show_ci=True, ignore_last=True,
    #                   x_type="s", plot_each=False, option_classes=ALL_OPTIONS)
    
    
    
    
    
 