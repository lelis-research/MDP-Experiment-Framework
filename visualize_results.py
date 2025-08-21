from RLBase.Evaluate import plot_experiments, gather_experiments

if __name__ == "__main__":
    OptionDQN_NoDistraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)/OptionDQN"
    OptionDQN_WithDistraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-30_seed-100)/OptionDQN"
    
    DQN_NoDistraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)/DQN"
    DQN_WithDistraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-30_seed-100)/DQN"
    
    Random_NoDistraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)/OptionRandom"
    Random_WithDistraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-30_seed-100)/OptionRandom"

    A2C_NoDistraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)/A2C"
    OptionA2C_NoDistraction="Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)/OptionA2C"
    
    agent_dict = {
        # "DQN": gather_experiments(DQN_WithDistraction),
        # "DecWhole": gather_experiments(OptionDQN_WithDistraction, name_string_conditions=["DecWhole_NoDistractor"], name_string_anti_conditions=[]),
        # "FineTune": gather_experiments(OptionDQN_WithDistraction, name_string_conditions=["FineTune_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-Input": gather_experiments(OptionDQN_WithDistraction, name_string_conditions=["Mask-input_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-Input-L1": gather_experiments(OptionDQN_WithDistraction, name_string_conditions=["Mask-input-l1_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-L1": gather_experiments(OptionDQN_WithDistraction, name_string_conditions=["Mask-l1_NoDistractor"], name_string_anti_conditions=[]),
        # "Transfer": gather_experiments(OptionDQN_WithDistraction, name_string_conditions=["Transfer_NoDistractor"], name_string_anti_conditions=[]),

        "A2C": gather_experiments(A2C_NoDistraction),
        "DecWhole": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        "FineTune": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        "Mask-Input": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        "Mask-Input-L1": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        "Mask-L1": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        "Transfer": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),


        # "DecWhole": gather_experiments(Random_WithDistraction, name_string_conditions=["DecWhole_NoDistractor"], name_string_anti_conditions=[]),
        # "FineTune": gather_experiments(Random_WithDistraction, name_string_conditions=["FineTune_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-Input": gather_experiments(Random_WithDistraction, name_string_conditions=["Mask-input_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-Input-L1": gather_experiments(Random_WithDistraction, name_string_conditions=["Mask-input-l1_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-L1": gather_experiments(Random_WithDistraction, name_string_conditions=["Mask-l1_NoDistractor"], name_string_anti_conditions=[]),
        # "Transfer": gather_experiments(Random_WithDistraction, name_string_conditions=["Transfer_NoDistractor"], name_string_anti_conditions=[]),
        
        # "DecWhole": gather_experiments(Random_NoDistraction, name_string_conditions=["DecWhole_NoDistractor"], name_string_anti_conditions=[]),
        # "FineTune": gather_experiments(Random_NoDistraction, name_string_conditions=["FineTune_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-Input": gather_experiments(Random_NoDistraction, name_string_conditions=["Mask-input_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-Input-L1": gather_experiments(Random_NoDistraction, name_string_conditions=["Mask-input-l1_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-L1": gather_experiments(Random_NoDistraction, name_string_conditions=["Mask-l1_NoDistractor"], name_string_anti_conditions=[]),
        # "Transfer": gather_experiments(Random_NoDistraction, name_string_conditions=["Transfer_NoDistractor"], name_string_anti_conditions=[]),

        
    }

    plot_experiments(agent_dict, "Runs/Figures", name="FourRoom-NoDistractions-NoDistraction_Options_A2C", window_size=50, show_ci=False)
