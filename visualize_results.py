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
    
    A2C_10Distraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C"
    OptionA2C_10Distraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-10_seed-100)/OptionA2C"
    
    A2C_20Distraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-20_seed-100)/A2C"
    OptionA2C_20Distraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-20_seed-100)/OptionA2C"
    
    A2C_40Distraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-40_seed-100)/A2C"
    OptionA2C_40Distraction = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-40_seed-100)/OptionA2C"
    
    agent_dict = {
        # "DQN": gather_experiments(DQN_NoDistraction),
        # "DecWhole": gather_experiments(OptionDQN_NoDistraction, name_string_conditions=["DecWhole_NoDistractor"], name_string_anti_conditions=[]),
        # "FineTune": gather_experiments(OptionDQN_NoDistraction, name_string_conditions=["FineTune_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-Input": gather_experiments(OptionDQN_NoDistraction, name_string_conditions=["Mask-input_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-Input-L1": gather_experiments(OptionDQN_NoDistraction, name_string_conditions=["Mask-input-l1_NoDistractor"], name_string_anti_conditions=[]),
        # "Mask-L1": gather_experiments(OptionDQN_NoDistraction, name_string_conditions=["Mask-l1_NoDistractor"], name_string_anti_conditions=[]),
        # "Transfer": gather_experiments(OptionDQN_NoDistraction, name_string_conditions=["Transfer_NoDistractor"], name_string_anti_conditions=[]),

        "A2C": gather_experiments(A2C_40Distraction),
        "DecWhole": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        "FineTune": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        "Mask-Input": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        "Mask-Input-Layer1": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        "Mask-Layer1": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        "Mask-Input-Reg": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]),
        "Mask-Input-Layer1-Reg": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]),
        "Mask-Layer1-Reg": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        "Transfer": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),

        # "DecWhole_NoDistractions": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        # "DecWhole_10Distractions": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        # "DecWhole_20Distractions": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        # "DecWhole_40Distractions": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        
        # "FineTune_NoDistractions": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        # "FineTune_10Distractions": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        # "FineTune_20Distractions": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        # "FineTune_40Distractions": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),

        # "Mask-Input_NoDistractions": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        # "Mask-Input_10Distractions": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        # "Mask-Input_20Distractions": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        # "Mask-Input_40Distractions": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),

        # "Mask-Layer1_NoDistractions": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        # "Mask-Layer1_10Distractions": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        # "Mask-Layer1_20Distractions": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        # "Mask-Layer1_40Distractions": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),

        # "Mask-Input-Layer1_NoDistractions": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        # "Mask-Input-Layer1_10Distractions": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        # "Mask-Input-Layer1_20Distractions": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        # "Mask-Input-Layer1_40Distractions": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),

        # "Transfer_NoDistractions": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),
        # "Transfer_10Distractions": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),
        # "Transfer_20Distractions": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),
        # "Transfer_40Distractions": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),

        

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

    plot_experiments(agent_dict, "Runs/Figures", name="FourRoom_40Distractions-Options_NoDistraction-A2C", window_size=1, show_ci=True, ignore_last=True)
