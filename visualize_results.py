from RLBase.Evaluate import plot_experiments, gather_experiments

if __name__ == "__main__":
    PPO_SimpleCrossing = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-2000)/PPO"
    
    A2C_SimpleCrossing_1 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-1000)_FixedRandomDistractor(num_distractors-25_seed-100)/A2C"
    A2C_SimpleCrossing_2 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-2000)_FixedRandomDistractor(num_distractors-25_seed-100)/A2C"
    A2C_SimpleCrossing_3 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-3000)_FixedRandomDistractor(num_distractors-25_seed-100)/A2C"
    A2C_SimpleCrossing_4 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-4000)_FixedRandomDistractor(num_distractors-25_seed-100)/A2C"
    A2C_SimpleCrossing_5 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-25_seed-100)/A2C"
    A2C_SimpleCrossing_6 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-6000)_FixedRandomDistractor(num_distractors-25_seed-100)/A2C"
    A2C_SimpleCrossing_7 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-7000)_FixedRandomDistractor(num_distractors-25_seed-100)/A2C"
    A2C_SimpleCrossing_8 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-8000)_FixedRandomDistractor(num_distractors-25_seed-100)/A2C"
    A2C_SimpleCrossing_9 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-9000)_FixedRandomDistractor(num_distractors-25_seed-100)/A2C"
    A2C_SimpleCrossing_10 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-10000)_FixedRandomDistractor(num_distractors-25_seed-100)/A2C"
    

    
    
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
    
    Ant_Test = "Runs/Train/Ant-v5_/A2C"
    agent_dict = {

        # "A2C": gather_experiments(A2C_40Distraction),
        # "DecWhole": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        # "FineTune": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        # # "Mask-Input": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Layer1": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Layer1": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Reg": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), #best
        # # "Mask-Input-Layer1-Reg": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]),
        # # "Mask-Layer1-Reg": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Transfer": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),
        # "Didec": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), #best
        
        # "A2C": gather_experiments(A2C_20Distraction),
        # "DecWhole": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        # "FineTune": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        # # "Mask-Input": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Layer1": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Layer1": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Reg": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), 
        # # "Mask-Input-Layer1-Reg": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]), #best
        # # "Mask-Layer1-Reg": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Transfer": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),
        # "Didec": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]), #best

        
        # "A2C": gather_experiments(A2C_10Distraction),
        # "DecWhole": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        # "FineTune": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        # # "Mask-Input": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Layer1": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]), #best
        # # "Mask-Layer1": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Reg": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), 
        # # "Mask-Input-Layer1-Reg": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]),
        # # "Mask-Layer1-Reg": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Transfer": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),
        # "Didec": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]), #best

        
        # "A2C": gather_experiments(A2C_NoDistraction),
        # "DecWhole": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        # "FineTune": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        # # "Mask-Input": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Layer1": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Layer1": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Reg": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), #best
        # # "Mask-Input-Layer1-Reg": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]),
        # # "Mask-Layer1-Reg": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Transfer": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),
        # "Didec": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), #best

        
        "A2C_1": gather_experiments(A2C_SimpleCrossing_1),
        "A2C_2": gather_experiments(A2C_SimpleCrossing_2),
        "A2C_3": gather_experiments(A2C_SimpleCrossing_3),
        "A2C_4": gather_experiments(A2C_SimpleCrossing_4),
        "A2C_5": gather_experiments(A2C_SimpleCrossing_5),
        "A2C_6": gather_experiments(A2C_SimpleCrossing_6),
        "A2C_7": gather_experiments(A2C_SimpleCrossing_7),
        "A2C_8": gather_experiments(A2C_SimpleCrossing_8),
        "A2C_9": gather_experiments(A2C_SimpleCrossing_9),
        "A2C_10": gather_experiments(A2C_SimpleCrossing_10),

        
    }

    plot_experiments(agent_dict, "Runs/Figures", name="Simple_Crossing_5_distractors", window_size=5, show_ci=True, ignore_last=True, plt_configs=["r_s", "r_e"], plot_each=True)
