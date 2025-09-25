from RLBase.Evaluate import plot_experiments, gather_experiments

if __name__ == "__main__":
    PPO_SimpleCrossing1 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-1000)/PPO"
    PPO_SimpleCrossing2 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-2000)/PPO"
    PPO_SimpleCrossing3 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-3000)/PPO"
    PPO_SimpleCrossing4 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-4000)/PPO"
    PPO_SimpleCrossing5 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-5000)/PPO"
    
    A2C_SimpleCrossing_1 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-1000)/A2C"
    A2C_SimpleCrossing_2 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-2000)/A2C"
    A2C_SimpleCrossing_3 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-3000)/A2C"
    A2C_SimpleCrossing_4 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-4000)/A2C"
    A2C_SimpleCrossing_5 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)/A2C"
    A2C_SimpleCrossing_6 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-6000)/A2C"
    A2C_SimpleCrossing_7 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-7000)/A2C"
    A2C_SimpleCrossing_8 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-8000)/A2C"
    A2C_SimpleCrossing_9 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-9000)/A2C"
    A2C_SimpleCrossing_10 = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-10000)/A2C"
    
    PPO_SimpleCrossing_1_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgObs_FixedSeed(seed-1000)/PPO"
    PPO_SimpleCrossing_2_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgObs_FixedSeed(seed-2000)/PPO"
    PPO_SimpleCrossing_3_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgObs_FixedSeed(seed-3000)/PPO"
    PPO_SimpleCrossing_4_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgObs_FixedSeed(seed-4000)/PPO"
    PPO_SimpleCrossing_5_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgObs_FixedSeed(seed-5000)/PPO"
    PPO_SimpleCrossing_6_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgObs_FixedSeed(seed-6000)/PPO"
    PPO_SimpleCrossing_7_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgObs_FixedSeed(seed-7000)/PPO"
    PPO_SimpleCrossing_8_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgObs_FixedSeed(seed-8000)/PPO"
    PPO_SimpleCrossing_9_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgObs_FixedSeed(seed-9000)/PPO"
    PPO_SimpleCrossing_10_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgObs_FixedSeed(seed-10000)/PPO"

    A2C_SimpleCrossing_1_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-1000)/A2C"
    A2C_SimpleCrossing_2_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-2000)/A2C"
    A2C_SimpleCrossing_3_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-3000)/A2C"
    A2C_SimpleCrossing_4_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-4000)/A2C"
    A2C_SimpleCrossing_5_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-5000)/A2C"
    A2C_SimpleCrossing_6_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-6000)/A2C"
    A2C_SimpleCrossing_7_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-7000)/A2C"
    A2C_SimpleCrossing_8_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-8000)/A2C"
    A2C_SimpleCrossing_9_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-9000)/A2C"
    A2C_SimpleCrossing_10_RGB = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-10000)/A2C"
    
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
    
    A2C_RGB = "Runs/Train/MiniGrid-FourRooms-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-5000)/A2C"
    OptionA2C_RGB = "Runs/Train/MiniGrid-FourRooms-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-5000)/OptionA2C"
    
    OptionPPO_RGB = "Runs/Train/MiniGrid-FourRooms-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-5000)/OptionPPO"
    PPO_RGB = "Runs/Train/MiniGrid-FourRooms-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-5000)/PPO"
    
    num_distractors = 25
    Ant_Test = "Runs/Train/Ant-v5_/A2C"
    
    Test = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)/OptionA2C"
    # Test2 = "Runs/Train/MiniGrid-FourRooms-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_AgentPos/A2C"
    
    PPO_minihack1 = "Runs/Train/MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-12)/PPO"
    PPO_minihack2 = "Runs/Train/MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-37)/PPO"
    PPO_minihack3 = "Runs/Train/MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-44)/PPO"
    PPO_minihack4 = "Runs/Train/MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-49)/PPO"
    PPO_minihack5 = "Runs/Train/MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-52)/PPO"
    PPO_minihack6 = "Runs/Train/MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-57)/PPO"
    PPO_minihack7 = "Runs/Train/MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-10)/PPO"
    PPO_minihack8 = "Runs/Train/MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-20)/PPO"
    # PPO_minihack9 = "Runs/Train/MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-90)/PPO"
    # PPO_minihack10 = "Runs/Train/MiniHack-Corridor-R2-v0_reward_win-1.0_reward_lose-0.0_penalty_step-0.0/OneHotChars_FixedSeed(seed-100)/PPO"
    
    agent_dict = {
        "Test": gather_experiments(Test, name_string_conditions=["test2"], name_string_anti_conditions=[]),
        # "Test2": gather_experiments(Test2, name_string_conditions=[], name_string_anti_conditions=[]),
        
        # "Didec": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=["Distractor"]), #best
        # "Vanilla (A2C)": gather_experiments(A2C_40Distraction),
        # "Neural-Augmented": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=["Distractor"]),
        # "Dec-Whole": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=["Distractor"]),
        # "Fine-Tuned Policy": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=["Distractor"]),
        # # "Mask-Input": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Layer1": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Layer1": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Reg": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), #best
        # # "Mask-Input-Layer1-Reg": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]),
        # # "Mask-Layer1-Reg": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Input Only": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), 
        # "Neuron Only": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Input and Neuron": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]),
        
        
        # "Didec": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=["Distractor"]), #best
        # "Vanilla (A2C)": gather_experiments(A2C_20Distraction),
        # "Neural-Augmented": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=["Distractor"]),
        # "Dec-Whole": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=["Distractor"]),
        # "Fine-Tuned Policy": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=["Distractor"]),
        # # "Mask-Input": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Layer1": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Layer1": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Reg": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), 
        # # "Mask-Input-Layer1-Reg": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]), #best
        # # "Mask-Layer1-Reg": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Input Only": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), 
        # "Neuron Only": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Input and Neuron": gather_experiments(OptionA2C_20Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]),
        

        # "Didec": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]), #best
        # "Vanilla (A2C)": gather_experiments(A2C_10Distraction),
        # "Neural-Augmented": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=["Distractor"]),
        # "Dec-Whole": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=["Distractor"]),
        # "Fine-Tuned Policy": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=["Distractor"]),
        # # "Mask-Input": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Layer1": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Layer1": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=[]),
        # # "Mask-Input-Reg": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), 
        # # "Mask-Input-Layer1-Reg": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]),#best
        # # "Mask-Layer1-Reg": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Input Only": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), 
        # "Neuron Only": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Input and Neuron": gather_experiments(OptionA2C_10Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]),
        

        # "Didec": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=["Distractor"]), #best
        # "Vanilla (A2C)": gather_experiments(A2C_NoDistraction),
        # "Neural-Augmented": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Transfer_"], name_string_anti_conditions=["Distractor"]),
        # "Dec-Whole": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["DecWhole_"], name_string_anti_conditions=["Distractor"]),
        # "Fine-Tuned Policy": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["FineTune_"], name_string_anti_conditions=["Distractor"]),
        # # "Mask-Input": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input_"], name_string_anti_conditions=["Distractor"]),
        # # "Mask-Input-Layer1": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-l1_"], name_string_anti_conditions=["Distractor"]),
        # # "Mask-Layer1": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-l1_"], name_string_anti_conditions=["Distractor"]),
        # # "Mask-Input-Reg": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=["Distractor"]), #best
        # # "Mask-Input-Layer1-Reg": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=["Distractor"]),
        # # "Mask-Layer1-Reg": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=["Distractor"]),
        # "Input Only": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=["Distractor"]),
        # "Neuron Only": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=["Distractor"]),
        # "Input and Neuron": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=["Distractor"]),

        
        # "A2C_1": gather_experiments(A2C_SimpleCrossing_1),
        # "A2C_2": gather_experiments(A2C_SimpleCrossing_2),
        # "A2C_3": gather_experiments(A2C_SimpleCrossing_3),
        # "A2C_4": gather_experiments(A2C_SimpleCrossing_4),
        # "A2C_5": gather_experiments(A2C_SimpleCrossing_5),
        # "A2C_6": gather_experiments(A2C_SimpleCrossing_6),
        # "A2C_7": gather_experiments(A2C_SimpleCrossing_7),
        # "A2C_8": gather_experiments(A2C_SimpleCrossing_8),
        # "A2C_9": gather_experiments(A2C_SimpleCrossing_9),
        # "A2C_10": gather_experiments(A2C_SimpleCrossing_10),
        
        # f"A2C": gather_experiments(A2C_NoDistraction),
        # f"DecWhole-{num_distractors}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"DecWhole_Distractor-{num_distractors}"], name_string_anti_conditions=[]),
        # f"FineTune-{num_distractors}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"FineTune_Distractor-{num_distractors}"], name_string_anti_conditions=[]),
        # f"Transfer-{num_distractors}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Transfer_Distractor-{num_distractors}"], name_string_anti_conditions=[]),

        # f"Mask-Input-{num_distractors}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input_Distractor-{num_distractors}"], name_string_anti_conditions=[]),
        # f"Mask-Input-Reg-{num_distractors}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input_Reg01_Distractor-{num_distractors}"], name_string_anti_conditions=[]),
        
        # f"Mask-Layer1-{num_distractors}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-l1_Distractor-{num_distractors}"], name_string_anti_conditions=[]),
        # f"Mask-Layer1-Reg-{num_distractors}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-l1_Reg01_Distractor-{num_distractors}"], name_string_anti_conditions=[]),

        # f"Mask-Input-Layer1-{num_distractors}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input-l1_Distractor-{num_distractors}"], name_string_anti_conditions=[]),
        # f"Mask-Input-Layer1-Reg-{num_distractors}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input-l1_Reg01_Distractor-{num_distractors}"], name_string_anti_conditions=[]),
        
        
        # *********
        
        # f"DecWhole-{5}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"DecWhole_Distractor-{5}"], name_string_anti_conditions=[]),
        # f"DecWhole-{15}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"DecWhole_Distractor-{15}"], name_string_anti_conditions=[]),
        # f"DecWhole-{25}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"DecWhole_Distractor-{25}"], name_string_anti_conditions=[]),

        # f"FineTune-{5}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"FineTune_Distractor-{5}"], name_string_anti_conditions=[]),
        # f"FineTune-{15}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"FineTune_Distractor-{15}"], name_string_anti_conditions=[]),
        # f"FineTune-{25}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"FineTune_Distractor-{25}"], name_string_anti_conditions=[]),

        # f"Transfer-{5}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Transfer_Distractor-{5}"], name_string_anti_conditions=[]),
        # f"Transfer-{15}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Transfer_Distractor-{15}"], name_string_anti_conditions=[]),
        # f"Transfer-{25}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Transfer_Distractor-{25}"], name_string_anti_conditions=[]),

        # f"Mask-Input-{5}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input_Distractor-{5}"], name_string_anti_conditions=[]),
        # f"Mask-Input-{15}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input_Distractor-{15}"], name_string_anti_conditions=[]),
        # f"Mask-Input-{25}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input_Distractor-{25}"], name_string_anti_conditions=[]),
        # f"Mask-Input-Reg-{5}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input_Reg01_Distractor-{5}"], name_string_anti_conditions=[]),
        # f"Mask-Input-Reg-{15}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input_Reg01_Distractor-{15}"], name_string_anti_conditions=[]),
        # f"Mask-Input-Reg-{25}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input_Reg01_Distractor-{25}"], name_string_anti_conditions=[]),
        
        # f"Mask-Layer1-{5}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-l1_Distractor-{5}"], name_string_anti_conditions=[]),
        # f"Mask-Layer1-{15}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-l1_Distractor-{15}"], name_string_anti_conditions=[]),
        # f"Mask-Layer1-{25}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-l1_Distractor-{25}"], name_string_anti_conditions=[]),
        # f"Mask-Layer1-Reg-{5}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-l1_Reg01_Distractor-{5}"], name_string_anti_conditions=[]),
        # f"Mask-Layer1-Reg-{15}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-l1_Reg01_Distractor-{15}"], name_string_anti_conditions=[]),
        # f"Mask-Layer1-Reg-{25}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-l1_Reg01_Distractor-{25}"], name_string_anti_conditions=[]),
        
        # f"Mask-Input-Layer1-{5}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input-l1_Distractor-{5}"], name_string_anti_conditions=[]),
        # f"Mask-Input-Layer1-{15}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input-l1_Distractor-{15}"], name_string_anti_conditions=[]),
        # f"Mask-Input-Layer1-{25}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input-l1_Distractor-{25}"], name_string_anti_conditions=[]),
        # f"Mask-Input-Layer1-Reg-{5}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input-l1_Reg01_Distractor-{5}"], name_string_anti_conditions=[]),
        # f"Mask-Input-Layer1-Reg-{15}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input-l1_Reg01_Distractor-{15}"], name_string_anti_conditions=[]),
        # f"Mask-Input-Layer1-Reg-{25}": gather_experiments(OptionA2C_NoDistraction, name_string_conditions=[f"Mask-input-l1_Reg01_Distractor-{25}"], name_string_anti_conditions=[]),

        # f"A2C_1": gather_experiments(A2C_SimpleCrossing_1, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"A2C_2": gather_experiments(A2C_SimpleCrossing_2, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"A2C_3": gather_experiments(A2C_SimpleCrossing_3, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"A2C_4": gather_experiments(A2C_SimpleCrossing_4, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"A2C_5": gather_experiments(A2C_SimpleCrossing_5, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"A2C_6": gather_experiments(A2C_SimpleCrossing_6, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"A2C_7": gather_experiments(A2C_SimpleCrossing_7, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"A2C_8": gather_experiments(A2C_SimpleCrossing_8, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"A2C_9": gather_experiments(A2C_SimpleCrossing_9, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"A2C_10": gather_experiments(A2C_SimpleCrossing_10, name_string_conditions=[], name_string_anti_conditions=[]),
        
        # f"A2C": gather_experiments(A2C_RGB, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"Transfer": gather_experiments(OptionA2C_RGB, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),
        # f"DecWhole": gather_experiments(OptionA2C_RGB, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        # f"FineTune": gather_experiments(OptionA2C_RGB, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        # f"Mask-l1-l3-l5": gather_experiments(OptionA2C_RGB, name_string_conditions=["Masked-l1-l3-l5_"], name_string_anti_conditions=["Reg001"]),
        # f"Mask-l1-l3-l5_Reg": gather_experiments(OptionA2C_RGB, name_string_conditions=["Masked-l1-l3-l5_Reg001_"], name_string_anti_conditions=[]),
        # f"Mask-l1-l3-l5-l8": gather_experiments(OptionA2C_RGB, name_string_conditions=["Masked-l1-l3-l5-l8_"], name_string_anti_conditions=["Reg001"]),
        # f"Mask-l1-l3-l5-l8_Reg": gather_experiments(OptionA2C_RGB, name_string_conditions=["Masked-l1-l3-l5-l8_Reg001_"], name_string_anti_conditions=[]),
        # f"Mask-l8_": gather_experiments(OptionA2C_RGB, name_string_conditions=["Masked-l8_"], name_string_anti_conditions=["Reg001"]),
        # f"Mask-l8_Reg": gather_experiments(OptionA2C_RGB, name_string_conditions=["Masked-l8_Reg_"], name_string_anti_conditions=[]),
        
        # f"A2C": gather_experiments(A2C_RGB, name_string_conditions=["Long_"], name_string_anti_conditions=[]),
        # f"Transfer": gather_experiments(OptionA2C_RGB, name_string_conditions=["Transfer_Long_"], name_string_anti_conditions=[]),
        # f"DecWhole": gather_experiments(OptionA2C_RGB, name_string_conditions=["DecWhole_Long_"], name_string_anti_conditions=[]),
        # f"FineTune": gather_experiments(OptionA2C_RGB, name_string_conditions=["FineTune_Long_"], name_string_anti_conditions=[]),
        # f"Mask-l1-l3-l5": gather_experiments(OptionA2C_RGB, name_string_conditions=["Mask-l1-l3-l5_Long_"], name_string_anti_conditions=["Reg001"]),
        # f"Mask-l1-l3-l5_Reg": gather_experiments(OptionA2C_RGB, name_string_conditions=["Mask-l1-l3-l5_Reg001_Long_"], name_string_anti_conditions=[]),
        # f"Mask-l1-l3-l5-l8": gather_experiments(OptionA2C_RGB, name_string_conditions=["Mask-l1-l3-l5-l8_Long_"], name_string_anti_conditions=["Reg001"]),
        # f"Mask-l1-l3-l5-l8_Reg": gather_experiments(OptionA2C_RGB, name_string_conditions=["Mask-l1-l3-l5-l8_Reg001_Long_"], name_string_anti_conditions=[]),
        # f"Mask-l8_": gather_experiments(OptionA2C_RGB, name_string_conditions=["Mask-l8_Long_"], name_string_anti_conditions=["Reg001"]),
        # f"Mask-l8_Reg": gather_experiments(OptionA2C_RGB, name_string_conditions=["Mask-l8_Reg001_Long_"], name_string_anti_conditions=[]),
        
        # f"PPO1": gather_experiments(PPO_SimpleCrossing1, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"PPO2": gather_experiments(PPO_SimpleCrossing2, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"PPO3": gather_experiments(PPO_SimpleCrossing3, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"PPO4": gather_experiments(PPO_SimpleCrossing4, name_string_conditions=[], name_string_anti_conditions=[]),
        # f"PPO5": gather_experiments(PPO_SimpleCrossing5, name_string_conditions=[], name_string_anti_conditions=[]),
        
        # "Didec": gather_experiments(OptionPPO_RGB, name_string_conditions=["Mask-l1-l3-l5-l8_Reg01_"], name_string_anti_conditions=[]), #best
        # "Vanilla (PPO)": gather_experiments(PPO_RGB),
        # "Transferred Policy": gather_experiments(OptionPPO_RGB, name_string_conditions=["Transfer_"], name_string_anti_conditions=[]),
        # "DecWhole": gather_experiments(OptionPPO_RGB, name_string_conditions=["DecWhole_"], name_string_anti_conditions=[]),
        # "FineTuned Policy": gather_experiments(OptionPPO_RGB, name_string_conditions=["FineTune_"], name_string_anti_conditions=[]),
        # "Mask-l1-l3-l5": gather_experiments(OptionPPO_RGB, name_string_conditions=["Mask-l1-l3-l5_"], name_string_anti_conditions=["Reg"]),
        # "Mask-l1-l3-l5_Reg": gather_experiments(OptionPPO_RGB, name_string_conditions=["Mask-l1-l3-l5_Reg01_"], name_string_anti_conditions=[]),
        # "Mask-l1-l3-l5-l8": gather_experiments(OptionPPO_RGB, name_string_conditions=["Mask-l1-l3-l5-l8_"], name_string_anti_conditions=["Reg"]),
        # "Mask-l1-l3-l5-l8_Reg": gather_experiments(OptionPPO_RGB, name_string_conditions=["Mask-l1-l3-l5-l8_Reg01_"], name_string_anti_conditions=[]), #best
        # "Mask-l8": gather_experiments(OptionPPO_RGB, name_string_conditions=["Mask-l8_"], name_string_anti_conditions=[]),
        # "Mask-l8_Reg": gather_experiments(OptionPPO_RGB, name_string_conditions=["Mask-l8_reg01_"], name_string_anti_conditions=[]),
        # "Input Only": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-reg01_"], name_string_anti_conditions=[]), 
        # "Neuron Only": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-l1-reg01_"], name_string_anti_conditions=[]),
        # "Input and Neuron": gather_experiments(OptionA2C_40Distraction, name_string_conditions=["Mask-input-l1-reg01_"], name_string_anti_conditions=[]),
        
        # "PPO1": gather_experiments(PPO_minihack1, name_string_conditions=[], name_string_anti_conditions=[]),
        # "PPO2": gather_experiments(PPO_minihack2, name_string_conditions=[], name_string_anti_conditions=[]),
        # "PPO3": gather_experiments(PPO_minihack3, name_string_conditions=[], name_string_anti_conditions=[]),
        # "PPO4": gather_experiments(PPO_minihack4, name_string_conditions=[], name_string_anti_conditions=[]),
        # "PPO5": gather_experiments(PPO_minihack5, name_string_conditions=[], name_string_anti_conditions=[]),
        # "PPO6": gather_experiments(PPO_minihack6, name_string_conditions=[], name_string_anti_conditions=[]),
        # "PPO7": gather_experiments(PPO_minihack7, name_string_conditions=[], name_string_anti_conditions=[]),
        # "PPO8": gather_experiments(PPO_minihack8, name_string_conditions=[], name_string_anti_conditions=[]),
        # "PPO9": gather_experiments(PPO_minihack9, name_string_conditions=[], name_string_anti_conditions=[]),
        # "PPO10": gather_experiments(PPO_minihack10, name_string_conditions=[], name_string_anti_conditions=[]),
    
    }

    plot_experiments(agent_dict, "Runs/Figures", name=f"FourRooms-Repeat", window_size=10, show_ci=True, ignore_last=True, plt_configs=["r_s"], plot_each=False)
    # plot_experiments(agent_dict, "Runs/Figures", name=f"FourRoom_Transfer_train", window_size=10, show_ci=True, ignore_last=True, plt_configs=["r_s"], plot_each=False)
