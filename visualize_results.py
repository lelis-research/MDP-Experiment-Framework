from RLBase.Evaluate import plot_experiments, gather_experiments

if __name__ == "__main__":
    PPO_SimpleCrossing = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-2000)/PPO"
    
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
    
    num_distractors = 25
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

        f"A2C_1": gather_experiments(A2C_SimpleCrossing_1, name_string_conditions=[], name_string_anti_conditions=[]),
        f"A2C_2": gather_experiments(A2C_SimpleCrossing_2, name_string_conditions=[], name_string_anti_conditions=[]),
        f"A2C_3": gather_experiments(A2C_SimpleCrossing_3, name_string_conditions=[], name_string_anti_conditions=[]),
        f"A2C_4": gather_experiments(A2C_SimpleCrossing_4, name_string_conditions=[], name_string_anti_conditions=[]),
        f"A2C_5": gather_experiments(A2C_SimpleCrossing_5, name_string_conditions=[], name_string_anti_conditions=[]),
        f"A2C_6": gather_experiments(A2C_SimpleCrossing_6, name_string_conditions=[], name_string_anti_conditions=[]),
        f"A2C_7": gather_experiments(A2C_SimpleCrossing_7, name_string_conditions=[], name_string_anti_conditions=[]),
        f"A2C_8": gather_experiments(A2C_SimpleCrossing_8, name_string_conditions=[], name_string_anti_conditions=[]),
        f"A2C_9": gather_experiments(A2C_SimpleCrossing_9, name_string_conditions=[], name_string_anti_conditions=[]),
        f"A2C_10": gather_experiments(A2C_SimpleCrossing_10, name_string_conditions=[], name_string_anti_conditions=[]),

    }

    plot_experiments(agent_dict, "Runs/Figures", name=f"SimpleCrossing A2C", window_size=10, show_ci=True, ignore_last=True, plt_configs=["r_s"], plot_each=False)
    # plot_experiments(agent_dict, "Runs/Figures", name=f"FourRoom_Transfer_train", window_size=10, show_ci=True, ignore_last=True, plt_configs=["r_s"], plot_each=False)
