from RLBase.Evaluate import plot_experiments, gather_experiments, plot_option_usage
from RLBase.Evaluate import SingleExpAnalyzer

def multi_exp_results():
    PPO_SimpleCrossing = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/OneHotImageDir/PPO"
    VQ_SimpleCrossing = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/OneHotImageDir/VQOptionCritic"
    
    PPO_Maze = "Runs/Train/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/OptionPPO"
    VQ_Maze = "Runs/Train/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/VQOptionCritic"
    
    Random_SF_Test = "Runs/Train/MiniGrid-TestOptionRoom-v0_/FullyObs_OneHotImageDir_FixedSeed(seed-10)/OptionRandomSFCodebook"
    
    UnLockPickUp_VQ = "Runs/Train/MiniGrid-UnlockPickupLimitedColor-v0_/FullyObs_OneHotImageDirCarry/VQOptionCritic"
    
    Online_UnlockPickup_PPO = "Runs/Train/MiniGrid-UnlockPickupLimitedColor-v0_curriculum_steps-100000/FullyObs_OneHotImageDirCarry/OptionPPO"
    Online_UnLockPickUp_VQ = "Runs/Train/MiniGrid-UnlockPickupLimitedColor-v0_curriculum_steps-100000/FullyObs_OneHotImageDirCarry/VQOptionCritic"
    agent_dict = {      
        # "emb[onehot-d19]": gather_experiments(UnLockPickUp_VQ, name_string_conditions=["emb[onehot-d19]"], name_string_anti_conditions=[]),
        # "emb[uniform-d8]": gather_experiments(UnLockPickUp_VQ, name_string_conditions=["emb[uniform-d8]"], name_string_anti_conditions=[]),
        # "emb[uniform-d4]": gather_experiments(UnLockPickUp_VQ, name_string_conditions=["emb[uniform-d4]"], name_string_anti_conditions=[]),
        
        # "emb[fixed-d4]": gather_experiments(UnLockPickUp_VQ, name_string_conditions=["emb[repr[fixed-d4]]"], name_string_anti_conditions=[]),
        # "emb[fixed-d7]": gather_experiments(UnLockPickUp_VQ, name_string_conditions=["emb[repr[fixed-d7]]"], name_string_anti_conditions=[]),
        
        # "PPO_Offline": gather_experiments(Online_UnlockPickup_PPO, name_string_conditions=["Options_Add[all]_Curr[100K]"], name_string_anti_conditions=[]),
        # "PPO_Expand": gather_experiments(Online_UnlockPickup_PPO, name_string_conditions=["Options_Add[expand]_Reset[false]_Count[20]_Curr[100K]"], name_string_anti_conditions=[]),
        # "PPO_Recreate": gather_experiments(Online_UnlockPickup_PPO, name_string_conditions=["Options_Add[recreate]_Reset[false]_Count[20]_Curr[100K]"], name_string_anti_conditions=[]),
        
        # "VQ_LLM_Online": gather_experiments(Online_UnLockPickUp_VQ, name_string_conditions=["Options_Add[expand]_Reset[false]_Count[20]_Curr[100K]_CB[dim8-grad-l2]_Emb[llm]"], name_string_anti_conditions=[]),
        # "VQ_LLM_Offline": gather_experiments(Online_UnLockPickUp_VQ, name_string_conditions=["Options_Add[all]_Curr[100K]_CB[dim8-grad-l2]_Emb[llm]"], name_string_anti_conditions=[]),
        
        # "VQ_Uniform_Online": gather_experiments(Online_UnLockPickUp_VQ, name_string_conditions=["Options_Add[expand]_Reset[false]_Count[20]_Curr[100K]_CB[dim8-grad-l2]_Emb[uniform]"], name_string_anti_conditions=[]),
        # "VQ_Uniform_Offline": gather_experiments(Online_UnLockPickUp_VQ, name_string_conditions=["Options_Add[all]_Curr[100K]_CB[dim8-grad-l2]_Emb[uniform]"], name_string_anti_conditions=[]),
        
        # "VQ_OneHot_Online": gather_experiments(Online_UnLockPickUp_VQ, name_string_conditions=["Options_Add[expand]_Reset[false]_Count[20]_Curr[100K]_CB[dim80-grad-l2]_Emb[onehot]"], name_string_anti_conditions=[]),
        # "VQ_OneHot_Offline": gather_experiments(Online_UnLockPickUp_VQ, name_string_conditions=["Options_Add[all]_Curr[100K]_CB[dim80-grad-l2]_Emb[onehot]"], name_string_anti_conditions=[]),
        
        "Offline": gather_experiments(PPO_Maze, name_string_conditions=["Options_Add[all]"], name_string_anti_conditions=[]),
        "recreate": gather_experiments(PPO_Maze, name_string_conditions=["Options_Add[recreate]_Count[20]_Reset[False]"], name_string_anti_conditions=[]),
        "recreate-reset": gather_experiments(PPO_Maze, name_string_conditions=["Options_Add[recreate]_Count[20]_Reset[True]"], name_string_anti_conditions=[]),
        "expand": gather_experiments(PPO_Maze, name_string_conditions=["Options_Add[expand]_Count[20]_Reset[False]"], name_string_anti_conditions=[]),
        "expand-reset": gather_experiments(PPO_Maze, name_string_conditions=["Options_Add[expand]_Count[20]_Reset[True]"], name_string_anti_conditions=[]),
    }
    
    
    
    plot_experiments(agent_dict, "Runs/Figures/Maze", name=f"OptionPPO", 
                     window_size=1, show_ci=True, ignore_last=True, 
                     plt_configs=["r_s", "uni_ou_s", "no_s"], plot_each=False)
    
    # plot_experiments(agent_dict, "Runs/Figures", f"TestOption_nce[0.01-#]", 
    #                  plt_configs=["cb_hm_first", "cb_hm_last"])


def single_exp_results():
    for i in range(1):
        exp_dir = f"Runs/Train/MiniGrid-EmptyTwoGoals-v0_/FullyObs_OneHotImageDir/VQOptionCritic/_seed[123123]"
        try:
            analyzer = SingleExpAnalyzer(exp_path=exp_dir)
        except:
            print(f"{i} has error")
            continue

        analyzer.plot_combined(plt_configs=["cb_hm_first", "cb_hm_last"], save_dir="Runs/Figures", window_size=1, show_ci=False, ignore_last=False, plot_each=False)

if __name__ == "__main__":
    multi_exp_results()
    # single_exp_results()    
    
    
    
    
    
    
 
