from RLBase.Evaluate import plot_experiments, gather_experiments, plot_option_usage
from RLBase.Evaluate import SingleExpAnalyzer

def multi_exp_results():
    PPO_SimpleCrossing = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/OneHotImageDir/PPO"
    VQ_SimpleCrossing = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/OneHotImageDir/VQOptionCritic"
    
    PPO_Maze = "Runs/Train/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/OptionPPO"
    VQ_Maze = "Runs/Train/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/VQOptionCritic"
    
    Random_SF_Test = "Runs/Train/MiniGrid-TestOptionRoom-v0_/FullyObs_OneHotImageDir_FixedSeed(seed-10)/OptionRandomSFCodebook"
    
    UnLockPickUp_VQ = "Runs/Train/MiniGrid-UnlockPickupLimitedColor-v0_/FullyObs_OneHotImageDirCarry/VQOptionCritic"
    agent_dict = {      
        "emb[onehot-d19]": gather_experiments(UnLockPickUp_VQ, name_string_conditions=["emb[onehot-d19]"], name_string_anti_conditions=[]),
        "emb[uniform-d8]": gather_experiments(UnLockPickUp_VQ, name_string_conditions=["emb[uniform-d8]"], name_string_anti_conditions=[]),
        "emb[uniform-d4]": gather_experiments(UnLockPickUp_VQ, name_string_conditions=["emb[uniform-d4]"], name_string_anti_conditions=[]),
        
    }
    
    
    
    plot_experiments(agent_dict, "Runs/Figures", name=f"UnLockPickUp_VQ", 
                     window_size=1, show_ci=True, ignore_last=True, 
                     plt_configs=["r_s", "ou_s", "uni_ou_s", "no_s"], plot_each=False)
    
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
    
    
    
    
    
    
 
