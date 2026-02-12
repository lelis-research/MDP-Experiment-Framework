from RLBase.Evaluate import plot_experiments, gather_experiments, plot_option_usage
from RLBase.Evaluate import SingleExpAnalyzer

def multi_exp_results():
    PPO_SimpleCrossing = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/OneHotImageDir/PPO"
    VQ_SimpleCrossing = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/OneHotImageDir/VQOptionCritic"
    
    PPO_Maze = "Runs/Train/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/OptionPPO"
    VQ_Maze = "Runs/Train/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/VQOptionCritic"
    
    Random_SF_Test = "Runs/Train/MiniGrid-TestOptionRoom-v0_/FullyObs_OneHotImageDir_FixedSeed(seed-10)/OptionRandomSFCodebook"
    
    agent_dict = {      
        # "emb[dim_2]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+emb[dim_2]"], name_string_anti_conditions=[]),
        # "emb[dim_4]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+emb[dim_4]"], name_string_anti_conditions=[]),
        # "emb[dim_8]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+emb[dim_8]"], name_string_anti_conditions=[]),
        # "emb[dim_16]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]"], name_string_anti_conditions=[]),
        
        # "emb[init_u0.01]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+emb[init_u0.01]"], name_string_anti_conditions=[]),
        # "emb[init_u0.05]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]"], name_string_anti_conditions=[]),
        # "emb[init_u0.1]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+emb[init_u0.1]"], name_string_anti_conditions=[]),
        # "emb[init_u0.5]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+emb[init_u0.5]"], name_string_anti_conditions=[]),
        # "emb[init_u1.0]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+emb[init_u1.0]"], name_string_anti_conditions=[]),
        # "emb[init_onehot]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+emb[init_onehot]"], name_string_anti_conditions=[]),
        
        # "sf[64]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+sf[64]"], name_string_anti_conditions=[]),
        # "sf[128]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+sf[128]"], name_string_anti_conditions=[]),
        # "sf[256]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]"], name_string_anti_conditions=[]),
        # "sf[256_256]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+sf[256_256]"], name_string_anti_conditions=[]),
        
        # "obs[128_8]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+obs[128_8]"], name_string_anti_conditions=[]),
        # "obs[128_16]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+obs[128_16]"], name_string_anti_conditions=[]),
        # "obs[128_32]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+obs[128_32]"], name_string_anti_conditions=[]),
        # "obs[256_8]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+obs[256_8]"], name_string_anti_conditions=[]),
        # "obs[256_16]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]"], name_string_anti_conditions=[]),
        # "obs[256_32]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+obs[256_32]"], name_string_anti_conditions=[]),
        
        # "dp[0.0]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+dp[0.0]"], name_string_anti_conditions=[]),
        # "dp[0.1]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+dp[0.1]"], name_string_anti_conditions=[]),
        # "dp[0.2]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]"], name_string_anti_conditions=[]),
        # "dp[0.3]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+dp[0.3]"], name_string_anti_conditions=[]),
        # "dp[0.5]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+dp[0.5]"], name_string_anti_conditions=[]),
        
        # "inp[obs_emb]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]"], name_string_anti_conditions=[]),
        # "inp[emb]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+inp[emb]"], name_string_anti_conditions=[]),
        
        # "nce[0.0-1.0]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+nce[0.0-1.0]"], name_string_anti_conditions=[]),
        # "nce[0.001-1.0]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+nce[0.001-1.0]"], name_string_anti_conditions=[]),
        # "nce[0.01-1.0]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]"], name_string_anti_conditions=[]),
        # "nce[0.05-1.0]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+nce[0.05-1.0]"], name_string_anti_conditions=[]),
        
        "nce[0.01-0.2]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+nce[0.01-0.2]"], name_string_anti_conditions=[]),
        "nce[0.01-0.5]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+nce[0.01-0.5]"], name_string_anti_conditions=[]),
        "nce[0.01-1.0]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]"], name_string_anti_conditions=[]),
        "nce[0.01-2.0]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+nce[0.01-2.0]"], name_string_anti_conditions=[]),
        "nce[0.01-5.0]": gather_experiments(Random_SF_Test, name_string_conditions=["Baseline+nce[0.01-5.0]"], name_string_anti_conditions=[]),
        
    }
    
    
    
    # plot_experiments(agent_dict, "Runs/Figures", name=f"SF_Rand_Res", 
    #                  window_size=1, show_ci=True, ignore_last=True, 
    #                  plt_configs=["r_s", "ou_s", "uni_ou_s", "no_s"], plot_each=False)
    
    plot_experiments(agent_dict, "Runs/Figures", f"TestOption_nce[0.01-#]", 
                     plt_configs=["cb_hm_first", "cb_hm_last"])


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
    
    
    
    
    
    
 
