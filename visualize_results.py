from RLBase.Evaluate import plot_experiments, gather_experiments, plot_option_usage, plot_option_embeddings
from RLBase.Evaluate import SingleExpAnalyzer

def multi_exp_results():
    PPO_SimpleCrossing = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/OneHotImageDir/PPO"
    VQ_SimpleCrossing = "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/OneHotImageDir/VQOptionCritic"
    
    PPO_Maze = "Runs/Train/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/OptionPPO"
    VQ_Maze = "Runs/Train/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/VQOptionCritic"
    agent_dict = {       
        # "PPO_Maze": gather_experiments(PPO_Maze, name_string_conditions=[], name_string_anti_conditions=[]),
        # "VQ_Maze-d2": gather_experiments(VQ_Maze, name_string_conditions=["dim-2"], name_string_anti_conditions=[]),
        # "VQ_Maze-d4": gather_experiments(VQ_Maze, name_string_conditions=["dim-4"], name_string_anti_conditions=[]),
        # "VQ_Maze-d8": gather_experiments(VQ_Maze, name_string_conditions=["dim-8"], name_string_anti_conditions=[]),
        # "VQ_Maze-d16": gather_experiments(VQ_Maze, name_string_conditions=["dim-16"], name_string_anti_conditions=[]),
        
        # "PPO_SimpleCrossing": gather_experiments(PPO_SimpleCrossing, name_string_conditions=[], name_string_anti_conditions=[]),
        # "VQ_SimpleCrossing": gather_experiments(VQ_SimpleCrossing, name_string_conditions=[], name_string_anti_conditions=[]),
        
        "PPO_Maze": gather_experiments(PPO_Maze, name_string_conditions=[], name_string_anti_conditions=[]),
        "VQ_Maze-d8": gather_experiments(VQ_Maze, name_string_conditions=["conv_dim-8"], name_string_anti_conditions=["online-c20", "manual-emb"]),
        "VQ_Maze-d8_Manual-emb": gather_experiments(VQ_Maze, name_string_conditions=["conv_dim-8_manual-emb"], name_string_anti_conditions=["online-c20"]),
        "VQ_Maze-d8_online": gather_experiments(VQ_Maze, name_string_conditions=["online-c20_conv_dim-8"], name_string_anti_conditions=["manual-emb"]),
        "VQ_Maze-d8_online_Manual-emb": gather_experiments(VQ_Maze, name_string_conditions=["online-c20_conv_dim-8_manual-emb"], name_string_anti_conditions=[]),
        
        

    }
    
    plot_experiments(agent_dict, "Runs/Figures", name=f"MazeRoom_ManualOption_GoalKeyDoor_comparison", window_size=1, show_ci=True, ignore_last=True, 
                     plt_configs=["r_s", "ou_s", "uni_ou_s", "no_s"], plot_each=False)
    # plot_option_embeddings(agent_dict, "Runs/Figures", name=f"Option_Emb_Codebook_Commit", show_legend=True, min_ep=5000


def single_exp_results():
    for i in range(50):
        exp_dir = f"Runs/Train/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/VQOptionCritic/conv-dim2_{i}_seed[{i}]"
        try:
            analyzer = SingleExpAnalyzer(exp_path=exp_dir)
        except:
            print(f"{i} has error")
            continue

        # analyzer.plot_option_embedding(min_ep=998, max_ep=999)
        analyzer.plot_option_embedding2(save_dir=exp_dir)
        analyzer.plot_option_embedding_gif(every=10)

if __name__ == "__main__":
    multi_exp_results()
    # single_exp_results()    
    
    
    
    
    
    
 