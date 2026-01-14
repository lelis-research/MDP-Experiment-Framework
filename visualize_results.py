from RLBase.Evaluate import plot_experiments, gather_experiments, plot_option_usage, plot_option_embeddings
from RLBase.Evaluate import SingleExpAnalyzer

def multi_exp_results():
    VQOptionCritic = "Runs/Train/MiniGrid-EmptyTwoGoals-v0_/VQOptionCritic"

    
    agent_dict = {       
        "VQ": gather_experiments(VQOptionCritic, name_string_conditions=[], name_string_anti_conditions=["fake_update_01_"]),
        "VQ_fake1": gather_experiments(VQOptionCritic, name_string_conditions=["fake_update_01_"], name_string_anti_conditions=[]),
    }
    
    plot_experiments(agent_dict, "Runs/Figures", name=f"Performance_CodeBook_Commit", window_size=1, show_ci=True, ignore_last=True, 
                     plt_configs=["r_s"], plot_each=False)
    # plot_option_embeddings(agent_dict, "Runs/Figures", name=f"Option_Emb_Codebook_Commit", show_legend=True, min_ep=5000)


def single_exp_results():
    for i in range(50):
        exp_dir = f"Runs/Train/MiniGrid-EmptyTwoGoals-v0_/VQOptionCritic/fake_update_01_{i}_seed[{i}]"
        analyzer = SingleExpAnalyzer(exp_path=exp_dir)

        # analyzer.plot_option_embedding(min_ep=998, max_ep=999)
        analyzer.plot_option_embedding2(save_dir=exp_dir)

if __name__ == "__main__":
    multi_exp_results()
    single_exp_results()    
    
    
    
    
    
    
 