from RLBase.Evaluate import plot_experiments, gather_experiments

if __name__ == "__main__":
    Test = "Runs/Train/MiniGrid-DoorKey-5x5-v0_/ImgObs/QLearning"
    
    agent_dict = {
        "Test": gather_experiments(Test, name_string_conditions=[], name_string_anti_conditions=[]),
            
    }

    plot_experiments(agent_dict, "Runs/Figures", name=f"QLearning", window_size=10, show_ci=True, ignore_last=True, plt_configs=["r_s"], plot_each=True)
