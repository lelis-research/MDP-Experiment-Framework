from RLBase.Evaluate import plot_experiments, gather_experiments

if __name__ == "__main__":
    DQN_ChainEnv0_len20_dir = "Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 20}/DQN"
    DQN_ChainEnv0_len40_dir = "Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 40}/DQN"
    DQN_ChainEnv1_len40_dir = "Runs/Train/MiniGrid-ChainEnv-v1_{'chain_length': 40}/DQN"

    MaskedDQN_input_ChainEnv0_len20_dir = "Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 20}/MaskedDQN"
    MaskedDQN_input_ChainEnv0_len40_dir = "Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 40}/MaskedDQN"
    MaskedDQN_input_ChainEnv1_len40_dir = "Runs/Train/MiniGrid-ChainEnv-v1_{'chain_length': 40}/MaskedDQN"

    agent_dict = {
        "DQN": gather_experiments(DQN_ChainEnv1_len40_dir),

        # "T5_N1": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T5_N1_", "L[1]"]),
        # "T5_N5": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T5_N5_", "L[1]"]),
        # "T5_N10": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T5_N10_", "L[1]"]),
        # "T5_N20": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T5_N20_", "L[1]"]),

        # "T50_N1": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T50_N1_", "L[1]"]),
        # "T50_N5": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T50_N5_", "L[1]"]),
        # "T50_N10": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T50_N10_", "L[1]"]),
        # "T50_N20": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T50_N20_", "L[1]"]),

        # "T100_N1": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N1_", "L[1]"]),
        # "T100_N5": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N5_", "L[1]"]),
        # "T100_N10": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N10_", "L[1]"]),
        # "T100_N20": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N20_", "L[1]"]),
    }

    agent_dict = {
        "DQN": gather_experiments(DQN_ChainEnv1_len40_dir),
        "T100_N1": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N1_"], name_string_anti_conditions=["L[1]"]),
        "T100_N5": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N5_"], name_string_anti_conditions=["L[1]"]),
        # "T100_N10": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N10_"], name_string_anti_conditions=["L[1]"]),
        # "T100_N20": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N20_"], name_string_anti_conditions=["L[1]"]),

        "T100_N1_L1": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N1_", "L[1]"]),
        "T100_N5_L1": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N5_", "L[1]"]),
        # "T100_N10_L1": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N10_", "L[1]"]),
        # "T100_N20_L1": gather_experiments(MaskedDQN_input_ChainEnv1_len40_dir, name_string_conditions=["T100_N20_", "L[1]"]),

        
    }

    plot_experiments(agent_dict, "Runs/Test", name="ChainV1_Len40_Comparison_T100")
