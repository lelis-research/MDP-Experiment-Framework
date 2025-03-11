import os
import pickle
from matplotlib import pyplot as plt

from RLBase.Experiments import BaseExperiment
from RLBase.Evaluate import AnalyzeMultiExp, SingleExpAnalyzer
from RLBase.Environments import get_env
from RLBase import load_policy, load_agent

def visualize(experiment_path, run_number, episode_number):
    analyzer = SingleExpAnalyzer(exp_path=experiment_path)
    analyzer.generate_video(run_number, episode_number)


if __name__ == "__main__":
    agent_dict = {
            "DQN": "Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 30}_DQN_seed[123123]_20250310_185658",
            "Masked_Input": "Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 30}_MaskedDQN_seed[123123]_20250310_190218",
            "Masked_1": "Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 30}_MaskedDQN_seed[123123]_20250310_190314",
            "Masked_3": "Runs/Train/MiniGrid-ChainEnv-v0_{'chain_length': 30}_MaskedDQN_seed[123123]_20250310_190341"
            # Add more experiments as needed.
        }
    AnalyzeMultiExp(agent_dict, "Runs/Test", name_tag="Chain")
    agent_dict = {
        "DQN": "Runs/Train/MiniGrid-ChainEnv-v1_{'chain_length': 40}_DQN_seed[123123]_20250310_192233",
        "Masked_Input": "Runs/Train/MiniGrid-ChainEnv-v1_{'chain_length': 40}_MaskedDQN_seed[123123]_20250310_192349",
        "Masked_1": "Runs/Train/MiniGrid-ChainEnv-v1_{'chain_length': 40}_MaskedDQN_seed[123123]_20250310_192309",
        "Masked_3": "Runs/Train/MiniGrid-ChainEnv-v1_{'chain_length': 40}_MaskedDQN_seed[123123]_20250310_192336"
        # Add more experiments as needed.
        }
    AnalyzeMultiExp(agent_dict, "Runs/Test", name_tag="Door Chain")
    exit(0)


    exp_name = "MiniGrid-Empty-5x5-v0_{}_DQN_seed[123123]_20250306_113125"
    train_path = f"Runs/Train/{exp_name}"
    test_path = f"Runs/Test/{exp_name}"
    # visualize(train_path, 1, 200)

    args = BaseExperiment.load_args(train_path)
    config = BaseExperiment.load_config(train_path)

    env = get_env(
        env_name=args.env,
        num_envs=args.num_envs,
        max_steps=args.episode_max_steps,
        render_mode="rgb_array_list", #args.render_mode,
        env_params=config.env_params,
        wrapping_lst=config.env_wrapping,
        wrapping_params=config.wrapping_params,
    )
    agent = load_agent(os.path.join(train_path, "Run1_Last_agent.t"))
    # policy = load_policy(os.path.join(train_path, "Run1_Last_policy.t"))
    # print(policy)
    # agent.hp.update(epsilon=0.0)

    experiment = BaseExperiment(env, agent, test_path, train=False, args=args)
    metrics = experiment.multi_run(num_runs=1, num_episodes=5, seed_offset=1)
    analyzer = SingleExpAnalyzer(exp_path=test_path)
    analyzer.generate_video(1, 1)
    analyzer.save_seeds(test_path)
