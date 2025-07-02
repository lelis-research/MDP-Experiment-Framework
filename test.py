import os

from RLBase.Experiments import BaseExperiment
from RLBase.Evaluate import SingleExpAnalyzer
from RLBase.Environments import get_env
from RLBase import load_policy, load_agent


if __name__ == "__main__":

    exp_name = "MiniGrid-SimpleCrossingS9N1-v0_{}/A2C/_seed[123123]_20250612_152550"
    train_path = f"Runs/Train/{exp_name}"
    test_path = f"Runs/Test/{exp_name}"

    # If saved the frames during training you can visualize them like this
    # analyzer = SingleExpAnalyzer(exp_path=train_path)
    # analyzer.generate_video(1, 1)

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
    analyzer.generate_video(1, 2)
    analyzer.generate_video(1, 3)
    analyzer.generate_video(1, 4)
    analyzer.generate_video(1, 5)


    analyzer.save_seeds(test_path)
