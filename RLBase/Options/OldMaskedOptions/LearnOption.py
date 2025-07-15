def learn_options(exp_path):
    agent_lst, trajectories_lst = [], []
    for exp_path in exp_path_list:
        args = BaseExperiment.load_args(exp_path)
        config = BaseExperiment.load_config(exp_path)

        # Load Agent
        agent = load_agent(os.path.join(exp_path, f"Run{run_ind}_Last_agent.t"))

        env = get_env(
            env_name=args.env,
            num_envs=args.num_envs,
            max_steps=args.episode_max_steps,
            render_mode="rgb_array_list", #args.render_mode,
            env_params=config.env_params,
            wrapping_lst=config.env_wrapping,
            wrapping_params=config.wrapping_params,
        )
        experiment = BaseExperiment(env, agent, config=config, args=args, train=False)
        metrics = experiment.multi_run(num_runs=1, num_episodes=1, dump_transitions=True, dump_metrics=False)
        transitions = metrics[0][0]['transitions']
        trajectory = [(obs, action) for obs, action, *_ in transitions] #Only observations and actions
        agent_lst.append(agent)
        trajectories_lst.append(trajectory)