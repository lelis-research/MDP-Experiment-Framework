from RLBase.Agents.Utils import (
    TabularFeature,
    FLattenFeature,
    ImageFeature,
    HyperParameters,
)
from RLBase.Options.TransferOptions import TransferOptionLearner
from RLBase.Options.DecWholeOptions import DecWholeOptionsLearner
from RLBase.Options.FineTuneOptions import FineTuneOptionLearner
from RLBase.Options.MaskedOptions import MaskedOptionLearner
from RLBase.Experiments import BaseExperiment
from RLBase import load_policy, load_agent
from RLBase.Environments import get_env

import os
def exp_path_lst_to_agent_and_trajectory(exp_path_lst, run_ind_lst):
    agent_lst, trajectories_lst = [], []
    for exp_path, run_ind in zip(exp_path_lst, run_ind_lst):
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
    return agent_lst, trajectories_lst

device = "cpu" #cpu, mps, cuda


OPTION_DICT = {
    TransferOptionLearner.name: lambda exp_path_lst, run_ind_lst: TransferOptionLearner(
        agent_lst = exp_path_lst_to_agent_and_trajectory(exp_path_lst, run_ind_lst)[0],
        trajectories_lst = exp_path_lst_to_agent_and_trajectory(exp_path_lst, run_ind_lst)[1],
        hyper_params = HyperParameters(max_option_len=1)),
    
    DecWholeOptionsLearner.name: lambda exp_path_lst, run_ind_lst: DecWholeOptionsLearner(
        agent_lst = exp_path_lst_to_agent_and_trajectory(exp_path_lst, run_ind_lst)[0],
        trajectories_lst = exp_path_lst_to_agent_and_trajectory(exp_path_lst, run_ind_lst)[1],
        hyper_params = HyperParameters(max_option_len=20, max_num_options=10, 
                                       n_neighbours=1, n_restarts=3, n_iteration=5)),
    
    FineTuneOptionLearner.name: lambda exp_path_lst, run_ind_lst: FineTuneOptionLearner(
        agent_lst = exp_path_lst_to_agent_and_trajectory(exp_path_lst, run_ind_lst)[0],
        trajectories_lst = exp_path_lst_to_agent_and_trajectory(exp_path_lst, run_ind_lst)[1],
        hyper_params = HyperParameters(max_option_len=20, max_num_options=10, 
                                       n_neighbours=1, n_restarts=3, n_iteration=5,
                                       n_epochs=100, actor_lr=1e-3)),
    
    MaskedOptionLearner.name: lambda exp_path_lst, run_ind_lst: MaskedOptionLearner(
        agent_lst = exp_path_lst_to_agent_and_trajectory(exp_path_lst, run_ind_lst)[0],
        trajectories_lst = exp_path_lst_to_agent_and_trajectory(exp_path_lst, run_ind_lst)[1],
        hyper_params = HyperParameters(max_option_len=20, max_num_options=10, 
                                       n_neighbours=1, n_restarts=3, n_iteration=5,
                                       n_epochs=100, mask_lr=1e-3, 
                                       masked_layers=['input', '1'])),
}
   