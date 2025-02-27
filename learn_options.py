 
import os

from Experiments import BaseExperiment
from Environments import get_env
from OfflineOptionLearner.MaskOptions import MaskedOptionLearner_v1


if __name__ == "__main__":

    agent_type = "DQN"
    exp_path = "Runs/Train/MiniGrid-Empty-5x5-v0_DQN_seed[123123]_20250226_150731"

    # Load Environment
    env_config = BaseExperiment.load_environment(exp_path)
    env = get_env(**env_config, render_mode="rgb_array_list")

    # Load Agent
    config = BaseExperiment.load_config(exp_path)
    agent = config.AGENT_DICT[agent_type](env)
    agent.reset(123123)    
    agent.load(os.path.join(exp_path, "Policy_Run1_Last.t"))

    # Load Transitions
    all_transitions = BaseExperiment.load_transitions(exp_path)

    # Learning Options
    option_learner = MaskedOptionLearner_v1(agent, all_transitions)
    option_learner.random_search()