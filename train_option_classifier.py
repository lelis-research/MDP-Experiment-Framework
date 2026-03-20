from RLBase.Trainers import OnlineTrainer
import pickle

@staticmethod
def load_dumped_option_rollouts(path):
    records = []
    with open(path, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except EOFError:
                break
    return records

path = "Runs/Train/MiniGrid-UnlockPickupLimitedColor-v0_auto_drop-False/FullyObs_OneHotImageDirCarry/VQOptionCritic/test_seed[123123]"
metrics = OnlineTrainer.load_all_run_metrics(path)
agent_logs = OnlineTrainer.load_agent_logs(path)

for r in range(len(agent_logs)):
    for e in range(len(agent_logs[r])):
        print(agent_logs[r][e].keys())
        # print(agent_logs[r][e]["code_book"].shape)
        # print(agent_logs[r][e]["num_options"].shape)
        # print(agent_logs[r][e]["option_index"].shape)
        print(agent_logs[r][e]["option_rollout"][0][0].keys())
        exit(0)


