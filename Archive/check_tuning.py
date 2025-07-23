import os
import optuna

# 1) Point to your DB file
db_path = os.path.expanduser("Runs/Tune/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-10000)/A2C/grid_search_seed[1]/optuna_study.db")
storage_url = f"sqlite:///{db_path}"

# 2) Load the study (use the same name_tag & seed you used when creating it)
study = optuna.load_study(
    study_name="grid_search_seed[1]",
    storage=storage_url
)

# 3) Inspect the best trial
best = study.best_trial
print(f"▶ Best trial number: {best.number}")
print(f"▶ Best objective value: {best.value}")
print("▶ Best hyperparameters:")
for k, v in best.params.items():
    print(f"    {k}: {v}")