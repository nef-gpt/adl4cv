import pandas as pd
import wandb

api = wandb.Api()
wandb.login()
entity, project = "adl-for-cv", "shapenet_token_transformer"
run_name = "run-2024-07-14-11-42-04"

runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains output keys/values for
    # metrics such as accuracy.
    #  We call ._json_dict to omit large files
    if run.name == run_name:
        summary_list.append(run.history())

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

summery_df = pd.DataFrame(summary_list[0])

config_df = pd.DataFrame(
    {"config": config_list, "name": name_list}
)

summery_df.to_csv("run_large.csv")
#config_df.to_csv("wandb_config_df.csv")

