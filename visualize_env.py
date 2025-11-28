import argparse
import argcomplete
import os
from PIL import Image
import json
import matplotlib.pyplot as plt

from RLBase.Environments import get_env, ENV_LST
from Configs.loader import load_config

def parse():
    parser = argparse.ArgumentParser()
    # Environment name
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0", choices=ENV_LST, help="which environment")
    # Add a name tag
    parser.add_argument("--name_tag", type=str, default="", help="name tag for saved image")
    # List of wrappers for the environment
    parser.add_argument("--env_wrapping",   type=json.loads, default="[]", help="list of wrappers")
    # A list of dictionary of the parameters for each wrapper
    parser.add_argument("--wrapping_params", type=json.loads, default="[]", help="list of dictionary represeting the parameters for each wrapper")
    # A dictionary of the environment parameters
    parser.add_argument("--env_params",     type=json.loads, default="{}", help="dictionary of the env parameters")
    argcomplete.autocomplete(parser)
    return parser.parse_args()

def main():
    args = parse()
    
    runs_dir = "Runs/Figures/"
    os.makedirs(runs_dir, exist_ok=True)  
    
    env = get_env(env_name=args.env, 
                  render_mode="rgb_array",
                  env_params   = args.env_params,
                  wrapping_lst = args.env_wrapping,
                  wrapping_params = args.wrapping_params,
                  )
    env.reset()
    frame = env.envs[0].render()
    
    # fig, ax = plt.subplots(figsize=(6,6))
    # ax.text(0.5, 0.5, frame, family="monospace", fontsize=10,
    #         ha="center", va="center")
    # ax.axis("off")

    # path = os.path.join(runs_dir, f"{args.env}_{args.name_tag}.pdf")
    # fig.savefig(path, bbox_inches="tight", format="pdf")


    img = Image.fromarray(frame)
    path = os.path.join(runs_dir, f"{args.env}_{args.name_tag}.png")
    img.save(path)

if __name__ == "__main__":
    main()