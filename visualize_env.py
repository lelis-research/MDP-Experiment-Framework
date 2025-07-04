import argparse
import argcomplete
import os
from PIL import Image


from RLBase.Environments import get_env, ENV_LST
from Configs.loader import load_config

def parse():
    parser = argparse.ArgumentParser()
    # Config file name
    parser.add_argument("--config", type=str, default="base_config", help="path to the experiment config file")
    # Environment name
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0", choices=ENV_LST, help="which environment")
    # Add a name tag
    parser.add_argument("--name_tag", type=str, default="", help="name tag for saved image")

    argcomplete.autocomplete(parser)
    return parser.parse_args()

def main():
    args = parse()
    config_path = os.path.join("Configs", f"{args.config}.py")
    config = load_config(config_path)
    
    runs_dir = "Runs/Figures/"
    os.makedirs(runs_dir, exist_ok=True)  
    
    env = get_env(env_name=args.env, 
                  render_mode="rgb_array",
                  env_params   = config.env_params,
                  wrapping_lst = config.env_wrapping,
                  wrapping_params = config.wrapping_params,
                  )
    env.reset()
    frame = env.render()
    img = Image.fromarray(frame)
    path = os.path.join(runs_dir, f"{args.env}_{args.name_tag}.png")
    img.save(path)

if __name__ == "__main__":
    main()