import argparse
import os
import pickle

from sympy import false
import torch.multiprocessing as mp

from dobot_control.agents.dp_agent import BimanualDPAgent


def eval_ckpts(ckpt_paths, eval_dir, save_path, sampling = False):
    mse_dict = {}

    # Check if the save path already exists
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            mse_dict = pickle.load(f)
        print(f"Loaded previous MSE dict from {save_path}")
    else:
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Allow existing directories
        mse_dict = {}  # Initialize an empty dictionary if no previous data
        print(f"Created directory for saving results at {os.path.dirname(save_path)}")


    eval_loader = None
    last_arg = None

    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.basename(os.path.dirname(ckpt_path))
        ckpt_num = os.path.basename(ckpt_path)
        agent = BimanualDPAgent(ckpt_path)
        if eval_loader is None:
            eval_loader = agent.dp.get_eval_loader(eval_dir, prefix=None)
        elif (
            agent.dp_args["representation_type"] != last_arg["representation_type"]
            or agent.dp_args["camera_indices"] != last_arg["camera_indices"]
        ):
            eval_loader = agent.dp.get_eval_loader(eval_dir)
        last_arg = agent.dp_args
        mse, action_mse = agent.dp.eval_dir(eval_loader, sampling=sampling)
        if mse_dict.get(ckpt_name) is None:
            mse_dict[ckpt_name] = {}
        mse_dict[ckpt_name]["config"] = agent.dp_args

        mse_dict[ckpt_name][ckpt_num] = {}
        mse_dict[ckpt_name][ckpt_num]["mse"] = mse
        mse_dict[ckpt_name][ckpt_num]["action_mse"] = action_mse

        print(f"MSE for {ckpt_name}: {mse}")

    with open(save_path, "wb") as f:
        pickle.dump(mse_dict, f)
    print(f"Saved MSE dict to {save_path}")



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = argparse.ArgumentParser()
    args.add_argument(
        "--ckpt_path",
        nargs="+",
        type=str,
        default=[
            "/home/zhuoli/dobot_xtrainer/model/dp_tidying_up_bowls_a_0920/1006_203352_ZbrD-camera=012-identity=False-repr=IP-oh=1-ah=8-ph=16-prefix=None-do=0.0-imgos=32-wd=1e-05-use_ddim=False-binarize_touch=False/last.ckpt",
        ],
    )
    args.add_argument(
        "--eval_dir",
        type=str,
        default="/home/zhuoli/dobot_xtrainer/ModelTrain/dp/split_data/collect_data_test",
    )
    args.add_argument("--save_path", type=str, default=None)
    args.add_argument("--sampling", type=bool, default=True)


    args = args.parse_args()

    if args.save_path is None:
        data_name = os.path.basename(args.eval_dir)
        args.save_path = "/home/zhuoli/dobot_xtrainer/ModelTrain/dp/eval_results/eval_{}.pkl".format(data_name)
    eval_ckpts(args.ckpt_path, args.eval_dir, args.save_path, args.sampling)
