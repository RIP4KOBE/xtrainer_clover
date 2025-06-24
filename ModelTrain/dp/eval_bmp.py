"""
Usage:
python eval_bmp.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o
data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import time
import wandb
import json
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from omegaconf import  open_dict
from typing import Dict, Union
from ModelTrain.dp.transform_utils import quat_multiply
from ModelTrain.dp.bimanual_motion_prior.base_workspace import BaseWorkspace
from ModelTrain.dp.learner import BaseLowdimPolicy
from ModelTrain.dp.bimanual_motion_prior.pytorch_util import dict_apply
from ModelTrain.dp.utils import quaternion_from_6d_rotation
from scipy.spatial.transform import Rotation as R

def quaternion_multiply(q1: Union[torch.Tensor, np.ndarray], q2: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Batch quaternion multiplication q1 * q2
    """
    if isinstance(q1, np.ndarray):
        q1 = torch.from_numpy(q1).float()
    if isinstance(q2, np.ndarray):
        q2 = torch.from_numpy(q2).float()

    # q1, q2: [..., 4] where last dim is [x, y, z, w]
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack([x, y, z, w], dim=-1)


def strip_prefix_from_state_dict(state_dict, prefix="model."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # 去掉前缀
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def load_policy(checkpoint_path, device="cuda:0"):
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    model_cfg = cfg.policy.model
    model_cfg.cond_dim = 0
    model_cls = hydra.utils.get_class(model_cfg._target_)
    model_kwargs = OmegaConf.to_container(model_cfg, resolve=True)
    model_kwargs.pop("_target_", None)
    model = model_cls(**model_kwargs)
    state_dict = payload['state_dicts']['model']
    state_dict = strip_prefix_from_state_dict(state_dict, prefix="model.")
    model.load_state_dict(state_dict)

    if cfg.training.use_ema:
        ema_cfg = cfg.ema
        ema_cls = hydra.utils.get_class(ema_cfg._target_)
        # ema_model = ema_cls(model, **OmegaConf.to_container(ema_cfg, resolve=True))
        ema_model_kwargs = OmegaConf.to_container(ema_cfg, resolve=True)
        ema_model_kwargs.pop("_target_", None)
        ema_model = ema_cls(**ema_model_kwargs)
        ema_model.load_state_dict(payload['state_dicts']['ema_model'])
        model = ema_model.ema_model  # 获取内部的平滑模型参数

    model.to(device)
    model.eval()
    return model

def vis_action(sample, l_visualize_strat=None, r_visualize_strat=None):
    """
    Visualize predicted bimanual delta pose.

    Args:
        action (B, T, action_dim): Predicted action tensor, where B is batch size, T is sequence length, and action_dim is the dimension of the action space.
        l_visualize_strat: target pose for the left arm
        r_visualize_strat: target pose for the right arm
    """

    l_visualize_strat = l_visualize_strat
    r_visualize_strat = r_visualize_strat
    traj_len = sample.shape[1]

    # Setup figure
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.set_title(f"Left Arm -  Trajectories Distribution")
    ax2.set_title(f"Right Arm - Trajectories Distribution")

    for ax in (ax1, ax2):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True, alpha=0.3)

    # Color map for different starting poses
    colors = plt.cm.rainbow(np.linspace(0, 1))

    # Plot starting poses with larger markers
    ax1.scatter(*l_visualize_strat[0:3], color="r", marker='o', s=100,
                edgecolors='black', linewidth=2,
                label=f'Left Start', alpha=1.0)
    ax2.scatter(*r_visualize_strat[0:3], color="b", marker='o', s=100,
                edgecolors='black', linewidth=2,
                label=f'Right Start', alpha=1.0)


    for i in range(sample.shape[0]):
        action = sample[i]  # Get the i-th action sequence
        # extract left and right arm actions
        l_traj_pos = l_visualize_strat[0:3] + action[:, 0:3]
        r_traj_pos = r_visualize_strat[0:3] + action[:, 10:13]
        l_quat_delta = quaternion_from_6d_rotation(action[:, 3:9])
        r_quat_delta = quaternion_from_6d_rotation(action[:, 13:19])
        l_traj_quat = quaternion_multiply(np.broadcast_to(l_visualize_strat[3:7], (traj_len, 4)),
                                          l_quat_delta).cpu().numpy()
        r_traj_quat = quaternion_multiply(np.broadcast_to(r_visualize_strat[3:7], (traj_len, 4)),
                                          r_quat_delta).cpu(

        ).numpy()

        # Plot trajectory positions
        alpha = 0.3
        color_index = 1
        ax1.plot(l_traj_pos[:, 0], l_traj_pos[:, 1], l_traj_pos[:, 2],
                 color='b', alpha=alpha, linewidth=1.5)
        ax2.plot(r_traj_pos[:, 0], r_traj_pos[:, 1], r_traj_pos[:, 2],
                 color='g', alpha=alpha, linewidth=1.5)

        # Plot trajectory orientations as quaternions
        # Left arm orientation arrow
        l_traj_rot = R.from_quat(l_traj_quat[-1])  # latest quaternion
        l_direction = l_traj_rot.apply([1, 0, 0])  # x-axis direction in base frame

        ax1.quiver(l_traj_pos[-1][0], l_traj_pos[-1][1], l_traj_pos[-1][2],
                   l_direction[0], l_direction[1], l_direction[2],
                   length=0.05, normalize=True, color='r')

        # Right arm orientation arrow
        r_traj_rot = R.from_quat(r_traj_quat[-1])
        r_direction = r_traj_rot.apply([1, 0, 0])  # x-axis

        ax2.quiver(r_traj_pos[-1][0], r_traj_pos[-1][1], r_traj_pos[-1][2],
                   r_direction[0], r_direction[1], r_direction[2],
                   length=0.05, normalize=True, color='r')

        plt.tight_layout()

        # Save figure
        # if save_fig:
        #     save_path = self.output_dir / 'trajectory_visualization.png'
        #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
        #     print(f"\nVisualization saved to: {save_path}")

    plt.show()


def eval_policy(policy: BaseLowdimPolicy, batch_size=128, l_visualize_strat=None, r_visualize_strat=None):
    device = policy.device
    dtype = policy.dtype

    # start evaluation
    policy.reset()

    #record time
    time0 = time.time()
    with torch.no_grad():
        action_dict = policy.predict_action(batch_size)

    # device_transfer
    np_action_dict = dict_apply(action_dict,
                                lambda x: x.detach().to('cpu').numpy())

    action = np_action_dict['action_pred']
    time1 = time.time()

    print("Predicted action shape:", action.shape)
    print("Time taken for prediction:", time1 - time0, "seconds")

    vis_action(action, l_visualize_strat, r_visualize_strat)

@click.command()
@click.option('-c', '--checkpoint', default='/home/zhuoli/dobot_xtrainer/model/bimanual_motion_prior/2025.06.24/00.33.33_train_bimanual_motion_prior/checkpoints/latest.ckpt', required=True)
@click.option('-o', '--output_dir', default='eval/bimanual_motion_prior', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    with open_dict(cfg):
        cfg._target_ = 'ModelTrain.dp.train_bmp.TrainBimanualMotionPrior'
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # load policy from checkpoint
    # policy = load_policy(checkpoint, device=device)
    # print("Loaded policy from checkpoint:", checkpoint)

    # evaluate the policy
    l_visualize_strat = np.array([-0.11750221, -0.3450092, 0.41539302,
                                  3.89601281525054e-06, -0.707109378514511, -0.707104183808496, -6.49338013368839e-06])
    r_visualize_strat = np.array([0.11749968, -0.34500929, 0.41539644,
                                  -1.29865980772817e-06, -0.707106781180585, 0.707106781190125, 1.29866934832097e-06])

    # l_visualize_strat = np.array([-0.1198, -0.3435, 0.4162,  0.0009, -0.7068, -0.7073, -0.0006])
    # r_visualize_strat = np.array([ 0.1181, -0.3470, 0.4125, -0.0013, -0.7072,  0.7069,  0.0019])

    eval_policy(policy, batch_size=128,l_visualize_strat=l_visualize_strat, r_visualize_strat=r_visualize_strat)

    # env_runner = hydra.utils.instantiate(
    #     cfg.task.env_runner,
    #     output_dir=output_dir)
    # runner_log = env_runner.run(policy)
    #
    # # dump log to json
    # json_log = dict()
    # for key, value in runner_log.items():
    #     if isinstance(value, wandb.sdk.data_types.video.Video):
    #         json_log[key] = value._path
    #     else:
    #         json_log[key] = value
    # out_path = os.path.join(output_dir, 'eval_log.json')
    # json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
