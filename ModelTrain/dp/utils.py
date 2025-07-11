import json
import os
import pprint
import random
import string
import tempfile
import time

import torch
import torch.nn.functional as F
import yaml
from copy import copy
from typing import Union
from datetime import datetime
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath.base import q2r

import absl.flags
import numpy as np
import quaternion
import wandb
from absl import logging
from dataset import unnormalize_6d_pose
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags
from spatialmath.base import q2r
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
)


def save_args(args, output_dir):
    args_dict = vars(args)
    args_str = "\n".join(f"{key}: {value}" for key, value in args_dict.items())
    with open(os.path.join(output_dir, "args_log.txt"), "w") as file:
        file.write(args_str)
    with open(os.path.join(output_dir, "args_log.json"), "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

def generate_random_string(length=4, characters=string.ascii_letters + string.digits):
    """
    Generate a random string of the specified length using the given characters.

    :param length: The length of the random string (default is 12).
    :param characters: The characters to choose from when generating the string
                      (default is uppercase letters, lowercase letters, and digits).
    :return: A random string of the specified length.
    """
    return "".join(random.choice(characters) for _ in range(length))

def get_eef_delta(eef_pose, eef_pose_target):
    pos_delta = eef_pose_target[:3] - eef_pose[:3]
    # axis angle to quaternion
    ee_rot = quaternion.from_rotation_vector(eef_pose[3:])
    ee_rot_target = quaternion.from_rotation_vector(eef_pose_target[3:])
    # calculate the quaternion difference
    rot_delta = quaternion.as_rotation_vector(ee_rot_target * ee_rot.inverse())
    return np.concatenate((pos_delta, rot_delta))


def rotation_vector_to_sixd(rot_vec):
    """Convert 3D rotation vector to 6D rotation representation."""
    rot_mat = R.from_rotvec(rot_vec).as_matrix()
    return rot_mat[:, :2].flatten()  # [R11, R21, R31, R12, R22, R32]


def sixd_to_rotation_vector(sixd):
    """Convert 6D rotation representation to 3D rotation vector using SciPy."""
    sixd = np.array(sixd).reshape(3, 2)
    a1, a2 = sixd[:, 0], sixd[:, 1]

    # Gram-Schmidt orthogonalization
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rot_mat = np.column_stack([b1, b2, b3])

    # Convert rotation matrix to rotation vector using SciPy
    rot = R.from_matrix(rot_mat)
    rot_vec = rot.as_rotvec()
    return rot_vec


def sixd_to_rotation_matrix(sixd: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix using numpy."""
    sixd = np.array(sixd).reshape(3, 2)
    a1, a2 = sixd[:, 0], sixd[:, 1]

    # Gram-Schmidt orthogonalization
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rot_mat = np.column_stack([b1, b2, b3])

    return rot_mat


# def pose_to_SE3(pose):
#     """
#     Convert 10D pose (3D pos + 6D rot) to SE(3) matrix (4x4).
#     pose: (10,) or (B, 10)
#     Assumes rotation is 6D representation -> converts to rotation matrix.
#     """
#     pos = pose[..., :3]
#     rot_6d = pose[..., 3:9]
#     rot_mat = rotation_6d_to_matrix(rot_6d)  # shape (..., 3, 3)
#     T = torch.eye(4, device=pose.device).expand(*pose.shape[:-1], 4, 4).clone()
#     T[..., :3, :3] = rot_mat
#     T[..., :3, 3] = pos
#     return T

def pose_to_SE3(pose):
    """
    Convert 10D pose (3D pos + 6D rot) to torchlie-compatible SE(3) matrix (3x4).

    Args:
        pose: Tensor of shape (..., 10)
              where pose[..., :3] is position,
                    pose[..., 3:9] is 6D rotation representation

    Returns:
        T: Tensor of shape (..., 3, 4), compatible with torchlie.SE3
    """
    pos = pose[..., :3]  # (..., 3)
    rot_6d = pose[..., 3:9]  # (..., 6)
    rot_mat = rotation_6d_to_matrix(rot_6d)  # (..., 3, 3)

    # Combine rotation and translation into a (3, 4) matrix
    T = torch.cat([rot_mat, pos.unsqueeze(-1)], dim=-1)  # (..., 3, 4)
    return T


def batch_rotation_matrix_from_6d_rotation(sixd_array: Union[torch.tensor, np.ndarray]) -> np.ndarray:
    """
    Convert a batch of 6D rotation representations to quaternions.

    Args:
        sixd_array: np.ndarray of shape (batch, horizon, 6)

    Returns:
        quaternions: np.ndarray of shape (batch, horizon, 4)
    """
    device = None

    # if isinstance(sixd_array, torch.Tensor):
    #     device = sixd_array.device
    #     sixd_array = sixd_array.cpu().numpy()

    batch, horizon, _ = sixd_array.shape
    quaternions = np.zeros((batch, horizon, 4), dtype=np.float32)

    for b in range(batch):
        for t in range(horizon):
            sixd = sixd_array[b, t]
            rot_mat = sixd_to_rotation_matrix(sixd)
            quat = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
            quaternions[b, t] = quat

    return quaternions


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Modified PyTorch version to match NumPy's behavior exactly.
    """
    # 重塑为(*, 3, 2)以匹配NumPy的reshape(3, 2)
    d6_reshaped = d6.view(*d6.shape[:-1], 3, 2)

    # 取第0列和第1列
    a1 = d6_reshaped[..., :, 0]  # 相当于NumPy的 sixd[:, 0]
    a2 = d6_reshaped[..., :, 1]  # 相当于NumPy的 sixd[:, 1]

    # Gram-Schmidt正交化
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    # 使用dim=-1来匹配column_stack的行为
    return torch.stack((b1, b2, b3), dim=-1)

def quaternion_from_6d_rotation(rot6d: Union[torch.tensor, np.ndarray]) -> torch.Tensor:
    """
    用 PyTorch3D 快速把 6D 旋转表示转回四元数。
    Args:
        rot6d: (..., 6) 6D 旋转表示
    Returns:
        q: (..., 4) 四元数，格式 [x, y, z, w]
    """

    if isinstance(rot6d, np.ndarray):
        rot6d = torch.from_numpy(rot6d).float()

    R = rotation_6d_to_matrix(rot6d) # customized rotation_6d_to_matrix() function to align with numpy's convention
    # R = R.transpose(-1, -2)  # align with numpy convention (column-major order)
    q = matrix_to_quaternion(R)              # (..., 4)
    return q

# === Approximate Jacobian matrix for transforming pose noise (single sample)
def compute_approx_jacobian(R_ref: torch.Tensor) -> torch.Tensor:
    """
    R_ref: [3, 3] - reference rotation matrix
    return: [9, 9] - approximate Jacobian matrix for pose transformation
    """
    J = torch.zeros(9, 9, device=R_ref.device)
    R_T = R_ref.T
    J[0:3, 0:3] = R_T                            # Translation part
    J[3:9, 3:9] = torch.kron(torch.eye(2, device=R_ref.device), R_T)  # Rotation part (6D)
    return J

# === Main function: transform dual-arm action noise from world frame to reference frame
def noise_jocabian_transform(epsilon_abs: torch.Tensor, ref_action: torch.Tensor) -> torch.Tensor:
    """
    Transform dual-arm action noise from the world frame to a reference frame using Jacobian Transformation.

    Parameters:
        epsilon_abs: [B, 16, 20] - noise in world/global frame for a single action sequence
        ref_action:  [20,]    - reference action (used to extract reference rotation matrices)

    Returns:
        epsilon_rel: [B, 16, 20] - transformed noise in reference frame
    """
    B, T, _ = epsilon_abs.shape

    # Extract left/right 6D rotation from single reference action
    left_rot6d = ref_action[3:9]  # [6,]
    right_rot6d = ref_action[13:19]  # [6,]

    # Convert to rotation matrices
    R_left = rotation_6d_to_matrix(left_rot6d)  # [3, 3]
    R_right = rotation_6d_to_matrix(right_rot6d)  # [3, 3]

    # Compute Jacobians
    J_left = compute_approx_jacobian(R_left)  # [9, 9]
    J_right = compute_approx_jacobian(R_right)  # [9, 9]

    # Expand to match batch size
    J_left_exp = J_left.unsqueeze(0).expand(B * T, 9, 9)  # [B*T, 9, 9]
    J_right_exp = J_right.unsqueeze(0).expand(B * T, 9, 9)  # [B*T, 9, 9]

    # Split noise
    eps_left = epsilon_abs[:, :, 0:9]  # [B, T, 9]
    eps_gripL = epsilon_abs[:, :, 9:10]  # [B, T, 1]
    eps_right = epsilon_abs[:, :, 10:19]  # [B, T, 9]
    eps_gripR = epsilon_abs[:, :, 19:20]  # [B, T, 1]

    # Reshape for batched matmul
    eps_left_flat = eps_left.reshape(B * T, 9, 1)    # [B*T, 9, 1]
    eps_right_flat = eps_right.reshape(B * T, 9, 1)  # [B*T, 9, 1]

    # Apply Jacobian transformation: ε_rel = J^T @ ε_abs
    eps_left_rel_flat = torch.bmm(J_left_exp.transpose(1, 2), eps_left_flat)
    eps_right_rel_flat = torch.bmm(J_right_exp.transpose(1, 2), eps_right_flat)

    # Reshape back to [B, T, 9]
    eps_left_rel = eps_left_rel_flat.squeeze(-1).view(B, T, 9)  # [B, T, 9]
    eps_right_rel = eps_right_rel_flat.squeeze(-1).view(B, T, 9)  # [B, T, 9]

    # Gripper noise doesn't need transformation (scalar values)
    # Reassemble full transformed noise
    epsilon_rel = torch.cat([
        eps_left_rel,  # [B, T, 9]
        eps_gripL,  # [B, T, 1]
        eps_right_rel,  # [B, T, 9]
        eps_gripR  # [B, T, 1]
    ], dim=-1)  # [B, T, 20]

    return epsilon_rel

def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, "automatically defined flag")
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, "automatically defined flag")
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, "automatically defined flag")
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, "automatically defined flag")
        else:
            raise ValueError("Incorrect value type")
    return kwargs

def _parse(s, variant):
    orig_s = copy(s)
    final_s = []

    while len(s) > 0:
        indx = s.find("{")
        if indx == -1:
            final_s.append(s)
            break
        final_s.append(s[:indx])
        s = s[indx + 1 :]
        indx = s.find("}")
        assert indx != -1, "can't find the matching right bracket for {}".format(orig_s)
        final_s.append(str(variant[s[:indx]]))
        s = s[indx + 1 :]

    return "".join(final_s)

def get_user_flags(flags, flags_def):
    output = {}
    for key in sorted(flags_def):
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            flatten_config_dict(val, prefix=key, output=output)
        else:
            output[key] = val

    return output

def get_config(config_path=None):
    if config_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_file_dir, 'configs/configs.yaml')
    assert config_path and os.path.exists(config_path), f'configs file does not exist ({config_path})'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def flatten_config_dict(config, prefix=None, output=None):
    if output is None:
        output = {}
    for key, val in sorted(config.items()):
        if prefix is not None:
            next_prefix = "{}.{}".format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            flatten_config_dict(val, prefix=next_prefix, output=output)
        else:
            output[next_prefix] = val
    return output

def to_config_dict(flattened):
    config = config_dict.ConfigDict()
    for key, val in flattened.items():
        c_config = config
        ks = key.split(".")
        for k in ks[:-1]:
            if k not in c_config:
                c_config[k] = config_dict.ConfigDict()
            c_config = c_config[k]
        c_config[ks[-1]] = val
    return config.to_dict()

def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}

def fk_solver(joint_angles, ee_link=None):
    """
    Compute forward kinematics (FK) to transform joint angles into end-effector pose in Cartesian space.

    Parameters:
    - joint_angles: (batch_size, prediction_horizon, 7) -> 7 joint angles per arm

    Returns:
    - End-effector pose (batch_size, prediction_horizon, action_dim).
    """
    # load the URDF files for the left and right arms
    # urdf = "/home/zhuoli/xtrainer_clover/assets/urdf/nova2_robot.urdf"
    urdf = "/home/zhuoli/dobot_xtrainer/assets/urdf/nova2_robot.urdf"
    xtrainer_arm = rtb.robot.ERobot.URDF(urdf)

    if joint_angles.ndim == 2:
        joint_angles = joint_angles[np.newaxis, :, :]  # Add batch dimension
    elif joint_angles.ndim == 1:
        joint_angles = joint_angles[np.newaxis, np.newaxis, :]

    assert joint_angles.shape[2] == 6, "Joint angles must have 6 values."

    batch_size, prediction_horizon, _ = joint_angles.shape
    ee_position = np.zeros((batch_size, prediction_horizon, 3))  # (x, y, z)
    ee_orientations = np.zeros((batch_size, prediction_horizon, 3, 3))  # 3x3 rotation matrix

    for b in range(batch_size):
        for t in range(prediction_horizon):
            angles = joint_angles[b, t, :]
            T_ee = xtrainer_arm.fkine(q=angles, end=ee_link)  # Forward kinematics to get end-effector pose
            T_ee = np.array(T_ee.A)
            ee_position[b, t, :] = T_ee[:3, 3]
            ee_orientations[b, t, :, :] = T_ee[:3, :3]

    return ee_position, ee_orientations

def ik_solver(ee_position, ee_orientation, initial_joint=None):
    """
    Solve IK for end-effector pose (position + orientation), with flexible orientation input.

    Args:
        ee_position (np.ndarray): shape (3,)
        ee_orientation: one of the following:
            - 3x3 rotation matrix
            - 4D quaternion (x, y, z, w)
            - scipy.spatial.transform.Rotation
        initial_joint (np.ndarray): optional initial joint guess

    Returns:
        joint_angle (np.ndarray): shape (6,)
        success_flag (bool)
    """
    urdf = "/home/zhuoli/dobot_xtrainer/assets/urdf/nova2_robot.urdf"
    robot = rtb.robot.ERobot.URDF(urdf)

    # Ensure ee_position is a flat 3D vector
    ee_position = np.asarray(ee_position).flatten()

    # --- Normalize & convert orientation ---
    if isinstance(ee_orientation, R):  # scipy Rotation
        rot_matrix = ee_orientation.as_matrix()
    elif isinstance(ee_orientation, np.ndarray):
        ee_orientation = np.asarray(ee_orientation)
        if ee_orientation.shape == (3, 3):
            rot_matrix = ee_orientation
        elif ee_orientation.shape == (4,):  # quaternion
            rot_matrix = q2r(ee_orientation)  # spatialmath.base
        else:
            raise ValueError(f"Unsupported orientation shape: {ee_orientation.shape}")
    else:
        raise TypeError(f"Unsupported ee_orientation type: {type(ee_orientation)}")

    # Construct SE3 pose
    target_pose = SE3.Rt(R=rot_matrix, t=ee_position)

    # Call inverse kinematics
    solution = robot.ikine_LM(
        target_pose,
        q0=initial_joint,
        ilimit=15,
        slimit=50,
    )

    if solution.success:
        return solution.q, True
    else:
        print("IK failed")
        return np.zeros(robot.n), False


def align_trajs_to_origin(population_trajectories, traj_origin):
    """
    Align a batch of trajectories such that their Cartesian EE start points match the origin trajectory's start point.

    Args:
        population_trajectories: np.ndarray of shape (B, T, 14), batch of joint trajectories
        traj_origin: np.ndarray of shape (1, T, 14), reference trajectory to align to
        forward_kinematics: function that takes (B, T, 7) joint angles and returns (positions, orientations)
        inverse_kinematics: function that takes (B, T, 3) positions and returns (joint_angles, success_flags)

    Returns:
        aligned_population_trajectories: np.ndarray of shape (B, T, 14)
        aligned_left_ee_positions: np.ndarray of shape (B, T, 3)
        aligned_right_ee_positions: np.ndarray of shape (B, T, 3)
    """

    B, T, _ = population_trajectories.shape

    left_arm_trajs = population_trajectories[:, :, :6]    # (B, T, 7)
    right_arm_trajs = population_trajectories[:, :, 7:13] # (B, T, 7)

    left_ee_positions, left_ee_orientations = fk_solver(left_arm_trajs)   # (B, T, 3)
    right_ee_positions, right_ee_orientations = fk_solver(right_arm_trajs) # (B, T, 3)

    left_ee_positions = np.array(left_ee_positions)
    right_ee_positions = np.array(right_ee_positions)
    left_ee_orientations = np.array(left_ee_orientations)
    right_ee_orientations = np.array(right_ee_orientations)

    origin_left_arm = traj_origin[:6]
    origin_right_arm = traj_origin[7:13]

    origin_left_pos, origin_left_orientations = fk_solver(origin_left_arm)   # (1, T, 3)
    origin_right_pos, origin_right_orientations = fk_solver(origin_right_arm) # (1, T, 3)

    origin_left_pos = np.array(origin_left_pos)
    origin_right_pos = np.array(origin_right_pos)
    origin_left_orientations = np.array(origin_left_orientations)
    origin_right_orientations = np.array(origin_right_orientations)

    left_offsets = origin_left_pos[:, 0, :] - left_ee_positions[:, 0, :]    # (B, 3)
    right_offsets = origin_right_pos[:, 0, :] - right_ee_positions[:, 0, :] # (B, 3)
    # print("left_offsets", left_offsets, "right_offsets", right_offsets)

    aligned_left_ee_positions = left_ee_positions + left_offsets[:, np.newaxis, :]    # (B, T, 3)
    aligned_right_ee_positions = right_ee_positions + right_offsets[:, np.newaxis, :] # (B, T, 3)

    aligned_left_joints = np.zeros((B, T, 6))  # (B, T, 6)
    aligned_right_joints = np.zeros((B, T, 6)) # (B, T, 6)
    for b in range(B):
        for t in range(T):
            left_q, success_left = ik_solver(aligned_left_ee_positions[b, t, :], left_ee_orientations[b, t, :, :], origin_left_arm)
            right_q, success_right = ik_solver(aligned_right_ee_positions[b, t, :], right_ee_orientations[b, t, :, :], origin_right_arm)

            if success_left and success_right:
                aligned_left_joints[b, t, :] = left_q
                aligned_right_joints[b, t, :] = right_q

            else:
                print(f"IK failed at batch {b}, step {t}")

    # aligned_left_joints, success_left = ik_solver(aligned_left_ee_positions, left_ee_orientations, origin_left_arm)
    # aligned_right_joints, success_right = ik_solver(aligned_right_ee_positions, right_ee_orientations, origin_right_arm)

    aligned_population_trajectories = population_trajectories.copy()
    aligned_population_trajectories[:, :, :6] = aligned_left_joints
    aligned_population_trajectories[:, :, 7:13] = aligned_right_joints

    return aligned_population_trajectories, aligned_left_ee_positions, aligned_right_ee_positions


def bimanual_coordinator(mode: str,
                         traj_origin: np.ndarray,
                         best_trajectory: np.ndarray) -> np.ndarray:
    """
    Update traj_origin based on control mode and best_trajectory.

    Args:
        mode (str): one of ['left', 'right', 'bimanual']
        traj_origin (np.ndarray): shape (1, T, 20), original trajectory to update
        best_trajectory (np.ndarray): shape (1, T, 20), new best trajectory (usually B=1)

    Returns:
        np.ndarray: updated traj_origin with selected parts from best_trajectory
    """
    assert mode in ["left_eef_pos", "left_eef_rot", "left_gripper", "right_eef_pos", "right_eef_rot", "right_gripper", "bimanual"], f"Invalid mode: {mode}"
    # assert traj_origin.shape == 14 and best_trajectory.shape[2] == 14, "Trajectory dim must be 14"
    coordinate_traj = np.tile(traj_origin, (best_trajectory.shape[0], 1))

    print("coordinate_traj", coordinate_traj.shape, "best_trajectory", best_trajectory.shape)

    if mode == "left_eef_pos":
        # Only update left arm eef pose
        coordinate_traj[:, :3] = best_trajectory[:, :3]

    elif mode == "left_eef_rot":
        # Only update left arm eef rotation
        coordinate_traj[:, 3:9] = best_trajectory[:, 3:9]

    elif mode == "left_gripper":
        # Only update left arm gripper
        coordinate_traj[:, 9] = best_trajectory[:, 9]

    elif mode == "right_eef_pos":
        # Only update right arm jeef pose
        coordinate_traj[:, 10:13] = best_trajectory[:, 10:13]

    elif mode == "right_eef_rot":
        # Only update right arm eef rotation
        coordinate_traj[:, 13:19] = best_trajectory[:, 13:19]

    elif mode == "right_gripper":
        # Only update right arm gripper
        coordinate_traj[:, 19] = best_trajectory[:, 19]

    elif mode == "bimanual":
        # Update both arms
        coordinate_traj[:, :3] = best_trajectory[:, :3]
        coordinate_traj[:, 10:13] = best_trajectory[:, 10:13]

    return coordinate_traj

def bimanual_frame_transform(left_positions, right_positions):
    """
    Transform right arm end-effector positions to the left arm base frame.

    Parameters:
    - left_positions: np.ndarray of shape (batch, 16, 3)
    - right_positions: np.ndarray of shape (batch, 16, 3)

    Returns:
    - left_positions_transformed: same as input (batch, 16, 3)
    - right_positions_transformed: transformed to left base frame (batch, 16, 3)
    """

    assert left_positions.shape == right_positions.shape, \
        "Left and right positions must have the same shape (batch, 16, 3)"
    assert left_positions.shape[1:] == (16, 3), \
        "Positions must have shape (batch, 16, 3)"

    # Rotation matrix: 180 deg about Z (clockwise)
    R = np.array([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ])

    # Translation vector from right base to left base
    t = np.array([0.0, -1.08, 0.0])  # in meters

    # Apply transformation: p_left = R * p_right + t
    # right_positions: (batch, 16, 3)
    # R: (3, 3) → use einsum to apply across batch
    right_transformed = np.einsum('ij,btj->bti', R, right_positions) + t

    return left_positions, right_transformed


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


def quaternion_to_6d_rotation(q: Union[torch.tensor, np.ndarray]) -> torch.Tensor:
    """
    用 PyTorch3D 快速把四元数转成 6D 表示。
    Args:
        q: (..., 4) 四元数张量，格式 [x, y, z, w]
    Returns:
        rot6d: (..., 6) 6D 旋转表示
    """


    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q).float()

    # 1) 四元数 -> 3x3 旋转矩阵
    R = quaternion_to_matrix(q)              # (..., 3, 3)
    # 2) 矩阵 -> 6D 表示 (取前两列)
    rot6d = matrix_to_rotation_6d(R)         # (..., 6)
    return rot6d


def get_abs_traj_from_delta(traj_delta, initial_traj, device=None):
    """
    Compute absolute trajectory from delta and initial pose with 6D rotation.

    Args:
        traj_delta: (B, T, 20), torch.Tensor or np.ndarray
        initial_traj: (20,), torch.Tensor or np.ndarray
        device: torch.device to put the result

    Returns:
        Same type as input traj_delta: torch.Tensor or np.ndarray
    """
    if isinstance(traj_delta, np.ndarray):
        traj_delta = torch.from_numpy(traj_delta).to(device)
    if isinstance(initial_traj, np.ndarray):
        initial_traj = torch.from_numpy(initial_traj).to(device)

    # Ensure float32 dtype for consistency
    traj_delta = traj_delta.float()
    initial_traj = initial_traj.float()

    B, T, _ = traj_delta.shape
    init = initial_traj.unsqueeze(0).unsqueeze(0)  # (1, 1, 20)

    # Positions
    left_pos_abs = traj_delta[:, :, :3] + init[:, :, :3]
    right_pos_abs = traj_delta[:, :, 10:13] + init[:, :, 10:13]

    # Gripper
    left_gripper = traj_delta[:, :, 9:10]
    right_gripper = traj_delta[:, :, 19:20]

    # Rotations
    left_rot_delta = quaternion_from_6d_rotation(traj_delta[:, :, 3:9])
    right_rot_delta = quaternion_from_6d_rotation(traj_delta[:, :, 13:19])
    left_rot_init = quaternion_from_6d_rotation(init[:, :, 3:9].expand(B, T, -1))
    right_rot_init = quaternion_from_6d_rotation(init[:, :, 13:19].expand(B, T, -1))

    left_quat_abs = quaternion_multiply(left_rot_init, left_rot_delta)
    right_quat_abs = quaternion_multiply(right_rot_init, right_rot_delta)

    left_rot_abs = quaternion_to_6d_rotation(left_quat_abs)
    right_rot_abs = quaternion_to_6d_rotation(right_quat_abs)

    abs_traj = torch.cat([
        left_pos_abs, left_rot_abs, left_gripper,
        right_pos_abs, right_rot_abs, right_gripper
    ], dim=-1)

    return abs_traj



def kinematic_func_test():
    """
    Test the correctness of forward and inverse kinematics functions.
    """
    # Configuration
    batch_size = 2
    prediction_horizon = 3
    joint_dim = 6

    # Generate random joint angles within valid range
    np.random.seed(42)
    random_joint_angles = np.random.uniform(
        low=-np.pi, high=np.pi,
        size=(batch_size, prediction_horizon, joint_dim)
    )

    # Step 1: Forward Kinematics to obtain end-effector poses
    ee_pos, ee_orient = fk_solver(random_joint_angles)

    # Step 2: Inverse Kinematics to recover joint angles from poses
    recovered_joint_angles, success_flags = ik_solver(
        ee_pos, ee_orient, initial_joint=None
    )

    # Step 3: Forward Kinematics again on recovered joint angles
    ee_pos_recovered, ee_orient_recovered = fk_solver(recovered_joint_angles)

    # Step 4: Compute position and orientation errors
    pos_error = np.linalg.norm(ee_pos - ee_pos_recovered, axis=-1)  # Euclidean distance

    orient_error = np.zeros((batch_size, prediction_horizon))
    for b in range(batch_size):
        for t in range(prediction_horizon):
            R1 = ee_orient[b, t]
            R2 = ee_orient_recovered[b, t]
            dR = R1 @ R2.T
            # Compute orientation error using angle between rotation matrices
            angle_error = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))
            orient_error[b, t] = angle_error

    # Report results
    print("\n==== Kinematics Test Results ====")
    print("Joint angles (original):\n", random_joint_angles)
    print("Joint angles (recovered):\n", recovered_joint_angles)

    print("\nSuccess rate: {:.2f}%".format(100 * np.mean(success_flags)))
    print("Max position error: {:.6f} m".format(np.max(pos_error)))
    print("Mean position error: {:.6f} m".format(np.mean(pos_error)))
    print("Max orientation error: {:.6f} rad".format(np.max(orient_error)))
    print("Mean orientation error: {:.6f} rad".format(np.mean(orient_error)))
    print("=================================\n")

    # Optional assertions for automated testing
    assert np.all(pos_error < 1e-3), "Position error too large"
    assert np.all(orient_error < 1e-2), "Orientation error too large"
    assert np.all(success_flags), "Some IK solutions failed"



class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.mode = "online"
        config.project = "hato"
        config.entity = "user"
        config.output_dir = "."
        config.exp_name = str(datetime.now())[:19].replace(" ", "_")
        config.random_delay = 0.5
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)
        config.time = str(datetime.now())[:19].replace(" ", "_")

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant, prefix=None):
        self.config = self.get_default_config(config)

        for key, val in sorted(self.config.items()):
            if type(val) != str:
                continue
            new_val = _parse(val, variant)
            if val != new_val:
                logging.info(
                    "processing configs: {}: {} => {}".format(key, val, new_val)
                )
                setattr(self.config, key, new_val)

                output = flatten_config_dict(self.config, prefix=prefix)
                variant.update(output)

        if self.config.output_dir == "":
            self.config.output_dir = tempfile.mkdtemp()

        output = flatten_config_dict(self.config, prefix=prefix)
        variant.update(output)

        self._variant = copy(variant)

        logging.info(
            "wandb logging with hyperparameters: \n{}".format(
                pprint.pformat(
                    ["{}: {}".format(key, val) for key, val in self.variant.items()]
                )
            )
        )

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0.1, 0.1 + self.config.random_delay))

        self.run = wandb.init(
            entity=self.config.entity,
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            name=self.config.exp_name,
            anonymous=self.config.anonymous,
            monitor_gym=False,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode=self.config.mode,
        )

        self.logging_step = 0

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs, step=self.logging_step)

    def step(self):
        self.logging_step += 1

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self._variant

    @property
    def output_dir(self):
        return self.config.output_dir


if __name__ == "__main__":
    # kinmeatic function test

    #fk
    nova_left_init_joint_pose = [-1.5708, 0, -1.5708, 0, 1.5708, 1.5708]
    nova_right_init_joint_pose = [1.5708, 0, 1.5708, 0, -1.5708, -1.5708]
    # l_ee_pos, l_ee_rot = fk_solver(np.array(nova_left_init_joint_pose))
    # r_ee_pos, r_ee_rot = fk_solver(np.array(nova_right_init_joint_pose))
    # # transform the 3x3 rotation matrix to quaternion
    # l_ee_pos = l_ee_pos[0][0]
    # r_ee_pos = r_ee_pos[0][0]
    # l_ee_rot = quaternion.from_rotation_matrix(l_ee_rot[0][0])
    # r_ee_rot = quaternion.from_rotation_matrix(r_ee_rot[0][0])
    #
    # print("Left EE Position:", l_ee_pos, "Left EE Rotation:", l_ee_rot)
    # print("Right EE Position:", r_ee_pos, "Right EE Rotation:", r_ee_rot)


    # ik
    l_ee_pos = np.array([-1.1750e-01, -3.4501e-01,  4.1539e-01])
    l_ee_rot = R.from_quat([3.89601281525054e-06, -0.707109378514511, -0.707104183808496, -6.49338013368839e-06])
    l_joints = ik_solver(l_ee_pos, l_ee_rot, initial_joint=np.array(nova_left_init_joint_pose))
    print("Left Joint Angles:", l_joints[0], "Success:", l_joints[1])



    # Example usage
    # traj_origin = np.random.rand(1, 16, 14)  # Example trajectory
    # best_trajectory = np.random.rand(1, 16, 14)  # Example best trajectory
    # mode = "left"  # or "right", "bimanual"
    # updated_trajectory = bimanual_coordinator(mode, traj_origin, best_trajectory)
    # print(updated_trajectory)