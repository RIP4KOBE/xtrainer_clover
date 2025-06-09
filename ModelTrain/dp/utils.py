import json
import os
import pprint
import random
import string
import tempfile
import time
import yaml
from copy import copy
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
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags
from spatialmath.base import q2r
from scipy.spatial.transform import Rotation as R






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
        traj_origin (np.ndarray): shape (1, T, 14), original trajectory to update
        best_trajectory (np.ndarray): shape (1, T, 14), new best trajectory (usually B=1)

    Returns:
        np.ndarray: updated traj_origin with selected parts from best_trajectory
    """
    assert mode in ["left", "right", "bimanual"], f"Invalid mode: {mode}"
    # assert traj_origin.shape == 14 and best_trajectory.shape[2] == 14, "Trajectory dim must be 14"
    coordinate_traj = np.tile(traj_origin, (best_trajectory.shape[0], 1))

    print("coordinate_traj", coordinate_traj.shape, "best_trajectory", best_trajectory.shape)

    if mode == "left":
        # Only update left arm joints (0:6)
        coordinate_traj[:, :6] = best_trajectory[:, :6]

    elif mode == "right":
        # Only update right arm joints (7:13)
        coordinate_traj[:, 7:13] = best_trajectory[:, 7:13]

    elif mode == "bimanual":
        # Update both arms
        coordinate_traj[:, :6] = best_trajectory[:, :6]
        coordinate_traj[:, 7:13] = best_trajectory[:, 7:13]

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