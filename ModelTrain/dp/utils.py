import json
import os
import pprint
import random
import string
import tempfile
import time
from copy import copy
from datetime import datetime
import matplotlib.pyplot as plt
import roboticstoolbox as rtb

import absl.flags
import numpy as np
import quaternion
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags


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



def forward_kinematics(joint_angles):
    """
    Compute forward kinematics (FK) to transform joint angles into end-effector pose in Cartesian space.

    Parameters:
    - joint_angles: (batch_size, prediction_horizon, 7) -> 7 joint angles per arm

    Returns:
    - End-effector pose (batch_size, prediction_horizon, action_dim).
    """
    # load the URDF files for the left and right arms
    urdf = "/home/zhuoli/xtrainer_clover/assets/urdf/nova2_robot.urdf"
    xtrainer_arm = rtb.robot.ERobot.URDF(urdf)

    if joint_angles.ndim == 2:
        joint_angles = joint_angles[np.newaxis, :, :]  # Add batch dimension

    assert joint_angles.shape[2] == 6, "Joint angles must have 6 values."

    batch_size, prediction_horizon, _ = joint_angles.shape
    ee_position = np.zeros((batch_size, prediction_horizon, 3))  # (x, y, z)
    ee_orientations = np.zeros((batch_size, prediction_horizon, 3, 3))  # 3x3 rotation matrix

    for b in range(batch_size):
        for t in range(prediction_horizon):
            angles = joint_angles[b, t, :]
            T_ee = xtrainer_arm.fkine(angles)
            T_ee = np.array(T_ee.A)
            ee_position[b, t, :] = T_ee[:3, 3]
            ee_orientations[b, t, :, :] = T_ee[:3, :3]

    return ee_position, ee_orientations


def visualize_trajectory(population_trajectories, best_trajectory):
    """
    Visualize the trajectories generated by Diffusion-ES.

    Parameters:
    - population_trajectories: (batch_size, prediction_horizon, 14)
      128 trajectories, each with 16 steps, 14 joint angles.
    - best_trajectory: (prediction_horizon, 14)
      The best trajectory (1 trajectory).
    """
    batch_size, prediction_horizon, action_dim = population_trajectories.shape

    # Separate left and right arm joint angles
    left_arm_trajectories = population_trajectories[:, :, :6]  # (batch_size, prediction_horizon, 7)
    right_arm_trajectories = population_trajectories[:, :, 7:13]  # (batch_size, prediction_horizon, 7)

    best_left_arm = best_trajectory[:, :6]  # (prediction_horizon, 7)
    best_right_arm = best_trajectory[:, 7:13]  # (prediction_horizon, 7)

    # Compute FK end-effector positions
    left_ee_positions, _ = forward_kinematics(left_arm_trajectories)
    right_ee_positions, _ = forward_kinematics(right_arm_trajectories)
    left_ee_positions = np.array(left_ee_positions)
    right_ee_positions = np.array(right_ee_positions)

    best_left_ee_positions, _ = forward_kinematics(best_left_arm)  # (prediction_horizon, 3)
    best_right_ee_positions, _ = forward_kinematics(best_right_arm)
    best_left_ee_positions = np.array(best_left_ee_positions)
    best_right_ee_positions = np.array(best_right_ee_positions)

    # Visualization of left and right arm trajectories
    fig = plt.figure(figsize=(12, 6))

    # Left arm trajectories
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(batch_size):
        ax1.plot(left_ee_positions[i, :, 0],
                 left_ee_positions[i, :, 1],
                 left_ee_positions[i, :, 2],
                 color="blue", alpha=0.2)  # Semi-transparent trajectories

    # Best trajectory (left arm)
    ax1.plot(best_left_ee_positions[0][:, 0],
             best_left_ee_positions[0][:, 1],
             best_left_ee_positions[0][:, 2],
             color="red", linewidth=2, label="Best Left Trajectory")

    # Mark start and end points for left arm
    ax1.scatter(best_left_ee_positions[0, 0, 0],  # Start X
                best_left_ee_positions[0, 0, 1],  # Start Y
                best_left_ee_positions[0, 0, 2],  # Start Z
                color='green', marker='o', s=100, label="Start Point")  # Green Start

    ax1.scatter(best_left_ee_positions[0, -1, 0],  # End X
                best_left_ee_positions[0, -1, 1],  # End Y
                best_left_ee_positions[0, -1, 2],  # End Z
                color='red', marker='X', s=150, label="End Point")  # Red End

    ax1.view_init(elev=10, azim=180)

    ax1.set_title("Left Arm End-Effector Trajectories")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # Right arm trajectories
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(batch_size):
        ax2.plot(right_ee_positions[i, :, 0],
                 right_ee_positions[i, :, 1],
                 right_ee_positions[i, :, 2],
                 color="green", alpha=0.2)  # Semi-transparent trajectories

    # Best trajectory (right arm)
    ax2.plot(best_right_ee_positions[0][:, 0],
             best_right_ee_positions[0][:, 1],
             best_right_ee_positions[0][:, 2],
             color="red", linewidth=2, label="Best Right Trajectory")

    # Mark start and end points for right arm
    ax2.scatter(best_right_ee_positions[0, 0, 0],  # Start X
                best_right_ee_positions[0, 0, 1],  # Start Y
                best_right_ee_positions[0, 0, 2],  # Start Z
                color='green', marker='o', s=100, label="Start Point")  # Green Start

    ax2.scatter(best_right_ee_positions[0, -1, 0],  # End X
                best_right_ee_positions[0, -1, 1],  # End Y
                best_right_ee_positions[0, -1, 2],  # End Z
                color='red', marker='X', s=150, label="End Point")  # Red End

    ax2.view_init(elev=10, azim=0)
    ax2.set_title("Right Arm End-Effector Trajectories")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def align_trajs_to_origin(trajs, traj_origin):


    return aligned_trajs