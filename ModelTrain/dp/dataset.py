import os
import pickle
import pathlib
import numpy as np
import torch
import copy

from typing import Dict
from ModelTrain.dp.bimanual_motion_prior.normalizer import LinearNormalizer


from ModelTrain.dp.bimanual_motion_prior.pytorch_util import dict_apply
from ModelTrain.dp.bimanual_motion_prior.replay_buffer import ReplayBuffer
from ModelTrain.dp.bimanual_motion_prior.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from ModelTrain.dp.bimanual_motion_prior.normalizer import LinearNormalizer, QuatSafeNormalizer


LEFT_ARM_6D_INDICES = slice(0, 10)
RIGHT_ARM_6D_INDICES = slice(10, 20)

def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    if np.any(stats["max"] > 1e5) or np.any(stats["min"] < -1e5):
        raise ValueError("data out of range")
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def normalize_6d_pose(pose, stats):
    """
        Batch normalization for dual-arm pose data (shape [batch_size, 20])

        Args:
            pose: Input dual-arm pose data (left 10D + right 10D)
            stats: Precomputed statistics dictionary containing min/max/mean etc.

        Returns:
            Normalized pose data with same shape as input
        """

    assert pose.shape[1] == 20, "Input should be 20 dimensional dual-arm data"

    # Initialize output array
    normalized = np.zeros_like(pose)

    # Process both arms using same normalization logic
    for arm_slice in [LEFT_ARM_6D_INDICES, RIGHT_ARM_6D_INDICES]:
        arm_data = pose[:, arm_slice]
        arm_stats_min = stats["min"][arm_slice]  # shape (10,)
        arm_stats_max = stats["max"][arm_slice]  # shape (10,)

        # Position dimensions (first 3 dims per arm)
        pos_indices = slice(0, 3)
        pos_data = arm_data[:, pos_indices]
        pos_stats = {
            "min": arm_stats_min[pos_indices],  # 只取前3维min
            "max": arm_stats_max[pos_indices]  # 只取前3维max
        }
        normalized_pos = normalize_data(pos_data, pos_stats)

        # Rotation dimensions (dims 3-8 per arm)
        rot_indices = slice(3, 9)
        rot_data = arm_data[:, rot_indices]
        normalized_rot = rot_data

        # Gripper dimension (last dim per arm)
        gripper_index = 9
        gripper_data = arm_data[:, gripper_index]
        gripper_stats = {
            "min": arm_stats_min[gripper_index],
            "max": arm_stats_max[gripper_index]
        }
        normalized_gripper = normalize_data(gripper_data, gripper_stats)

        # Combine normalized components
        normalized[:, arm_slice] = np.hstack([
            normalized_pos,
            normalized_rot,
            normalized_gripper.reshape(-1, 1)  # Ensure gripper is 2D
        ])

    # normalized = np.zeros_like(pose)
    # for i in range(len(pose)):
    #     if (stats["max"][i] - stats["min"][i]) < 1e-6:  # 零方差情况
    #         normalized[i] = pose[i] - stats["mean"][i]
    #     elif 3 <= i < len(pose)-1:    # 6D旋转维度（假设前3维是位置）
    #         normalized[i] = pose[i]  # 不归一化
    #     else:  # 位置维度+gripper维度
    #         normalized[i] = normalize_data(pose[i], stats)
    return normalized


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"] + 1e-8) + stats["min"]
    return data


def unnormalize_6d_pose(normalized, stats):
    """
        Batch denormalization for dual-arm pose data (shape [batch_size, 20])

        Args:
            normalized: Normalized pose data to be denormalized
            stats: Same statistics dictionary used for normalization

        Returns:
            Denormalized pose data in original scale
        """
    assert normalized.shape[1] == 20, "Input should be 20D normalized data"

    original = np.zeros_like(normalized)

    for arm_slice in [LEFT_ARM_6D_INDICES, RIGHT_ARM_6D_INDICES]:
        norm_arm = normalized[:, arm_slice]
        arm_stats_min = stats["min"][arm_slice]  # shape (10,)
        arm_stats_max = stats["max"][arm_slice]  # shape (10,)

        # Position denormalization
        pos_indices = slice(0, 3)
        norm_pos = norm_arm[:, pos_indices]
        pos_stats = {
            "min": arm_stats_min[pos_indices],  # 只取前3维min
            "max": arm_stats_max[pos_indices]  # 只取前3维max
        }
        original_pos = unnormalize_data(norm_pos, pos_stats)

        # Rotation denormalization
        rot_indices = slice(3, 9)
        original_rot = norm_arm[:, rot_indices]  # no change for rotation

        # Gripper denormalization
        gripper_index = 9
        norm_gripper = norm_arm[:, gripper_index]
        gripper_stats = {
            "min": arm_stats_min[gripper_index],
            "max": arm_stats_max[gripper_index]
        }
        original_gripper = unnormalize_data(norm_gripper, gripper_stats)

        original[:, arm_slice] = np.hstack([
            original_pos,
            original_rot,
            original_gripper.reshape(-1, 1)  # Ensure gripper is 2D
        ])

    return original

    # original = np.zeros_like(normalized)
    # for i in range(len(normalized)):
    #     if (stats["max"][i] - stats["min"][i]) < 1e-6:
    #         original[i] = normalized[i] + stats["mean"][i]
    #     elif 3 <= i < len(normalized)-1:
    #         original[i] = normalized[i]
    #     else:
    #         original[i] = unnormalize_data(normalized[i], stats)
    # return original

class MemmapLoader:
    def __init__(self, path):
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            meta_data = pickle.load(f)

        print("Meta Data:", meta_data)
        self.fps = {}

        self.length = None
        for key, (shape, dtype) in meta_data.items():
            self.fps[key] = np.memmap(
                os.path.join(path, key + ".dat"), dtype=dtype, shape=shape, mode="r"
            )
            if self.length is None:
                self.length = shape[0]
            else:
                assert self.length == shape[0]

    def __getitem__(self, index):
        rets = {}
        for key in self.fps.keys():
            value = self.fps[key]
            value = value[index]
            value_cp = np.empty(dtype=value.dtype, shape=value.shape)
            value_cp[:] = value
            rets[key] = value_cp
        return rets

    def __length__(self):
        return self.length


# dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: dict,
        representation_type: list,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        stats: dict = None,
        transform=None,
        get_img=None,
        load_img: bool = False,
        hand_grip_range: int = 110,
        binarize_touch: bool = False,
        predict_eef_6d: bool = False,
        state_noise: float = 0.0,
    ):
        self.state_noise = state_noise
        self.memmap_loader = None
        if "memmap_loader_path" in data.keys():
            self.memmap_loader = MemmapLoader(data["memmap_loader_path"])
        self.representation_type = representation_type
        self.transform = transform
        self.get_img = get_img
        self.load_img = load_img

        print("Representation type: ", representation_type)
        except_img_representation_type = representation_type.copy()

        if "img" in representation_type:
            train_image_data = data["data"]["img"][:]
            except_img_representation_type.remove("img")

        train_data = {
            rt: data["data"][rt][:, :] for rt in except_img_representation_type
        }
        train_data["action"] = data["data"]["action"][:]
        episode_ends = data["meta"]["episode_ends"][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        normalized_train_data = dict()

        # compute statistics and normalized data to [-1,1]
        if stats is None:
            stats = dict()
            for key, data in train_data.items():
                stats[key] = get_data_stats(data)

        # normalize the training data
        for key, data in train_data.items():
            if predict_eef_6d:
                if key == "action":
                    normalized_train_data[key] = normalize_6d_pose(data, stats[key])
                else:
                    normalized_train_data[key] = normalize_data(data, stats[key])
            else:
                if key == "touch" and binarize_touch:
                    normalized_train_data[key] = (
                        data  # don't normalize if binarize touch in model
                    )
                else:
                    normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        if "img" in representation_type:
            normalized_train_data["img"] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.binarize_touch = binarize_touch

    def __len__(self):
        return len(self.indices)

    def read_img(self, image_pathes, idx):
        if self.memmap_loader is not None:
            # using memmap loader
            indices = range(idx, idx + self.obs_horizon)
            data = self.memmap_loader[indices]
            data = [
                {"base_rgb": data["base_rgb"][i], "base_depth": data["base_depth"][i]}
                for i in range(data["base_rgb"].shape[0])
            ]
        else:
            # not using memmap loader and loading images while training
            data = [pickle.load(open(image_path, "rb")) for image_path in image_pathes]
        imgs = self.get_img(data)
        return imgs

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        for k in self.representation_type:
            # discard unused observations
            nsample[k] = nsample[k][: self.obs_horizon]
            if k == "img":
                if not self.load_img:
                    nsample["img"] = self.read_img(nsample["img"], idx)
                else:
                    nsample["img"] = torch.tensor(
                        nsample["img"].astype(np.float32), dtype=torch.float32
                    )
                nsample_shape = nsample["img"].shape
                # transform the img
                nsample["img"] = nsample["img"].reshape(
                    nsample_shape[0] * nsample_shape[1], *nsample_shape[2:]
                )                                                                           # (Batch * num_cam, Channel, Height, Width)
                nsample["img"] = self.transform(nsample["img"])
                nsample["img"] = nsample["img"].reshape(nsample_shape[:3] + (216, 288))     # (Batch, num_cam, Channel, Height, Width)

            else:
                nsample[k] = torch.tensor(nsample[k], dtype=torch.float32)
                if self.state_noise > 0.0:
                    # add noise to the state
                    nsample[k] = nsample[k] + torch.randn_like(nsample[k]) * self.state_noise
        nsample["action"] = torch.tensor(nsample["action"], dtype=torch.float32)
        return nsample


class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()


# class BimanualMotionPriorDataset(BaseLowdimDataset):
#     def __init__(self,
#                  dataset_dir,
#                  horizon=1,
#                  pad_before=0,
#                  pad_after=0,
#                  seed=42,
#                  val_ratio=0.0
#                  ):
#         super().__init__()
#
#         # data_directory = pathlib.Path(dataset_dir)
#         # observations = np.load(data_directory / "observations_seq.npy")
#         # actions = np.load(data_directory / "actions_seq.npy")
#         # masks = np.load(data_directory / "existence_mask.npy")
#
#         self.replay_buffer = ReplayBuffer.create_empty_numpy()
#
#         for i, epi in enumerate(dataset_dir):
#             print("loading {}-th data from {}\r".format(i, epi), end="")
#             epi_data = data_processing.iterate(epi, load_img=False)
#             obs = np.stack([d["ee_pos_quat"] for d in epi_data]).astype(np.float32)
#             action = np.stack([d["control"] for d in epi_data]).astype(np.float32)
#             data = {
#                 'obs': obs,
#                 'action': action
#             }
#             self.replay_buffer.add_episode(data)
#
#             if len(data) == 0:
#                 continue
#
#         val_mask = get_val_mask(
#             n_episodes=self.replay_buffer.n_episodes,
#             val_ratio=val_ratio,
#             seed=seed)
#         train_mask = ~val_mask
#
#         self.sampler = SequenceSampler(
#             replay_buffer=self.replay_buffer,
#             sequence_length=horizon,
#             pad_before=pad_before,
#             pad_after=pad_after,
#             episode_mask=train_mask)
#
#         self.train_mask = train_mask
#         self.horizon = horizon
#         self.pad_before = pad_before
#         self.pad_after = pad_after
#
#     def get_validation_dataset(self):
#         val_set = copy.copy(self)
#         val_set.sampler = SequenceSampler(
#             replay_buffer=self.replay_buffer,
#             sequence_length=self.horizon,
#             pad_before=self.pad_before,
#             pad_after=self.pad_after,
#             episode_mask=~self.train_mask
#         )
#         val_set.train_mask = ~self.train_mask
#         return val_set
#
#     def get_normalizer(self, mode='limits', **kwargs):
#         data = {
#             'obs': self.replay_buffer['obs'],
#             'action': self.replay_buffer['action']
#         }
#         if 'range_eps' not in kwargs:
#             # to prevent blowing up dims that barely change
#             kwargs['range_eps'] = 5e-2
#         normalizer = LinearNormalizer()
#         normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
#         return normalizer
#
#     def get_all_actions(self) -> torch.Tensor:
#         return torch.from_numpy(self.replay_buffer['action'])
#
#     def __len__(self) -> int:
#         return len(self.sampler)
#
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         sample = self.sampler.sample_sequence(idx)
#         data = sample
#
#         torch_data = dict_apply(data, torch.from_numpy)
#         return torch_data


class BimanualMotionPriorDataset(BaseLowdimDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='ee_pose',
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
            )
        # self.obs_key = obs_key
        # self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        quaternion_dims = list(range(3, 7)) + list(range(11, 15))
        normalizer = QuatSafeNormalizer(quaternion_dims)

        # normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # keypoint = sample[self.obs_key]
        # state = sample[self.state_key]
        # agent_pos = state[:,:2]
        # obs = np.concatenate([
        #     keypoint.reshape(keypoint.shape[0], -1),
        #     agent_pos], axis=-1)

        data = {
            'action': sample[self.action_key], # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data