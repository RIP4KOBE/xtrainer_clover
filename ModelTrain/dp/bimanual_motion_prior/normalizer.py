from typing import Union, Dict, List

import unittest
import zarr
import numpy as np
import torch
import torch.nn as nn
from ModelTrain.dp.bimanual_motion_prior.pytorch_util import dict_apply
from ModelTrain.dp.bimanual_motion_prior.dict_of_tensor_mixin import DictOfTensorMixin



# class QuatSafeNormalizer:
#     def __init__(self, quaternion_dims: List[int]):
#         self.quaternion_dims = quaternion_dims
#         self.normalizer = LinearNormalizer()
#         self._original_quat_values = None
#
#     def fit(self,
#             data: Union[Dict, torch.Tensor],
#             **kwargs):
#         """
#         Fit LinearNormalizer while excluding quaternion dims
#         """
#         data = torch.as_tensor(data).clone()  # clone to avoid modifying original
#         self._original_quat_values = data[:, self.quaternion_dims].clone()
#
#         # Replace quaternion dims with constant (e.g., 0)
#         data[:, self.quaternion_dims] = 0.0
#
#         # Fit using internal LinearNormalizer
#         self.normalizer.fit(data, **kwargs)
#
#     def normalize(self, x: torch.Tensor) -> torch.Tensor:
#         x = torch.as_tensor(x).clone()
#         quat = x[:, self.quaternion_dims].clone()
#         x_norm = self.normalizer.normalize(x)
#         x_norm[:, self.quaternion_dims] = quat  # restore original quaternion
#         return x_norm
#
#     def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
#         x = torch.as_tensor(x).clone()
#         quat = x[:, self.quaternion_dims].clone()
#         x_raw = self.normalizer.unnormalize(x)
#         x_raw[:, self.quaternion_dims] = quat  # restore quaternion as-is
#         return x_raw
#
#     def get_input_stats(self):
#         return self.normalizer.get_input_stats()
#
#     def get_output_stats(self):
#         return self.normalizer.get_output_stats()
#
#     def __call__(self, x):
#         return self.normalize(x)

# class QuatSafeNormalizer:
#     def __init__(self, quaternion_dims: Union[List[int], Dict[str, List[int]]]):
#         """
#         Args:
#             quaternion_dims:
#                 - List[int] for flat tensor input
#                 - Dict[str, List[int]] for dict input
#         """
#         self.quaternion_dims = quaternion_dims
#         self.normalizer = LinearNormalizer()
#         self._original_quat_values = None  # store raw quaternion values if needed
#
#     def fit(self, data: Union[Dict[str, torch.Tensor], torch.Tensor], **kwargs):
#         """
#         Fit inner normalizer while skipping quaternion dims
#         """
#         if isinstance(data, dict):
#             data = {k: torch.as_tensor(v).clone() for k, v in data.items()}
#
#             replaced_data = {}
#             for k, v in data.items():
#                 v = v.clone()
#                 for dim in self.quaternion_dims:
#                     if dim < v.shape[1]:  # 安全检查：防止维度越界
#                         v[:, dim] = 0.0
#                 replaced_data[k] = v
#
#             self.normalizer.fit(replaced_data, **kwargs)
#
#         else:
#             data = torch.as_tensor(data).clone()
#             if isinstance(self.quaternion_dims, dict):
#                 raise ValueError("For tensor input, quaternion_dims should be a List[int]")
#             data[:, self.quaternion_dims] = 0.0
#             self.normalizer.fit(data, **kwargs)
#
#     def normalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
#         if isinstance(x, dict):
#             x = {k: torch.as_tensor(v).clone() for k, v in x.items()}
#             x_norm = self.normalizer.normalize(x)
#
#             # Replace quaternion dims with original values
#             for k, v in x.items():
#                 quat_dims = self.quaternion_dims.get(k, [])
#                 if quat_dims:
#                     x_norm[k][:, quat_dims] = v[:, quat_dims]
#             return x_norm
#
#         else:
#             x = torch.as_tensor(x).clone()
#             x_norm = self.normalizer.normalize(x)
#             x_norm[:, self.quaternion_dims] = x[:, self.quaternion_dims]
#             return x_norm
#
#     def unnormalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
#         if isinstance(x, dict):
#             x = {k: torch.as_tensor(v).clone() for k, v in x.items()}
#             x_raw = self.normalizer.unnormalize(x)
#
#             for k, v in x.items():
#                 quat_dims = self.quaternion_dims.get(k, [])
#                 if quat_dims:
#                     x_raw[k][:, quat_dims] = v[:, quat_dims]
#             return x_raw
#
#         else:
#             x = torch.as_tensor(x).clone()
#             x_raw = self.normalizer.unnormalize(x)
#             x_raw[:, self.quaternion_dims] = x[:, self.quaternion_dims]
#             return x_raw
#
#     def get_input_stats(self):
#         return self.normalizer.get_input_stats()
#
#     def get_output_stats(self):
#         return self.normalizer.get_output_stats()
#
#     def __call__(self, x):
#         return self.normalize(x)

class QuatSafeNormalizer(DictOfTensorMixin):
    def __init__(self, quaternion_dims: Union[List[int], Dict[str, List[int]]]):
        """支持跳过 quaternion 维度归一化的 Normalizer

        Args:
            quaternion_dims:
                - List[int]: 对于 flat tensor 输入（如 [N, D]）
                - Dict[str, List[int]]: 对于 dict 输入（如 {'obs': tensor, ...}）
        """
        super().__init__()
        self.quaternion_dims = quaternion_dims
        self.normalizer = LinearNormalizer()

    @torch.no_grad()
    def fit(self, data: Union[Dict[str, torch.Tensor], torch.Tensor], **kwargs):
        if isinstance(data, dict):
            replaced_data = {}
            for k, v in data.items():
                v = torch.as_tensor(v).clone()
                v = v.clone()
                for dim in self.quaternion_dims:
                    if dim < v.shape[1]:
                        v[:, dim] = 0.0
                replaced_data[k] = v
            self.normalizer.fit(replaced_data, **kwargs)
        else:
            data = torch.as_tensor(data).clone()
            if isinstance(self.quaternion_dims, dict):
                raise ValueError("For flat tensor input, quaternion_dims should be List[int]")
            data[:, self.quaternion_dims] = 0.0
            self.normalizer.fit(data, **kwargs)

    def _restore_quaternion(self, orig, normalized):
        if isinstance(orig, dict):
            restored = {}
            for k, v in orig.items():
                v_norm = normalized[k].clone()
                for dim in self.quaternion_dims:
                    if dim < v.shape[1]:
                        v_norm[:, dim] = v[:, dim]
                restored[k] = v_norm
        else:
            restored = normalized.clone()
            for dim in self.quaternion_dims:
                if dim < orig.shape[1]:
                    restored[:, dim] = orig[:, dim]
            return restored

    def normalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        # x = {k: torch.as_tensor(v).clone() for k, v in x.items()} if isinstance(x, dict) else torch.as_tensor(x).clone()
        x_norm = self.normalizer.normalize(x)
        # return self._restore_quaternion(x, x_raw)
        return x_norm

    def unnormalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        # x = {k: torch.as_tensor(v).clone() for k, v in x.items()} if isinstance(x, dict) else torch.as_tensor(x).clone()
        x_raw = self.normalizer.unnormalize(x)
        # return self._restore_quaternion(x, x_raw)
        return x_raw

    def get_input_stats(self):
        return self.normalizer.get_input_stats()

    def get_output_stats(self):
        return self.normalizer.get_output_stats()

    def __call__(self, x):
        return self.normalize(x)

    def __getitem__(self, key: str):
        return self.normalizer[key]

    def __setitem__(self, key: str, value):
        self.normalizer[key] = value

    # def state_dict(self):
    #     return self.normalizer.state_dict()
    #
    # def load_state_dict(self, state_dict):
    #     return self.normalizer.load_state_dict(state_dict)

class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] =  _fit(value, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
        else:
            self.params_dict['_default'] = _fit(data, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
    
    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)
    
    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']
        
        result = dict()
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


    def get_output_stats(self, key='_default'):
        input_stats = self.get_input_stats()
        if 'min' in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)
        
        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key:value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
            data: Union[torch.Tensor, np.ndarray, zarr.Array],
            last_n_dims=1,
            dtype=torch.float32,
            mode='limits',
            output_max=1.,
            output_min=-1.,
            range_eps=1e-4,
            fit_offset=True):
        self.params_dict = _fit(data, 
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset)
    
    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj
    
    @classmethod
    def create_manual(cls, 
            scale: Union[torch.Tensor, np.ndarray], 
            offset: Union[torch.Tensor, np.ndarray],
            input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x
        
        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype
        
        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict['input_stats']

    def get_output_stats(self):
        return dict_apply(self.params_dict['input_stats'], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)



def _fit(data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1,dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels 
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    # save
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def test():
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1., atol=1e-3)
    assert np.allclose(datan.min(), 0., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = torch.zeros((100,10,9,2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='gaussian', last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0., atol=1e-3)
    assert np.allclose(datan.std(), 1., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)


    # dict
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    data = {
        'obs': torch.zeros((1000,128,9,2)).uniform_() * 512,
        'action': torch.zeros((1000,128,2)).uniform_() * 512
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
    
    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
