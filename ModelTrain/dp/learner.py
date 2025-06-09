import collections
import copy
import os
import time
import yaml
import json


import numpy as np
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from einops import rearrange, reduce
from jsonschema.exceptions import best_match

from ModelTrain.dp.models import *
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Dict, Tuple
from utils import fk_solver, align_trajs_to_origin, bimanual_coordinator,bimanual_frame_transform, get_config
from vis_utils import visualize_trajectory
from keypoint_proposer import KeypointProposer

from ModelTrain.dp.bimanual_motion_prior.normalizer import LinearNormalizer, QuatSafeNormalizer
from ModelTrain.dp.bimanual_motion_prior.mask_generator import LowdimMaskGenerator



def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / ((stats["max"] - stats["min"]) + 1e-8)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"] + 1e-8) + stats["min"]
    return data


class DiffusionPolicy:
    def __init__(
        self,
        obs_horizon,
        obs_dim,
        pred_horizon,
        action_horizon,
        action_dim,
        representation_type,  # pos, img, touch, eef
        encoders,
        num_diffusion_iters=100,
        without_sampling=False,
        weight_decay=1e-6,
        use_ddim=True,
        binarize_touch=False,
        policy_dropout_rate=0.0,
    ):
        for rt in representation_type:
            assert rt in encoders, f"{rt} not in encoders"
        self.representation_type = representation_type
        self.encoders = encoders
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.data_stat = None
        self.writer = None
        self.without_sampling = without_sampling
        self.binarize_touch = binarize_touch

        if self.without_sampling:
            bc_actor = SimpleBCModel(
                input_dim=obs_dim * obs_horizon,
                output_dim=self.action_dim * self.pred_horizon,
                dropout_rate=policy_dropout_rate,
            )
            # the final arch has 2 parts
            self.nets = nn.ModuleDict({"bc_actor": bc_actor})
        else:
            noise_pred_net = ConditionalUnet1D(
                input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon
            )
            # the final arch has 2 parts: one for the state encoder and another for the noise prediction network
            self.nets = nn.ModuleDict({"noise_pred_net": noise_pred_net})

        for rt in representation_type:
            self.nets[f"{rt}_encoder"] = encoders[rt]

        self.num_diffusion_iters = num_diffusion_iters

        if use_ddim:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                # the choise of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule="squaredcos_cap_v2",
                # clip output to [-1,1] to improve stability
                clip_sample=True,
                # our network predicts noise (instead of denoised action)
                prediction_type="epsilon",
            )
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                # the choise of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule="squaredcos_cap_v2",
                # clip output to [-1,1] to improve stability
                clip_sample=True,
                # our network predicts noise (instead of denoised action)
                prediction_type="epsilon",
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Exponential Moving Average of the model weights
        self.ema = EMAModel(parameters=self.nets.parameters(), power=0.75)
        self.ema_nets = copy.deepcopy(self.nets)

        # Standard ADAM optimizer
        self.optimizer = torch.optim.AdamW(
            params=self.nets.parameters(), lr=1e-4, weight_decay=weight_decay
        )

    def set_lr_scheduler(self, num_training_steps):
        # Cosine LR schedule with linear warmup
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps,
        )

    def to(self, device):
        self.device = device
        self.nets.to(device)
        self.ema.to(device)
        self.ema_nets.to(device)

    def train(
        self,
        num_epochs,
        dataloader,
        eval_data=None,
        save_path=None,
        save_freq=10,
        eval_freq=10,
        wandb_logger=None,
        eval=False,
    ):
        if eval:
            nets = self.ema_nets
            nets.eval()
            action_mse = []
        else:
            nets = self.nets
            nets.train()

        if self.writer is None and save_path is not None:
            # get the name from save_path
            name = os.path.basename(save_path)
            self.writer = SummaryWriter(os.path.join("./runs", name))
        with tqdm(range(num_epochs), desc="Epoch") as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                # batch loop
                epoch_loss = list()
                with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer
                        naction = nbatch["action"].to(self.device)
                        B = naction.shape[0]
                        features = []

                        ### IMPT: make sure input is always in this order
                        # eef, hand_pos, img, pos, touch
                        for data_key in [
                            dk
                            for dk in ["eef", "hand_pos", "img", "pos", "touch"]
                            if dk in self.representation_type
                        ]:
                            nsample = nbatch[data_key][:, : self.obs_horizon].to(
                                self.device
                            )
                            if data_key == "img":
                                images = [
                                    nsample[:, :, i] for i in range(nsample.shape[2])
                                ]  # [B, obs_horizon, M, C, H, W]
                                image_features = [
                                    nets[f"{data_key}_encoder"][i](
                                        image.flatten(end_dim=1)
                                    )
                                    for i, image in enumerate(images)
                                ]
                                image_features = torch.stack(image_features, dim=2)
                                image_features = image_features.reshape(
                                    *nsample.shape[:2], -1
                                )
                                features.append(image_features)
                            else:
                                nfeat = nets[f"{data_key}_encoder"](
                                    nsample.flatten(end_dim=1)
                                )
                                nfeat = nfeat.reshape(*nsample.shape[:2], -1)
                                features.append(nfeat)

                        obs_features = torch.cat(features, dim=-1)
                        # (B, obs_horizon * obs_dim)
                        obs_cond = obs_features.flatten(start_dim=1)

                        if self.without_sampling:
                            action = nets["bc_actor"](obs_cond)
                            action = action.reshape(
                                -1, self.pred_horizon, self.action_dim
                            )
                            # L2 loss
                            loss = nn.functional.mse_loss(action, naction)
                        else:
                            # sample noise to add to actions
                            noise = torch.randn(naction.shape, device=self.device)

                            # Training
                            if not eval:
                                # sample a diffusion iteration for each data point
                                timesteps = torch.randint(
                                    0,
                                    self.noise_scheduler.config.num_train_timesteps,
                                    (B,),
                                    device=self.device,
                                ).long()

                                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                                # (this is the forward diffusion process)
                                noisy_actions = self.noise_scheduler.add_noise(
                                    naction, noise, timesteps
                                )
                                # predict the noise residual
                                noise_pred = nets["noise_pred_net"](
                                    noisy_actions, timesteps, global_cond=obs_cond
                                )

                                # L2 loss
                                loss = nn.functional.mse_loss(noise_pred, noise)

                            # Evaluation
                            else:
                                noisy_action = noise
                                pred_action = noisy_action

                                self.noise_scheduler.set_timesteps(
                                    self.num_diffusion_iters
                                )

                                for k in self.noise_scheduler.timesteps:
                                    # predict noise
                                    noise_pred = nets["noise_pred_net"](
                                        sample=pred_action,
                                        timestep=k,
                                        global_cond=obs_cond,
                                    )

                                    # inverse diffusion step (remove noise)
                                    pred_action = self.noise_scheduler.step(
                                        model_output=noise_pred,
                                        timestep=k,
                                        sample=pred_action,
                                    ).prev_sample

                                loss = nn.functional.mse_loss(naction, pred_action)

                                unnormalized_naction = unnormalize_data(
                                    naction.detach().cpu().numpy(),
                                    self.data_stat["action"],
                                )
                                unnormalized_pred_action = unnormalize_data(
                                    pred_action.detach().cpu().numpy(),
                                    self.data_stat["action"],
                                )
                                unnormalized_loss = nn.functional.mse_loss(
                                    torch.tensor(unnormalized_naction),
                                    torch.tensor(unnormalized_pred_action),
                                )

                        if not eval:
                            # optimize
                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # step lr scheduler every batch
                            # this is different from standard pytorch behavior
                            self.lr_scheduler.step()

                            # update Exponential Moving Average of the model weights
                            self.ema.step(nets.parameters())
                        else:
                            action_mse.append(unnormalized_loss.item())

                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))
                if self.writer is not None:
                    self.writer.add_scalar("Loss", np.mean(epoch_loss), epoch_idx)

                if eval:
                    return np.mean(epoch_loss), np.mean(action_mse)

                if wandb_logger is not None:
                    wandb_logger.step()
                    wandb_logger.log({"Loss": np.mean(epoch_loss), "epoch": epoch_idx})
                if (
                    save_path is not None
                    and epoch_idx % save_freq == 0
                    and epoch_idx != 0
                ):
                    model_path = os.path.join(
                        save_path, f"model_epoch_{epoch_idx}.ckpt"
                    )
                    self.save(model_path)

                # save last checkpoint
                model_path = os.path.join(save_path, f"last.ckpt")
                self.save(model_path)

                if eval_data is not None and epoch_idx % eval_freq == 0:
                    self.to_ema()
                    self.ema_nets.eval()
                    print("Evaluating one trajectory...")
                    obs, action = eval_data
                    _, mse, normalized_mse = self.eval(obs, action)
                    self.writer.add_scalar("Action_MSE", mse, epoch_idx)
                    self.writer.add_scalar("Normalized_MSE", normalized_mse, epoch_idx)

                    if wandb_logger is not None:
                        wandb_logger.log({"Action_MSE": mse})
                        wandb_logger.log({"Normalized_MSE": normalized_mse})
                    print(f"Action_MSE: {mse}, Normalized_MSE: {normalized_mse}")
                    self.ema_nets.train()
                    return None
                return None
            return None

    def train_cfg(
        self,
        num_epochs,
        dataloader,
        eval_data=None,
        save_path=None,
        save_freq=10,
        eval_freq=10,
        wandb_logger=None,
        eval=False,
        conditional_pdrop=0.1,
        cfg_options=None,
    ):
        print("Dffusion Training with CFG Mode")

        if cfg_options is None:
            cfg_dict = dict()
        else:
            with open(cfg_options, 'r') as f:
                cfg_dict = yaml.safe_load(f)

        use_large_drop_prob = cfg_dict.get("use_large_drop_prob", False)
        use_batch_split = cfg_dict.get("use_batch_split", False)
        use_mixed_loss = cfg_dict.get("use_mixed_loss", False)
        use_two_stage = cfg_dict.get("use_two_stage", False)
        unconditional_training = cfg_dict.get("unconditional_training", False)

        print(f"cfg_dict: {cfg_dict}")


        if eval:
            nets = self.ema_nets
            nets.eval()
            action_mse = []
        else:
            nets = self.nets
            nets.train()

        if self.writer is None and save_path is not None:
            # get the name from save_path
            name = os.path.basename(save_path)
            self.writer = SummaryWriter(os.path.join("./runs", name))
        with tqdm(range(num_epochs), desc="Epoch") as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                # batch loop
                epoch_loss = list()
                with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer
                        naction = nbatch["action"].to(self.device)
                        B = naction.shape[0]
                        features = []

                        ### IMPT: make sure input is always in this order
                        # eef, hand_pos, img, pos, touch
                        for data_key in [
                            dk
                            for dk in ["eef", "hand_pos", "img", "pos", "touch"]
                            if dk in self.representation_type
                        ]:
                            nsample = nbatch[data_key][:, : self.obs_horizon].to(
                                self.device
                            )
                            if data_key == "img":
                                images = [
                                    nsample[:, :, i] for i in range(nsample.shape[2])
                                ]  # [B, obs_horizon, M, C, H, W]
                                image_features = [
                                    nets[f"{data_key}_encoder"][i](
                                        image.flatten(end_dim=1)
                                    )
                                    for i, image in enumerate(images)
                                ]
                                image_features = torch.stack(image_features, dim=2)
                                image_features = image_features.reshape(
                                    *nsample.shape[:2], -1
                                )
                                features.append(image_features)
                            else:
                                nfeat = nets[f"{data_key}_encoder"](
                                    nsample.flatten(end_dim=1)
                                )
                                nfeat = nfeat.reshape(*nsample.shape[:2], -1)
                                features.append(nfeat)

                        obs_features = torch.cat(features, dim=-1)
                        # (B, obs_horizon * obs_dim)
                        obs_cond = obs_features.flatten(start_dim=1)

                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device)

                        # Training
                        if not eval:
                            # sample a diffusion iteration for each data point
                            timesteps = torch.randint(
                                0,
                                self.noise_scheduler.config.num_train_timesteps,
                                (B,),
                                device=self.device,
                            ).long()

                            # add noise to the clean images according to the noise magnitude at each diffusion iteration
                            # (this is the forward diffusion process)
                            noisy_actions = self.noise_scheduler.add_noise(
                                naction, noise, timesteps
                            )

                            # different CFG training stragety
                            cond_prob = conditional_pdrop
                            if use_two_stage:
                                if epoch_idx < num_epochs * 0.8:
                                    cond_prob = 0.3 if use_large_drop_prob else 0.1
                                else:
                                    cond_prob = 1.0
                            elif use_large_drop_prob:
                                cond_prob = 0.3

                            if use_batch_split:
                                ratio = 0.2
                                B_uncond = int(B * ratio)
                                obs_cond = obs_cond.clone()
                                obs_cond[:B_uncond] = 0.0
                                is_cond_mask = torch.ones(B).bool()
                                is_cond_mask[:B_uncond] = False
                            elif unconditional_training:
                                obs_cond = torch.zeros_like(obs_cond)
                            else:
                                drop_mask = torch.rand(B, device=self.device) < cond_prob
                                obs_cond = obs_cond.clone()
                                obs_cond[drop_mask] = 0.0

                            if not use_mixed_loss:
                                # 常规 CFG loss（单一预测）
                                noise_pred = nets["noise_pred_net"](noisy_actions, timesteps, global_cond=obs_cond)
                                loss = nn.functional.mse_loss(noise_pred, noise)
                            else:
                                # 混合 loss
                                cond_out = nets["noise_pred_net"](noisy_actions, timesteps, global_cond=obs_cond)
                                uncond_out = nets["noise_pred_net"](noisy_actions, timesteps,
                                                                    global_cond=torch.zeros_like(obs_cond))
                                loss_cond = nn.functional.mse_loss(cond_out, noise)
                                loss_uncond = nn.functional.mse_loss(uncond_out, noise)
                                loss = 0.5 * loss_cond + 0.5 * loss_uncond

                            # # random dropout for classifier-free guidance
                            # if torch.rand(1) < conditional_pdrop:
                            #     obs_cond.zero_()
                            #
                            # # predict the noise residual
                            # noise_pred = nets["noise_pred_net"](
                            #     noisy_actions, timesteps, global_cond=obs_cond
                            # )
                            #
                            # # L2 loss
                            # loss = nn.functional.mse_loss(noise_pred, noise)

                        # Evaluation
                        else:
                            noisy_action = noise
                            pred_action = noisy_action

                            self.noise_scheduler.set_timesteps(
                                self.num_diffusion_iters
                            )

                            for k in self.noise_scheduler.timesteps:
                                # predict noise
                                noise_pred = nets["noise_pred_net"](
                                    sample=pred_action,
                                    timestep=k,
                                    global_cond=obs_cond,
                                )

                                # inverse diffusion step (remove noise)
                                pred_action = self.noise_scheduler.step(
                                    model_output=noise_pred,
                                    timestep=k,
                                    sample=pred_action,
                                ).prev_sample

                            loss = nn.functional.mse_loss(naction, pred_action)

                            unnormalized_naction = unnormalize_data(
                                naction.detach().cpu().numpy(),
                                self.data_stat["action"],
                            )
                            unnormalized_pred_action = unnormalize_data(
                                pred_action.detach().cpu().numpy(),
                                self.data_stat["action"],
                            )
                            unnormalized_loss = nn.functional.mse_loss(
                                torch.tensor(unnormalized_naction),
                                torch.tensor(unnormalized_pred_action),
                            )

                        if not eval:
                            # optimize
                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # step lr scheduler every batch
                            # this is different from standard pytorch behavior
                            self.lr_scheduler.step()

                            # update Exponential Moving Average of the model weights
                            self.ema.step(nets.parameters())
                        else:
                            action_mse.append(unnormalized_loss.item())

                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))
                if self.writer is not None:
                    self.writer.add_scalar("Loss", np.mean(epoch_loss), epoch_idx)

                if eval:
                    return np.mean(epoch_loss), np.mean(action_mse)

                if wandb_logger is not None:
                    wandb_logger.step()
                    wandb_logger.log({"Loss": np.mean(epoch_loss), "epoch": epoch_idx})
                if (
                    save_path is not None
                    and epoch_idx % save_freq == 0
                    and epoch_idx != 0
                ):
                    model_path = os.path.join(
                        save_path, f"model_epoch_{epoch_idx}.ckpt"
                    )
                    self.save(model_path)

                # save last checkpoint
                model_path = os.path.join(save_path, f"last.ckpt")
                self.save(model_path)

                if eval_data is not None and epoch_idx % eval_freq == 0:
                    self.to_ema()
                    self.ema_nets.eval()
                    print("Evaluating one trajectory...")
                    obs, action = eval_data
                    _, mse, normalized_mse = self.eval(obs, action)
                    self.writer.add_scalar("Action_MSE", mse, epoch_idx)
                    self.writer.add_scalar("Normalized_MSE", normalized_mse, epoch_idx)

                    if wandb_logger is not None:
                        wandb_logger.log({"Action_MSE": mse})
                        wandb_logger.log({"Normalized_MSE": normalized_mse})
                    print(f"Action_MSE: {mse}, Normalized_MSE: {normalized_mse}")
                    self.ema_nets.train()
                    return None
                return None
            return None

    def eval(self, obs, action):
        obs_deque = collections.deque(
            [obs[0]] * self.obs_horizon, maxlen=self.obs_horizon
        )
        actions_pred = []

        i = 0
        while i < len(obs) - self.action_horizon:
            action_pred = self.forward(self.data_stat, obs_deque)
            for j in range(self.action_horizon):
                actions_pred.append(action_pred[j])
                obs_deque.append(obs[i + j])
            i += self.action_horizon

        actions_pred = np.array(actions_pred)
        action = np.array(action)
        mse = mse_loss(
            torch.tensor(actions_pred), torch.tensor(action[: len(actions_pred)])
        )

        normalized_action = normalize_data(action, self.data_stat["action"])
        normalized_action_pred = normalize_data(actions_pred, self.data_stat["action"])

        normalized_mse = mse_loss(
            torch.tensor(normalized_action_pred),
            torch.tensor(normalized_action[: len(actions_pred)]),
        )

        return actions_pred, mse, normalized_mse

    def to_ema(self):
        # Weights of the EMA model
        # is used for inference
        self.ema.copy_to(self.ema_nets.parameters())

    def load(self, path):
        def rename_key(old_key):
            new_key = old_key.replace("image", "img")
            unexpected = [
                "pos_encoder.encoder.mlp.0.weight",
                "pos_encoder.encoder.mlp.0.bias",
                "pos_encoder.encoder.mlp.2.weight",
                "pos_encoder.encoder.mlp.2.bias",
                "touch_encoder.encoder.mlp.0.weight",
                "touch_encoder.encoder.mlp.0.bias",
                "touch_encoder.encoder.mlp.2.weight",
                "touch_encoder.encoder.mlp.2.bias",
            ]
            missing = [
                "pos_encoder.linear.mlp.0.weight",
                "pos_encoder.linear.mlp.0.bias",
                "pos_encoder.linear.mlp.2.weight",
                "pos_encoder.linear.mlp.2.bias",
                "touch_encoder.linear.mlp.0.weight",
                "touch_encoder.linear.mlp.0.bias",
                "touch_encoder.linear.mlp.2.weight",
                "touch_encoder.linear.mlp.2.bias",
            ]
            for i, u in enumerate(unexpected):
                new_key = new_key.replace(u, missing[i])
            return new_key

        basename = os.path.basename(path)
        dirname = os.path.dirname(path)

        state_dict = torch.load(path, map_location="cuda")
        # rename model keys for backward compatibility
        state_dict = {rename_key(k): v for k, v in state_dict.items()}

        self.nets.load_state_dict(state_dict)

        if os.path.exists(os.path.join(dirname, "ema_" + basename)):
            ema_state_dict = torch.load(
                os.path.join(dirname, "ema_" + basename), map_location="cuda"
            )
            ema_state_dict = {rename_key(k): v for k, v in ema_state_dict.items()}
            self.ema_nets.load_state_dict(ema_state_dict)
        else:
            self.ema_nets.load_state_dict(state_dict)

    def save(self, path):
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.to_ema()
        torch.save(self.ema_nets.state_dict(), os.path.join(dirname, "ema_" + basename))
        torch.save(self.nets.state_dict(), path)

    def _get_data_forward(self, stats, obs_deque, data_key):
        sample = np.stack([x[data_key] for x in obs_deque])
        if data_key != "img" and (data_key != "touch" or not self.binarize_touch):
            # image is already normalized
            sample = normalize_data(sample, stats=stats[data_key])
        sample = (
            torch.from_numpy(sample).to(self.device, dtype=torch.float32).unsqueeze(0)
        )
        return sample

    def forward(self, stats, obs_deque, num_diffusion_iters=None):
        self.ema_nets.eval()

        if not num_diffusion_iters:
            num_diffusion_iters = self.num_diffusion_iters

        with torch.no_grad():
            features = []

            ### IMPT: make sure input is always in this order
            # eef, hand_pos, img, pos, touch
            for data_key in [
                dk
                for dk in ["eef", "hand_pos", "img", "pos", "touch"]
                if dk in self.representation_type
            ]:
                sample = self._get_data_forward(stats, obs_deque, data_key)
                if data_key == "img":
                    images = [
                        sample[:, :, i] for i in range(sample.shape[2])
                    ]  # [1, obs_horizon, M, C, H, W]
                    image_features = [
                        self.ema_nets[f"{data_key}_encoder"][i](
                            image.flatten(end_dim=1)
                        )
                        for i, image in enumerate(images)
                    ]
                    image_features = torch.stack(image_features, dim=2)
                    image_features = image_features.reshape(*sample.shape[:2], -1)
                    features.append(image_features)
                else:
                    feat = self.ema_nets[f"{data_key}_encoder"](
                        sample.flatten(end_dim=1)
                    )
                    feat = feat.reshape(*sample.shape[:2], -1)
                    features.append(feat)

            obs_features = torch.cat(features, dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)

            if self.without_sampling:
                action = self.ema_nets["bc_actor"](obs_cond)
                naction = action.reshape(-1, self.pred_horizon, self.action_dim)
            else:
                noisy_action = torch.randn(
                    (1, self.pred_horizon, self.action_dim), device=self.device
                )
                naction = noisy_action

                self.noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in self.noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = self.ema_nets["noise_pred_net"](
                        sample=naction, timestep=k, global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=naction
                    ).prev_sample

        # unnormalize action
        naction = naction.detach().to("cpu").numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats["action"])

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end, :]

        return action

    def eval_loader(self, eval_loader):
        self.ema_nets.eval()
        mse = self.train(num_epochs=1, dataloader=eval_loader, eval=True)
        return mse


    def run_diffusion_es(self, stats, obs_deque, obj_img, num_diffusion_iters=None, constraints=None, traj_origin=None,
                         use_cem=False, cem_iters=20,
                         num_elites=32,
                         temperature=0.1, visualize=False):
        self.ema_nets.eval()

        if not num_diffusion_iters:
            num_diffusion_iters = self.num_diffusion_iters

        if constraints is None:
            constraints = self.generate_constraints(stats)

        with torch.no_grad():
            features = []
            self.sampling_batch_size = 128

            ### IMPT: make sure input is always in this order
            # eef, hand_pos, img, pos, touch
            for data_key in [
                dk
                for dk in ["eef", "hand_pos", "img", "pos", "touch"]
                if dk in self.representation_type
            ]:
                sample = self._get_data_forward(stats, obs_deque, data_key)
                if data_key == "img":
                    images = [
                        sample[:, :, i] for i in range(sample.shape[2])
                    ]  # [1, obs_horizon, M, C, H, W]
                    image_features = [
                        self.ema_nets[f"{data_key}_encoder"][i](
                            image.flatten(end_dim=1)
                        )
                        for i, image in enumerate(images)
                    ]
                    image_features = torch.stack(image_features, dim=2)
                    image_features = image_features.reshape(*sample.shape[:2], -1)
                    features.append(image_features)
                else:
                    feat = self.ema_nets[f"{data_key}_encoder"](
                        sample.flatten(end_dim=1)
                    )
                    feat = feat.reshape(*sample.shape[:2], -1)
                    features.append(feat)

            obs_features = torch.cat(features, dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
            obs_cond = obs_cond.repeat(self.sampling_batch_size, 1)
            # Add Gaussian noise to obs condition to enhance trajectory diversity
            # obs_noise_level = 1.0
            # obs_cond = obs_cond + obs_noise_level * torch.randn_like(obs_cond)
            # obs_cond = torch.randn_like(obs_cond)

            # scaling_factor = 0.25
            # obs_cond = obs_cond * scaling_factor

            # alpha = 0.25 # [0.3, 0.7]
            # obs_cond = alpha * obs_cond + (1 - alpha) * torch.randn_like(obs_cond)

            # object-related keypoints extraction

            # get keypoints

            # get object-related keypoints
            # keypoint_config = get_config(config_path="/home/zhuoli/xtrainer_clover/configs/keypoint_config.yaml")
            # keypoint_proposer = KeypointProposer(keypoint_config['keypoint_proposer'])
            # self.keypoints = keypoint_proposer.run(visualize_projection=True)

            # Diffusion-es parameter initialization
            trunc_step_schedule = np.linspace(5, 1, cem_iters).astype(int)
            noise_scale = 0.1

            noisy_action = torch.randn(
                (self.sampling_batch_size, self.pred_horizon, self.action_dim), device=self.device
            )

            naction = noisy_action
            self.noise_scheduler.set_timesteps(num_diffusion_iters)

            # Initialize elite set
            population_trajectories, population_scores, population_info = self.rollout(
                obs_cond,
                naction,
                constraints,
                initial_rollout=True,
                deterministic=False,
                noise_scale=noise_scale,
            )

            time1 = time.time()
            for i in range(cem_iters):
                n_trunc_steps = trunc_step_schedule[i]

                """
                Local MPPI update
                """
                # Compute reward-probabilities
                reward_probs = torch.exp(temperature * -population_scores)
                reward_probs = reward_probs / reward_probs.sum()
                probs = reward_probs

                """
                Resample and mutate (renoise-denoise)
                """
                if use_cem:
                    elites = torch.argsort(population_scores)[:num_elites]
                    indices = torch.randint(0, num_elites, (self.sampling_batch_size,), device=self.device)
                    population_trajectories = population_trajectories[elites[indices]]
                    population_trajectories = self.renoise(population_trajectories, n_trunc_steps)
                else:
                    indices = torch.multinomial(probs, self.sampling_batch_size,
                                                replacement=True)  # torch.multinomial(probs, 1).squeeze(1)
                    population_trajectories = population_trajectories[indices]
                    population_trajectories = self.renoise(population_trajectories, n_trunc_steps)

                # Denoise
                population_trajectories, population_scores, population_info = self.rollout(
                    obs_cond,
                    population_trajectories,
                    constraints,
                    initial_rollout=False,
                    deterministic=False,
                    n_trunc_steps=n_trunc_steps,
                    noise_scale=noise_scale,
                )

        time2 = time.time()
        print("Diffusion-ES planning time", time2 - time1)
        print("population_scores", population_scores)
        print("best score", population_scores.min())
        print("traj_origin shape", traj_origin.shape)
        # unnormalize action
        population_trajectories = population_trajectories.detach().to("cpu").numpy()
        population_trajectories = unnormalize_data(population_trajectories, stats["action"])

        # align the trajectory
        population_trajectories, left_ee_positions, right_ee_positions = align_trajs_to_origin(population_trajectories, traj_origin)

        # select the best trajectory
        best_trajectory = population_trajectories[population_scores.argmin()]

        # visualize the trajectory
        if visualize:
            visualize_trajectory(left_ee_positions, right_ee_positions, best_trajectory)

        # schedule the executed trajectory
        best_trajectory = bimanual_coordinator(
            mode="left",
            traj_origin=traj_origin,
            best_trajectory=best_trajectory
        )

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = best_trajectory[start:end, :]

        out = {
            "trajectory": best_trajectory,
            "multimodal_trajectories": population_trajectories,
            "scores": population_scores,
        }

        return action

    def rollout(
            self,
            obs_cond,
            naction,
            constraints,
            initial_rollout=True,
            deterministic=True,
            n_trunc_steps=5,
            noise_scale=1.0,
            ablate_diffusion=False,
            gamma = 0.5
    ):
        if initial_rollout:
            timesteps = self.noise_scheduler.timesteps
        else:
            timesteps = self.noise_scheduler.timesteps[-n_trunc_steps:]

        if ablate_diffusion and not initial_rollout:
            timesteps = []

        for k in timesteps:

            # predict noise with classifier-free guidance
            noise_pred = self.ema_nets["noise_pred_net"](
                sample=naction, timestep=k, global_cond=obs_cond
            )
            uncond_noise_pred = self.ema_nets["noise_pred_net"](
                sample=naction, timestep=k, global_cond=torch.zeros_like(obs_cond)
            )
            # noise_pred = (1 + gamma) * noise_pred - gamma * uncond_noise_pred
            noise_pred = uncond_noise_pred

            if deterministic:
                eta = 0.0
            else:
                prev_alpha = self.noise_scheduler.alphas[k-1]
                alpha = self.noise_scheduler.alphas[k]
                eta = noise_scale * torch.sqrt((1 - prev_alpha) / (1 - alpha)) * \
                        torch.sqrt((1 - alpha) / prev_alpha)

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction, eta=eta
            ).prev_sample

            # scores, info = compute_constraint_scores(constraints, naction)
            scores, info = constraints(naction)

        return naction, scores, info


    def renoise(self, population_trajectories, t):
        noise = torch.randn(population_trajectories.shape, device=self.device)
        population_trajectories = self.noise_scheduler.add_noise(population_trajectories, noise, self.noise_scheduler.timesteps[-t])
        return population_trajectories

    def generate_constraints(self, stats):
        """
        Each constraint is a non-differentiable black-box cost function that maps bimanual trajectory to some scalar cost to be minimized.
        """
        keypoints_path = '/home/zhuoli/xtrainer_clover/configs/metadata.json'
        with open(keypoints_path, 'r') as f:
            data = json.load(f)
        keypoints_list = data['init_keypoint_positions']
        keypoints = np.array(keypoints_list)
        # print("keypoints for NBCFs", keypoints)# Shape: (5, 3)

        # <editor-fold desc="utils">
        def unnormalize_traj(trajectory):
            device = trajectory.device
            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1, 16, 14)
            trajectory = unnormalize_data(trajectory, stats["action"])
            return trajectory
        # </editor-fold>

        # <editor-fold desc="single-arm cartesian NBCFs">
        def left_arm_height_upward(trajectory):
            """
            Compute the reward for "raising the left arm slightly" based on joint angles.

            :param trajectory: Tensor of shape (batch, 16, 14), representing bimanual motion trajectories.
                               Each trajectory consists of 16 timesteps, and each timestep has 14 joint angles
                               (7 for the left arm, 7 for the right arm).
            :return: Tensor of shape (batch,), representing the reward scores for each trajectory.
            """
            # Convert trajectory to numpy and unnormalize
            device = trajectory.device
            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1, 16, 14)
            trajectory = unnormalize_data(trajectory, stats["action"])
            left_trajectory = trajectory[:, :, :6]

            # Extract predicted left arm ee positions
            ee_position, _ = fk_solver(left_trajectory)  # Shape: (batch, 16)
            scores = np.zeros(self.sampling_batch_size)

            # Iterate scoring each trajectory in the batch
            for i in range(self.sampling_batch_size):
                initial_height = ee_position[i, 0, 2]  # First timestep
                final_height = ee_position[i, -1, 2]  # Last timestep

                # Compute reward as the height increase from the first to the last timestep
                scores[i] = final_height - initial_height
                # scores[i] = initial_height - final_height
            scores = -torch.as_tensor(scores, device=device)
            return scores, {}

        def right_arm_height_downward(trajectory):
            """
            Compute the reward for "lowering the right arm slightly" based on joint angles.

            :param trajectory: Tensor of shape (batch, 16, 14), representing bimanual motion trajectories.
                               Each trajectory consists of 16 timesteps, and each timestep has 14 joint angles
                               (7 for the left arm, 7 for the right arm).
            :return: Tensor of shape (batch,), representing the reward scores for each trajectory.
            """
            # Convert trajectory to numpy and unnormalize
            device = trajectory.device
            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1, 16, 14)
            trajectory = unnormalize_data(trajectory, stats["action"])
            right_trajectory = trajectory[:, :, 7:13]  # Extract right arm joint angles

            # Extract predicted right arm end-effector positions
            ee_position, _  = fk_solver(right_trajectory)  # Shape: (batch, 16, 3)
            scores = np.zeros(self.sampling_batch_size)

            # Iterate scoring each trajectory in the batch
            for i in range(self.sampling_batch_size):
                initial_height = ee_position[i, 0, 2]  # First timestep height (z-axis)
                final_height = ee_position[i, -1, 2]  # Last timestep height (z-axis)

                # Compute reward as the height decrease from the first to the last timestep
                scores[i] = initial_height - final_height
            scores = -torch.as_tensor(scores, device=device)
            return scores, {}

        def right_wrist_rotate_outward(trajectory):
            """
            Compute the reward for "rotating the right wrist slightly outward" based on joint angles.

            :param trajectory: Tensor of shape (batch, 16, 14), representing bimanual motion trajectories.
                               Each trajectory consists of 16 timesteps, and each timestep has 14 joint angles
                               (7 for the left arm, 7 for the right arm).
            :return: Tensor of shape (batch,), representing the reward scores for each trajectory.
            """
            # Convert trajectory to numpy and unnormalize
            device = trajectory.device
            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1, 16, 14)
            trajectory = unnormalize_data(trajectory, stats["action"])

            # Extract right arm trajectory
            right_trajectory = trajectory[:, :, 7:13]  # Right arm: joints 7 to 13

            # Assuming the right wrist rotation corresponds to the last joint (index 6 of the 7 right arm joints)
            wrist_joint_index = 5
            scores = np.zeros(self.sampling_batch_size)

            # Iterate through the batch and compute wrist rotation change
            for i in range(self.sampling_batch_size):
                initial_angle = right_trajectory[i, 0, wrist_joint_index]
                final_angle = right_trajectory[i, -1, wrist_joint_index]

                # Outward rotation is assumed to be a positive change in joint angle
                scores[i] = final_angle - initial_angle

            scores = -torch.as_tensor(scores, device=device)
            return scores, {}

        # </editor-fold>

        # <editor-fold desc="bimanual carteisan NBCFs">

        def both_arms_forward_motion(trajectory):
            """
            Compute the reward for "moving both hands forward" based on joint angles.

            :param trajectory: Tensor of shape (batch, 16, 14), representing bimanual motion trajectories.
                               Each trajectory consists of 16 timesteps, and each timestep has 14 joint angles
                               (7 for the left arm, 7 for the right arm).
            :return: Tensor of shape (batch,), representing the reward scores for each trajectory.
            """
            # Convert trajectory to numpy and unnormalize
            device = trajectory.device
            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1, 16, 14)
            trajectory = unnormalize_data(trajectory, stats["action"])

            left_trajectory = trajectory[:, :, :6]
            right_trajectory = trajectory[:, :, 7:13]

            # Extract predicted end-effector positions for both arms
            left_ee_position, _ = fk_solver(left_trajectory)  # Shape: (batch, 16, 3)
            right_ee_position, _ = fk_solver(right_trajectory)  # Shape: (batch, 16, 3)

            scores = np.zeros(self.sampling_batch_size)

            # Iterate scoring each trajectory in the batch
            for i in range(self.sampling_batch_size):
                # Compute forward (x-axis) displacement for both hands
                left_initial_x = left_ee_position[i, 0, 0]
                left_final_x = left_ee_position[i, -1, 0]
                right_initial_x = right_ee_position[i, 0, 0]
                right_final_x = right_ee_position[i, -1, 0]

                # Sum of forward displacements
                left_forward = left_final_x - left_initial_x
                right_forward = right_final_x - right_initial_x

                scores[i] = left_forward + right_forward

            scores = -torch.as_tensor(scores, device=device)
            return scores, {}

        def widen_hands_distance(trajectory):
            """
            Compute the reward for "Spread your hands" based on end-effector positions.

            :param trajectory: Tensor of shape (batch, 16, 14), representing bimanual motion trajectories.
                               Each trajectory has 16 timesteps, and each timestep has 14 joint angles
                               (7 for the left arm, 7 for the right arm).
            :return: Tensor of shape (batch,), representing the reward scores for each trajectory.
            """
            # Convert trajectory to numpy and unnormalize
            device = trajectory.device
            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1, 16, 14)
            trajectory = unnormalize_data(trajectory, stats["action"])

            # Extract left and right arm joint angles
            left_trajectory = trajectory[:, :, :6]
            right_trajectory = trajectory[:, :, 7:13]

            # Forward kinematics to get end-effector positions
            left_ee_position, _ = fk_solver(left_trajectory)  # Shape: (batch, 16, 3)
            right_ee_position, _ = fk_solver(right_trajectory)  # Shape: (batch, 16, 3)

            left_ee_position, right_ee_position = bimanual_frame_transform(left_ee_position, right_ee_position)

            scores = np.zeros(left_ee_position.shape[0])  # batch size

            for i in range(left_ee_position.shape[0]):
                # Compute initial and final distances between left and right hands
                initial_dist = np.linalg.norm(left_ee_position[i, 0] - right_ee_position[i, 0])
                final_dist = np.linalg.norm(left_ee_position[i, -1] - right_ee_position[i, -1])

                # initial_dist = left_ee_position[i, 0, 1] - right_ee_position[i, 0, 1]
                # final_dist = left_ee_position[i, -1, 1] - right_ee_position[i, -1, 1]

                # Reward is the increase in distance
                scores[i] = final_dist - initial_dist
                # scores[i] = initial_dist - final_dist


            scores = -torch.as_tensor(scores, device=device)
            return scores, {}

        def lower_both_hands_evenly(trajectory):
            """
            Compute the reward for "lower both hands evenly" based on vertical displacement (z-axis)
            and symmetry between arms.

            :param trajectory: Tensor of shape (batch, 16, 14), representing bimanual motion trajectories.
                               Each trajectory has 16 timesteps, and each timestep has 14 joint angles
                               (7 for the left arm, 7 for the right arm).
            :return: Tensor of shape (batch,), representing the reward scores for each trajectory.
            """
            # Convert trajectory to numpy and unnormalize
            device = trajectory.device
            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1, 16, 14)
            trajectory = unnormalize_data(trajectory, stats["action"])

            # Extract joint angles for each arm
            left_trajectory = trajectory[:, :, :7]
            right_trajectory = trajectory[:, :, 7:]

            # Compute end-effector positions
            left_ee_position, _ = fk_solver(left_trajectory)  # Shape: (batch, 16, 3)
            right_ee_position, _ = fk_solver(right_trajectory)  # Shape: (batch, 16, 3)

            scores = np.zeros(left_ee_position.shape[0])  # batch size

            for i in range(left_ee_position.shape[0]):
                # Get initial and final z-positions (vertical) of both hands
                left_z_start = left_ee_position[i, 0, 2]
                left_z_end = left_ee_position[i, -1, 2]
                right_z_start = right_ee_position[i, 0, 2]
                right_z_end = right_ee_position[i, -1, 2]

                # Compute amount lowered
                left_drop = left_z_start - left_z_end
                right_drop = right_z_start - right_z_end

                # Compute discrepancy between hands (should be minimal for even lowering)
                symmetry_penalty = np.abs(left_drop - right_drop)

                # Reward: total lowering minus symmetry penalty
                lowering_reward = (left_drop + right_drop) - symmetry_penalty
                scores[i] = lowering_reward

            scores = -torch.as_tensor(scores, device=device)
            return scores, {}

        # </editor-fold>

        # <editor-fold desc="bimanual joint-space NBCFs">
        def lift_the_elbows(trajectory):
            """
            Compute the reward for 'lifting the elbows a bit higher' by using forward kinematics
            to get elbow positions in Cartesian space.

            :param trajectory: Tensor of shape (batch, 16, 14), representing bimanual joint angle trajectories.
                               Each trajectory has 16 timesteps and 14 joint angles (7 left + 7 right arm).
            :return: Tensor of shape (batch,), reward scores for each trajectory.
            """
            # Step 1: Convert tensor to numpy and unnormalize
            device = trajectory.device
            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1, 16, 14)
            trajectory = unnormalize_data(trajectory, stats["action"])

            # Step 2: Separate left and right arm joint angles
            left_trajectory = trajectory[:, :, :6]  # (batch, 16, 7)
            right_trajectory = trajectory[:, :, 7:13]  # (batch, 16, 7)

            # Step 3: Run FK to get joint positions of the whole kinematic chain
            left_joint_positions, _ = fk_solver(left_trajectory, ee_link="Link3")
            right_joint_positions, _ = fk_solver(right_trajectory, ee_link="Link3")

            # Step 5: 计算每个trajectory中肘部高度的变化
            scores = np.zeros(self.sampling_batch_size)

            for i in range(self.sampling_batch_size):
                left_initial_z = left_joint_positions[i, 0, 2]
                left_final_z = left_joint_positions[i, -1, 2]
                right_initial_z = right_joint_positions[i, 0, 2]
                right_final_z = right_joint_positions[i, -1, 2]

                # 奖励是：两个肘部高度提升的总和
                # left_lift = left_final_z - left_initial_z
                # right_lift = right_final_z - right_initial_z

                left_lift = left_initial_z - left_final_z
                right_lift = right_initial_z - right_final_z

                scores[i] = left_lift + right_lift

            # 转换为 torch tensor 并返回负的 cost
            scores = -torch.as_tensor(scores, device=device)
            return scores, {}

        def bend_arms_into_holding_pose(trajectory):
            """
            Compute the reward for 'bend both arms into a curved holding pose' by evaluating elbow flexion
            in joint space. A curved holding pose typically involves elbow joints bending toward ~90 degrees.

            :param trajectory: Tensor of shape (batch, 16, 14), representing bimanual joint angle trajectories.
                               Each trajectory contains 16 timesteps, and each timestep has 14 joint angles
                               (7 for the left arm, 7 for the right arm).
            :return: Tensor of shape (batch,), reward scores for each trajectory.
            """
            # Step 1: Convert to numpy and unnormalize
            device = trajectory.device
            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1, 16, 14)
            trajectory = unnormalize_data(trajectory, stats["action"])

            # Step 2: Extract elbow joint angles (assumed to be joint index 2 for both arms)
            # Modify index if elbow joint is mapped differently in your robot
            left_elbow_angles = trajectory[:, :, 2]  # Shape: (batch, 16)
            right_elbow_angles = trajectory[:, :, 9]  # Shape: (batch, 16)

            scores = np.zeros(left_elbow_angles.shape[0])  # batch size

            for i in range(left_elbow_angles.shape[0]):
                # Use final timestep to evaluate the holding pose
                left_final = left_elbow_angles[i, -1]
                right_final = right_elbow_angles[i, -1]

                # Ideal elbow angle for curved holding pose is around 90 degrees ≈ 1.57 rad
                target_angle = 1.57

                # Compute deviation from desired bend angle
                left_error = np.abs(left_final - target_angle)
                right_error = np.abs(right_final - target_angle)

                # Reward is negative of total deviation
                scores[i] = -(left_error + right_error)

            scores = torch.as_tensor(scores, device=device)
            return scores, {}

        # </editor-fold>

        # <editor-fold desc="object-related NBCFs">
        def avoid_left_collision(trajectory):
            """
            Compute the reward for "watching out for the vase on the left hand" by maintaining a safe distance.

            :param trajectory: Tensor of shape (batch, 16, 14), representing bimanual motion trajectories.
                               Each trajectory consists of 16 timesteps, and each timestep has 14 joint angles
                               (7 for the left arm, 7 for the right arm).
            :param vase_position: Numpy array of shape (3,), the fixed world coordinates of the vase.
            :param safe_distance: Minimum allowable distance to the vase to avoid collision.
            :return: Tensor of shape (batch,), representing the reward scores for each trajectory.
            """
            device = trajectory.device
            trajectory = trajectory.detach().cpu().numpy()
            trajectory = trajectory.reshape(-1, 16, 14)
            trajectory = unnormalize_data(trajectory, stats["action"])
            left_trajectory = trajectory[:, :, :6]

            # Extract predicted left arm end-effector positions
            ee_positions, _ = fk_solver(left_trajectory)  # Shape: (batch, 16, 3)
            scores = np.zeros(trajectory.shape[0])

            for i in range(trajectory.shape[0]):
                distance = np.linalg.norm(ee_positions[i] - keypoints[5], axis=1)
                mean_distance = np.mean(distance)
                scores[i] = mean_distance

            scores = -torch.as_tensor(scores, device=device)
            return scores, {}

        # </editor-fold>

        return avoid_left_collision

# def compute_constraint_scores(constraints, trajectory):
#     all_info = {}
#     total_cost = torch.zeros(trajectory.shape[0], device=trajectory.device)
#     for constraint in constraints:
#         cost, info = constraint(trajectory)
#         total_cost += cost
#         all_info.update(info)
#     return total_cost, all_info


class BaseLowdimPolicy(ModuleAttrMixin):
    # ========= inference  ============
    # also as self.device and self.dtype for inference device transfer
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            obs: B,To,Do
        return:
            action: B,Ta,Da
        To = 3
        Ta = 4
        T = 6
        |o|o|o|
        | | |a|a|a|a|
        |o|o|
        | |a|a|a|a|a|
        | | | | |a|a|
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()



class ConditionalBimanualMotionPrior(BaseLowdimPolicy):
    def __init__(self,
                 noise_scheduler: DDPMScheduler,
                 # task parameters
                 horizon,
                 obs_dim,
                 action_dim,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 # arch
                 causal_attn=False,
                 time_as_cond=True,
                 obs_as_cond=True,
                 pred_action_steps_only=False,
                 # parameters passed to step
                 **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        eef_dim = obs_dim
        eef_feature_dim = 64
        eef_encoder = StateEncoder(
            input_size=eef_dim,
            output_size=eef_feature_dim,
            hidden_size=128,
            dropout=0.0,
        )

        input_dim = action_dim if obs_as_cond else (eef_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = eef_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            # n_layer=n_layer,
            # n_head=n_head,
            # n_emb=n_emb,
            # p_drop_emb=p_drop_emb,
            # p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            # n_cond_layers=n_cond_layers
        )

        self.model = model
        self.eef_encoder = eef_encoder
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data, condition_mask,
                           cond=None, generator=None,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict  # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = nobs[:, :To]
            nobs_features = self.eef_encoder(this_nobs)
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:, :self.n_obs_steps, :]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


class UnConditionalBimanualMotionPrior(BaseLowdimPolicy):
    def __init__(self,
                 noise_scheduler: DDPMScheduler,
                 # task parameters
                 horizon,
                 obs_dim,
                 action_dim,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 # arch
                 causal_attn=False,
                 time_as_cond=False,
                 obs_as_cond=False,
                 pred_action_steps_only=False,
                 # parameters passed to step
                 **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        # eef_dim = obs_dim
        # eef_feature_dim = 64
        # eef_encoder = StateEncoder(
        #     input_size=eef_dim,
        #     output_size=eef_feature_dim,
        #     hidden_size=128,
        #     dropout=0.0,
        # )

        input_dim = action_dim
        output_dim = input_dim
        cond_dim = 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            # n_layer=n_layer,
            # n_head=n_head,
            # n_emb=n_emb,
            # p_drop_emb=p_drop_emb,
            # p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            # n_cond_layers=n_cond_layers
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.quat_dims = list(range(3, 7)) + list(range(11, 15))
        self.normalizer = QuatSafeNormalizer(self.quat_dims)
        # self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        # print("Received kwargs:", kwargs)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def unconditional_sample(self,
                             action_data, generator=None,
                             # keyword arguments to scheduler.step
                             **kwargs
                             ):
        model = self.model
        scheduler = self.noise_scheduler

        print("Noise Scheduler Config:")
        print(self.noise_scheduler.config)

        trajectory = torch.randn(
            size=action_data.shape,
            dtype=action_data.dtype,
            device=action_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # # 1. apply conditioning
            # trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t)

            # print("kwargs:", kwargs)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
            ).prev_sample

            # trajectory = scheduler.step(
            #     model_output, t, trajectory,
            #     generator=generator,
            #     **kwargs
            # ).prev_sample

        # finally make sure conditioning is enforced
        # trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, action_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        action_dict: must include "action" key
        result: must include "action" key
        """

        # assert 'obs' in obs_dict
        # assert 'past_action' not in obs_dict  # not implemented yet
        # nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        # B, _, Do = nobs.shape
        # To = self.n_obs_steps
        # assert Do == self.obs_dim

        B, _, Da = action_dict['action'].shape
        T = self.horizon
        assert Da == self.action_dim

        # build input
        device = self.device
        dtype = self.dtype
        shape = (B, T, Da)
        action_data = torch.zeros(size=shape, device=device, dtype=dtype)

        nsample = self.unconditional_sample(
            action_data,
            **self.kwargs)

        # # handle different ways of passing observation
        # cond = None
        # cond_data = None
        # cond_mask = None
        # if self.obs_as_cond:
        #     this_nobs = nobs[:, :To]
        #     nobs_features = self.eef_encoder(this_nobs)
        #     cond = nobs_features.reshape(B, To, -1)
        #     shape = (B, T, Da)
        #     if self.pred_action_steps_only:
        #         shape = (B, self.n_action_steps, Da)
        #     cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        #     cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        # else:
        #     # condition through impainting
        #     shape = (B, T, Da + Do)
        #     cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        #     cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        #     cond_data[:, :To, Da:] = nobs[:, :To]
        #     cond_mask[:, :To, Da:] = True

        # # run sampling
        # nsample = self.unconditional_sample(
        #     cond_data,
        #     cond_mask,
        #     cond=cond,
        #     **self.kwargs)


        # unnormalize prediction

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        action = action_pred[:, :self.n_action_steps]

        # # get action
        # if self.pred_action_steps_only:
        #     action = action_pred
        # else:
        #     start = To - 1
        #     end = start + self.n_action_steps
        #     action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }

        # if not self.obs_as_cond:
        #     nobs_pred = nsample[..., Da:]
        #     obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
        #     action_obs_pred = obs_pred[:, start:end]
        #     result['action_obs_pred'] = action_obs_pred
        #     result['obs_pred'] = obs_pred

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        # obs = nbatch['obs']
        action = nbatch['action']
        trajectory = action
        # print("trajectory:", trajectory.min().item(), trajectory.max().item(), torch.isnan(trajectory).any().item(),
        #       torch.isinf(trajectory).any().item())

        # # handle different ways of passing observation
        # cond = None
        # trajectory = action
        # if self.obs_as_cond:
        #     cond = obs[:, :self.n_obs_steps, :]
        #     if self.pred_action_steps_only:
        #         To = self.n_obs_steps
        #         start = To - 1
        #         end = start + self.n_action_steps
        #         trajectory = action[:, start:end]
        # else:
        #     trajectory = torch.cat([action, obs], dim=-1)


        # generate impainting mask
        # if self.pred_action_steps_only:
        #     condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        # else:
        #     condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        # print("noisy_trajectory has NaN:", torch.isnan(noisy_trajectory).any().item())

        # # compute loss mask
        # loss_mask = ~condition_mask
        #
        # # apply conditioning
        # noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # print("pred stats:", pred.min().item(), pred.max().item(), torch.isnan(pred).any().item(),
        #       torch.isinf(pred).any().item())
        # print("target stats:", target.min().item(), target.max().item(), torch.isnan(target).any().item(),
        #       torch.isinf(target).any().item())
        loss = F.mse_loss(pred, target, reduction='none')
        # loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

