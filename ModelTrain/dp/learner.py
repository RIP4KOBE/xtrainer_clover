import collections
import copy
import os

import numpy as np
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from ModelTrain.dp.scheduler.Sampling_DDPMScheduler import SamplingDDPMScheduler
from ModelTrain.manipulability.bimanual_manip_learning import GMRModel
from ModelTrain.dp.optimizer.manipulability_optimizer import ManipulabilityOptimizer
from diffusers.training_utils import EMAModel
from ModelTrain.dp.models import *
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Tuple
from omegaconf import OmegaConf


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
        use_ddim=False,
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

        # Manipulability optimizer
        self.sampling_optimizer = ManipulabilityOptimizer()
        # Parameters for Manipulability-guided sampling
        cfg = OmegaConf.load("/home/zhuoli/dobot_xtrainer/ModelTrain/config/optimizer.yaml")
        self.gmm_ckpt_path = cfg.gmm_ckpt_path
        self.gmr_model = GMRModel(self.gmm_ckpt_path)
        self.dt = 1E-1  # Time step for GMR prediction


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
        sampling_eval=False,
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
                        pred_count = 1

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

                                if sampling_eval:
                                    print("Manipulability-guided sampling")
                                    xIn = torch.tensor(pred_count * self.action_horizon * self.dt)
                                    # Predict manipulability ellipsoid for the current time step using GMR
                                    M_t_pred, _ = self.gmr_model.gmr_regression(xIn)  # Predicted manipulability
                                    # ellipsoid (B, 3, 3)
                                    M_t_pred = self.gmr_model.vec2symmat(M_t_pred)
                                    pred_action = self.p_sample_loop(pred_action, obs_cond, M_t_pred)
                                    pred_count += 1

                                else:
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


                                if sampling_eval:
                                    loss = nn.functional.mse_loss(naction[0], pred_action[0])
                                    unnormalized_naction = unnormalize_data(
                                        naction[0].detach().cpu().numpy(),
                                        self.data_stat["action"],
                                    )
                                    unnormalized_pred_action = unnormalize_data(
                                        pred_action[0].detach().cpu().numpy(),
                                        self.data_stat["action"],
                                    )
                                    unnormalized_loss = nn.functional.mse_loss(
                                        torch.tensor(unnormalized_naction),
                                        torch.tensor(unnormalized_pred_action),
                                    )

                                else:
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

    @torch.no_grad()
    def sample_with_optimizer(self, stats, obs_deque, pred_count):
        self.ema_nets.eval()

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

            # sample noise to add to actions
            noisy_action = torch.randn(
                (1, self.pred_horizon, self.action_dim), device=self.device
            )
            naction = noisy_action

            xIn = torch.tensor(pred_count * self.action_horizon * self.dt)
            # Predict manipulability ellipsoid for the current time step using GMR
            M_t_pred, _ = self.gmr_model.gmr_regression(xIn)  # Predicted 3x3 manipulability ellipsoid
            M_t_pred = self.gmr_model.vec2symmat(M_t_pred)
            naction = self.p_sample_loop(naction, obs_cond, M_t_pred)

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

    @torch.no_grad()
    def p_sample_loop(self, noisy_action, obs_cond, M_t_pred) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape

        Return:
            Sampled data, <B, T, ...>
        """
        pred_action = noisy_action
        for t in reversed(range(0, self.num_diffusion_iters)):
            x_t = pred_action
            pred_action = self.p_sample(x_t, t, obs_cond, M_t_pred)

        return pred_action

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, obs_cond: torch.Tensor, M_t_pred: torch.Tensor) -> torch.Tensor:
        """ One step of reverse diffusion process

        $x_{t-1} = \tilde{\mu} + \sqrt{\tilde{\beta}} * z$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature

        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        model_mean, model_variance = self.p_mean_variance(x_t, t, obs_cond)
        ## sampling with mean updated by optimizer and planner
        ## openai guided diffusion uses the input x to compute gradient, see
        ## https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L436
        gradient = self.sampling_optimizer.gradient(model_mean, model_variance, M_t_pred)
        print(f"Gradient: {gradient[0]}")
        model_mean[0] = model_mean[0] + gradient[0]
        # model_mean = model_mean + gradient

        # equation for sampling from Gaussian distribution: x=μ+L⋅z
        # pred_x = model_mean + (0.5 * model_log_variance).exp() * noise
        pred_x = model_mean + model_variance

        return pred_x

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Calculate the mean and variance, we adopt the following first equation.

        $\tilde{\mu} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0$
        $\tilde{\mu} = \frac{1}{\sqrt{\alpha}_t}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor

        Return:
            (model_mean, posterior_variance, posterior_log_variance)
        """
        pred_noise = self.ema_nets["noise_pred_net"](
            sample=x_t, timestep=t, global_cond=cond
        )

        self.noise_scheduler = SamplingDDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        # pred_x0 = self.scheduler.step(
        #     model_output=pred_noise, timestep=t, sample=x_t
        # ).pred_original_sample

        model_mean = self.noise_scheduler.step(
            model_output=pred_noise, timestep=t, sample=x_t
        ).prev_sample

        posterior_variance = self.noise_scheduler.step(
            model_output=pred_noise, timestep=t, sample=x_t
        ).variance

        return model_mean, posterior_variance

    def eval_loader(self, eval_loader, sampling=False):
        self.ema_nets.eval()
        mse = self.train(num_epochs=1, dataloader=eval_loader, eval=True, sampling_eval=sampling)
        return mse
