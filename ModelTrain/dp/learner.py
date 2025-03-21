import collections
import copy
import os
import time

import numpy as np
import torch
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from jsonschema.exceptions import best_match

from ModelTrain.dp.models import *
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from vis_utils import visualize_trajectory, forward_kinematics


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


    def run_diffusion_es(self, stats, obs_deque, num_diffusion_iters=None, constraints=None,
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
            obs_noise_level = 1.5
            obs_cond = obs_cond + obs_noise_level * torch.randn_like(obs_cond)
            # obs_cond = torch.randn_like(obs_cond)

            # scaling_factor = 0.3
            # obs_cond = obs_cond * scaling_factor

            # alpha = 0.4 # [0.3, 0.7]
            # obs_cond = alpha * obs_cond + (1 - alpha) * torch.randn_like(obs_cond)

            # Diffusion-es parameter initialization
            trunc_step_schedule = np.linspace(5, 1, cem_iters).astype(int)
            noise_scale = 2.3

            # Initialize elite set
            noisy_action = torch.randn(
                (self.sampling_batch_size, self.pred_horizon, self.action_dim), device=self.device
            )

            naction = noisy_action
            self.noise_scheduler.set_timesteps(num_diffusion_iters)

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
        # unnormalize action
        population_trajectories = population_trajectories.detach().to("cpu").numpy()
        population_trajectories = unnormalize_data(population_trajectories, stats["action"])
        best_trajectory = population_trajectories[population_scores.argmin()]

        if visualize:
            visualize_trajectory(population_trajectories, best_trajectory)

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = best_trajectory[0][start:end, :]

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
            ablate_diffusion=False
    ):
        if initial_rollout:
            timesteps = self.noise_scheduler.timesteps
        else:
            timesteps = self.noise_scheduler.timesteps[-n_trunc_steps:]

        if ablate_diffusion and not initial_rollout:
            timesteps = []

        for k in timesteps:
            # predict noise
            noise_pred = self.ema_nets["noise_pred_net"](
                sample=naction, timestep=k, global_cond=obs_cond
            )

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
        Each constraint is a function that maps an bimanual trajectory to some scalar cost to be minimized.
        """
        ## A set of tested NBCFs for bimanual manipulation
        def left_arm_height_upward(trajectory):
            """
            Compute the reward for "raising the left arm slightly" based on joint angles.

            :param trajectory: Tensor of shape (batch, 16, 14), representing bimanual motion trajectories.
                               Each trajectory consists of 16 timesteps, and each timestep has 14 joint angles
                               (7 for the left arm, 7 for the right arm).
            :return: Tensor of shape (batch,), representing the reward scores for each trajectory.
            """
            # Extract left arm joint angles from the trajectory
            # Assuming the left arm's vertical movement is primarily affected by the 3rd joint (index 2)
            trajectory = unnormalize_data(trajectory, stats["action"])

            ee_pose = forward_kinematics(trajectory)  # Shape: (batch, 16)

            # Initialize reward tensor
            scores = torch.zeros(self.sampling_batch_size, device=trajectory.device)

            # Iterate through each trajectory in the batch
            for i in range(self.sampling_batch_size):
                initial_height = ee_pose[i, 0, 2]  # First timestep
                final_height = ee_pose[i, -1, 2]  # Last timestep

                # Compute reward as the height increase from the first to the last timestep
                scores[i] = final_height - initial_height
                # scores[i] = initial_height - final_height

            return -scores, {}  # Return in (cost, info) format # Negative sign since Diffusion-ES minimizes the cost


        return left_arm_height_upward



# def compute_constraint_scores(constraints, trajectory):
#     all_info = {}
#     total_cost = torch.zeros(trajectory.shape[0], device=trajectory.device)
#     for constraint in constraints:
#         cost, info = constraint(trajectory)
#         total_cost += cost
#         all_info.update(info)
#     return total_cost, all_info




