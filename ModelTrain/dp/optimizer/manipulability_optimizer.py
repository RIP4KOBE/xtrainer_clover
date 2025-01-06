import torch
from omegaconf import OmegaConf
from ModelTrain.dp.optimizer.optimizer import Optimizer
from ModelTrain.manipulability.bimanual_manip_learning import GMRModel, BimanualRobot

class ManipulabilityOptimizer(Optimizer):
    def __init__(self):
        cfg = OmegaConf.load("/home/zhuoli/dobot_xtrainer/ModelTrain/config/optimizer.yaml")
        self.device = cfg.device

        # Optimizer configurations
        self.scale = cfg.scale
        self.scale_type = cfg.scale_type
        self.clip_grad_by_value = cfg.clip_grad_by_value
        self.action_horizon = cfg.action_horizon

        # Manipulability-related configurations
        self.task_dim = cfg.task_dim  # Task space dimension, typically 3 for 3D space
        self.scaling = cfg.scaling  # Whether to apply scaling in Jacobian computation
        self.manipulability_weight = cfg.manipulability_weight  # Weight for manipulability loss
        self.gmm_ckpt_path = cfg.gmm_ckpt_path
        self.robot_urdf_path = cfg.robot_urdf_path

        self.gmr_model = GMRModel(self.gmm_ckpt_path)
        self.robot = BimanualRobot(self.robot_urdf_path)

    def optimize(self, x: torch.Tensor, M_t_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the manipulability loss for the current time step t.

        Args:
            x: The denoised signal at the current step (requires gradient).
            t: The current time step.
            data: A dictionary containing additional scene data.

        Returns:
            The manipulability loss value for the current time step.
        """

        action = x[0]
        loss_t = 0.0

        for i in range(action.shape[0]):

            # Extract joint angles for the left and right arms from x at time step t
            q_left = action[i,:6]
            q_right = action[i, 7:13]

            # Compute the Jacobians for left and right arms
            J_left, J_right = self.robot.compute_bimanual_jacobian(q_left, q_right)

            JR = self.robot.compute_bimanual_relative_jacobian(q_left, q_right, J_left, J_right,
                                                               task_dim=self.task_dim, scaling=False)
            M_t = self.robot.compute_bimanual_relative_manipulability(JR)

            log_M = self.gmr_model.logmap(M_t_pred, M_t)

            # Compute Frobenius norm of logarithmic map
            loss_t = loss_t + torch.norm(log_M, p='fro').pow(2)  # (B,)

            print("loss_t", loss_t)

        return (-1.0) * loss_t # Return the negative loss for gradient-based optimization


    def gradient(self, x: torch.Tensor, variance: torch.Tensor, M_t_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the manipulability loss for the current time step t.

        Args:
            x: The denoised signal at the current step.
            t: The current time step.
            variance: Variance at the current step.

        Returns:
            The computed gradient for manipulability optimization.
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            obj = self.optimize(x_in, M_t_pred)
            grad = torch.autograd.grad(obj, x_in)[0]

            # Clip the gradient by value
            # grad = grad / (torch.norm(grad) + 1e-8)  # 归一化梯度
            # grad = torch.clip(grad, -1.0, 1.0)  # 裁剪梯度
            grad = torch.clip(grad, **self.clip_grad_by_value)

            # Scale the gradient based on variance
            if self.scale_type == 'normal':
                grad = self.scale * grad * variance
            elif self.scale_type == 'div_var':
                grad = self.scale * grad
            else:
                raise Exception('Unsupported scale type!')

            return grad