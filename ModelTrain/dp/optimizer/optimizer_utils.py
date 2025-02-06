from scipy.linalg import logm
import numpy as np
import torch


def matrix_log(M: torch.Tensor) -> torch.Tensor:
    """
    使用特征值分解计算矩阵对数。
    """
    eigvals, eigvecs = torch.linalg.eig(M)  # 特征值和特征向量（实数）
    log_eigvals = torch.log(eigvals)

    return eigvecs @ torch.diag(log_eigvals) @ eigvecs.T


def log_map_frobenius_norm(M: torch.Tensor, M_hat: torch.Tensor) -> torch.Tensor:
    """
    Compute the Frobenius norm of the logarithmic map between two SPD matrices.

    Args:
        M: The current manipulability ellipsoid (3, 3).
        M_hat: The target manipulability ellipsoid (3, 3).

    Returns:
        The Frobenius norm of log_M(M_hat) for each batch (B,).
    """

    # M_inv = torch.linalg.inv(M)  # (3, 3)
    # log_M = matrix_log(M_inv @ M_hat)

    log_M = matrix_log(torch.linalg.solve(M, M_hat))
    frobenius_norm_sq = torch.norm(log_M, p='fro').pow(2)

    return frobenius_norm_sq

if __name__ == "__main__":
    # 测试用例
    # 示例输入：两个 3x3 的对称正定矩阵
    M = torch.tensor([[2.0, 0.5, 0.0],
                      [0.5, 2.0, 0.2],
                      [0.0, 0.2, 1.5]])
    M_hat = torch.tensor([[1.5, 0.3, 0.0],
                          [0.3, 1.8, 0.1],
                          [0.0, 0.1, 1.2]])

    # 转换为对称正定矩阵
    M = M @ M.T
    M_hat = M_hat @ M_hat.T

    # 转移到 CUDA（可选）
    M = M.to('cuda')
    M_hat = M_hat.to('cuda')

    # 计算 Frobenius 范数的平方
    result = log_map_frobenius_norm(M, M_hat)
    print("Frobenius norm squared:", result)